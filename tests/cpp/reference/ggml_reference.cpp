#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "gguf.h"
#include <cuda_runtime_api.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct GgufDeleter {
    void operator()(gguf_context * ctx) const {
        if (ctx != nullptr) {
            gguf_free(ctx);
        }
    }
};

struct GgmlDeleter {
    void operator()(ggml_context * ctx) const {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct BackendDeleter {
    void operator()(ggml_backend * backend) const {
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
    }
};

struct BackendBufferDeleter {
    void operator()(ggml_backend_buffer * buffer) const {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
    }
};

using GgufPtr = std::unique_ptr<gguf_context, GgufDeleter>;
using GgmlPtr = std::unique_ptr<ggml_context, GgmlDeleter>;
using BackendPtr = std::unique_ptr<ggml_backend, BackendDeleter>;
using BackendBufferPtr = std::unique_ptr<ggml_backend_buffer, BackendBufferDeleter>;

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error(message);
}

void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) fail(cudaGetErrorString(status));
}

struct TimingEvent {
    cudaEvent_t handle{};
    TimingEvent() { check_cuda(cudaEventCreate(&handle)); }
    ~TimingEvent() { cudaEventDestroy(handle); }
    TimingEvent(const TimingEvent &) = delete;
    TimingEvent & operator=(const TimingEvent &) = delete;

    void record(ggml_backend_t backend) {
        // ggml's public event constructor disables timing. Wrap our timing-enabled
        // CUDA event using the pinned event ABI to record on the backend's stream.
        ggml_backend_event event{ggml_backend_get_device(backend), handle};
        ggml_backend_event_record(&event, backend);
    }
};

uint64_t parse_unsigned(const char * text, const char * name, bool allow_zero = false) {
    const std::string value(text);
    uint64_t result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result, 10);
    if (value.empty() || parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
        (!allow_zero && result == 0)) {
        fail(std::string(name) + (allow_zero ? " must be a nonnegative integer" : " must be a positive integer"));
    }
    return result;
}

std::size_t checked_multiply(uint64_t a, uint64_t b, const char * label) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        fail(std::string(label) + " is too large");
    }
    return static_cast<std::size_t>(a * b);
}

bool is_little_endian() {
    const uint16_t value = 1;
    return *reinterpret_cast<const uint8_t *>(&value) == 1;
}

void read_exact_at(const std::filesystem::path & path,
                   uint64_t offset,
                   void * destination,
                   std::size_t size) {
    if (offset > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
        fail("GGUF tensor data offset is too large for this platform");
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        fail("cannot open GGUF file: " + path.string());
    }
    input.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!input) {
        fail("cannot seek to tensor data in GGUF file");
    }
    if (size > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        fail("tensor data is too large for this platform");
    }
    input.read(static_cast<char *>(destination), static_cast<std::streamsize>(size));
    if (input.gcount() != static_cast<std::streamsize>(size)) {
        fail("truncated GGUF tensor data: expected " + std::to_string(size) + " bytes");
    }
}

std::vector<float> read_stdin_f32(std::size_t expected_values) {
    const std::size_t expected_bytes = checked_multiply(expected_values, sizeof(float), "input byte count");
    if (expected_bytes > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        fail("input is too large for this platform's stream interface");
    }
    std::vector<float> values(expected_values);
    std::cin.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(expected_bytes));
    if (std::cin.gcount() != static_cast<std::streamsize>(expected_bytes)) {
        fail("stdin input size mismatch: expected exactly " + std::to_string(expected_bytes) +
             " bytes of row-major [M,K] float32 little-endian input, received fewer bytes");
    }
    char extra = 0;
    std::cin.read(&extra, 1);
    if (std::cin.gcount() == 1) {
        fail("stdin input size mismatch: expected exactly " + std::to_string(expected_bytes) +
             " bytes, received trailing bytes");
    }
    if (std::cin.bad()) {
        fail("failed while reading stdin");
    }
    return values;
}

std::string json_escape(const std::string & value) {
    std::ostringstream out;
    for (const unsigned char c : value) {
        switch (c) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) {
                    static constexpr char hex[] = "0123456789abcdef";
                    out << "\\u00" << hex[c >> 4] << hex[c & 0xf];
                } else {
                    out << static_cast<char>(c);
                }
        }
    }
    return out.str();
}

std::string array_descriptor(const char * name,
                             uint64_t shape0,
                             uint64_t shape1,
                             std::size_t element_count) {
    const std::size_t nbytes = checked_multiply(element_count, sizeof(float), "output array size");
    std::ostringstream out;
    out << "{\"name\":\"" << name << "\",\"dtype\":\"float32\",\"shape\":["
        << shape0 << "," << shape1 << "],\"nbytes\":" << nbytes << "}";
    return out.str();
}

std::string make_header(const std::string & tensor_name,
                        const std::string & tensor_type,
                        int64_t k,
                        int64_t n,
                        std::size_t row_bytes,
                        std::size_t tensor_bytes,
                        uint64_t absolute_offset,
                        int64_t dequant_rows,
                        int64_t m,
                        std::size_t dequant_count,
                        std::size_t output_count,
                        const std::string & backend_name) {
    std::ostringstream out;
    out << "{\"schema_version\":1"
        << ",\"op\":\"matmul\""
        << ",\"tensor_name\":\"" << json_escape(tensor_name) << "\""
        << ",\"ggml_type\":\"" << json_escape(tensor_type) << "\""
        << ",\"shape_ggml_ne_order\":[" << k << "," << n << "]"
        << ",\"row_bytes\":" << row_bytes
        << ",\"tensor_bytes\":" << tensor_bytes
        << ",\"data_offset_absolute\":" << absolute_offset
        << ",\"ggml_backend\":\"" << json_escape(backend_name) << "\""
        << ",\"arrays\":["
        << array_descriptor("dequant_rows", static_cast<uint64_t>(dequant_rows), static_cast<uint64_t>(k), dequant_count) << ","
        << array_descriptor("ggml_dequant_f64", static_cast<uint64_t>(m), static_cast<uint64_t>(n), output_count) << ","
        << array_descriptor("ggml_cuda", static_cast<uint64_t>(m), static_cast<uint64_t>(n), output_count)
        << "]}";
    return out.str();
}

void write_bytes(const void * data, std::size_t size, const char * label) {
    if (size > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        fail(std::string(label) + " is too large for this platform's stream interface");
    }
    std::cout.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
    if (!std::cout) {
        fail("failed while writing " + std::string(label) + " to stdout");
    }
}

void write_npy_header(const std::string & dtype, const std::string & shape) {
    std::string header = "{'descr': '" + dtype + "', 'fortran_order': False, 'shape': " + shape + ", }";
    header.append((16 - ((10 + header.size() + 1) % 16)) % 16, ' ');
    header.push_back('\n');
    if (header.size() > std::numeric_limits<uint16_t>::max()) {
        fail("NumPy header is too large");
    }
    static constexpr char magic[] = "\x93NUMPY";
    write_bytes(magic, 6, "NumPy magic");
    const uint8_t version_and_length[] = {1, 0, static_cast<uint8_t>(header.size() & 0xffU),
                                          static_cast<uint8_t>((header.size() >> 8) & 0xffU)};
    write_bytes(version_and_length, sizeof(version_and_length), "NumPy version and header length");
    write_bytes(header.data(), header.size(), "NumPy header");
}

void write_npy_f32(const std::vector<float> & values, uint64_t rows, uint64_t cols) {
    if (checked_multiply(rows, cols, "NumPy array elements") != values.size()) {
        fail("NumPy array shape does not match data size");
    }
    write_npy_header("<f4", "(" + std::to_string(rows) + ", " + std::to_string(cols) + ")");
    write_bytes(values.data(), checked_multiply(values.size(), sizeof(float), "NumPy array bytes"), "NumPy array");
}

void write_result(const std::string & header, uint64_t m, uint64_t k, uint64_t n, uint64_t rows,
                  const std::vector<float> & dequant_rows,
                  const std::vector<float> & y_dequant_f64,
                  const std::vector<float> & y_ggml_cuda) {
    write_npy_header("|u1", "(" + std::to_string(header.size()) + ",)");
    write_bytes(header.data(), header.size(), "JSON metadata array");
    write_npy_f32(dequant_rows, rows, k);
    write_npy_f32(y_dequant_f64, m, n);
    write_npy_f32(y_ggml_cuda, m, n);
    std::cout.flush();
    if (!std::cout) {
        fail("failed while flushing NumPy arrays to stdout");
    }
}

void compute_ggml_cuda(const std::vector<uint8_t> & weights,
                      const std::vector<float> & input,
                      int64_t k,
                      int64_t n,
                      int64_t m,
                      bool benchmark,
                      int benchmark_iters,
                      std::vector<float> & output) {
    ggml_init_params params{};
    params.mem_size = 1U << 20;
    params.no_alloc = true;
    GgmlPtr ctx(ggml_init(params));
    if (!ctx) {
        fail("ggml_init failed");
    }

    ggml_tensor * w = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_IQ3_S, k, n);
    ggml_tensor * x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, m);
    ggml_tensor * y = ggml_mul_mat(ctx.get(), w, x);
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, y);

    BackendPtr backend(ggml_backend_cuda_init(0));
    if (!backend) {
        fail("ggml_backend_cuda_init(0) failed");
    }
    if (!ggml_backend_supports_op(backend.get(), y)) {
        fail("ggml CUDA backend does not support this IQ3_S matmul");
    }

    BackendBufferPtr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!buffer) {
        fail("ggml_backend_alloc_ctx_tensors failed");
    }
    ggml_backend_tensor_set(w, weights.data(), 0, weights.size());
    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));

    auto compute_async = [&]() {
        const ggml_status status = ggml_backend_graph_compute_async(backend.get(), graph);
        if (status != GGML_STATUS_SUCCESS) {
            fail("ggml CUDA graph compute failed with status " + std::to_string(status));
        }
    };
    auto compute = [&]() {
        compute_async();
        ggml_backend_synchronize(backend.get());
    };
    if (benchmark) {
        constexpr int warmup_iters = 5;
        for (int i = 0; i < warmup_iters; ++i) compute();
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < benchmark_iters; ++i) compute();
        const auto stop = std::chrono::steady_clock::now();
        const double mean_us = std::chrono::duration<double, std::micro>(stop - start).count() / benchmark_iters;
        for (int i = 0; i < warmup_iters; ++i) compute_async();
        ggml_backend_synchronize(backend.get());
        const auto batch_start = std::chrono::steady_clock::now();
        for (int i = 0; i < benchmark_iters; ++i) compute_async();
        ggml_backend_synchronize(backend.get());
        const auto batch_stop = std::chrono::steady_clock::now();
        const double batched_us = std::chrono::duration<double, std::micro>(batch_stop - batch_start).count() / benchmark_iters;
        TimingEvent event_start, event_stop;
        event_start.record(backend.get());
        for (int i = 0; i < benchmark_iters; ++i) compute_async();
        event_stop.record(backend.get());
        check_cuda(cudaEventSynchronize(event_stop.handle));
        float elapsed_ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&elapsed_ms, event_start.handle, event_stop.handle));
        std::cerr << "benchmark backend=cuda op=matmul m=" << m << " n=" << n << " k=" << k
                  << " warmup=" << warmup_iters << " iterations=" << benchmark_iters
                  << " mean_us=" << mean_us << " batched_us=" << batched_us
                  << " event_us=" << elapsed_ms * 1000.0 / benchmark_iters << '\n';
        return;
    }
    compute();
    output.resize(checked_multiply(static_cast<uint64_t>(m),
                                   checked_multiply(static_cast<uint64_t>(n), 1, "output elements"),
                                   "output elements"));
    ggml_backend_tensor_get(y, output.data(), 0, output.size() * sizeof(float));
}

int run(int argc, char ** argv) {
    const char * program = argc > 0 ? argv[0] : "mambo_ggml_reference";
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cerr << "Usage: " << program
                  << " --op matmul --model PATH --weight TENSOR_NAME --m M"
                     " [--dequant-rows N] [--benchmark] [--benchmark-iters N]\n"
                     "Reads exactly M*K little-endian float32 values from stdin.\n"
                     "Normally writes four NumPy .npy arrays to stdout; --benchmark writes timing to stderr only.\n";
        return 0;
    }
    if (!is_little_endian()) {
        fail("the reference binary protocol requires a little-endian host");
    }

    std::optional<std::string> op;
    std::optional<std::filesystem::path> model_path;
    std::optional<std::string> tensor_name;
    std::optional<uint64_t> m_argument;
    std::optional<uint64_t> requested_dequant_rows_argument;
    uint64_t benchmark_iters_u64 = 20;
    bool benchmark_iters_was_set = false;
    bool benchmark = false;
    auto require_value = [&](int & index, const char * flag) -> const char * {
        if (index + 1 >= argc) {
            fail(std::string("missing value for ") + flag);
        }
        return argv[++index];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string flag(argv[i]);
        if (flag == "--op") {
            if (op) fail("--op specified more than once");
            op = require_value(i, "--op");
        } else if (flag == "--model") {
            if (model_path) fail("--model specified more than once");
            model_path = require_value(i, "--model");
        } else if (flag == "--weight") {
            if (tensor_name) fail("--weight specified more than once");
            tensor_name = require_value(i, "--weight");
        } else if (flag == "--m") {
            if (m_argument) fail("--m specified more than once");
            m_argument = parse_unsigned(require_value(i, "--m"), "M");
        } else if (flag == "--dequant-rows") {
            if (requested_dequant_rows_argument) fail("--dequant-rows specified more than once");
            requested_dequant_rows_argument = parse_unsigned(require_value(i, "--dequant-rows"), "dequant-rows", true);
        } else if (flag == "--benchmark") {
            if (benchmark) fail("--benchmark specified more than once");
            benchmark = true;
        } else if (flag == "--benchmark-iters") {
            if (benchmark_iters_was_set) fail("--benchmark-iters specified more than once");
            benchmark_iters_u64 = parse_unsigned(require_value(i, "--benchmark-iters"), "benchmark-iters");
            benchmark_iters_was_set = true;
        } else {
            fail("unknown argument: " + flag);
        }
    }
    if (!op || !model_path || !tensor_name || !m_argument) {
        fail(std::string("missing required argument; usage: ") + program +
             " --op matmul --model PATH --weight TENSOR_NAME --m M [--dequant-rows N] [--benchmark] [--benchmark-iters N]");
    }
    if (*op != "matmul") {
        fail("unsupported op: " + *op + " (supported op: matmul)");
    }
    const uint64_t m_u64 = *m_argument;
    if (m_u64 > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        benchmark_iters_u64 > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        fail("M or benchmark-iters exceed supported integer range");
    }
    if (benchmark_iters_was_set && !benchmark) fail("--benchmark-iters requires --benchmark");
    const int64_t m = static_cast<int64_t>(m_u64);
    const int benchmark_iters = static_cast<int>(benchmark_iters_u64);

    std::error_code ec;
    const uint64_t model_bytes = std::filesystem::file_size(*model_path, ec);
    if (ec || model_bytes < 4) {
        fail("cannot stat GGUF model or file is too small: " + model_path->string());
    }

    gguf_init_params gguf_params{};
    gguf_params.no_alloc = true;
    GgufPtr gguf(gguf_init_from_file(model_path->string().c_str(), gguf_params));
    if (!gguf) {
        fail("gguf_init_from_file failed: " + model_path->string());
    }
    const int64_t tensor_id = gguf_find_tensor(gguf.get(), tensor_name->c_str());
    if (tensor_id < 0) {
        fail("tensor not found in GGUF: " + *tensor_name);
    }
    const ggml_type tensor_type = gguf_get_tensor_type(gguf.get(), tensor_id);
    if (tensor_type != GGML_TYPE_IQ3_S) {
        fail("unsupported ggml tensor type for matmul: " +
             std::string(ggml_type_name(tensor_type)) + " (currently supported type: iq3_s)");
    }

    const int64_t * ne = gguf_get_tensor_ne(gguf.get(), tensor_id);
    const int64_t k = ne[0];
    const int64_t n = ne[1];
    if (k <= 0 || n <= 0 || ne[2] != 1 || ne[3] != 1) {
        fail("tensor must be a nonempty rank-2 GGUF tensor");
    }
    const int64_t block_elements = ggml_blck_size(tensor_type);
    const std::size_t block_bytes = ggml_type_size(tensor_type);
    if (k % block_elements != 0) {
        fail("IQ3_S tensor row width is not divisible by the quantization block size");
    }
    const std::size_t row_bytes = ggml_row_size(tensor_type, k);
    const std::size_t expected_tensor_bytes = checked_multiply(static_cast<uint64_t>(row_bytes),
                                                                static_cast<uint64_t>(n),
                                                                "tensor byte count");
    const std::size_t tensor_bytes = gguf_get_tensor_size(gguf.get(), tensor_id);
    if (tensor_bytes != expected_tensor_bytes) {
        fail("GGUF tensor byte size does not match its dimensions and IQ3_S row size");
    }
    const uint64_t requested_dequant_rows = requested_dequant_rows_argument.value_or(
        std::min<uint64_t>(4, static_cast<uint64_t>(n)));
    if (requested_dequant_rows > static_cast<uint64_t>(n)) {
        fail("dequant-rows cannot exceed tensor output width");
    }

    const uint64_t data_offset = gguf_get_data_offset(gguf.get());
    const uint64_t tensor_relative_offset = gguf_get_tensor_offset(gguf.get(), tensor_id);
    if (tensor_relative_offset > std::numeric_limits<uint64_t>::max() - data_offset) {
        fail("absolute tensor offset overflow");
    }
    const uint64_t absolute_offset = data_offset + tensor_relative_offset;
    if (absolute_offset > model_bytes || tensor_bytes > model_bytes - absolute_offset) {
        fail("tensor data range extends past the GGUF file size");
    }
    if (tensor_bytes > std::numeric_limits<std::size_t>::max()) {
        fail("tensor data is too large for host address space");
    }
    const std::size_t input_values = checked_multiply(m_u64, static_cast<uint64_t>(k), "input element count");
    const std::size_t output_values = checked_multiply(m_u64, static_cast<uint64_t>(n), "output element count");

    std::vector<uint8_t> weights(tensor_bytes);
    read_exact_at(*model_path, absolute_offset, weights.data(), weights.size());
    std::vector<float> input = read_stdin_f32(input_values);
    if (!std::all_of(input.begin(), input.end(), [](float x) { return std::isfinite(x); })) {
        fail("stdin input contains a NaN or infinity");
    }
    std::vector<float> y_ggml;
    compute_ggml_cuda(weights, input, k, n, m, benchmark, benchmark_iters, y_ggml);
    if (benchmark) return 0;

    const ggml_type_traits * traits = ggml_get_type_traits(tensor_type);
    if (traits == nullptr || traits->to_float == nullptr || traits->blck_size != block_elements ||
        traits->type_size != block_bytes) {
        fail("ggml tensor dequantization traits are unavailable or inconsistent for iq3_s");
    }
    std::vector<float> dequant_rows(checked_multiply(requested_dequant_rows, static_cast<uint64_t>(k),
                                                      "dequant row elements"));
    std::vector<float> dequantized_row(static_cast<std::size_t>(k));
    std::vector<float> y_f64(output_values, 0.0f);
    const std::size_t m_size = static_cast<std::size_t>(m);
    const std::size_t k_size = static_cast<std::size_t>(k);
    const std::size_t n_size = static_cast<std::size_t>(n);
    const std::size_t count_rows = static_cast<std::size_t>(requested_dequant_rows);
    for (std::size_t out_row = 0; out_row < n_size; ++out_row) {
        const uint8_t * qrow = weights.data() + out_row * row_bytes;
        traits->to_float(qrow, dequantized_row.data(), k);
        if (out_row < count_rows) {
            std::memcpy(dequant_rows.data() + out_row * k_size,
                        dequantized_row.data(), k_size * sizeof(float));
        }
        for (std::size_t token = 0; token < m_size; ++token) {
            const float * xrow = input.data() + token * k_size;
            double sum = 0.0;
            for (std::size_t col = 0; col < k_size; ++col) {
                sum += static_cast<double>(dequantized_row[col]) * static_cast<double>(xrow[col]);
            }
            y_f64[token * n_size + out_row] = static_cast<float>(sum);
        }
    }

    const std::string header = make_header(*tensor_name, ggml_type_name(tensor_type), k, n,
                                           row_bytes, tensor_bytes, absolute_offset,
                                           static_cast<int64_t>(requested_dequant_rows), m,
                                           dequant_rows.size(), output_values, "cuda");
    write_result(header, m_u64, static_cast<uint64_t>(k), static_cast<uint64_t>(n),
                 requested_dequant_rows, dequant_rows, y_f64, y_ggml);
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception & error) {
        std::cerr << "mambo_ggml_reference: " << error.what() << '\n';
        return 1;
    }
}
