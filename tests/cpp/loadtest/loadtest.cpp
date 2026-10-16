#include <cuda.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

std::string cuda_error_message(CUresult result, const char *expression) {
    const char *name = nullptr;
    const char *description = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &description);

    std::string message = expression;
    message += " failed";
    if (name != nullptr) {
        message += " (";
        message += name;
        message += ")";
    }
    if (description != nullptr) {
        message += ": ";
        message += description;
    }
    return message;
}

void check_cuda(CUresult result, const char *expression) {
    if (result != CUDA_SUCCESS) {
        throw std::runtime_error(cuda_error_message(result, expression));
    }
}

#define CUDA_CHECK(expression) check_cuda((expression), #expression)

uint64_t parse_positive_integer(const char *text, const char *label) {
    std::size_t parsed_chars = 0;
    const std::string value(text);
    unsigned long long parsed = 0;
    try {
        parsed = std::stoull(value, &parsed_chars, 10);
    } catch (const std::exception &) {
        throw std::invalid_argument(std::string(label) + " must be a positive integer");
    }
    if (parsed_chars != value.size() || parsed == 0) {
        throw std::invalid_argument(std::string(label) + " must be a positive integer");
    }
    return static_cast<uint64_t>(parsed);
}

struct CudaResources {
    CUcontext context = nullptr;
    CUmodule module = nullptr;
    CUdeviceptr a = 0;
    CUdeviceptr b = 0;
    CUdeviceptr out = 0;
    std::vector<CUdeviceptr> extra_allocations;

    ~CudaResources() {
        if (out != 0) cuMemFree(out);
        if (b != 0) cuMemFree(b);
        if (a != 0) cuMemFree(a);
        for (auto it = extra_allocations.rbegin(); it != extra_allocations.rend(); ++it) {
            if (*it != 0) cuMemFree(*it);
        }
        if (module != nullptr) cuModuleUnload(module);
        if (context != nullptr) cuCtxDestroy(context);
    }
};

using Json = nlohmann::json;

struct ArrayAbi {
    std::string name;
};

struct AbiManifest {
    std::string cubin_path;
    std::string symbol;
    std::vector<ArrayAbi> arrays;
    std::vector<std::pair<std::string, std::string>> parameters;
    std::array<unsigned int, 3> grid{};
    std::array<unsigned int, 3> block{};
    std::array<std::string, 2> inputs{};
    std::string output;
    int32_t length = 0;
};

[[noreturn]] void invalid_abi(const std::string &message) {
    throw std::invalid_argument("invalid ABI JSON: " + message);
}

void require_exact_keys(const Json &object, const std::set<std::string> &expected,
                        const std::string &where) {
    if (!object.is_object()) invalid_abi(where + " must be an object");
    std::set<std::string> actual;
    for (auto it = object.begin(); it != object.end(); ++it) actual.insert(it.key());
    if (actual != expected) {
        std::vector<std::string> extra;
        std::vector<std::string> missing;
        std::set_difference(actual.begin(), actual.end(), expected.begin(), expected.end(),
                            std::back_inserter(extra));
        std::set_difference(expected.begin(), expected.end(), actual.begin(), actual.end(),
                            std::back_inserter(missing));
        std::string message = where + " has unsupported fields";
        if (!extra.empty()) message += "; unknown=" + extra.front();
        if (!missing.empty()) message += "; missing=" + missing.front();
        invalid_abi(message);
    }
}

int64_t json_integer(const Json &value, const std::string &where) {
    if (!value.is_number_integer() && !value.is_number_unsigned()) {
        invalid_abi(where + " must be an integer");
    }
    try {
        return value.get<int64_t>();
    } catch (const Json::exception &) {
        invalid_abi(where + " is outside the supported integer range");
    }
}

std::array<unsigned int, 3> parse_extent(const Json &value, const std::string &where) {
    if (!value.is_array() || value.size() != 3) {
        invalid_abi(where + " must contain exactly three dimensions");
    }
    std::array<unsigned int, 3> result{};
    for (size_t i = 0; i < result.size(); ++i) {
        const int64_t dimension = json_integer(value[i], where + "[" + std::to_string(i) + "]");
        if (dimension <= 0 || static_cast<uint64_t>(dimension) > std::numeric_limits<unsigned int>::max()) {
            invalid_abi(where + " dimensions must be positive unsigned integers");
        }
        result[i] = static_cast<unsigned int>(dimension);
    }
    return result;
}

std::vector<Json> constraint_vector(const Json &value, const std::string &where) {
    if (value.is_array()) {
        if (value.size() != 1) invalid_abi(where + " must describe one dimension");
        return {value[0]};
    }
    return {value};
}

void require_one(const Json &value, const std::string &where, int64_t expected) {
    auto values = constraint_vector(value, where);
    if (!values[0].is_number_integer() || json_integer(values[0], where) != expected) {
        invalid_abi(where + " uses an unsupported value");
    }
}

void require_one_or_null(const Json &value, const std::string &where, int64_t expected) {
    auto values = constraint_vector(value, where);
    if (!values[0].is_null() &&
        (!values[0].is_number_integer() || json_integer(values[0], where) != expected)) {
        invalid_abi(where + " uses an unsupported value");
    }
}

void validate_array_constraint(const Json &constraint, const std::string &name) {
    const std::set<std::string> keys = {
        "name", "kind", "dtype", "ndim", "index_dtype", "stride_lower_bound_incl",
        "alias_groups", "may_alias_internally", "stride_constant", "shape_constant",
        "stride_divisible_by", "shape_divisible_by", "base_addr_divisible_by"};
    require_exact_keys(constraint, keys, "constraint " + name);
    if (constraint.at("kind") != "array" || constraint.at("dtype") != "float32" ||
        constraint.at("index_dtype") != "int32" || json_integer(constraint.at("ndim"), "ndim") != 1) {
        invalid_abi("only one-dimensional float32 arrays with int32 indexing are supported (" + name + ")");
    }
    const Json &groups = constraint.at("alias_groups");
    if (!groups.is_array() || !groups.empty() || constraint.at("may_alias_internally") != false) {
        invalid_abi("array aliasing constraints are unsupported (" + name + ")");
    }
    // cuTile removes a lower bound made redundant by an exact stride constant.
    require_one_or_null(constraint.at("stride_lower_bound_incl"), name + ".stride_lower_bound_incl", 1);
    require_one(constraint.at("stride_constant"), name + ".stride_constant", 1);
    require_one(constraint.at("stride_divisible_by"), name + ".stride_divisible_by", 1);
    require_one(constraint.at("shape_divisible_by"), name + ".shape_divisible_by", 1);
    require_one(constraint.at("base_addr_divisible_by"), name + ".base_addr_divisible_by", 1);

    auto shape_constants = constraint_vector(constraint.at("shape_constant"), name + ".shape_constant");
    if (!shape_constants[0].is_null()) {
        invalid_abi("fixed array shapes are unsupported (" + name + ")");
    }
}

AbiManifest load_abi_manifest(const std::string &path) {
    Json document;
    try {
        std::ifstream input(path);
        if (!input) invalid_abi("cannot open " + path);
        input >> document;
    } catch (const Json::exception &error) {
        invalid_abi(std::string("malformed JSON: ") + error.what());
    }
    if (!document.is_object()) invalid_abi("root must be an object");
    if (!document.contains("schema_version") || json_integer(document.at("schema_version"), "schema_version") != 1) {
        invalid_abi("schema_version 1 is required");
    }
    if (document.contains("constants")) {
        invalid_abi("compile-time constants must be omitted from runtime ABI JSON");
    }
    if (!document.contains("calling_convention") || document.at("calling_convention") != "cutile_python_v2") {
        invalid_abi("unsupported calling convention");
    }
    if (!document.contains("name") || !document.at("name").is_string() ||
        !document.contains("symbol") || !document.at("symbol").is_string() ||
        document.at("name") != document.at("symbol")) {
        invalid_abi("name and symbol must be matching strings");
    }
    if (!document.contains("cubin_path") || !document.at("cubin_path").is_string()) {
        invalid_abi("cubin_path must be a string");
    }
    if (!document.contains("gpu_code") || !document.at("gpu_code").is_string() ||
        document.at("gpu_code").get<std::string>().rfind("sm_", 0) != 0) {
        invalid_abi("gpu_code must use the sm_NN form");
    }
    if (!document.contains("toolchain") || !document.at("toolchain").is_object() ||
        !document.at("toolchain").contains("python") || !document.at("toolchain").contains("cuda_tile")) {
        invalid_abi("toolchain metadata is missing");
    }

    AbiManifest manifest;
    manifest.cubin_path = document.at("cubin_path").get<std::string>();
    manifest.symbol = document.at("symbol").get<std::string>();
    if (manifest.symbol.empty()) invalid_abi("symbol must not be empty");

    if (!document.contains("launch") || !document.at("launch").is_object() ||
        !document.at("launch").contains("grid") || !document.at("launch").contains("block")) {
        invalid_abi("launch grid and block are required");
    }
    const Json &launch = document.at("launch");
    manifest.grid = parse_extent(launch.at("grid"), "launch.grid");
    manifest.block = parse_extent(launch.at("block"), "launch.block");
    if (!document.contains("grid") || !document.contains("block") ||
        parse_extent(document.at("grid"), "grid") != manifest.grid ||
        parse_extent(document.at("block"), "block") != manifest.block) {
        invalid_abi("legacy grid/block fields must match launch metadata");
    }
    const uint64_t block_threads = static_cast<uint64_t>(manifest.block[0]) * manifest.block[1] * manifest.block[2];
    if (block_threads > 1024) invalid_abi("block contains more than 1024 threads");

    if (!document.contains("constraints") || !document.at("constraints").is_array()) {
        invalid_abi("constraints must be an array");
    }
    std::set<std::string> constraint_names;
    for (const Json &constraint : document.at("constraints")) {
        if (!constraint.is_object() || !constraint.contains("name") || !constraint.at("name").is_string()) {
            invalid_abi("each constraint must have a string name");
        }
        const std::string name = constraint.at("name").get<std::string>();
        if (name.empty() || !constraint_names.insert(name).second) invalid_abi("duplicate or empty constraint name");
        validate_array_constraint(constraint, name);
        manifest.arrays.push_back({name});
    }
    if (manifest.arrays.size() != 3) invalid_abi("the P0 vector-add probe requires three arrays");

    if (!document.contains("parameters") || !document.at("parameters").is_array()) {
        invalid_abi("parameters must be an array");
    }
    std::vector<std::pair<std::string, std::string>> expected_parameters;
    for (const ArrayAbi &array : manifest.arrays) {
        expected_parameters.emplace_back(array.name, "float32*");
        expected_parameters.emplace_back(array.name + "_shape_0", "int32");
        expected_parameters.emplace_back(array.name + "_stride_0", "int32");
    }
    const Json &parameters = document.at("parameters");
    if (parameters.size() != expected_parameters.size()) invalid_abi("parameter count does not match array constraints");
    for (size_t i = 0; i < parameters.size(); ++i) {
        const Json &parameter = parameters[i];
        require_exact_keys(parameter, {"name", "type"}, "parameter");
        if (!parameter.at("name").is_string() || !parameter.at("type").is_string()) {
            invalid_abi("parameter name and type must be strings");
        }
        const auto actual = std::make_pair(parameter.at("name").get<std::string>(),
                                           parameter.at("type").get<std::string>());
        if (actual != expected_parameters[i]) invalid_abi("unsupported or reordered ABI parameter at index " + std::to_string(i));
        manifest.parameters.push_back(actual);
    }

    if (!document.contains("probe") || !document.at("probe").is_object()) invalid_abi("probe metadata is required");
    const Json &probe = document.at("probe");
    require_exact_keys(probe, {"operation", "inputs", "output", "length"}, "probe");
    if (probe.at("operation") != "vector_add" || !probe.at("inputs").is_array() || probe.at("inputs").size() != 2 ||
        !probe.at("inputs")[0].is_string() || !probe.at("inputs")[1].is_string() || !probe.at("output").is_string()) {
        invalid_abi("only the vector_add probe operation is supported");
    }
    manifest.inputs = {probe.at("inputs")[0].get<std::string>(), probe.at("inputs")[1].get<std::string>()};
    manifest.output = probe.at("output").get<std::string>();
    if (!probe.at("length").is_number_integer()) invalid_abi("probe.length must be a positive integer");
    const int64_t length = json_integer(probe.at("length"), "probe.length");
    if (length <= 0 || length > std::numeric_limits<int32_t>::max()) invalid_abi("probe.length is outside int32 range");
    manifest.length = static_cast<int32_t>(length);

    std::set<std::string> expected_arrays = {manifest.inputs[0], manifest.inputs[1], manifest.output};
    if (expected_arrays.size() != 3 || constraint_names != expected_arrays ||
        manifest.inputs[0] == manifest.inputs[1] || manifest.output == manifest.inputs[0] || manifest.output == manifest.inputs[1]) {
        invalid_abi("probe inputs/output do not match distinct array constraints");
    }
    return manifest;
}

struct RuntimeArgument {
    CUdeviceptr pointer = 0;
    int32_t integer = 0;
};

int run_abi_probe(const AbiManifest &manifest) {
    const int32_t n = manifest.length;
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    std::map<std::string, std::vector<float>> host_arrays;
    std::map<std::string, CUdeviceptr> device_arrays;
    for (const ArrayAbi &array : manifest.arrays) {
        host_arrays.emplace(array.name, std::vector<float>(n, 0.0f));
    }
    for (int32_t i = 0; i < n; ++i) {
        host_arrays.at(manifest.inputs[0])[i] = static_cast<float>(i) * 0.25f;
        host_arrays.at(manifest.inputs[1])[i] = static_cast<float>(n - i) * 0.5f;
    }

    CUDA_CHECK(cuInit(0));
    CUdevice device = 0;
    CUDA_CHECK(cuDeviceGet(&device, 0));
    CudaResources cuda;
#if CUDA_VERSION >= 13000
    CUctxCreateParams context_params{};
    CUDA_CHECK(cuCtxCreate(&cuda.context, &context_params, 0, device));
#else
    CUDA_CHECK(cuCtxCreate(&cuda.context, 0, device));
#endif
    CUDA_CHECK(cuModuleLoad(&cuda.module, manifest.cubin_path.c_str()));
    CUfunction kernel = nullptr;
    CUDA_CHECK(cuModuleGetFunction(&kernel, cuda.module, manifest.symbol.c_str()));

    for (const ArrayAbi &array : manifest.arrays) {
        CUdeviceptr address = 0;
        CUDA_CHECK(cuMemAlloc(&address, bytes));
        cuda.extra_allocations.push_back(address);
        device_arrays.emplace(array.name, address);
        if (array.name != manifest.output) {
            CUDA_CHECK(cuMemcpyHtoD(address, host_arrays.at(array.name).data(), bytes));
        }
    }

    std::vector<RuntimeArgument> values;
    values.reserve(manifest.parameters.size());
    for (const auto &parameter : manifest.parameters) {
        RuntimeArgument value;
        const std::string &name = parameter.first;
        if (parameter.second == "float32*") {
            value.pointer = device_arrays.at(name);
        } else if (parameter.second == "int32") {
            const std::string shape_suffix = "_shape_0";
            const std::string stride_suffix = "_stride_0";
            if (name.size() > shape_suffix.size() && name.compare(name.size() - shape_suffix.size(), shape_suffix.size(), shape_suffix) == 0) {
                value.integer = n;
            } else if (name.size() > stride_suffix.size() && name.compare(name.size() - stride_suffix.size(), stride_suffix.size(), stride_suffix) == 0) {
                value.integer = 1;
            } else {
                invalid_abi("unrecognized integer parameter name " + name);
            }
        } else {
            invalid_abi("unrecognized parameter type " + parameter.second);
        }
        values.push_back(value);
    }
    std::vector<void *> kernel_parameters;
    kernel_parameters.reserve(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        kernel_parameters.push_back(manifest.parameters[i].second == "float32*"
                                        ? static_cast<void *>(&values[i].pointer)
                                        : static_cast<void *>(&values[i].integer));
    }

    CUDA_CHECK(cuLaunchKernel(kernel,
                              manifest.grid[0], manifest.grid[1], manifest.grid[2],
                              manifest.block[0], manifest.block[1], manifest.block[2],
                              0, nullptr, kernel_parameters.data(), nullptr));
    CUDA_CHECK(cuCtxSynchronize());
    CUDA_CHECK(cuMemcpyDtoH(host_arrays.at(manifest.output).data(),
                            device_arrays.at(manifest.output), bytes));
    const auto &result = host_arrays.at(manifest.output);
    const auto &left = host_arrays.at(manifest.inputs[0]);
    const auto &right = host_arrays.at(manifest.inputs[1]);
    for (int32_t i = 0; i < n; ++i) {
        const float expected = left[i] + right[i];
        if (std::fabs(result[i] - expected) > 1.0e-5f) {
            std::cerr << "Vector-add mismatch at index " << i << ": got " << result[i]
                      << ", expected " << expected << '\n';
            return 1;
        }
    }
    std::cout << "ABI-driven vector add passed for " << n << " elements (kernel "
              << manifest.symbol << ", grid=" << manifest.grid[0] << "x" << manifest.grid[1]
              << "x" << manifest.grid[2] << ", block=" << manifest.block[0] << "x"
              << manifest.block[1] << "x" << manifest.block[2] << ").\n";
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc == 3 && std::string(argv[1]) == "--validate-abi-json") {
        try {
            const AbiManifest manifest = load_abi_manifest(argv[2]);
            std::cout << "ABI JSON valid for kernel " << manifest.symbol << ".\n";
            return 0;
        } catch (const std::exception &error) {
            std::cerr << "loadtest: " << error.what() << '\n';
            return 1;
        }
    }
    if (argc >= 3 && std::string(argv[1]) == "--abi-json") {
        try {
            const AbiManifest manifest = load_abi_manifest(argv[2]);
            if (argc == 5 && std::string(argv[3]) == "--n") {
                const uint64_t requested = parse_positive_integer(argv[4], "n");
                if (requested != static_cast<uint64_t>(manifest.length)) {
                    invalid_abi("--n must match probe.length");
                }
            } else if (argc != 3) {
                throw std::invalid_argument("usage: loadtest --abi-json <path> [--n <probe-length>]");
            }
            return run_abi_probe(manifest);
        } catch (const std::exception &error) {
            std::cerr << "loadtest: " << error.what() << '\n';
            return 1;
        }
    }
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <cubin-path> <kernel-name> <n> <grid-x> <block-x>\n"
                  << "       " << argv[0] << " --abi-json <path> [--n <probe-length>]\n"
                  << "       " << argv[0] << " --validate-abi-json <path>\n"
                  << "Loads a cuTile vector-add cubin with the Driver API. Legacy mode uses "
                     "three 1D array arguments are passed as (pointer, shape, stride); "
                     "ABI mode reads parameters and launch metadata from schema_version 1 JSON.\n";
        return 2;
    }

    try {
        const std::string cubin_path(argv[1]);
        const std::string kernel_name(argv[2]);
        const uint64_t n64 = parse_positive_integer(argv[3], "n");
        const uint64_t grid_x64 = parse_positive_integer(argv[4], "grid-x");
        const uint64_t block_x64 = parse_positive_integer(argv[5], "block-x");

        if (n64 > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
            throw std::invalid_argument("n must fit in the cubin's int32 shape argument");
        }
        if (grid_x64 > std::numeric_limits<unsigned int>::max() ||
            block_x64 > std::numeric_limits<unsigned int>::max()) {
            throw std::invalid_argument("grid-x and block-x must fit in unsigned int");
        }
        if (n64 > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
            throw std::invalid_argument("n is too large for host allocation");
        }

        auto n = static_cast<int32_t>(n64);
        const auto grid_x = static_cast<unsigned int>(grid_x64);
        const auto block_x = static_cast<unsigned int>(block_x64);
        const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);

        std::vector<float> host_a(n);
        std::vector<float> host_b(n);
        std::vector<float> host_out(n, 0.0f);
        for (int32_t i = 0; i < n; ++i) {
            host_a[i] = static_cast<float>(i) * 0.25f;
            host_b[i] = static_cast<float>(n - i) * 0.5f;
        }

        CUDA_CHECK(cuInit(0));
        CUdevice device = 0;
        CUDA_CHECK(cuDeviceGet(&device, 0));

        CudaResources cuda;
#if CUDA_VERSION >= 13000
        CUctxCreateParams context_params{};
        CUDA_CHECK(cuCtxCreate(&cuda.context, &context_params, 0, device));
#else
        CUDA_CHECK(cuCtxCreate(&cuda.context, 0, device));
#endif
        CUDA_CHECK(cuModuleLoad(&cuda.module, cubin_path.c_str()));

        CUfunction kernel = nullptr;
        CUDA_CHECK(cuModuleGetFunction(&kernel, cuda.module, kernel_name.c_str()));

        CUDA_CHECK(cuMemAlloc(&cuda.a, bytes));
        CUDA_CHECK(cuMemAlloc(&cuda.b, bytes));
        CUDA_CHECK(cuMemAlloc(&cuda.out, bytes));
        CUDA_CHECK(cuMemcpyHtoD(cuda.a, host_a.data(), bytes));
        CUDA_CHECK(cuMemcpyHtoD(cuda.b, host_b.data(), bytes));

        // cuTile's AOT ABI expands each 1D array to pointer, shape, and stride.
        // The kernel's ct.Constant tile parameter is baked into the cubin.
        int32_t stride = 1;
        void *kernel_params[] = {
            &cuda.a, &n, &stride,
            &cuda.b, &n, &stride,
            &cuda.out, &n, &stride,
        };
        CUDA_CHECK(cuLaunchKernel(kernel,
                                  grid_x, 1, 1,
                                  block_x, 1, 1,
                                  0, nullptr, kernel_params, nullptr));
        CUDA_CHECK(cuCtxSynchronize());
        CUDA_CHECK(cuMemcpyDtoH(host_out.data(), cuda.out, bytes));

        for (int32_t i = 0; i < n; ++i) {
            const float expected = host_a[i] + host_b[i];
            if (std::fabs(host_out[i] - expected) > 1.0e-5f) {
                std::cerr << "Vector-add mismatch at index " << i << ": got "
                          << host_out[i] << ", expected " << expected << '\n';
                return 1;
            }
        }

        std::cout << "Vector add passed for " << n << " elements (kernel "
                  << kernel_name << ", grid-x=" << grid_x
                  << ", block-x=" << block_x << ").\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "loadtest: " << error.what() << '\n';
        return 1;
    }
}
