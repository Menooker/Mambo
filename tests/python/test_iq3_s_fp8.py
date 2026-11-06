"""GPU checks for IQ3_S decoding, MXFP8 activations, and FP8 matmul."""

from __future__ import annotations

from typing import Any

import cupy as cp
import numpy as np
import pytest

from utils.paths import LLAMA_SOURCE, ensure_test_paths, model_reference_config

ensure_test_paths()

from mambo.kernels.iq3_s_fp8 import (  # noqa: E402
    iq3_s_fp8_table,
    iq3_s_table,
    launch_dequant_iq3_s,
    launch_iq3_s_fp8_matmul,
)
from mambo.kernels.quantize_fp8 import launch_quantize_fp8  # noqa: E402
from mambo.utils.iq3_s_weight import prepare_iq3_s_fields  # noqa: E402
from utils.compute_reference_gguf import (  # noqa: E402
    _load_gguf_decoder,
    compute_reference_gguf,
)
from utils.fp8_reference import (  # noqa: E402
    FP8_GROUP_K,
    decode_e4m3fn,
    decode_mxfp8,
    encode_mxfp8_reference,
)
from utils.gguf_weight import read_model_weight  # noqa: E402


WEIGHT_NAME = "blk.1.attn_qkv.weight"
IQ3_S_BLOCK_K = 256
IQ3_S_BLOCK_BYTES = 110


def make_synthetic_iq3_s(n_rows: int, k: int, seed: int) -> np.ndarray:
    if k % IQ3_S_BLOCK_K:
        raise ValueError("IQ3_S synthetic K must be divisible by 256")
    blocks = k // IQ3_S_BLOCK_K
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, size=(n_rows, blocks, IQ3_S_BLOCK_BYTES), dtype=np.uint8)
    # IQ3_S starts each 110-byte block with an FP16 base scale. Keep it finite,
    # positive, and varied; the remaining fields are valid arbitrary bit codes.
    base_scales = rng.uniform(0.002, 0.04, size=(n_rows, blocks)).astype("<f2")
    raw[..., :2] = base_scales.view(np.uint8).reshape(n_rows, blocks, 2)
    return raw


@pytest.fixture(scope="module")
def require_cuda() -> None:
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("IQ3_S FP8 tests require a CUDA device")
    except Exception as exc:
        pytest.skip(f"IQ3_S FP8 tests require a usable CUDA device: {exc}")


@pytest.fixture(scope="module")
def gguf_decoder() -> tuple[Any, Any]:
    package_root = LLAMA_SOURCE / "gguf-py" / "gguf" / "quants.py"
    if not package_root.is_file():
        pytest.skip("initialize the pinned llama.cpp submodule with gguf-py")
    return _load_gguf_decoder(LLAMA_SOURCE)


def test_fp8_codebook_preserves_every_iq3_s_magnitude() -> None:
    raw = iq3_s_table()
    encoded = iq3_s_fp8_table()
    assert raw.shape == encoded.shape == (512,)
    assert raw.dtype == encoded.dtype == np.uint32

    shifts = (0, 8, 16, 24)
    magnitudes = np.stack([(raw >> shift) & 0xFF for shift in shifts], axis=-1).astype(np.float32)
    bits = np.stack([(encoded >> shift) & 0xFF for shift in shifts], axis=-1).astype(np.uint8)
    np.testing.assert_array_equal(decode_e4m3fn(bits), magnitudes)
    np.testing.assert_array_equal(decode_e4m3fn(bits | np.uint8(0x80)), -magnitudes)


@pytest.mark.usefixtures("require_cuda")
def test_iq3_s_decode_matches_gguf_py(
    gguf_decoder: tuple[Any, Any]
) -> None:
    quant_type, dequantize = gguf_decoder
    n_rows, k = 5, 3 * IQ3_S_BLOCK_K
    raw = make_synthetic_iq3_s(n_rows, k, seed=0x1A3)
    fields = prepare_iq3_s_fields(cp.asarray(raw))

    q_cp, scale_cp = launch_dequant_iq3_s(fields, tile_n=4)
    q = cp.asnumpy(q_cp)
    scales = cp.asnumpy(scale_cp)
    assert q.dtype == np.uint8 and q.shape == (n_rows, k)
    assert scales.dtype == np.float32 and scales.shape == (n_rows, k // FP8_GROUP_K)

    rows = np.ascontiguousarray(raw.reshape(n_rows, -1))
    reference = dequantize(rows, quant_type.IQ3_S)
    decoded = decode_e4m3fn(q) * np.repeat(scales, FP8_GROUP_K, axis=1)
    np.testing.assert_allclose(decoded, reference, rtol=2e-7, atol=2e-7)


@pytest.mark.usefixtures("require_cuda")
def test_packed_decode_all_codes_and_sign_nibbles(gguf_decoder: tuple[Any, Any]) -> None:
    quant_type, dequantize = gguf_decoder
    # 8192 four-weight groups cover every codebook entry with all 16 signs.
    n_rows, blocks = 8, 16
    raw = np.zeros((n_rows, blocks, IQ3_S_BLOCK_BYTES), dtype=np.uint8)
    raw[..., :2] = np.array([1.0], dtype=np.float16).view(np.uint8)
    code = (np.arange(8192, dtype=np.uint16) % 512).reshape(n_rows, blocks, 64)
    sign = (np.arange(8192, dtype=np.uint16) // 512).reshape(n_rows, blocks, 64)
    raw[..., 2:66] = code.astype(np.uint8)
    raw[..., 66:74] = np.packbits((code >> 8).astype(np.uint8), axis=-1, bitorder="little")
    raw[..., 74:106] = (sign[..., 0::2] | (sign[..., 1::2] << 4)).astype(np.uint8)
    expected = dequantize(raw.reshape(n_rows, -1), quant_type.IQ3_S)
    aligned = cp.asarray(raw)
    # The raw GGUF-to-SoA conversion must accept odd-address and strided source
    # views while producing aligned contiguous field arrays for the decoder.
    allocation = cp.empty(raw.size + 1, dtype=cp.uint8)
    unaligned = allocation[1:].reshape(raw.shape)
    unaligned[...] = aligned
    strided_allocation = cp.empty((n_rows, blocks, 2 * IQ3_S_BLOCK_BYTES), dtype=cp.uint8)
    strided = strided_allocation[..., ::2]
    strided[...] = aligned
    for source in (aligned, unaligned, strided):
        fields = prepare_iq3_s_fields(source)
        bits, scales = launch_dequant_iq3_s(fields, tile_n=4)
        actual = decode_e4m3fn(cp.asnumpy(bits)) * np.repeat(cp.asnumpy(scales), FP8_GROUP_K, axis=1)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.usefixtures("require_cuda")
@pytest.mark.parametrize("n_rows,blocks", [(7, 3), (1, 1)])
def test_soa_fields_preserve_every_byte(n_rows: int, blocks: int) -> None:
    raw = make_synthetic_iq3_s(n_rows, blocks * IQ3_S_BLOCK_K, seed=0x50A)
    fields = prepare_iq3_s_fields(cp.asarray(raw))
    reconstructed = np.empty_like(raw)
    for field, start, end in ((fields.d, 0, 2), (fields.qs, 2, 66),
                               (fields.qh, 66, 74), (fields.signs, 74, 106),
                               (fields.scales, 106, 110)):
        assert field.flags.c_contiguous
        assert field.data.ptr % field.dtype.itemsize == 0
        data = cp.asnumpy(field).view(np.uint8).reshape(blocks, n_rows, end - start)
        reconstructed[..., start:end] = data.transpose(1, 0, 2)
    np.testing.assert_array_equal(reconstructed, raw)
    assert sum(field.nbytes for field in (fields.d, fields.qs, fields.qh, fields.signs, fields.scales)) == raw.nbytes


@pytest.mark.usefixtures("require_cuda")
@pytest.mark.parametrize("tokens", [5, 33, 128])
def test_f32_to_mxfp8_quantization_matches_reference(tokens: int) -> None:
    cp.random.seed(20261015)
    x_cp = cp.random.standard_normal((tokens, 160), dtype=cp.float32)
    x_cp[0, :] = 0.0
    x_cp[1, 0] = 32.0
    x_cp[2, 32 + 7] = -64.0
    x_cp[3, :32] *= 1.0e-4
    x_cp[3, 32:64] = np.float32(2.0**-130)
    x_cp[4, 64:96] = cp.linspace(-12.0, 12.0, FP8_GROUP_K, dtype=cp.float32)

    q_cp, scale_cp = launch_quantize_fp8(x_cp)
    q = cp.asnumpy(q_cp)
    scale_codes = cp.asnumpy(scale_cp)
    assert q.dtype == np.uint8 and q.shape == (tokens, 160)
    assert scale_codes.dtype == np.uint8 and scale_codes.shape == (tokens, 5)

    expected_q, expected_scales = encode_mxfp8_reference(cp.asnumpy(x_cp))
    np.testing.assert_array_equal(q, expected_q)
    np.testing.assert_array_equal(scale_codes, expected_scales)
    reconstructed = decode_mxfp8(q, scale_codes)
    expected_reconstructed = decode_mxfp8(expected_q, expected_scales)
    np.testing.assert_array_equal(reconstructed, expected_reconstructed)
    assert np.count_nonzero(reconstructed[0]) == 0


@pytest.mark.usefixtures("require_cuda")
@pytest.mark.parametrize("m", [3, 4, 5, 8, 9, 33, 128, 129])
def test_synthetic_iq3_s_fp8_matmul_matches_gguf_py(
    gguf_decoder: tuple[Any, Any], m: int
) -> None:
    quant_type, dequantize = gguf_decoder
    n_rows, k = 7, 768
    raw = make_synthetic_iq3_s(n_rows, k, seed=0x3F8)
    fields = prepare_iq3_s_fields(cp.asarray(raw))
    cp.random.seed(20261016)
    x_cp = cp.random.standard_normal((m, k), dtype=cp.float32)
    x_q_cp, x_scale_cp = launch_quantize_fp8(x_cp)
    output_cp = launch_iq3_s_fp8_matmul(x_q_cp, x_scale_cp, fields)

    decoded_weight = dequantize(
        np.ascontiguousarray(raw.reshape(n_rows, -1)), quant_type.IQ3_S
    )
    effective_x = decode_mxfp8(cp.asnumpy(x_q_cp), cp.asnumpy(x_scale_cp))
    expected = (effective_x.astype(np.float64) @ decoded_weight.astype(np.float64).T).astype(
        np.float32
    )
    output = cp.asnumpy(output_cp)
    assert output.dtype == np.float32 and output.shape == (m, n_rows)
    np.testing.assert_allclose(output, expected, rtol=5e-4, atol=5e-3)
    # The matmul reads activation tensor strides directly; it need not copy or reshape them.
    strided_values = cp.empty((m, 2 * k), dtype=cp.uint8)
    strided_values[:, ::2] = x_q_cp
    strided_scales = cp.empty((m, 2 * k // FP8_GROUP_K), dtype=cp.uint8)
    strided_scales[:, ::2] = x_scale_cp
    strided_output = launch_iq3_s_fp8_matmul(
        strided_values[:, ::2], strided_scales[:, ::2], fields,
    )
    np.testing.assert_allclose(cp.asnumpy(strided_output), expected, rtol=5e-4, atol=5e-3)


@pytest.mark.parametrize("m", [1, 4, 8, 128])
@pytest.mark.usefixtures("require_cuda")
def test_model_projection_matches_f32_and_fp8_references(m: int, record_property) -> None:
    model_path, reference_executable, llama_source = model_reference_config()
    raw, n_rows, k = read_model_weight(model_path, WEIGHT_NAME, expected_type="iq3_s")
    assert raw.shape[2] == IQ3_S_BLOCK_BYTES

    cp.random.seed(20261017)
    x_cp = cp.random.standard_normal((m, k), dtype=cp.float32)
    x_f32 = cp.asnumpy(x_cp)
    weight = prepare_iq3_s_fields(cp.asarray(raw))
    x_values, x_scales = launch_quantize_fp8(x_cp)
    output = cp.asnumpy(
        launch_iq3_s_fp8_matmul(
            x_values, x_scales, weight,
        )
    )
    assert output.shape == (m, n_rows)

    # The original F32 activation goes directly to both independent reference
    # paths. This checks the full quantization-and-matmul result.
    original = compute_reference_gguf(
        "matmul",
        {"x": x_f32, "weight": WEIGHT_NAME},
        model_path=model_path,
        reference_executable=reference_executable,
        llama_source_dir=llama_source,
    )
    assert original.metadata["ggml_backend"] == "cuda"

    def nmse(reference: np.ndarray) -> float:
        diff = output.astype(np.float64) - reference.astype(np.float64)
        return float(np.sum(diff * diff) / np.sum(reference.astype(np.float64) ** 2))

    for name, reference in (("gguf_py", original.gguf_py), ("ggml_cuda", original.ggml_cuda)):
        assert reference.shape == output.shape
        error = nmse(reference)
        record_property(f"{name}_original_f32_nmse", error)
        if name == "ggml_cuda":
            record_property("meets_llama_cpp_mul_mat_nmse_5e-4", error <= 5e-4)
        # FP8 activation quantization is less precise than llama.cpp's Q8
        # activation path; use a measured FP8 regression bound here.
        assert error < 1e-3, (name, m, error)
        np.testing.assert_allclose(output, reference, rtol=0.05, atol=0.18)

    # A second comparison removes activation quantization error. NumPy gets
    # the reconstructed FP8 activation, so this checks the IQ3 decode and MMA
    # arithmetic against a full-precision dot product on the real weight.
    effective_x = decode_mxfp8(cp.asnumpy(x_values), cp.asnumpy(x_scales))
    effective = compute_reference_gguf(
        "matmul",
        {"x": effective_x, "weight": WEIGHT_NAME},
        model_path=model_path,
        reference_executable=reference_executable,
        llama_source_dir=llama_source,
    )
    np.testing.assert_allclose(output, effective.gguf_py, rtol=5e-4, atol=5e-3)
