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
    launch_quantize_fp8,
)
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
    raw_cp = cp.asarray(raw)

    q_cp, scale_cp = launch_dequant_iq3_s(
        raw_cp, n_rows, k, tile_n=4
    )
    q = cp.asnumpy(q_cp)
    scales = cp.asnumpy(scale_cp)
    assert q.dtype == np.uint8 and q.shape == (n_rows, k)
    assert scales.dtype == np.float32 and scales.shape == (n_rows, k // FP8_GROUP_K)

    rows = np.ascontiguousarray(raw.reshape(n_rows, -1))
    reference = dequantize(rows, quant_type.IQ3_S)
    decoded = decode_e4m3fn(q) * np.repeat(scales, FP8_GROUP_K, axis=1)
    np.testing.assert_allclose(decoded, reference, rtol=2e-7, atol=2e-7)


@pytest.mark.usefixtures("require_cuda")
def test_f32_to_mxfp8_quantization_matches_reference() -> None:
    cp.random.seed(20261015)
    x_cp = cp.random.standard_normal((5, 160), dtype=cp.float32)
    x_cp[0, :] = 0.0
    x_cp[1, 0] = 32.0
    x_cp[2, 32 + 7] = -64.0
    x_cp[3, :32] *= 1.0e-4
    x_cp[3, 32:64] = np.float32(2.0**-130)
    x_cp[4, 64:96] = cp.linspace(-12.0, 12.0, FP8_GROUP_K, dtype=cp.float32)

    q_cp, scale_cp = launch_quantize_fp8(x_cp)
    q = cp.asnumpy(q_cp)
    scale_codes = cp.asnumpy(scale_cp)
    assert q.dtype == np.uint8 and q.shape == (5, 160)
    assert scale_codes.dtype == np.uint8 and scale_codes.shape == (5, 5)

    expected_q, expected_scales = encode_mxfp8_reference(cp.asnumpy(x_cp))
    np.testing.assert_array_equal(q, expected_q)
    np.testing.assert_array_equal(scale_codes, expected_scales)
    reconstructed = decode_mxfp8(q, scale_codes)
    expected_reconstructed = decode_mxfp8(expected_q, expected_scales)
    np.testing.assert_array_equal(reconstructed, expected_reconstructed)
    assert np.count_nonzero(reconstructed[0]) == 0


@pytest.mark.usefixtures("require_cuda")
def test_synthetic_iq3_s_fp8_matmul_matches_gguf_py(
    gguf_decoder: tuple[Any, Any]
) -> None:
    quant_type, dequantize = gguf_decoder
    m, n_rows, k = 3, 7, 768
    raw = make_synthetic_iq3_s(n_rows, k, seed=0x3F8)
    raw_cp = cp.asarray(raw)
    cp.random.seed(20261016)
    x_cp = cp.random.standard_normal((m, k), dtype=cp.float32)
    x_q_cp, x_scale_cp = launch_quantize_fp8(x_cp)
    output_cp = launch_iq3_s_fp8_matmul(
        x_q_cp,
        x_scale_cp,
        raw_cp,
    )

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

    # The matmul reads tensor strides directly; it need not copy or reshape inputs.
    strided_weight = cp.empty((n_rows, k // IQ3_S_BLOCK_K, 2 * IQ3_S_BLOCK_BYTES), dtype=cp.uint8)
    strided_weight[..., ::2] = raw_cp
    strided_values = cp.empty((m, 2 * k), dtype=cp.uint8)
    strided_values[:, ::2] = x_q_cp
    strided_scales = cp.empty((m, 2 * k // FP8_GROUP_K), dtype=cp.uint8)
    strided_scales[:, ::2] = x_scale_cp
    strided_output = launch_iq3_s_fp8_matmul(
        strided_values[:, ::2], strided_scales[:, ::2], strided_weight[..., ::2],
    )
    np.testing.assert_allclose(cp.asnumpy(strided_output), expected, rtol=5e-4, atol=5e-3)


@pytest.mark.parametrize("m", [1, 8, 128])
@pytest.mark.usefixtures("require_cuda")
def test_model_projection_matches_f32_and_fp8_references(m: int, record_property) -> None:
    model_path, reference_executable, llama_source = model_reference_config()
    raw, n_rows, k = read_model_weight(model_path, WEIGHT_NAME, expected_type="iq3_s")
    assert raw.shape[2] == IQ3_S_BLOCK_BYTES

    cp.random.seed(20261017)
    x_cp = cp.random.standard_normal((m, k), dtype=cp.float32)
    x_f32 = cp.asnumpy(x_cp)
    x_values, x_scales = launch_quantize_fp8(x_cp)
    output = cp.asnumpy(
        launch_iq3_s_fp8_matmul(
            x_values, x_scales, cp.asarray(raw),
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
        threads=4,
    )
    assert original.metadata["ggml_cpu_input_dot_type"] == "q8_K"

    def nmse(reference: np.ndarray) -> float:
        diff = output.astype(np.float64) - reference.astype(np.float64)
        return float(np.sum(diff * diff) / np.sum(reference.astype(np.float64) ** 2))

    for name, reference in (("gguf_py", original.gguf_py), ("ggml_cpu", original.ggml_cpu)):
        assert reference.shape == output.shape
        error = nmse(reference)
        record_property(f"{name}_original_f32_nmse", error)
        if name == "ggml_cpu":
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
        threads=4,
    )
    np.testing.assert_allclose(output, effective.gguf_py, rtol=5e-4, atol=5e-3)
