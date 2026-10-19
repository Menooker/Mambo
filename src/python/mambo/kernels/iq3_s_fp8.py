"""IQ3_S block decode and block-scaled FP8 projection for SM120.

The projection reads packed GGUF weights and prequantized FP8 activations. It
decodes IQ3_S blocks inside each matmul tile. The standalone decoder and F32
activation quantizer are retained for independent correctness checks.
CuPy wrappers are for development and tests; the kernels themselves can be AOT
exported and called by the C++ runtime.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

import cupy as cp
import numpy as np
import cuda.tile as ct


IQ3_S_BLOCK_BYTES = 110
IQ3_S_BLOCK_VALUES = 256
IQ3_S_SCALE_VALUES = 32
_IQ3_TABLES: dict[int, cp.ndarray] = {}
# Hand-maintained results for the real IQ3_S QKV projection (N=10240, K=5120).
# The tuner reports candidates but never updates this table automatically.
BEST_CONFIGS = {
    "sm120": {
        "medium": {"tile_m": 16, "tile_n": 32, "x_allow_tma": True,
                   "store_allow_tma": False, "occupancy": 2},  # tuned at M=4
        "large": {"tile_m": 32, "tile_n": 16, "x_allow_tma": True,
                  "store_allow_tma": False, "occupancy": 4},  # tuned at M=128
    },
}
# The measurements cover M=4 and M=128; use 32 as the dispatch boundary.
LARGE_TOKEN_THRESHOLD = 32


@lru_cache(maxsize=1)
def iq3_s_table() -> np.ndarray:
    """Read the original 512-entry lookup table from the pinned ggml source."""
    header = Path(__file__).resolve().parents[4] / "thirdparty/llama.cpp/ggml/src/ggml-common.h"
    source = header.read_text(encoding="utf-8")
    match = re.search(r"GGML_TABLE_BEGIN\(uint32_t, iq3s_grid, 512\)(.*?)GGML_TABLE_END\(\)", source, re.S)
    if match is None:
        raise ValueError(f"IQ3_S lookup table was not found in {header}")
    values = np.asarray([int(s, 16) for s in re.findall(r"0x[0-9a-fA-F]+", match.group(1))], dtype=np.uint32)
    if values.shape != (512,):
        raise ValueError(f"expected 512 IQ3_S table entries, got {values.size}")
    return values


@lru_cache(maxsize=1)
def iq3_s_fp8_table() -> np.ndarray:
    """Pack positive magnitudes as E4M3 bits; pass this table to the AOT kernels."""
    magnitudes = np.array([1, 3, 5, 7, 9, 11, 13, 15], dtype=np.uint8)
    fp8_bits = np.array([0x38, 0x44, 0x4A, 0x4E, 0x51, 0x53, 0x55, 0x57], dtype=np.uint8)
    lookup = np.zeros(256, dtype=np.uint8)
    valid = np.zeros(256, dtype=np.bool_)
    lookup[magnitudes] = fp8_bits
    valid[magnitudes] = True

    raw_table = iq3_s_table()
    encoded_table = np.zeros_like(raw_table)
    for shift in (0, 8, 16, 24):
        magnitude = (raw_table >> shift) & 0xFF
        if not np.all(valid[magnitude]):
            raise ValueError("IQ3_S codebook contains a magnitude without an E4M3 encoding")
        encoded_table |= lookup[magnitude].astype(np.uint32) << shift
    return encoded_table


def _get_iq3_table() -> cp.ndarray:
    """Upload the FP8-encoded IQ3_S codebook once per CUDA device."""
    device = cp.cuda.runtime.getDevice()
    if device not in _IQ3_TABLES:
        _IQ3_TABLES[device] = cp.asarray(iq3_s_fp8_table())
    return _IQ3_TABLES[device]


@ct.function
def decode_iq3_s_tile(iq3_weight, iq3_table, row_tile_index, block_index, tile_rows: ct.Constant[int]):
    """Decode a tile of GGUF blocks into signed E4M3 values and F32 scales.

    One IQ3_S block along K (256 weights, 110 bytes):

        byte  0..1   | d: FP16 base scale                  |  2 bytes
        byte  2..65  | qs: 64 low 8-bit codebook indices   | 64 bytes
        byte 66..73  | qh: 64 high index bits              |  8 bytes
        byte 74..105 | signs: one bit per weight           | 32 bytes
        byte 106..109| scales: 8 packed 4-bit values      |  4 bytes
                    +-------------------------------------+ 110 bytes

    Each 9-bit (qs, qh) index selects four E4M3 magnitude bit patterns from iq3_table.
    Each group of 32 weights uses scale d * (1 + 2 * scale_nibble).
    """
    # Slice each byte range so a tile load can read it contiguously.
    qs_bytes = iq3_weight.slice(2, 2, 66)
    qh_bytes = iq3_weight.slice(2, 66, 74)
    sign_bytes_array = iq3_weight.slice(2, 74, 106)
    scale_bytes_array = iq3_weight.slice(2, 106, 110)
    q_col = ct.expand_dims(ct.arange(64, dtype=ct.int32), 0)
    qs = ct.astype(ct.reshape(ct.load(qs_bytes, (row_tile_index, block_index, 0),
                                      (tile_rows, 1, 64), padding_mode=ct.PaddingMode.ZERO),
                              (tile_rows, 64)), ct.int32)
    qh = ct.astype(ct.reshape(ct.load(qh_bytes, (row_tile_index, block_index, 0),
                                      (tile_rows, 1, 8), padding_mode=ct.PaddingMode.ZERO),
                              (tile_rows, 8)), ct.int32)
    qh_per_code = ct.reshape(ct.broadcast_to(ct.expand_dims(qh, 2), (tile_rows, 8, 8)), (tile_rows, 64))
    table_index = qs | (((qh_per_code >> (q_col % 8)) & 1) << 8)
    table_word = ct.astype(ct.gather(iq3_table, table_index), ct.int32)
    table_per_value = ct.reshape(ct.broadcast_to(ct.expand_dims(table_word, 2), (tile_rows, 64, 4)), (tile_rows, 256))
    sign_bytes = ct.astype(ct.reshape(ct.load(sign_bytes_array, (row_tile_index, block_index, 0),
                                              (tile_rows, 1, 32), padding_mode=ct.PaddingMode.ZERO),
                                      (tile_rows, 32)), ct.int32)
    sign_per_value = ct.reshape(ct.broadcast_to(ct.expand_dims(sign_bytes, 2), (tile_rows, 32, 8)), (tile_rows, 256))
    col = ct.expand_dims(ct.arange(256, dtype=ct.int32), 0)
    magnitude_bits = (table_per_value >> ((col % 4) * 8)) & 255
    sign_bits = ((sign_per_value >> (col % 8)) & 1) << 7
    fp8_bits = ct.astype(magnitude_bits | sign_bits, ct.uint8)

    groups = ct.expand_dims(ct.arange(8, dtype=ct.int32), 0)
    packed_scale = ct.astype(ct.reshape(ct.load(scale_bytes_array, (row_tile_index, block_index, 0),
                                               (tile_rows, 1, 4), padding_mode=ct.PaddingMode.ZERO),
                                       (tile_rows, 4)), ct.int32)
    scale_bytes = ct.reshape(ct.broadcast_to(ct.expand_dims(packed_scale, 2), (tile_rows, 4, 2)), (tile_rows, 8))
    nibble = (scale_bytes >> ((groups % 2) * 4)) & 15
    # Load both bytes together, then reinterpret their little-endian bits as FP16.
    d_bytes = ct.load(iq3_weight, (row_tile_index, block_index, 0), (tile_rows, 1, 2),
                      padding_mode=ct.PaddingMode.ZERO)
    d_fp16 = ct.unpack_from_bytes(ct.reshape(d_bytes, (tile_rows * 2,)), ct.float16)
    d = ct.expand_dims(ct.astype(d_fp16, ct.float32), 1)
    scales = d * ct.astype(1 + 2 * nibble, ct.float32)
    return fp8_bits, scales


@ct.kernel
def decode_iq3_s_blocks(iq3_weight, iq3_table, values, scales, tile_n: ct.Constant[int]):
    """Decode one K block per CTA for a power-of-two tile of N rows."""
    block_values, block_scales = decode_iq3_s_tile(iq3_weight, iq3_table, ct.bid(0), ct.bid(1), tile_n)
    ct.store(values, (ct.bid(0), ct.bid(1), 0), ct.reshape(block_values, (tile_n, 1, 256)))
    ct.store(scales, (ct.bid(0), ct.bid(1), 0), ct.reshape(block_scales, (tile_n, 1, 8)))


@ct.function
def quantize_fp8_group(data, axis: ct.Constant[int]):
    """Return E4M3 values and E8M0 scale for groups of 32 F32 values."""
    amax = ct.max(ct.abs(data), axis=axis)
    # The nonzero branch is the standard MXFP8 round-up power-of-two scale.
    safe_amax = ct.where(amax > 0.0, amax, 448.0)
    exponent = ct.maximum(ct.ceil(ct.log2(safe_amax / 448.0)), -127.0)
    scale_f32 = ct.exp2(exponent)
    q = ct.astype(data / ct.expand_dims(scale_f32, axis), ct.float8_e4m3fn)
    scale_fp8 = ct.astype(scale_f32, ct.float8_e8m0fnu, rounding_mode=ct.RoundingMode.RP)
    return q, scale_fp8


@ct.kernel
def quantize_activation_fp8(x, values, scales, tile_m: ct.Constant[int], tile_k_groups: ct.Constant[int]):
    """Quantize F32 [M, K] into E4M3 bytes and per-32 E8M0 scale bytes."""
    data = ct.load(x, (ct.bid(0), ct.bid(1), 0), (tile_m, tile_k_groups, 32), padding_mode=ct.PaddingMode.ZERO)
    q, scale_fp8 = quantize_fp8_group(data, 2)
    ct.store(values, (ct.bid(0), ct.bid(1), 0), ct.bitcast(q, ct.uint8))
    ct.store(scales, (ct.bid(0), ct.bid(1)), ct.bitcast(scale_fp8, ct.uint8))


@ct.kernel
def iq3_s_fp8_matmul(x_values, x_scales, iq3_weight, iq3_table, out,
                     tile_m: ct.Constant[int], tile_n: ct.Constant[int],
                     x_allow_tma: ct.Constant[bool], store_allow_tma: ct.Constant[bool]):
    """Decode IQ3_S within the tile and multiply prequantized FP8 activations."""
    total = ct.zeros((tile_m, tile_n), dtype=ct.float32)
    one = ct.full((1, tile_n), 1.0, dtype=ct.float8_e8m0fnu)
    for block in range(iq3_weight.shape[1]):
        weight_bits, weight_scales = decode_iq3_s_tile(iq3_weight, iq3_table, ct.bid(1), block, tile_n)
        x_block = ct.load(x_values, (ct.bid(0), block), (tile_m, 256),
                          padding_mode=ct.PaddingMode.ZERO, allow_tma=x_allow_tma)
        xs_block = ct.load(x_scales, (ct.bid(0), block), (tile_m, 8),
                           padding_mode=ct.PaddingMode.ZERO)
        for group in range(8):
            x = ct.bitcast(ct.extract(x_block, (0, group), shape=(tile_m, 32)), ct.float8_e4m3fn)
            xs = ct.bitcast(ct.extract(xs_block, (0, group), shape=(tile_m, 1)), ct.float8_e8m0fnu)
            w_bits = ct.extract(weight_bits, (0, group), shape=(tile_n, 32))
            w = ct.bitcast(w_bits, ct.float8_e4m3fn)
            ws = ct.transpose(ct.extract(weight_scales, (0, group), shape=(tile_n, 1)))
            partial = ct.mma_scaled(x, xs, ct.transpose(w), one,
                                    ct.zeros((tile_m, tile_n), dtype=ct.float32))
            total += partial * ws
    ct.store(out, (ct.bid(0), ct.bid(1)), total, allow_tma=store_allow_tma)


def _power_of_two(value: int, name: str) -> None:
    if value < 1 or value & (value - 1):
        raise ValueError(f"{name} must be a positive power of two")


def launch_dequant_iq3_s(iq3_weight: cp.ndarray, n_rows: int, k: int, *, tile_n: int = 1,
                         stream=None) -> tuple[cp.ndarray, cp.ndarray]:
    """Return [N,K] E4M3 bit patterns and [N,K/32] original F32 scales."""
    _power_of_two(tile_n, "tile_n")
    if k % 256 or n_rows < 1 or iq3_weight.dtype != cp.uint8 or iq3_weight.shape != (n_rows, k // 256, 110):
        raise ValueError("IQ3_S weight must be uint8 [N,K/256,110] with K divisible by 256")
    values = cp.empty((n_rows, k // 256, 256), dtype=cp.uint8)
    scales = cp.empty((n_rows, k // 256, 8), dtype=cp.float32)
    launch_grid = (ct.cdiv(n_rows, tile_n), k // 256)
    ct.launch(stream or cp.cuda.get_current_stream(), launch_grid, decode_iq3_s_blocks,
              (iq3_weight, _get_iq3_table(), values, scales, tile_n))
    return values.reshape(n_rows, k), scales.reshape(n_rows, k // 32)


def launch_quantize_fp8(x: cp.ndarray, *, tile_m: int = 1, tile_k_groups: int = 4,
                        stream=None) -> tuple[cp.ndarray, cp.ndarray]:
    """Return [M,K] E4M3 bytes and [M,K/32] E8M0 scale bytes."""
    _power_of_two(tile_m, "tile_m")
    _power_of_two(tile_k_groups, "tile_k_groups")
    if x.ndim != 2 or x.dtype != cp.float32 or x.shape[1] % 32:
        raise ValueError("activation must be float32 [M,K] with K divisible by 32")
    x = cp.ascontiguousarray(x).reshape(x.shape[0], x.shape[1] // 32, 32)
    values = cp.empty(x.shape, dtype=cp.uint8)
    scales = cp.empty(x.shape[:2], dtype=cp.uint8)
    launch_grid = (ct.cdiv(x.shape[0], tile_m), ct.cdiv(x.shape[1], tile_k_groups))
    ct.launch(stream or cp.cuda.get_current_stream(), launch_grid, quantize_activation_fp8,
              (x, values, scales, tile_m, tile_k_groups))
    return values.reshape(x.shape[0], x.shape[1] * 32), scales


@lru_cache(maxsize=None)
def _matmul_kernel_for_occupancy(occupancy: int):
    return iq3_s_fp8_matmul.replace_hints(occupancy=occupancy)


def launch_iq3_s_fp8_matmul(x_values: cp.ndarray, x_scales: cp.ndarray, iq3_weight: cp.ndarray,
                            *, stream=None) -> cp.ndarray:
    """Multiply packed IQ3_S weights by prequantized MXFP8 activations."""
    if (iq3_weight.ndim != 3 or iq3_weight.dtype != cp.uint8 or iq3_weight.shape[0] < 1
            or iq3_weight.shape[1] < 1 or iq3_weight.shape[2] != 110):
        raise ValueError("IQ3_S weight must be uint8 [N,K/256,110] with positive N and K")
    n_rows, k_blocks, _ = iq3_weight.shape
    k = k_blocks * 256
    if x_values.ndim != 2 or x_values.dtype != cp.uint8 or x_values.shape[0] < 1 or x_values.shape[1] != k:
        raise ValueError("x_values must be uint8 [M,K]")
    if x_scales.shape != (x_values.shape[0], k // 32) or x_scales.dtype != cp.uint8:
        raise ValueError("x_scales must be uint8 [M,K/32]")
    sm = f"sm{cp.cuda.Device().compute_capability}"
    if sm not in BEST_CONFIGS:
        raise RuntimeError(f"IQ3_S FP8 matmul has no tuned config for {sm}")
    token_size = "large" if x_values.shape[0] >= LARGE_TOKEN_THRESHOLD else "medium"
    config = BEST_CONFIGS[sm][token_size]
    tile_m, tile_n = config["tile_m"], config["tile_n"]
    out = cp.empty((x_values.shape[0], n_rows), dtype=cp.float32)
    ct.launch(stream or cp.cuda.get_current_stream(),
              (ct.cdiv(x_values.shape[0], tile_m), ct.cdiv(n_rows, tile_n)),
              _matmul_kernel_for_occupancy(config["occupancy"]),
              (x_values, x_scales, iq3_weight, _get_iq3_table(), out,
               tile_m, tile_n, config["x_allow_tma"], config["store_allow_tma"]))
    return out


def make_benchmark_cases(iq3_weight: cp.ndarray) -> list:
    """Describe the IQ3_S projection cases for the shared benchmark runner."""
    from mambo.utils.kernel_benchmark import BenchmarkCase

    if (iq3_weight.ndim != 3 or iq3_weight.dtype != cp.uint8
            or iq3_weight.shape[2] != IQ3_S_BLOCK_BYTES):
        raise ValueError("IQ3_S weight must be uint8 [N,K/256,110]")

    n_rows, k_blocks, _ = iq3_weight.shape
    k = k_blocks * IQ3_S_BLOCK_VALUES
    iq3_table = _get_iq3_table()
    stream = cp.cuda.get_current_stream()
    # On SM120, 16x16 lowers to FFMA instead of QMMA for this kernel.
    search_space = [
        {"tile_m": tile_m, "tile_n": tile_n,
         "x_allow_tma": x_allow_tma, "store_allow_tma": store_allow_tma,
         "occupancy": occupancy}
        for tile_m in (16, 32)
        for tile_n in (16, 32)
        for x_allow_tma in (False, True)
        for store_allow_tma in (False, True)
        for occupancy in (None, 1, 2, 4)
        if (tile_m, tile_n) != (16, 16)
    ]

    def make_case(m: int) -> BenchmarkCase:
        x = cp.random.standard_normal((m, k), dtype=cp.float32)
        x_values, x_scales = launch_quantize_fp8(x, stream=stream)
        out = cp.empty((m, n_rows), dtype=cp.float32)

        def grid(config):
            return (ct.cdiv(m, config["tile_m"]), ct.cdiv(n_rows, config["tile_n"]))

        def args(config):
            return (x_values, x_scales, iq3_weight, iq3_table, out,
                    config["tile_m"], config["tile_n"],
                    config["x_allow_tma"], config["store_allow_tma"])

        def hints(config):
            return {"occupancy": config["occupancy"]}

        return BenchmarkCase(
            name=f"M={m} N={n_rows} K={k}", stream=stream, kernel=iq3_s_fp8_matmul,
            search_space=search_space, grid_fn=grid, args_fn=args, hints_fn=hints,
            summary_keys=("tile_m", "tile_n"),
        )

    return [make_case(m) for m in (4, 128)]
