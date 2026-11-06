"""IQ3_S block decode and block-scaled FP8 projection for SM120.

The projection reads prepared SoA weight fields and prequantized FP8 activations. It
decodes IQ3_S blocks inside each matmul tile. A standalone decoder supports
independent correctness checks; F32 activation quantization lives in quantize_fp8.
The retained large-token implementation uses packed integer decoding, contiguous
full-block sign-word LDG, and a regular tiled output store.
CuPy wrappers are for development and tests; the kernels themselves can be AOT
exported and called by the C++ runtime.

One IQ3_S block along K (256 weights, 110 bytes):

    byte  0..1   | d: FP16 base scale                 |  2 bytes
    byte  2..65  | qs: 64 low 8-bit codebook indices  | 64 bytes
    byte 66..73  | qh: 64 high index bits             |  8 bytes
    byte 74..105 | signs: one bit per weight          | 32 bytes
    byte 106..109| scales: 8 packed 4-bit values      |  4 bytes
                +------------------------------------+ 110 bytes

Each 9-bit (qs, qh) index selects four magnitudes from the codebook.
Each group of 32 weights uses scale d * (1 + 2 * scale_nibble).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

import cupy as cp
import numpy as np
import cuda.tile as ct

from mambo.utils.iq3_s_weight import (
    IQ3SFields,
    prepare_iq3_s_fields,
    _weight_dimensions,
    _weight_kernel_args,
    _large_weight_kernel_args,
)


IQ3_S_BLOCK_BYTES = 110
IQ3_S_BLOCK_VALUES = 256
IQ3_S_SCALE_VALUES = 32
_IQ3_TABLES: dict[int, cp.ndarray] = {}
# Hand-maintained results for the real IQ3_S QKV projection (N=10240, K=5120).
# Tuner candidates are rechecked including output reset; this table is updated manually.
BEST_CONFIGS = {
    "sm120": {
        # Transposed small-token kernel, tuned at M=4. tile_m follows M.
        "medium": {"tile_n": 128, "x_allow_tma": True,
                   "store_allow_tma": False, "occupancy": 2, "split_k": 2},
        # Reuse the tuned tile/occupancy config with the regular output store.
        "large": {"tile_m": 128, "tile_n": 64, "x_allow_tma": True,
                  "store_allow_tma": False, "occupancy": 2},
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
def _signed_codebook_bytes(table_index, sign_nibbles, iq3_table,
                           tile_rows: ct.Constant[int], codes: ct.Constant[int]):
    """Apply sign bits to packed FP8 codebook entries.

    Tiles: table_index int32 [tile_rows,codes] in [0,511];
    sign_nibbles uint32 [tile_rows,codes] (only the low four bits are used).
    Global buffer: iq3_table uint32 [512], four E4M3 encodings per word.
    Returns: uint8 tile [tile_rows,codes*4] containing signed E4M3 bits.
    """
    # uint8 qs plus one high bit proves 0 <= index < 512, including padded rows.
    # The 2 KiB codebook is hot; avoid scheduling this small lookup as a
    # high-latency memory operation. Keep all four magnitude bytes packed.
    words = ct.gather(iq3_table, table_index, check_bounds=False, latency=1)
    signs = ct.astype(sign_nibbles, ct.uint32)
    mask = ((signs & 1) << 7) | ((signs & 2) << 14) | ((signs & 4) << 21) | ((signs & 8) << 28)
    return ct.reshape(ct.pack_to_bytes(words | mask), (tile_rows, codes * 4))


@ct.function
def _decode_iq3_codes(qs, qh, signs, iq3_table, tile_rows: ct.Constant[int], codes: ct.Constant[int]):
    """Decode byte indices and packed i32 signs through the FP8 codebook.

    Tiles: qs uint8 [tile_rows,codes]; qh uint8 [tile_rows,codes/8];
    signs int32 [tile_rows,1] for a 32-weight group, or [tile_rows,codes]
    with each word repeated across its eight codes for a full block.
    Each code represents four weights.
    Global buffer: iq3_table uint32 [512], four E4M3 encodings per word.
    Returns: uint8 tile [tile_rows,codes*4] containing signed E4M3 bits.
    """
    q_col = ct.expand_dims(ct.arange(codes, dtype=ct.int32), 0)
    qs = ct.astype(qs, ct.int32)
    qh = ct.astype(qh, ct.int32)
    qh_per_code = ct.reshape(ct.broadcast_to(ct.expand_dims(qh, 2), (tile_rows, codes // 8, 8)), (tile_rows, codes))
    table_index = qs | (((qh_per_code >> (q_col % 8)) & 1) << 8)
    sign_nibbles = (signs >> ((q_col % 8) * 4)) & 15
    return _signed_codebook_bytes(table_index, sign_nibbles, iq3_table, tile_rows, codes)


@ct.function
def _decode_iq3_group_words(qs_words, high_byte, sign_word, iq3_table,
                           tile_rows: ct.Constant[int]):
    """Decode 32 weights while keeping QS packed until index extraction.

    Tiles: qs_words uint32 [1,tile_rows,2]; high_byte uint8 [tile_rows,1];
    sign_word int32 [tile_rows,1]. Buffer: iq3_table uint32 [512].
    Return uint8 [tile_rows,32] containing signed E4M3 bits.
    Integer shifts avoid the byte reinterpretation and its QS layout exchange.
    """
    words = ct.reshape(qs_words, (tile_rows, 2))
    byte_shifts = ct.reshape(ct.arange(4, dtype=ct.uint32) * 8, (1, 1, 4))
    low_codes = ct.reshape((ct.expand_dims(words, 2) >> byte_shifts) & 255,
                           (tile_rows, 8))
    q_col = ct.expand_dims(ct.arange(8, dtype=ct.int32), 0)
    table_index = ct.astype(low_codes, ct.int32) | (
        ((ct.astype(high_byte, ct.int32) >> q_col) & 1) << 8)
    codebook_words = ct.gather(iq3_table, table_index, check_bounds=False, latency=1)
    sign_col = ct.expand_dims(ct.arange(8, dtype=ct.uint32), 0)
    sign_nibbles = (ct.astype(sign_word, ct.uint32) >> (sign_col * 4)) & 15
    # Spread each nibble to the four packed FP8 sign positions.
    sign_mask = (sign_nibbles * ct.astype(0x10204080, ct.uint32)) & ct.astype(0x80808080, ct.uint32)
    return ct.reshape(ct.pack_to_bytes(codebook_words | sign_mask), (tile_rows, 32))


@ct.function
def _decode_iq3_group(qs, qh, sign_word, iq3_table, row, block, group, tile_n: ct.Constant[int]):
    """Load and decode one 32-weight group from the SoA buffers.

    Global buffers: qs uint32 [B,N,16]; qh uint16 [B,N,4];
    iq3_table uint32 [512]. B is K/256.
    Tile: sign_word int32 [tile_n,1], one packed sign word per row.
    Returns: uint8 tile [tile_n,32] containing signed E4M3 bits.
    """
    # One 32-weight group: 8 code indices, 1 high-bit byte, 4 sign bytes.
    qs_words = ct.load(qs, (block, row, group), (1, tile_n, 2), padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    qh_word = ct.load(qh, (block, row, group // 2), (1, tile_n, 1), padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    q_bytes = ct.reshape(ct.pack_to_bytes(qs_words), (tile_n, 8))
    high_byte = ct.astype(ct.reshape(qh_word, (tile_n, 1)) >> ct.astype((group % 2) * 8, ct.uint16), ct.uint8)
    return _decode_iq3_codes(q_bytes, high_byte, sign_word, iq3_table, tile_n, 8)


@ct.function
def _decode_iq3_scales(d, scales, row, block, tile_n: ct.Constant[int]):
    """Expand eight packed scale nibbles per block into F32 group scales.

    Global buffers: d float16 [B,N]; scales uint32 [B,N], with eight
    4-bit scale nibbles per word. B is K/256.
    Returns: float32 tile [tile_n,8], one scale per 32 weights.
    """
    groups = ct.expand_dims(ct.arange(8, dtype=ct.uint32), 0)
    base = ct.reshape(ct.astype(ct.load(d, (block, row), (1, tile_n),
                                       padding_mode=ct.PaddingMode.ZERO, allow_tma=False), ct.float32), (tile_n, 1))
    packed = ct.reshape(ct.load(scales, (block, row), (1, tile_n),
                               padding_mode=ct.PaddingMode.ZERO, allow_tma=False), (tile_n, 1))
    return base * ct.astype(1 + 2 * ((packed >> (groups * 4)) & 15), ct.float32)


@ct.kernel
def decode_iq3_s_blocks(d, qs, qh, signs, scales, iq3_table, values, out_scales,
                        tile_n: ct.Constant[int]):
    """Standalone SoA decoder for tests; write [N,B,256] bytes and [N,B,8] scales."""
    row, block = ct.bid(0), ct.bid(1)
    qs_words = ct.load(qs, (block, row, 0), (1, tile_n, 16),
                       padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    qh_words = ct.load(qh, (block, row, 0), (1, tile_n, 4),
                       padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    sign_words = ct.load(signs, (block, row, 0), (1, tile_n, 8),
                         padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    # Each packed i32 sign word supplies eight four-weight codebook entries.
    sign_per_code = ct.reshape(ct.broadcast_to(
        ct.reshape(sign_words, (tile_n, 8, 1)), (tile_n, 8, 8)), (tile_n, 64))
    bits = _decode_iq3_codes(ct.reshape(ct.pack_to_bytes(qs_words), (tile_n, 64)),
                            ct.reshape(ct.pack_to_bytes(qh_words), (tile_n, 8)),
                            sign_per_code,
                            iq3_table, tile_n, 64)
    group_scales = _decode_iq3_scales(d, scales, row, block, tile_n)
    ct.store(values, (row, block, 0), ct.reshape(bits, (tile_n, 1, 256)))
    ct.store(out_scales, (row, block, 0), ct.reshape(group_scales, (tile_n, 1, 8)))


@ct.kernel
def iq3_s_fp8_matmul(x_values, x_scales, d, qs, qh, signs, scales, iq3_table, out,
                     tile_m: ct.Constant[int], tile_n: ct.Constant[int],
                     x_allow_tma: ct.Constant[bool], store_allow_tma: ct.Constant[bool]):
    """Large-token MMA with packed metadata loaded once per IQ3_S block.

    Buffers (B=K/256):
      x_values uint8 [M,K], E4M3 bits; x_scales uint8 [M,K/32], E8M0 bits.
      d float16 [B,N]; qs uint32 [B,N,16]; scales uint32 [B,N].
      qh uint32 [B,N,2], a view of prepared uint16 [B,N,4].
      signs int32 [B,N*8], a zero-copy view of the prepared sign words.
      iq3_table uint32 [512], four packed E4M3 values per word; out float32 [M,N].
    The QH and signs views change metadata only; no weight bytes are copied.
    """
    total = ct.zeros((tile_m, tile_n), dtype=ct.float32)
    one = ct.full((1, tile_n), 1.0, dtype=ct.float8_e8m0fnu)
    for block in range(d.shape[0]):
        qh_block = ct.reshape(ct.load(
            qh, (block, ct.bid(1), 0), (1, tile_n, 2),
            padding_mode=ct.PaddingMode.ZERO, allow_tma=False), (tile_n, 2))
        base_scale = ct.reshape(ct.astype(ct.load(
            d, (block, ct.bid(1)), (1, tile_n), padding_mode=ct.PaddingMode.ZERO,
            allow_tma=False), ct.float32), (tile_n, 1))
        packed_scale = ct.reshape(ct.load(
            scales, (block, ct.bid(1)), (1, tile_n), padding_mode=ct.PaddingMode.ZERO,
            allow_tma=False), (tile_n, 1))
        # One contiguous LDG tile supplies all eight groups, amortizing global
        # load issue and latency across the group loop.
        sign_block = ct.reshape(ct.load(
            signs, (block, ct.bid(1)), (1, tile_n * 8),
            padding_mode=ct.PaddingMode.ZERO, allow_tma=False), (tile_n, 8))
        xs_block = ct.load(x_scales, (ct.bid(0), block), (tile_m, 8),
                           padding_mode=ct.PaddingMode.ZERO)
        for group in range(8):
            x_group = ct.load(x_values, (ct.bid(0), block * 8 + group), (tile_m, 32),
                              padding_mode=ct.PaddingMode.ZERO, allow_tma=x_allow_tma)
            x = ct.bitcast(x_group, ct.float8_e4m3fn)
            sign_word = ct.extract(sign_block, (0, group), shape=(tile_n, 1))
            xs = ct.bitcast(ct.extract(xs_block, (0, group), shape=(tile_m, 1)),
                            ct.float8_e8m0fnu)
            qs_words = ct.load(qs, (block, ct.bid(1), group), (1, tile_n, 2),
                               padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
            qh_word = ct.extract(qh_block, (0, group // 4), shape=(tile_n, 1))
            high_byte = ct.astype(qh_word >> ct.astype((group % 4) * 8, ct.uint32), ct.uint8)
            bits = _decode_iq3_group_words(qs_words, high_byte, sign_word, iq3_table, tile_n)
            w = ct.bitcast(bits, ct.float8_e4m3fn)
            ws = ct.transpose(base_scale * ct.astype(
                1 + 2 * ((packed_scale >> ct.astype(group * 4, ct.uint32)) & 15), ct.float32))
            partial = ct.mma_scaled(x, xs, ct.transpose(w), one,
                                    ct.zeros((tile_m, tile_n), dtype=ct.float32))
            total += partial * ws
    ct.store(out, (ct.bid(0), ct.bid(1)), total, allow_tma=store_allow_tma)


@ct.kernel
def iq3_s_fp8_matmul_medium(x_values, x_scales, d, qs, qh, signs, scales, iq3_table, out,
                       tile_m: ct.Constant[int], tile_n: ct.Constant[int],
                       x_allow_tma: ct.Constant[bool], store_allow_tma: ct.Constant[bool], split_k: ct.Constant[int]):
    """Small-token projection with transposed MMA operands.

    Buffers: x_values uint8 [M,K], x_scales uint8 [M,K/32], M < LARGE_TOKEN_THRESHOLD;
    weight fields as in IQ3SFields, iq3_table uint32 [512], out float32 [M,N].
    tile_m is 4 for M <= 4 and 8 otherwise. Decode W [tile_n,32], then
    compute W @ X.T using SM120 m16n8k32. Replicate each activation row adjacently until
    tile_n * fake_tile_m covers at least four warp MMA tiles (4*16*8).
    With split_k > 1, grid.z partitions K blocks and atomically adds partials
    to a zeroed out buffer. The caller must reset out before every launch.
    split_k == 1 uses a regular store. Bounds checks mask padded rows.
    """
    fake_tile_m = 512 // tile_n if tile_n < 64 else 8
    copies_per_token = fake_tile_m // tile_m
    row, token_tile = ct.bid(0), ct.bid(1)
    # Adjacent copies put duplicate token results in each thread's MMA registers.
    # Select one copy before accumulating; convert the small tile for output later.
    total = ct.zeros((tile_n, tile_m), dtype=ct.float32)
    one = ct.full((tile_n, 1), 1.0, dtype=ct.float8_e8m0fnu)
    block_begin = ct.bid(2) if split_k > 1 else 0
    for block in range(block_begin, d.shape[0], split_k):
        weight_scales = _decode_iq3_scales(d, scales, row, block, tile_n)
        x_block = ct.load(x_values, (token_tile, block), (tile_m, 256),
                          padding_mode=ct.PaddingMode.ZERO, allow_tma=x_allow_tma)
        xs_block = ct.load(x_scales, (token_tile, block), (tile_m, 8),
                           padding_mode=ct.PaddingMode.ZERO)
        for group in range(8):
            sign_word = ct.reshape(ct.load(signs, (block, row, group), (1, tile_n, 1),
                                           padding_mode=ct.PaddingMode.ZERO, allow_tma=False), (tile_n, 1))
            bits = _decode_iq3_group(qs, qh, sign_word, iq3_table, row, block, group, tile_n)
            w = ct.bitcast(bits, ct.float8_e4m3fn)
            x = ct.extract(x_block, (0, group), shape=(tile_m, 32))
            x = ct.reshape(ct.broadcast_to(ct.expand_dims(x, 1),
                                           (tile_m, copies_per_token, 32)), (fake_tile_m, 32))
            xs = ct.extract(xs_block, (0, group), shape=(tile_m, 1))
            xs = ct.reshape(ct.broadcast_to(ct.expand_dims(xs, 1),
                                            (tile_m, copies_per_token, 1)), (fake_tile_m, 1))
            partial = ct.mma_scaled(w, one,
                                    ct.transpose(ct.bitcast(x, ct.float8_e4m3fn)),
                                    ct.transpose(ct.bitcast(xs, ct.float8_e8m0fnu)),
                                    ct.zeros((tile_n, fake_tile_m), dtype=ct.float32))
            partial = ct.reshape(partial, (tile_n, tile_m, copies_per_token))
            partial = ct.extract(partial, (0, 0, 0), shape=(tile_n, tile_m, 1))
            partial = ct.reshape(partial, (tile_n, tile_m))
            total += partial * ct.extract(weight_scales, (0, group), shape=(tile_n, 1))
    result = ct.transpose(total)
    if split_k == 1:
        ct.store(out, (token_tile, row), result, allow_tma=store_allow_tma)
    else:
        token_index = ct.expand_dims(token_tile * tile_m + ct.arange(tile_m, dtype=ct.int32), 1)
        row_index = ct.expand_dims(row * tile_n + ct.arange(tile_n, dtype=ct.int32), 0)
        ct.atomic_add(out, (token_index, row_index), result, memory_order=ct.MemoryOrder.RELAXED)


def _power_of_two(value: int, name: str) -> None:
    if value < 1 or value & (value - 1):
        raise ValueError(f"{name} must be a positive power of two")


def launch_dequant_iq3_s(iq3_weight: IQ3SFields, *, tile_n: int = 1,
                         stream=None) -> tuple[cp.ndarray, cp.ndarray]:
    """Return [N,K] E4M3 bit patterns and [N,K/32] original F32 scales."""
    _power_of_two(tile_n, "tile_n")
    n_rows, blocks = _weight_dimensions(iq3_weight)
    values = cp.empty((n_rows, blocks, 256), dtype=cp.uint8)
    scales = cp.empty((n_rows, blocks, 8), dtype=cp.float32)
    ct.launch(stream or cp.cuda.get_current_stream(), (ct.cdiv(n_rows, tile_n), blocks),
              decode_iq3_s_blocks,
              (*_weight_kernel_args(iq3_weight), _get_iq3_table(), values, scales, tile_n))
    return values.reshape(n_rows, blocks * 256), scales.reshape(n_rows, blocks * 8)


@lru_cache(maxsize=None)
def _matmul_kernel_for_occupancy(occupancy: int):
    return iq3_s_fp8_matmul.replace_hints(occupancy=occupancy)


@lru_cache(maxsize=None)
def _medium_kernel_for_occupancy(occupancy: int):
    return iq3_s_fp8_matmul_medium.replace_hints(occupancy=occupancy)


def launch_iq3_s_fp8_matmul(x_values: cp.ndarray, x_scales: cp.ndarray, iq3_weight: IQ3SFields,
                            *, stream=None) -> cp.ndarray:
    """Multiply prepared SoA weights by prequantized MXFP8 activations."""
    n_rows, k_blocks = _weight_dimensions(iq3_weight)
    k = k_blocks * 256
    if x_values.ndim != 2 or x_values.dtype != cp.uint8 or x_values.shape[0] < 1 or x_values.shape[1] != k:
        raise ValueError("x_values must be uint8 [M,K]")
    if x_scales.shape != (x_values.shape[0], k // 32) or x_scales.dtype != cp.uint8:
        raise ValueError("x_scales must be uint8 [M,K/32]")
    stream = stream or cp.cuda.get_current_stream()
    sm = f"sm{cp.cuda.Device().compute_capability}"
    if sm not in BEST_CONFIGS:
        raise RuntimeError(f"IQ3_S FP8 matmul has no tuned config for {sm}")
    if x_values.shape[0] < LARGE_TOKEN_THRESHOLD:
        config = BEST_CONFIGS[sm]["medium"]
        tile_m = 4 if x_values.shape[0] <= 4 else 8
        out = cp.empty((x_values.shape[0], n_rows), dtype=cp.float32)
        split_k = config["split_k"]
        if split_k > 1:
            cp.cuda.runtime.memsetAsync(out.data.ptr, 0, out.nbytes, stream.ptr)
        ct.launch(stream,
                  (ct.cdiv(n_rows, config["tile_n"]), ct.cdiv(x_values.shape[0], tile_m), split_k),
                  _medium_kernel_for_occupancy(config["occupancy"]),
                  (x_values, x_scales, *_weight_kernel_args(iq3_weight), _get_iq3_table(), out,
                   tile_m, config["tile_n"],
                   config["x_allow_tma"], config["store_allow_tma"], split_k))
        return out
    config = BEST_CONFIGS[sm]["large"]
    tile_m, tile_n = config["tile_m"], config["tile_n"]
    out = cp.empty((x_values.shape[0], n_rows), dtype=cp.float32)
    ct.launch(stream or cp.cuda.get_current_stream(),
              (ct.cdiv(x_values.shape[0], tile_m), ct.cdiv(n_rows, tile_n)),
              _matmul_kernel_for_occupancy(config["occupancy"]),
              (x_values, x_scales, *_large_weight_kernel_args(iq3_weight), _get_iq3_table(), out,
               tile_m, tile_n, config["x_allow_tma"], config["store_allow_tma"]))
    return out


def make_benchmark_cases(iq3_weight: cp.ndarray) -> list:
    """Prepare SoA weights and FP8 inputs once; benchmark only the matmul."""
    from mambo.kernels.quantize_fp8 import launch_quantize_fp8
    from mambo.utils.kernel_benchmark import BenchmarkCase

    iq3_weight = prepare_iq3_s_fields(iq3_weight)
    n_rows, k_blocks = _weight_dimensions(iq3_weight)
    k = k_blocks * IQ3_S_BLOCK_VALUES
    iq3_table = _get_iq3_table()
    kernel = iq3_s_fp8_matmul
    weight_args = _weight_kernel_args(iq3_weight)
    large_weight_args = _large_weight_kernel_args(iq3_weight)
    stream = cp.cuda.get_current_stream()
    # On SM120, 16x16 lowers to FFMA instead of QMMA for this kernel.
    search_space = [
        {"tile_m": tile_m, "tile_n": tile_n,
         "x_allow_tma": x_allow_tma, "store_allow_tma": store_allow_tma,
         "occupancy": occupancy}
        for tile_m in (16, 32, 64, 128)
        for tile_n in (16, 32, 64, 128)
        for x_allow_tma in (False, True)
        for store_allow_tma in (False, True)
        for occupancy in (1, 2, 4)
        if (tile_m, tile_n) != (16, 16)
    ]

    def make_case(m: int) -> BenchmarkCase:
        x = cp.random.standard_normal((m, k), dtype=cp.float32)
        x_values, x_scales = launch_quantize_fp8(x, stream=stream)
        out = cp.zeros((m, n_rows), dtype=cp.float32) if m == 4 else cp.empty((m, n_rows), dtype=cp.float32)

        def grid(config):
            if m == 4:
                return (ct.cdiv(n_rows, config["tile_n"]), ct.cdiv(m, config["tile_m"]), config["split_k"])
            return (ct.cdiv(m, config["tile_m"]), ct.cdiv(n_rows, config["tile_n"]))

        def args(config):
            common = (x_values, x_scales, *(weight_args if m == 4 else large_weight_args), iq3_table, out,
                      config["tile_m"], config["tile_n"],
                      config["x_allow_tma"], config["store_allow_tma"])
            return (*common, config["split_k"]) if m == 4 else common

        def hints(config):
            return {"occupancy": config["occupancy"]}

        def reset_output(config):
            if config["split_k"] > 1:
                cp.cuda.runtime.memsetAsync(out.data.ptr, 0, out.nbytes, stream.ptr)

        return BenchmarkCase(
            name=f"M={m} N={n_rows} K={k}", stream=stream,
            kernel=iq3_s_fp8_matmul_medium if m == 4 else kernel,
            search_space=[dict(tile_m=4 if m <= 4 else 8, tile_n=tn, x_allow_tma=xt,
                               store_allow_tma=st, occupancy=occ, split_k=sk)
                          for tn in (32, 64, 128) for sk in (1, 2, 4, 8)
                          for xt in (False, True)
                          for st in ((False, True) if sk == 1 else (False,))
                          for occ in (1, 2, 4)] if m == 4 else search_space,
            grid_fn=grid, args_fn=args, hints_fn=hints,
            summary_keys=("tile_m", "tile_n"),
            before_launch_fn=reset_output if m == 4 else None,
        )

    return [make_case(m) for m in (4, 128)]
