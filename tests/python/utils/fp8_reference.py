"""Independent NumPy reference codecs for E4M3 and block-scaled FP8."""

from __future__ import annotations

import numpy as np

FP8_GROUP_K = 32


def _e4m3fn_positive_table() -> tuple[np.ndarray, np.ndarray]:
    """Return finite nonnegative E4M3FN values and their 7-bit encodings."""
    codes = np.arange(0x80, dtype=np.uint8)
    exponent = (codes >> 3) & 0x0F
    mantissa = codes & 0x07
    values = np.empty(codes.shape, dtype=np.float64)
    subnormal = exponent == 0
    normal = (exponent > 0) & ((exponent < 15) | (mantissa < 7))
    values[subnormal] = mantissa[subnormal] * (2.0**-9)
    values[normal] = (1.0 + mantissa[normal] / 8.0) * np.exp2(
        exponent[normal].astype(np.float64) - 7.0
    )
    values[~(subnormal | normal)] = np.nan
    finite = np.isfinite(values)
    return values[finite], codes[finite]


_E4M3_POS_VALUES, _E4M3_POS_CODES = _e4m3fn_positive_table()


def decode_e4m3fn(raw: np.ndarray) -> np.ndarray:
    bits = np.asarray(raw, dtype=np.uint8)
    exponent = (bits >> 3) & 0x0F
    mantissa = bits & 0x07
    values = np.empty(bits.shape, dtype=np.float32)
    subnormal = exponent == 0
    normal = (exponent > 0) & ((exponent < 15) | (mantissa < 7))
    values[subnormal] = mantissa[subnormal].astype(np.float32) * (2.0**-9)
    values[normal] = (
        (1.0 + mantissa[normal].astype(np.float32) / 8.0)
        * np.exp2(exponent[normal].astype(np.float32) - 7.0)
    )
    values[~(subnormal | normal)] = np.nan
    return np.where((bits & 0x80) != 0, -values, values)


def _decode_e8m0(raw: np.ndarray) -> np.ndarray:
    codes = np.asarray(raw, dtype=np.uint8)
    if np.any(codes == 0xFF):
        raise AssertionError("E8M0 scale output must not contain the NaN encoding")
    return np.exp2(codes.astype(np.int16) - 127).astype(np.float32)


def decode_mxfp8(q: np.ndarray, scales: np.ndarray) -> np.ndarray:
    q_values = decode_e4m3fn(q)
    scale_values = _decode_e8m0(scales)
    if q_values.ndim != 2 or scale_values.shape != (q_values.shape[0], q_values.shape[1] // FP8_GROUP_K):
        raise AssertionError("MXFP8 activation data and scale shapes do not match")
    return q_values * np.repeat(scale_values, FP8_GROUP_K, axis=1)


def encode_mxfp8_reference(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Small independent CPU reference for E4M3FN values with E8M0 block scales."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] % FP8_GROUP_K:
        raise ValueError("reference input must have shape [M,K] with K divisible by 32")
    m, k = x.shape
    q = np.empty((m, k), dtype=np.uint8)
    scale_codes = np.empty((m, k // FP8_GROUP_K), dtype=np.uint8)
    for row in range(m):
        for block in range(k // FP8_GROUP_K):
            values = x[row, block * FP8_GROUP_K : (block + 1) * FP8_GROUP_K]
            amax = float(np.max(np.abs(values)))
            if amax == 0.0:
                exponent = 0
            else:
                exponent = int(np.ceil(np.log2(amax / 448.0)))
                exponent = min(127, max(-127, exponent))
            scale_codes[row, block] = exponent + 127
            normalized = values.astype(np.float64) / np.exp2(exponent)
            nearest = np.abs(normalized[:, None])
            nearest = np.argmin(np.abs(nearest - _E4M3_POS_VALUES[None, :]), axis=1)
            code = _E4M3_POS_CODES[nearest].copy()
            code |= np.where(np.signbit(normalized), 0x80, 0).astype(np.uint8)
            q[row, block * FP8_GROUP_K : (block + 1) * FP8_GROUP_K] = code
    return q, scale_codes

