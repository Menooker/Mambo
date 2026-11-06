"""Standalone F32 to block-scaled FP8 activation quantization.

Outputs E4M3 value bytes and one E8M0 scale byte per 32 values.
Quantization is separate from IQ3_S matmul and its benchmark timing.
"""

from __future__ import annotations

import cupy as cp
import cuda.tile as ct


# Independently tuned F32 -> MXFP8 kernel, K=5120; no matmul fusion.
BEST_CONFIGS = {
    "sm120": {
        "medium": {"tile_m": 4, "tile_k_groups": 4},
        "large": {"tile_m": 16, "tile_k_groups": 16},
    },
}
LARGE_TOKEN_THRESHOLD = 32


@ct.function
def quantize_fp8_group(data, axis: ct.Constant[int]):
    """Return E4M3 values and E8M0 scales for groups of 32 F32 values.

    Input tile: data float32, with extent 32 along axis; the activation
    kernel passes [tile_m,tile_k_groups,32] with axis=2. No global buffers.
    Returns: float8_e4m3fn tile with data's shape and float8_e8m0fnu
    scale tile with axis removed.
    """
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
    """Quantize F32 activations into E4M3 bytes and per-32 E8M0 scales.

    Global buffers: x float32 [M,K/32,32]; values uint8 [M,K/32,32]
    containing E4M3 bits; scales uint8 [M,K/32] containing E8M0 bits.
    """
    data = ct.load(x, (ct.bid(0), ct.bid(1), 0), (tile_m, tile_k_groups, 32), padding_mode=ct.PaddingMode.ZERO)
    q, scale_fp8 = quantize_fp8_group(data, 2)
    ct.store(values, (ct.bid(0), ct.bid(1), 0), ct.bitcast(q, ct.uint8))
    ct.store(scales, (ct.bid(0), ct.bid(1)), ct.bitcast(scale_fp8, ct.uint8))


def _power_of_two(value: int, name: str) -> None:
    if value < 1 or value & (value - 1):
        raise ValueError(f"{name} must be a positive power of two")


def _activation_config(m: int) -> dict:
    sm = f"sm{cp.cuda.Device().compute_capability}"
    if sm not in BEST_CONFIGS:
        raise RuntimeError(f"FP8 activation quantizer has no tuned config for {sm}")
    return BEST_CONFIGS[sm]["large" if m >= LARGE_TOKEN_THRESHOLD else "medium"]


def launch_quantize_fp8(x: cp.ndarray, *, tile_m: int | None = None, tile_k_groups: int | None = None,
                        stream=None) -> tuple[cp.ndarray, cp.ndarray]:
    """Return [M,K] E4M3 bytes and [M,K/32] E8M0 scale bytes."""
    if x.ndim != 2 or x.dtype != cp.float32 or x.shape[1] % 32:
        raise ValueError("activation must be float32 [M,K] with K divisible by 32")
    if tile_m is None or tile_k_groups is None:
        config = _activation_config(x.shape[0])
        tile_m = config["tile_m"] if tile_m is None else tile_m
        tile_k_groups = config["tile_k_groups"] if tile_k_groups is None else tile_k_groups
    _power_of_two(tile_m, "tile_m")
    _power_of_two(tile_k_groups, "tile_k_groups")
    x = cp.ascontiguousarray(x).reshape(x.shape[0], x.shape[1] // 32, 32)
    values = cp.empty(x.shape, dtype=cp.uint8)
    scales = cp.empty(x.shape[:2], dtype=cp.uint8)
    launch_grid = (ct.cdiv(x.shape[0], tile_m), ct.cdiv(x.shape[1], tile_k_groups))
    ct.launch(stream or cp.cuda.get_current_stream(), launch_grid, quantize_activation_fp8,
              (x, values, scales, tile_m, tile_k_groups))
    return values.reshape(x.shape[0], x.shape[1] * 32), scales
