"""Prepare persistent IQ3_S SoA weight fields for GPU kernels."""

from __future__ import annotations

from dataclasses import dataclass

import cupy as cp


@dataclass(frozen=True)
class IQ3SFields:
    """Byte-preserving SoA, with K-block first and N rows second.

    d: float16 [B,N]; qs: uint32 [B,N,16]; qh: uint16 [B,N,4];
    signs: int32 [B,N,8]; scales: uint32 [B,N]. Total: 110 bytes/block.
    Prepare once on the launch stream, then reuse for all projections.
    """
    d: cp.ndarray
    qs: cp.ndarray
    qh: cp.ndarray
    signs: cp.ndarray
    scales: cp.ndarray


def prepare_iq3_s_fields(iq3_weight: cp.ndarray) -> IQ3SFields:
    """Split GGUF uint8 [N,B,110] blocks into aligned SoA fields once.

    One IQ3_S block along K (256 weights, 110 bytes):

        byte  0..1   | d: FP16 base scale                 |  2 bytes
        byte  2..65  | qs: 64 low 8-bit codebook indices  | 64 bytes
        byte 66..73  | qh: 64 high index bits             |  8 bytes
        byte 74..105 | signs: one bit per weight          | 32 bytes
        byte 106..109| scales: 8 packed 4-bit values      |  4 bytes
                    +------------------------------------+ 110 bytes

    Each 9-bit (qs, qh) index selects four magnitudes from the codebook.
    Each group of 32 weights uses scale d * (1 + 2 * scale_nibble).
    Reorder each field to [B,N,...] for the kernels; preserve every bit.
    Keep these arrays for reuse. This conversion is outside kernel timing.
    """
    if iq3_weight.ndim != 3 or iq3_weight.dtype != cp.uint8 or iq3_weight.shape[2] != 110:
        raise ValueError("IQ3_S weight must be uint8 [N,K/256,110]")
    n, blocks, _ = iq3_weight.shape
    if n < 1 or blocks < 1:
        raise ValueError("IQ3_S weight must have positive N and K")

    def field(offset, size, dtype):
        data = cp.array(iq3_weight[..., offset:offset + size].transpose(1, 0, 2), copy=True, order="C")
        return data.view(dtype)

    return IQ3SFields(field(0, 2, cp.float16).reshape(blocks, n),
                      field(2, 64, cp.uint32), field(66, 8, cp.uint16),
                      field(74, 32, cp.int32), field(106, 4, cp.uint32).reshape(blocks, n))


def _weight_dimensions(iq3_weight: IQ3SFields) -> tuple[int, int]:
    """Return N and the number of K blocks in the prepared fields."""
    if not isinstance(iq3_weight, IQ3SFields):
        raise TypeError("prepare IQ3_S weights with prepare_iq3_s_fields before launch")
    blocks, n_rows = iq3_weight.d.shape
    return n_rows, blocks


def _weight_kernel_args(iq3_weight: IQ3SFields) -> tuple[cp.ndarray, ...]:
    return (iq3_weight.d, iq3_weight.qs, iq3_weight.qh, iq3_weight.signs, iq3_weight.scales)


def _large_weight_kernel_args(iq3_weight: IQ3SFields) -> tuple[cp.ndarray, ...]:
    """Make zero-copy views of the contiguous prepared fields for large MMA.

    QH becomes uint32 [B,N,2]; signs becomes int32 [B,N*8]. The stored bytes,
    IQ3SFields layout, and other fields are unchanged. Prepare once with
    prepare_iq3_s_fields; these views only change the kernel's array metadata.
    """
    return (iq3_weight.d, iq3_weight.qs, iq3_weight.qh.view(cp.uint32),
            iq3_weight.signs.reshape(iq3_weight.signs.shape[0], -1), iq3_weight.scales)
