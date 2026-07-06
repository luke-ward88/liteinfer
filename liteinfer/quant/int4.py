"""INT4 post-training quantization with nibble packing.

INT4 codes are stored two-per-byte (low nibble first) so an ``(N,)`` array of
4-bit values occupies ``ceil(N/2)`` bytes. This module provides the packing
primitives; group-wise quantization is layered on top.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["pack_nibbles", "unpack_nibbles", "PackedInt4", "quantize_group", "dequantize_group"]


def pack_nibbles(codes: np.ndarray) -> np.ndarray:
    """Pack an array of 4-bit unsigned codes (0..15) into bytes.

    Values are flattened, then each even-indexed code goes into the low nibble
    and the following odd-indexed code into the high nibble of a byte. If the
    number of codes is odd, the final high nibble is zero-padded.
    """
    flat = np.asarray(codes, dtype=np.uint8).ravel()
    if np.any(flat > 15):
        raise ValueError("int4 codes must be in [0, 15]")
    if flat.size % 2 == 1:
        flat = np.concatenate([flat, np.zeros(1, dtype=np.uint8)])
    low = flat[0::2]
    high = flat[1::2]
    return (low | (high << 4)).astype(np.uint8)


def unpack_nibbles(packed: np.ndarray, count: int) -> np.ndarray:
    """Inverse of :func:`pack_nibbles`, returning exactly ``count`` codes."""
    packed = np.asarray(packed, dtype=np.uint8)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    out = np.empty(packed.size * 2, dtype=np.uint8)
    out[0::2] = low
    out[1::2] = high
    return out[:count]


@dataclass(frozen=True)
class PackedInt4:
    """A group-wise INT4-quantized 1-D vector.

    Stores packed nibbles plus one ``scale`` and ``zero`` per group of
    ``group_size`` elements. ``length`` records the true (unpadded) element count
    so :func:`dequantize_group` can trim the tail.
    """

    data: np.ndarray  # packed bytes
    scale: np.ndarray  # (num_groups,) float32
    zero: np.ndarray  # (num_groups,) uint8 — quantized value of the group's min
    group_size: int
    length: int


def quantize_group(x: np.ndarray, group_size: int = 64) -> PackedInt4:
    """Asymmetric group-wise INT4 quantization of a 1-D array.

    Each contiguous block of ``group_size`` elements gets its own affine range
    mapped onto ``[0, 15]``. Small groups keep the per-group dynamic range tight,
    which is what makes 4-bit weight quantization usable in practice.
    """
    x = np.asarray(x, dtype=np.float32).ravel()
    length = x.size
    pad = (-length) % group_size
    if pad:
        x = np.concatenate([x, np.zeros(pad, dtype=np.float32)])
    groups = x.reshape(-1, group_size)

    gmin = groups.min(axis=1)
    gmax = groups.max(axis=1)
    scale = (gmax - gmin) / 15.0
    scale = np.where(scale > 0, scale, 1.0).astype(np.float32)
    zero = np.clip(np.round(-gmin / scale), 0, 15).astype(np.uint8)

    codes = np.clip(
        np.round(groups / scale[:, None]) + zero[:, None], 0, 15
    ).astype(np.uint8)
    return PackedInt4(
        data=pack_nibbles(codes),
        scale=scale,
        zero=zero,
        group_size=group_size,
        length=length,
    )


def dequantize_group(packed: PackedInt4) -> np.ndarray:
    """Inverse of :func:`quantize_group`."""
    num_groups = packed.scale.size
    codes = unpack_nibbles(packed.data, count=num_groups * packed.group_size)
    codes = codes.reshape(num_groups, packed.group_size).astype(np.float32)
    x = (codes - packed.zero[:, None].astype(np.float32)) * packed.scale[:, None]
    return x.ravel()[: packed.length]
