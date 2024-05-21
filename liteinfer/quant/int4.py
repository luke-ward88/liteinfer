"""INT4 post-training quantization with nibble packing.

INT4 codes are stored two-per-byte (low nibble first) so an ``(N,)`` array of
4-bit values occupies ``ceil(N/2)`` bytes. This module provides the packing
primitives; group-wise quantization is layered on top.
"""

from __future__ import annotations

import numpy as np

__all__ = ["pack_nibbles", "unpack_nibbles"]


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
