"""Paged storage for key/value tensors.

Physical memory is a single big array shaped
``(num_blocks, block_size, num_heads, head_dim)`` for keys, and likewise for
values. Logical token positions are mapped onto ``(block, slot)`` pairs by the
space manager; this module just owns the backing store and the read/write of
individual slots.
"""

from __future__ import annotations

import numpy as np


class PagedKVStore:
    """Backing store for paged keys and values."""

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        dtype: np.dtype = np.float32,
    ) -> None:
        shape = (num_blocks, block_size, num_heads, head_dim)
        self.keys = np.zeros(shape, dtype=dtype)
        self.values = np.zeros(shape, dtype=dtype)
        self.block_size = block_size

    def write(self, block_id: int, slot: int, k: np.ndarray, v: np.ndarray) -> None:
        """Write a single token's key/value into ``(block_id, slot)``."""
        self.keys[block_id, slot] = k
        self.values[block_id, slot] = v
