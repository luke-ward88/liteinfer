"""Physical block pool for the paged KV-cache.

The cache memory is carved into fixed-size *blocks*, each holding ``block_size``
token slots. A :class:`BlockAllocator` hands out free physical block ids and
takes them back when a sequence finishes — exactly the coarse-grained allocator
that lets paged attention avoid per-sequence contiguous reservations.
"""

from __future__ import annotations

from collections import deque

from liteinfer.errors import CacheFullError


class BlockAllocator:
    """A free-list allocator over ``num_blocks`` fixed-size physical blocks."""

    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free: deque[int] = deque(range(num_blocks))

    @property
    def num_free(self) -> int:
        return len(self._free)

    def allocate(self) -> int:
        """Pop and return a free physical block id."""
        if not self._free:
            raise CacheFullError("no free KV-cache blocks left")
        return self._free.popleft()

    def free(self, block_id: int) -> None:
        """Return a physical block id to the pool."""
        self._free.append(block_id)

    def free_many(self, block_ids: list[int]) -> None:
        self._free.extend(block_ids)
