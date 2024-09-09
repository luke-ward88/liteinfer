"""Block space manager: logical sequences → physical block tables.

Each sequence owns a *block table*: an ordered list of physical block ids. Token
position ``pos`` in a sequence lives in that sequence's ``pos // block_size``-th
block at slot ``pos % block_size``. The manager grows a sequence's table on
demand and releases its blocks when the sequence ends.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from liteinfer.errors import ShapeError
from liteinfer.kvcache.block import BlockAllocator


@dataclass
class SequenceMeta:
    """Bookkeeping for one live sequence."""

    seq_id: int
    length: int = 0
    block_table: list[int] = field(default_factory=list)


class BlockSpaceManager:
    """Owns the allocator and per-sequence block tables."""

    def __init__(self, allocator: BlockAllocator) -> None:
        self.allocator = allocator
        self.block_size = allocator.block_size
        self._seqs: dict[int, SequenceMeta] = {}

    def add_sequence(self, seq_id: int) -> None:
        if seq_id in self._seqs:
            raise ValueError(f"sequence {seq_id} already exists")
        self._seqs[seq_id] = SequenceMeta(seq_id=seq_id)

    def get(self, seq_id: int) -> SequenceMeta:
        return self._seqs[seq_id]

    def _ensure_capacity(self, meta: SequenceMeta, new_length: int) -> None:
        needed_blocks = (new_length + self.block_size - 1) // self.block_size
        while len(meta.block_table) < needed_blocks:
            meta.block_table.append(self.allocator.allocate())

    def append_token(self, seq_id: int) -> tuple[int, int]:
        """Reserve the slot for the next token, returning ``(block_id, slot)``."""
        meta = self._seqs[seq_id]
        pos = meta.length
        self._ensure_capacity(meta, pos + 1)
        block_id = meta.block_table[pos // self.block_size]
        slot = pos % self.block_size
        meta.length += 1
        return block_id, slot

    def slot_for(self, seq_id: int, pos: int) -> tuple[int, int]:
        """Return the ``(block_id, slot)`` backing an existing position."""
        meta = self._seqs[seq_id]
        if pos >= meta.length:
            raise ShapeError(f"position {pos} out of range for sequence {seq_id}")
        return meta.block_table[pos // self.block_size], pos % self.block_size

    def free_sequence(self, seq_id: int) -> None:
        """Release all blocks held by a finished sequence."""
        meta = self._seqs.pop(seq_id)
        self.allocator.free_many(meta.block_table)
