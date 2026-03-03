"""Walk through the paged KV-cache: allocate, write, gather, attend.

Run: ``python examples/paged_kvcache.py``
"""

import numpy as np

from liteinfer.kvcache.attention import paged_attention
from liteinfer.kvcache.blocks import BlockAllocator
from liteinfer.kvcache.manager import BlockSpaceManager
from liteinfer.kvcache.paged_cache import PagedKVStore

NUM_HEADS = 4
HEAD_DIM = 16
BLOCK_SIZE = 8
NUM_BLOCKS = 32


def main() -> None:
    rng = np.random.default_rng(0)
    store = PagedKVStore(NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_DIM)
    manager = BlockSpaceManager(BlockAllocator(NUM_BLOCKS, BLOCK_SIZE))

    # two interleaved sequences share the same physical block pool
    for seq_id, length in [(0, 20), (1, 13)]:
        manager.add_sequence(seq_id)
        for _ in range(length):
            block_id, slot = manager.append_token(seq_id)
            k = rng.standard_normal((NUM_HEADS, HEAD_DIM)).astype(np.float32)
            v = rng.standard_normal((NUM_HEADS, HEAD_DIM)).astype(np.float32)
            store.write(block_id, slot, k, v)

    for seq_id in (0, 1):
        meta = manager.get(seq_id)
        q = rng.standard_normal((NUM_HEADS, HEAD_DIM)).astype(np.float32)
        out = paged_attention(q, seq_id, store, manager)
        print(
            f"seq {seq_id}: len={meta.length:2d} "
            f"blocks={len(meta.block_table)} out_norm={np.linalg.norm(out):.4f}"
        )

    print(f"free blocks remaining: {manager.allocator.num_free}/{NUM_BLOCKS}")
    manager.free_sequence(0)
    print(f"after freeing seq 0  : {manager.allocator.num_free}/{NUM_BLOCKS}")


if __name__ == "__main__":
    main()
