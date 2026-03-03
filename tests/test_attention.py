import numpy as np

from liteinfer.kvcache.attention import paged_attention
from liteinfer.kvcache.blocks import BlockAllocator
from liteinfer.kvcache.manager import BlockSpaceManager
from liteinfer.kvcache.paged_cache import PagedKVStore


def _dense_attention(query, keys, values):
    scale = 1.0 / np.sqrt(query.shape[-1])
    scores = np.einsum("hd,lhd->hl", query, keys) * scale
    scores = scores - scores.max(axis=-1, keepdims=True)
    w = np.exp(scores)
    w = w / w.sum(axis=-1, keepdims=True)
    return np.einsum("hl,lhd->hd", w, values)


def test_paged_matches_dense_reference():
    rng = np.random.default_rng(0)
    num_heads, head_dim, block_size, length = 4, 8, 4, 10

    store = PagedKVStore(
        num_blocks=8, block_size=block_size, num_heads=num_heads, head_dim=head_dim
    )
    mgr = BlockSpaceManager(BlockAllocator(num_blocks=8, block_size=block_size))
    mgr.add_sequence(0)

    keys = rng.standard_normal((length, num_heads, head_dim)).astype(np.float32)
    values = rng.standard_normal((length, num_heads, head_dim)).astype(np.float32)
    for pos in range(length):
        block_id, slot = mgr.append_token(0)
        store.write(block_id, slot, keys[pos], values[pos])

    q = rng.standard_normal((num_heads, head_dim)).astype(np.float32)
    got = paged_attention(q, 0, store, mgr)
    want = _dense_attention(q, keys, values)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_empty_sequence_returns_zeros():
    store = PagedKVStore(2, 4, 2, 4)
    mgr = BlockSpaceManager(BlockAllocator(2, 4))
    mgr.add_sequence(0)
    q = np.ones((2, 4), dtype=np.float32)
    out = paged_attention(q, 0, store, mgr)
    assert np.allclose(out, 0.0)
