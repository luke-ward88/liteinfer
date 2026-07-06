"""Paged multi-head attention over the KV-cache.

Given a query for the newest token and a sequence's block table, this gathers the
cached keys/values through the paged store and runs standard scaled dot-product
attention. The point is to show that paging is transparent to the math: the
result is identical to attending over a contiguous KV buffer.
"""

from __future__ import annotations

import numpy as np

from liteinfer.kvcache.manager import BlockSpaceManager
from liteinfer.kvcache.paged_cache import PagedKVStore


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def paged_attention(
    query: np.ndarray,
    seq_id: int,
    store: PagedKVStore,
    manager: BlockSpaceManager,
) -> np.ndarray:
    """Attend a single ``(num_heads, head_dim)`` query over a cached sequence.

    Returns the ``(num_heads, head_dim)`` attention output. Attention is causal
    by construction: only the ``length`` tokens already written to the cache are
    visible.
    """
    meta = manager.get(seq_id)
    keys, values = store.gather(meta.block_table, meta.length)
    if meta.length == 0:
        return np.zeros_like(query)

    # keys/values: (length, num_heads, head_dim); query: (num_heads, head_dim)
    head_dim = query.shape[-1]
    scale = 1.0 / np.sqrt(head_dim)
    # scores per head over the length axis
    scores = np.einsum("hd,lhd->hl", query, keys) * scale
    weights = _softmax(scores, axis=-1)
    return np.einsum("hl,lhd->hd", weights, values)
