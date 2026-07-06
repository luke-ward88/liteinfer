"""Deterministic toy language models.

These are *not* trained networks — they are fixed, seedable probability tables
over a small vocabulary. That is exactly what the KV-cache and speculative
decoding demos need: something with a well-defined ``next_token_probs`` that runs
instantly, offline, and reproducibly, so the interesting behaviour on show is the
inference machinery rather than a model download.

``NGramModel`` builds a smoothed conditional table from a context window; a
``BigramModel`` is the ``n=2`` special case. A ``temperature`` knob lets a demo
derive a deliberately *weaker* draft model from the same table, which is what
makes speculative decoding show a non-trivial acceptance rate.
"""

from __future__ import annotations

import numpy as np


class NGramModel:
    """A fixed n-gram next-token distribution over ``vocab_size`` tokens."""

    def __init__(
        self,
        vocab_size: int,
        order: int = 2,
        *,
        seed: int = 0,
        temperature: float = 1.0,
    ) -> None:
        if order < 1:
            raise ValueError("order must be >= 1")
        self.vocab_size = vocab_size
        self.order = order
        self.temperature = temperature
        # A deterministic table keyed by the hash of the last (order-1) tokens.
        # We lazily materialize rows so the table stays tiny for large vocabs.
        self._seed = seed
        self._cache: dict[int, np.ndarray] = {}

    def _context_key(self, prefix: list[int]) -> int:
        ctx = tuple(prefix[-(self.order - 1) :]) if self.order > 1 else ()
        # stable hash independent of Python's randomized hashing
        h = self._seed
        for t in ctx:
            h = (h * 1000003 + (t + 1)) & 0xFFFFFFFF
        return h

    def _row(self, key: int) -> np.ndarray:
        row = self._cache.get(key)
        if row is None:
            rng = np.random.default_rng(key)
            logits = rng.standard_normal(self.vocab_size)
            row = self._softmax(logits)
            self._cache[key] = row
        return row

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        logits = logits / self.temperature
        logits = logits - logits.max()
        e = np.exp(logits)
        return e / e.sum()

    def next_token_probs(self, prefix: list[int]) -> np.ndarray:
        """Return the next-token probability vector given ``prefix``."""
        return self._row(self._context_key(prefix))


class BigramModel(NGramModel):
    """Convenience alias for an order-2 n-gram model."""

    def __init__(self, vocab_size: int, *, seed: int = 0, temperature: float = 1.0) -> None:
        super().__init__(vocab_size, order=2, seed=seed, temperature=temperature)
