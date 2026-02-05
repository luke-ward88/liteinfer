"""Sampling utilities for (speculative) decoding.

Everything here operates on logits or probability vectors over the vocabulary.
The functions are deliberately explicit — temperature, then optional top-k /
top-p filtering, then a categorical draw — so the speculative sampler can reuse
exactly the same distribution for both the draft and target models.
"""

from __future__ import annotations

import numpy as np


def make_rng(seed: int | np.random.Generator | None = None) -> np.random.Generator:
    """Coerce a seed / existing generator / ``None`` into a ``Generator``.

    Centralizing this keeps every entry point (CLI, examples, the decode loop)
    from re-implementing the same "seed or generator?" dance.
    """
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Temperature-scaled softmax over the last axis.

    ``temperature`` is applied to the logits (not the probabilities); as it goes
    to 0 the distribution approaches a one-hot argmax.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = np.asarray(logits, dtype=np.float64) / temperature
    logits = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=-1, keepdims=True)


def sample(probs: np.ndarray, rng: np.random.Generator) -> int:
    """Draw a single index from a 1-D probability vector."""
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    return int(rng.choice(probs.size, p=probs))


def top_k_filter(probs: np.ndarray, k: int) -> np.ndarray:
    """Zero out all but the ``k`` highest-probability entries and renormalize."""
    if k <= 0 or k >= probs.size:
        return probs
    probs = np.asarray(probs, dtype=np.float64).copy()
    # indices of the k largest probs
    keep = np.argpartition(probs, -k)[-k:]
    mask = np.ones_like(probs, dtype=bool)
    mask[keep] = False
    probs[mask] = 0.0
    total = probs.sum()
    return probs / total if total > 0 else probs


def top_p_filter(probs: np.ndarray, p: float) -> np.ndarray:
    """Nucleus filtering: keep the smallest set whose cumulative mass ≥ ``p``."""
    if p >= 1.0:
        return probs
    probs = np.asarray(probs, dtype=np.float64).copy()
    order = np.argsort(probs)[::-1]
    cumulative = np.cumsum(probs[order])
    # number of tokens needed to reach mass p (at least one)
    cutoff = int(np.searchsorted(cumulative, p) + 1)
    keep = order[:cutoff]
    mask = np.ones_like(probs, dtype=bool)
    mask[keep] = False
    probs[mask] = 0.0
    total = probs.sum()
    return probs / total if total > 0 else probs


def apply_filters(probs: np.ndarray, top_k: int = 0, top_p: float = 1.0) -> np.ndarray:
    """Apply top-k then top-p filtering in sequence."""
    probs = top_k_filter(probs, top_k)
    probs = top_p_filter(probs, top_p)
    return probs
