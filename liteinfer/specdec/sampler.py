"""Sampling utilities for (speculative) decoding.

Everything here operates on logits or probability vectors over the vocabulary.
The functions are deliberately explicit — temperature, then optional top-k /
top-p filtering, then a categorical draw — so the speculative sampler can reuse
exactly the same distribution for both the draft and target models.
"""

from __future__ import annotations

import numpy as np


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
