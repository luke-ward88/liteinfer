"""The speculative-sampling accept/reject test.

This implements the DeepMind speculative sampling correction (Chen et al., 2023;
Leviathan et al., 2023). A draft model proposes token ``x`` with probability
``q(x)``; the target model assigns it ``p(x)``. Accept ``x`` with probability
``min(1, p(x) / q(x))``. On rejection, resample from the residual distribution
``normalize(max(p - q, 0))``. The key property: the accepted stream is
distributed *exactly* as if it had been sampled from the target ``p`` directly —
speculation only changes the speed, not the output distribution.
"""

from __future__ import annotations

import numpy as np


def accept_token(
    token: int,
    p: np.ndarray,
    q: np.ndarray,
    rng: np.random.Generator,
) -> bool:
    """Return whether draft ``token`` is accepted under target ``p`` vs draft ``q``.

    Accept with probability ``min(1, p[token] / q[token])``. If the draft never
    proposes a token (``q[token] == 0``) we conservatively reject.
    """
    qi = float(q[token])
    if qi <= 0:
        return False
    # min(1, p/q): a draft token the target likes at least as much is always kept
    ratio = min(1.0, float(p[token]) / qi)
    return bool(rng.random() < ratio)


def residual_distribution(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Return the normalized residual ``max(p - q, 0)``.

    When a draft token is rejected we must sample the replacement from this
    residual so that, combined with the acceptances, the overall output matches
    the target distribution ``p`` exactly. Using ``max(p - q, 0)`` (a ReLU, not a
    symmetric clamp) is what makes the correction exact.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    resid = np.maximum(p - q, 0.0)
    total = resid.sum()
    if total <= 0:
        # p and q coincide on their support; fall back to p
        return p / p.sum()
    return resid / total


def resample_on_reject(
    p: np.ndarray,
    q: np.ndarray,
    rng: np.random.Generator,
) -> int:
    """Draw a replacement token from the residual distribution."""
    resid = residual_distribution(p, q)
    return int(rng.choice(resid.size, p=resid))
