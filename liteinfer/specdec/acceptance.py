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
    ratio = float(p[token]) / qi
    return bool(rng.random() < ratio)
