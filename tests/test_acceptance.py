import numpy as np

from liteinfer.specdec.acceptance import (
    accept_token,
    resample_on_reject,
    residual_distribution,
)


def _one_speculative_draw(p, q, rng):
    """Draw a single token via one round of draft-then-correct."""
    token = int(rng.choice(q.size, p=q))
    if accept_token(token, p, q, rng):
        return token
    return resample_on_reject(p, q, rng)


def test_residual_is_normalized_and_nonnegative():
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.2, 0.6, 0.2])
    r = residual_distribution(p, q)
    assert np.all(r >= 0)
    assert np.isclose(r.sum(), 1.0)


def test_acceptance_preserves_target_distribution():
    rng = np.random.default_rng(0)
    p = np.array([0.5, 0.2, 0.2, 0.1])
    q = np.array([0.1, 0.4, 0.3, 0.2])

    counts = np.zeros(p.size)
    n = 40000
    for _ in range(n):
        counts[_one_speculative_draw(p, q, rng)] += 1
    empirical = counts / n
    # empirical distribution of accepted tokens must match target p
    assert np.max(np.abs(empirical - p)) < 0.02


def test_identical_draft_always_accepts():
    rng = np.random.default_rng(1)
    p = np.array([0.25, 0.25, 0.25, 0.25])
    # when q == p, ratio is 1 everywhere and every proposal is accepted
    accepts = sum(accept_token(t, p, p, rng) for t in range(4) for _ in range(50))
    assert accepts == 200
