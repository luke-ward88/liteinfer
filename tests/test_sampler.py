import numpy as np

from liteinfer.specdec.sampler import (
    apply_filters,
    sample,
    softmax,
    top_k_filter,
    top_p_filter,
)


def test_softmax_sums_to_one():
    logits = np.array([1.0, 2.0, 3.0])
    p = softmax(logits)
    assert np.isclose(p.sum(), 1.0)
    assert np.argmax(p) == 2


def test_low_temperature_sharpens():
    logits = np.array([1.0, 2.0])
    sharp = softmax(logits, temperature=0.1)
    soft = softmax(logits, temperature=5.0)
    assert sharp[1] > soft[1]


def test_top_k_keeps_k():
    p = np.array([0.1, 0.2, 0.3, 0.4])
    out = top_k_filter(p, k=2)
    assert np.count_nonzero(out) == 2
    assert np.isclose(out.sum(), 1.0)


def test_top_p_nucleus():
    p = np.array([0.5, 0.3, 0.15, 0.05])
    out = top_p_filter(p, p=0.8)
    # 0.5 + 0.3 = 0.8 reaches the threshold with two tokens
    assert np.count_nonzero(out) == 2


def test_sample_respects_zero_mass():
    p = apply_filters(np.array([0.9, 0.1, 0.0, 0.0]), top_k=1)
    rng = np.random.default_rng(0)
    draws = {sample(p, rng) for _ in range(20)}
    assert draws == {0}
