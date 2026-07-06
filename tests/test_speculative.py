import numpy as np

from liteinfer.models.toy import NGramModel
from liteinfer.specdec.ngram import make_draft_target
from liteinfer.specdec.speculative import speculative_decode


def test_speculative_generates_requested_length():
    draft, target = make_draft_target(vocab_size=32, seed=1)
    rng = np.random.default_rng(0)
    res = speculative_decode([0, 1], draft, target, gamma=4, max_new_tokens=20, rng=rng)
    assert len(res.tokens) - 2 == 20
    assert 0.0 <= res.acceptance_rate <= 1.0


def test_identical_models_accept_everything():
    # if draft == target, every proposal should be accepted
    model = NGramModel(16, order=2, seed=3, temperature=1.0)
    rng = np.random.default_rng(0)
    res = speculative_decode([0], model, model, gamma=4, max_new_tokens=16, rng=rng)
    assert res.acceptance_rate > 0.95


def test_output_distribution_matches_target():
    # single-step generation: the first new token must follow the target dist
    vocab = 8
    target = NGramModel(vocab, order=1, seed=7, temperature=1.0)
    draft = NGramModel(vocab, order=1, seed=7, temperature=2.0)
    p_true = target.next_token_probs([0])

    counts = np.zeros(vocab)
    n = 6000
    for i in range(n):
        rng = np.random.default_rng(i)
        res = speculative_decode([0], draft, target, gamma=3, max_new_tokens=1, rng=rng)
        counts[res.tokens[-1]] += 1
    empirical = counts / n
    assert np.max(np.abs(empirical - p_true)) < 0.03


def test_gamma_zero_falls_back_to_target():
    # with no draft proposals the loop degrades to plain target sampling
    draft, target = make_draft_target(vocab_size=16, seed=2)
    rng = np.random.default_rng(0)
    res = speculative_decode([0], draft, target, gamma=0, max_new_tokens=8, rng=rng)
    assert len(res.tokens) - 1 == 8
    assert res.proposed == 0
    assert res.acceptance_rate == 0.0
