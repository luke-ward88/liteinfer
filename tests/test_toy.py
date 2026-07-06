import numpy as np

from liteinfer.models.toy import BigramModel, NGramModel


def test_probs_are_valid_distribution():
    model = NGramModel(vocab_size=32, order=2, seed=0)
    p = model.next_token_probs([3, 7])
    assert p.shape == (32,)
    assert np.isclose(p.sum(), 1.0)
    assert np.all(p >= 0)


def test_same_context_is_deterministic():
    a = NGramModel(20, order=3, seed=5)
    b = NGramModel(20, order=3, seed=5)
    np.testing.assert_array_equal(a.next_token_probs([1, 2, 3]), b.next_token_probs([9, 1, 2, 3]))


def test_higher_temperature_flattens():
    cold = NGramModel(50, order=2, seed=1, temperature=0.5)
    hot = NGramModel(50, order=2, seed=1, temperature=4.0)

    # a flatter distribution has higher entropy
    def entropy(p):
        return -np.sum(p * np.log(p + 1e-12))

    assert entropy(hot.next_token_probs([0])) > entropy(cold.next_token_probs([0]))


def test_bigram_is_order_two():
    assert BigramModel(10).order == 2
