import numpy as np

from liteinfer.quant.linear import QuantLinear


def test_quant_linear_matches_fp_within_tol():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((32, 64)).astype(np.float32)
    b = rng.standard_normal(32).astype(np.float32)
    x = rng.standard_normal((8, 64)).astype(np.float32)

    layer = QuantLinear(w, b)
    y_q = layer(x)
    y_fp = x @ w.T + b

    rel = np.abs(y_q - y_fp).max() / (np.abs(y_fp).max() + 1e-6)
    assert rel < 0.05


def test_quant_linear_saves_memory():
    w = np.ones((128, 256), dtype=np.float32)
    layer = QuantLinear(w)
    fp_bytes = w.nbytes
    assert layer.memory_bytes() < fp_bytes / 3


def test_quant_linear_rejects_non_2d():
    try:
        QuantLinear(np.ones(5, dtype=np.float32))
    except ValueError:
        return
    raise AssertionError("expected ValueError for 1-D weight")
