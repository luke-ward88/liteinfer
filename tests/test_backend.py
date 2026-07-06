import numpy as np

from liteinfer import backend


def test_asarray_is_contiguous():
    x = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2]
    out = backend.asarray(x)
    assert out.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(out, x)


def test_to_numpy_passthrough():
    x = np.ones((2, 2))
    assert backend.to_numpy(x) is x


def test_to_numpy_from_list():
    out = backend.to_numpy([1, 2, 3])
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, [1, 2, 3])
