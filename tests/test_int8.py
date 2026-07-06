import numpy as np

from liteinfer.quant.int8 import dequantize, quantize_symmetric


def test_symmetric_roundtrip_small_error():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((64, 128)).astype(np.float32)
    q, params = quantize_symmetric(x)
    xr = dequantize(q, params)
    # error is bounded by half a quantization step
    assert np.max(np.abs(x - xr)) <= params.scale * 1.001


def test_symmetric_dtype_and_range():
    x = np.linspace(-3, 3, 100, dtype=np.float32)
    q, _ = quantize_symmetric(x)
    assert q.dtype == np.int8
    assert q.min() >= -127 and q.max() <= 127


def test_zero_maps_to_zero():
    x = np.zeros((4, 4), dtype=np.float32)
    q, _ = quantize_symmetric(x)
    assert np.all(q == 0)
