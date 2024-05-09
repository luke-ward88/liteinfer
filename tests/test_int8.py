import numpy as np

from liteinfer.quant.int8 import (
    dequantize,
    quantize_symmetric,
    quantize_symmetric_per_channel,
)


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


def test_per_channel_beats_per_tensor_on_skewed_channels():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((8, 256)).astype(np.float32)
    # blow up the dynamic range of a single channel
    x[0] *= 50.0
    q_pt, p_pt = quantize_symmetric(x)
    q_pc, p_pc = quantize_symmetric_per_channel(x, axis=0)
    err_pt = np.abs(x - dequantize(q_pt, p_pt)).mean()
    err_pc = np.abs(x - dequantize(q_pc, p_pc)).mean()
    assert err_pc < err_pt


def test_per_channel_scale_shape():
    x = np.ones((5, 10), dtype=np.float32)
    _, params = quantize_symmetric_per_channel(x, axis=0)
    assert params.scale.shape == (5,)
    assert params.axis == 0
