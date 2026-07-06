import numpy as np

from liteinfer.quant.int4 import (
    dequantize_group,
    pack_nibbles,
    quantize_group,
    unpack_nibbles,
)


def test_pack_unpack_roundtrip_even():
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 16, size=64, dtype=np.uint8)
    packed = pack_nibbles(codes)
    assert packed.size == 32
    out = unpack_nibbles(packed, count=codes.size)
    np.testing.assert_array_equal(out, codes)


def test_pack_unpack_roundtrip_odd():
    codes = np.array([1, 2, 3, 15, 7], dtype=np.uint8)
    packed = pack_nibbles(codes)
    assert packed.size == 3  # ceil(5/2)
    out = unpack_nibbles(packed, count=5)
    np.testing.assert_array_equal(out, codes)


def test_pack_rejects_out_of_range():
    try:
        pack_nibbles(np.array([16], dtype=np.uint8))
    except ValueError:
        return
    raise AssertionError("expected ValueError for code > 15")


def test_group_quant_error_within_step():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(500).astype(np.float32)
    packed = quantize_group(x, group_size=64)
    xr = dequantize_group(packed)
    assert xr.shape == x.shape
    # per-group max error is bounded by one quantization step of that group
    step = float(packed.scale.max())
    assert np.max(np.abs(x - xr)) <= step * 1.01


def test_group_quant_smaller_groups_reduce_error():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(1024).astype(np.float32)
    err_big = np.abs(x - dequantize_group(quantize_group(x, 256))).mean()
    err_small = np.abs(x - dequantize_group(quantize_group(x, 32))).mean()
    assert err_small < err_big
