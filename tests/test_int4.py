import numpy as np

from liteinfer.quant.int4 import pack_nibbles, unpack_nibbles


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
