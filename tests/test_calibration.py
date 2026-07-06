import numpy as np

from liteinfer.quant.calibration import calibrate
from liteinfer.quant.observer import PercentileObserver


def test_calibrate_symmetric_range():
    rng = np.random.default_rng(0)
    batches = [rng.uniform(-2, 2, size=(16, 32)).astype(np.float32) for _ in range(5)]
    params = calibrate(batches, bits=8, symmetric=True)
    assert params.symmetric
    assert params.scale > 0
    # scale should map ~2.0 near the top of the int8 range
    assert 2.0 / params.scale <= 127.5


def test_calibrate_asymmetric_zero_point():
    batches = [np.linspace(1.0, 5.0, 100, dtype=np.float32)]
    params = calibrate(batches, bits=8, symmetric=False)
    assert not params.symmetric
    assert 0 <= int(params.zero_point) <= 255


def test_calibrate_requires_batches():
    try:
        calibrate([], bits=8)
    except ValueError:
        return
    raise AssertionError("expected ValueError on empty calibration set")


def test_percentile_observer_clips_outlier():
    x = np.concatenate([np.zeros(999, np.float32), np.array([1000.0], np.float32)])
    obs = PercentileObserver(percentile=0.5)
    params = calibrate([x], observer=obs, symmetric=False)
    # the 1000 spike should be clipped away, keeping the scale small
    assert params.scale < 10.0
