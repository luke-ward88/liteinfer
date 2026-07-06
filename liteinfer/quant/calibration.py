"""Calibration: drive an observer over sample batches and derive QuantParams."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from liteinfer.quant.observer import MinMaxObserver, Observer
from liteinfer.quant.params import QuantParams


def calibrate(
    batches: Iterable[np.ndarray],
    *,
    bits: int = 8,
    symmetric: bool = True,
    observer: Observer | None = None,
) -> QuantParams:
    """Observe ``batches`` and return per-tensor quantization parameters.

    A calibration pass just needs representative activations; we fold each batch
    into the observer, then convert the observed range into a scale (and a
    zero-point, for the asymmetric case).
    """
    observer = observer or MinMaxObserver()
    seen = False
    for batch in batches:
        observer.update(np.asarray(batch))
        seen = True
    if not seen:
        raise ValueError("no calibration batches provided")

    lo, hi = observer.range()
    lo = float(lo)
    hi = float(hi)

    if symmetric:
        qmax = (1 << (bits - 1)) - 1
        amax = max(abs(lo), abs(hi))
        scale = amax / qmax if amax > 0 else 1.0
        zero_point = 0
    else:
        qmax = (1 << bits) - 1
        scale = (hi - lo) / qmax if hi > lo else 1.0
        zero_point = int(np.clip(round(-lo / scale), 0, qmax))

    return QuantParams(
        scale=np.asarray(scale, dtype=np.float32),
        zero_point=np.asarray(zero_point, dtype=np.int32),
        bits=bits,
        symmetric=symmetric,
        axis=None,
    )
