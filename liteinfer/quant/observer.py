"""Range observers for calibration.

An observer watches a stream of tensors during calibration and accumulates the
statistics needed to pick quantization ranges (``min`` / ``max``). Concrete
observers are implemented later; this module currently defines the interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Observer(ABC):
    """Accumulates value ranges seen across calibration batches."""

    @abstractmethod
    def update(self, x: np.ndarray) -> None:
        """Fold a new tensor into the running statistics."""

    @abstractmethod
    def range(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the observed ``(min, max)`` arrays."""


class MinMaxObserver(Observer):
    """Tracks the running global min and max across all observed tensors."""

    def __init__(self) -> None:
        self._min: float | None = None
        self._max: float | None = None

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32)
        lo = float(x.min())
        hi = float(x.max())
        self._min = lo if self._min is None else min(self._min, lo)
        self._max = hi if self._max is None else max(self._max, hi)

    def range(self) -> tuple[np.ndarray, np.ndarray]:
        if self._min is None or self._max is None:
            raise RuntimeError("observer has not seen any data yet")
        return np.asarray(self._min, np.float32), np.asarray(self._max, np.float32)


class PercentileObserver(Observer):
    """Clips the range to a percentile band to shrug off outliers.

    Instead of the absolute min/max, the range is taken at ``[q, 100 - q]``
    percentiles of the pooled samples. This trades a bit of clipping error on the
    tails for a much tighter scale on the bulk of the distribution — usually a
    win for activations with a few large spikes. Samples are pooled (capped) so
    the percentile stays cheap to compute.
    """

    def __init__(self, percentile: float = 0.1, max_samples: int = 1 << 20) -> None:
        if not 0.0 <= percentile < 50.0:
            raise ValueError("percentile must be in [0, 50)")
        self._pct = percentile
        self._max_samples = max_samples
        self._buf: list[np.ndarray] = []
        self._count = 0

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32).ravel()
        if self._count < self._max_samples:
            self._buf.append(x)
            self._count += x.size

    def range(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._buf:
            raise RuntimeError("observer has not seen any data yet")
        pooled = np.concatenate(self._buf)
        lo = np.percentile(pooled, self._pct)
        hi = np.percentile(pooled, 100.0 - self._pct)
        return np.asarray(lo, np.float32), np.asarray(hi, np.float32)
