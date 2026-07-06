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
