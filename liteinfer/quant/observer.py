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

    # TODO: revisit per-channel reduction axis once QuantLinear lands
