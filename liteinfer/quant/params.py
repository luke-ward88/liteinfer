"""Quantization parameters (scale / zero-point) container."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QuantParams:
    """Affine quantization parameters.

    A real value ``x`` maps to an integer ``q`` via::

        q = round(x / scale) + zero_point

    and back via ``x ≈ (q - zero_point) * scale``.

    ``scale`` and ``zero_point`` are arrays so the same container works for
    per-tensor (scalar-shaped), per-channel (one entry per output channel) and
    group-wise (one entry per group) schemes. ``axis`` records the channel axis
    the parameters were computed along (``None`` for per-tensor).
    """

    scale: np.ndarray
    zero_point: np.ndarray
    bits: int
    symmetric: bool
    axis: int | None = None

    @property
    def qmin(self) -> int:
        return -(1 << (self.bits - 1)) if self.symmetric else 0

    @property
    def qmax(self) -> int:
        return (1 << (self.bits - 1)) - 1 if self.symmetric else (1 << self.bits) - 1
