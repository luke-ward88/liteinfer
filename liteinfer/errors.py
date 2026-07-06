"""Exception types shared across liteinfer."""

from __future__ import annotations


class LiteInferError(Exception):
    """Base class for all liteinfer errors."""


class QuantizationError(LiteInferError):
    """Raised when a tensor cannot be quantized with the requested scheme."""


class CacheFullError(LiteInferError):
    """Raised when the paged KV-cache runs out of free blocks."""


class ShapeError(LiteInferError):
    """Raised when array shapes are incompatible with a kernel's contract."""
