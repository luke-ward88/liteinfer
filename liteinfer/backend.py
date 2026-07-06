"""Array backend shim.

The reference kernels are written against a tiny surface of array operations so
that the same code can run on plain NumPy (the default, dependency-light path)
or, optionally, on PyTorch tensors. We deliberately keep this minimal — just the
handful of ops the kernels actually use — rather than trying to abstract all of
NumPy.
"""

from __future__ import annotations

import numpy as np

# The public array type. Kept as an alias so call sites read as ``backend.ndarray``
# and we have a single place to widen the type if a torch path is added.
ndarray = np.ndarray

__all__ = ["np", "ndarray", "asarray", "to_numpy"]


def asarray(x, dtype=None) -> np.ndarray:
    """Coerce ``x`` into a contiguous NumPy array."""
    return np.ascontiguousarray(x, dtype=dtype)


def to_numpy(x) -> np.ndarray:
    """Return a NumPy view of ``x``.

    Accepts NumPy arrays directly and anything exposing ``.detach().cpu().numpy()``
    (e.g. a torch tensor) without importing torch.
    """
    if isinstance(x, np.ndarray):
        return x
    detach = getattr(x, "detach", None)
    if detach is not None:  # torch-like
        return detach().cpu().numpy()
    return np.asarray(x)
