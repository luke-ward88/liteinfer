"""INT8 post-training quantization.

Starts with plain per-tensor symmetric quantization; per-channel and asymmetric
(zero-point) variants are layered on top in the same module.
"""

from __future__ import annotations

import numpy as np

from liteinfer.quant.params import QuantParams

__all__ = [
    "quantize_symmetric",
    "quantize_symmetric_per_channel",
    "quantize_asymmetric",
    "dequantize",
]

_INT8_QMAX = 127


def quantize_symmetric(x: np.ndarray) -> tuple[np.ndarray, QuantParams]:
    """Per-tensor symmetric INT8 quantization.

    The scale is ``max(|x|) / 127`` so that the largest-magnitude value maps to
    ±127. Zero maps exactly to 0 (that is the point of the symmetric scheme).
    """
    x = np.asarray(x, dtype=np.float32)
    amax = float(np.abs(x).max())
    scale = amax / _INT8_QMAX if amax > 0 else 1.0
    q = np.clip(np.round(x / scale), -_INT8_QMAX, _INT8_QMAX).astype(np.int8)
    params = QuantParams(
        scale=np.asarray(scale, dtype=np.float32),
        zero_point=np.asarray(0, dtype=np.int32),
        bits=8,
        symmetric=True,
        axis=None,
    )
    return q, params


def quantize_symmetric_per_channel(x: np.ndarray, axis: int = 0) -> tuple[np.ndarray, QuantParams]:
    """Symmetric INT8 quantization with one scale per channel along ``axis``.

    Per-channel scales track weight magnitude variation across output channels
    far better than a single global scale — this is the standard scheme for
    weight-only quantization of linear layers.
    """
    x = np.asarray(x, dtype=np.float32)
    reduce_axes = tuple(a for a in range(x.ndim) if a != axis)
    amax = np.abs(x).max(axis=reduce_axes, keepdims=True)
    scale = np.where(amax > 0, amax / _INT8_QMAX, 1.0).astype(np.float32)
    q = np.clip(np.round(x / scale), -_INT8_QMAX, _INT8_QMAX).astype(np.int8)
    params = QuantParams(
        scale=scale.reshape(-1),
        zero_point=np.zeros(scale.size, dtype=np.int32),
        bits=8,
        symmetric=True,
        axis=axis,
    )
    return q, params


def quantize_asymmetric(x: np.ndarray) -> tuple[np.ndarray, QuantParams]:
    """Per-tensor asymmetric (affine) UINT8 quantization.

    Unlike the symmetric scheme, the full ``[min, max]`` range is mapped onto
    ``[0, 255]`` with an integer zero-point, which is a better fit for tensors
    whose distribution is not centered at zero (e.g. post-ReLU activations).
    """
    x = np.asarray(x, dtype=np.float32)
    xmin = float(x.min())
    xmax = float(x.max())
    qmin, qmax = 0, 255
    if xmax == xmin:
        scale = 1.0
        zero_point = 0
    else:
        scale = (xmax - xmin) / (qmax - qmin)
        zero_point = int(np.clip(round(qmin - xmin / scale), qmin, qmax))
    q = np.clip(np.round(x / scale) + zero_point, qmin, qmax).astype(np.uint8)
    params = QuantParams(
        scale=np.asarray(scale, dtype=np.float32),
        zero_point=np.asarray(zero_point, dtype=np.int32),
        bits=8,
        symmetric=False,
        axis=None,
    )
    return q, params


def dequantize(q: np.ndarray, params: QuantParams) -> np.ndarray:
    """Reconstruct an approximate float tensor from integer codes."""
    q = q.astype(np.float32)
    if params.axis is None:
        return (q - params.zero_point.astype(np.float32)) * params.scale
    shape = [1] * q.ndim
    shape[params.axis] = -1
    scale = params.scale.reshape(shape)
    zp = params.zero_point.reshape(shape).astype(np.float32)
    return (q - zp) * scale
