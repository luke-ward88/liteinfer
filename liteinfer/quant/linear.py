"""A quantized linear (matmul + bias) layer.

Weight-only quantization: weights are stored as INT8 per-channel codes and
dequantized on the fly for the matmul. This is the simplest scheme that shows a
real memory saving while keeping accuracy close to fp32, and it mirrors how
weight-only INT8/INT4 inference paths work in practice.
"""

from __future__ import annotations

import numpy as np

from liteinfer.quant.int8 import dequantize, quantize_symmetric_per_channel
from liteinfer.quant.params import QuantParams


class QuantLinear:
    """``y = x @ W.T + b`` with INT8-quantized ``W``.

    Parameters
    ----------
    weight:
        ``(out_features, in_features)`` float array.
    bias:
        Optional ``(out_features,)`` float array, kept in fp32.
    """

    def __init__(self, weight: np.ndarray, bias: np.ndarray | None = None) -> None:
        weight = np.asarray(weight, dtype=np.float32)
        if weight.ndim != 2:
            raise ValueError("weight must be 2-D (out_features, in_features)")
        # per output-channel (row) scale
        self.qweight, self.wparams = quantize_symmetric_per_channel(weight, axis=0)
        self.bias = None if bias is None else np.asarray(bias, dtype=np.float32)
        self.out_features, self.in_features = weight.shape

    @property
    def weight(self) -> np.ndarray:
        """Dequantized weight (recomputed on access)."""
        return dequantize(self.qweight, self.wparams)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        y = x @ self.weight.T
        if self.bias is not None:
            y = y + self.bias
        return y

    def memory_bytes(self) -> int:
        """Bytes used by the quantized weight and its scales."""
        return self.qweight.nbytes + self.wparams.scale.nbytes

    @property
    def weight_params(self) -> QuantParams:
        return self.wparams
