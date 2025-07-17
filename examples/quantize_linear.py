"""Quantize a linear layer and compare against the full-precision output.

Run: ``python examples/quantize_linear.py``
"""

import numpy as np

from liteinfer.quant.linear import QuantLinear


def main() -> None:
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((256, 512)).astype(np.float32)
    bias = rng.standard_normal(256).astype(np.float32)
    x = rng.standard_normal((16, 512)).astype(np.float32)

    layer = QuantLinear(weight, bias)
    y_quant = layer(x)
    y_full = x @ weight.T + bias

    max_abs_err = np.abs(y_quant - y_full).max()
    rel_err = max_abs_err / np.abs(y_full).max()
    saving = 1.0 - layer.memory_bytes() / weight.nbytes

    print(f"output shape     : {y_quant.shape}")
    print(f"max abs error    : {max_abs_err:.4f}")
    print(f"max relative err : {rel_err:.4%}")
    print(f"weight memory    : {weight.nbytes} -> {layer.memory_bytes()} bytes")
    print(f"memory saving    : {saving:.1%}")


if __name__ == "__main__":
    main()
