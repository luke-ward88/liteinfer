"""Command-line interface for liteinfer.

Exposes the three toolkits as subcommands so the reference kernels can be poked
at without writing a script. Everything runs on the bundled toy models, so the
CLI is fully offline.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np

from liteinfer import __version__
from liteinfer.quant.int8 import dequantize, quantize_symmetric_per_channel
from liteinfer.quant.linear import QuantLinear


def _cmd_quant(args: argparse.Namespace) -> int:
    """Quantize a random weight matrix and report error / memory savings."""
    rng = np.random.default_rng(args.seed)
    w = rng.standard_normal((args.rows, args.cols)).astype(np.float32)
    q, params = quantize_symmetric_per_channel(w, axis=0)
    wr = dequantize(q, params)
    layer = QuantLinear(w)
    rel_err = float(np.abs(w - wr).max() / (np.abs(w).max() + 1e-9))
    saving = 1.0 - layer.memory_bytes() / w.nbytes
    print(f"weight shape : {w.shape}")
    print(f"scheme       : INT{params.bits} per-channel symmetric")
    print(f"max rel error: {rel_err:.4f}")
    print(f"memory saving: {saving * 100:.1f}%")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="liteinfer", description="Efficient LLM inference toolkit")
    parser.add_argument("--version", action="version", version=f"liteinfer {__version__}")
    parser.set_defaults(func=None)
    sub = parser.add_subparsers(dest="command", metavar="{quant,kv,spec,bench}")

    p_quant = sub.add_parser("quant", help="quantize a demo weight and report error")
    p_quant.add_argument("--rows", type=int, default=64)
    p_quant.add_argument("--cols", type=int, default=256)
    p_quant.add_argument("--seed", type=int, default=0)
    p_quant.set_defaults(func=_cmd_quant)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "func", None) is None:
        parser.print_help()
        return 1
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
