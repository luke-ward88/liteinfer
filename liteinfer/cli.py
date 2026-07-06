"""Command-line interface for liteinfer.

Exposes the three toolkits as subcommands so the reference kernels can be poked
at without writing a script. Everything runs on the bundled toy models, so the
CLI is fully offline.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from liteinfer import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="liteinfer", description="Efficient LLM inference toolkit")
    parser.add_argument("--version", action="version", version=f"liteinfer {__version__}")
    parser.set_defaults(func=None)
    parser.add_subparsers(dest="command", metavar="{quant,kv,spec,bench}")
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
