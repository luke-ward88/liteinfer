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
from liteinfer.kvcache.attention import paged_attention
from liteinfer.kvcache.block import BlockAllocator
from liteinfer.kvcache.manager import BlockSpaceManager
from liteinfer.kvcache.paged_cache import PagedKVStore
from liteinfer.quant.int8 import dequantize, quantize_symmetric_per_channel
from liteinfer.quant.linear import QuantLinear
from liteinfer.specdec.ngram import make_draft_target
from liteinfer.specdec.speculative import speculative_decode


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


def _cmd_kv(args: argparse.Namespace) -> int:
    """Fill a paged KV-cache and run one paged-attention step."""
    rng = np.random.default_rng(args.seed)
    store = PagedKVStore(
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
    )
    mgr = BlockSpaceManager(BlockAllocator(args.num_blocks, args.block_size))
    mgr.add_sequence(0)
    for _ in range(args.length):
        block_id, slot = mgr.append_token(0)
        k = rng.standard_normal((args.num_heads, args.head_dim)).astype(np.float32)
        v = rng.standard_normal((args.num_heads, args.head_dim)).astype(np.float32)
        store.write(block_id, slot, k, v)
    q = rng.standard_normal((args.num_heads, args.head_dim)).astype(np.float32)
    out = paged_attention(q, 0, store, mgr)
    blocks_used = len(mgr.get(0).block_table)
    print(f"sequence length : {args.length}")
    print(f"block size      : {args.block_size}")
    print(f"blocks used     : {blocks_used} / {args.num_blocks}")
    print(f"attn output norm: {float(np.linalg.norm(out)):.4f}")
    return 0


def _cmd_spec(args: argparse.Namespace) -> int:
    """Run speculative decoding on the toy models and report acceptance rate."""
    draft, target = make_draft_target(
        args.vocab, order=args.order, seed=args.seed, draft_temperature=args.draft_temp
    )
    rng = np.random.default_rng(args.seed)
    res = speculative_decode(
        [0], draft, target, gamma=args.gamma, max_new_tokens=args.tokens, rng=rng
    )
    print(f"generated tokens: {len(res.tokens) - 1}")
    print(f"gamma           : {args.gamma}")
    print(f"proposed        : {res.proposed}")
    print(f"accepted        : {res.accepted}")
    print(f"acceptance rate : {res.acceptance_rate:.3f}")
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

    p_kv = sub.add_parser("kv", help="demo the paged KV-cache and paged attention")
    p_kv.add_argument("--length", type=int, default=20)
    p_kv.add_argument("--block-size", type=int, default=4)
    p_kv.add_argument("--num-blocks", type=int, default=16)
    p_kv.add_argument("--num-heads", type=int, default=4)
    p_kv.add_argument("--head-dim", type=int, default=8)
    p_kv.add_argument("--seed", type=int, default=0)
    p_kv.set_defaults(func=_cmd_kv)

    p_spec = sub.add_parser("spec", help="run speculative decoding on toy models")
    p_spec.add_argument("--vocab", type=int, default=64)
    p_spec.add_argument("--order", type=int, default=3)
    p_spec.add_argument("--gamma", type=int, default=4)
    p_spec.add_argument("--tokens", type=int, default=32)
    p_spec.add_argument("--draft-temp", type=float, default=1.5)
    p_spec.add_argument("--seed", type=int, default=0)
    p_spec.set_defaults(func=_cmd_spec)

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
