"""Metrics for evaluating speculative decoding runs."""

from __future__ import annotations

from dataclasses import dataclass

from liteinfer.specdec.speculative import SpecDecodeResult


@dataclass(frozen=True)
class SpecDecodeMetrics:
    acceptance_rate: float
    mean_accepted_len: float
    speedup_estimate: float


def summarize(result: SpecDecodeResult, gamma: int) -> SpecDecodeMetrics:
    """Derive summary metrics from a decode result.

    ``speedup_estimate`` is the idealized target-forward reduction: without
    speculation each token costs one target pass, whereas each speculative step
    verifies ``gamma`` proposals in one pass and emits ``mean_accepted_len + 1``
    tokens. We report the ratio of the two as a first-order estimate — real
    speedups also depend on the draft/target cost ratio, which this ignores.
    """
    steps = result.proposed / gamma if gamma else 0.0
    emitted = len(result.tokens)
    mean_accepted_len = result.accepted / steps if steps else 0.0
    tokens_per_step = emitted / steps if steps else 0.0
    return SpecDecodeMetrics(
        acceptance_rate=result.acceptance_rate,
        mean_accepted_len=mean_accepted_len,
        speedup_estimate=tokens_per_step,
    )
