"""The speculative decoding loop.

A *draft* model proposes ``gamma`` tokens cheaply; the *target* model scores them
in a single batched pass; the accept/reject correction keeps only a verified
prefix and either resamples the first rejected position from the residual or, if
all are accepted, samples one bonus token from the target. Each outer iteration
therefore emits between ``1`` and ``gamma + 1`` tokens while producing output
distributed exactly as the target model.

Models are duck-typed: anything with ``next_token_probs(prefix) -> np.ndarray``
(a probability vector over the vocabulary) works, which keeps the loop decoupled
from any particular network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from liteinfer.specdec.acceptance import accept_token, resample_on_reject


class ProbModel(Protocol):
    def next_token_probs(self, prefix: list[int]) -> np.ndarray: ...


@dataclass
class SpecDecodeResult:
    tokens: list[int]
    proposed: int
    accepted: int

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0


def speculative_decode(
    prefix: list[int],
    draft: ProbModel,
    target: ProbModel,
    *,
    gamma: int = 4,
    max_new_tokens: int = 32,
    rng: np.random.Generator | None = None,
) -> SpecDecodeResult:
    """Generate tokens from ``target`` accelerated by ``draft`` proposals."""
    rng = rng or np.random.default_rng()
    tokens = list(prefix)
    proposed = 0
    accepted = 0

    while len(tokens) - len(prefix) < max_new_tokens:
        # 1. draft proposes gamma tokens autoregressively, remembering its probs
        draft_tokens: list[int] = []
        draft_probs: list[np.ndarray] = []
        scratch = list(tokens)
        for _ in range(gamma):
            q = draft.next_token_probs(scratch)
            tok = int(rng.choice(q.size, p=q))
            draft_tokens.append(tok)
            draft_probs.append(q)
            scratch.append(tok)

        # 2. target scores every prefix position in one conceptual pass
        target_probs = [target.next_token_probs(tokens + draft_tokens[:i]) for i in range(gamma)]

        # 3. walk the proposals, accepting under the correction
        emitted_bonus = False
        for i in range(gamma):
            proposed += 1
            p = target_probs[i]
            q = draft_probs[i]
            if accept_token(draft_tokens[i], p, q, rng):
                accepted += 1
                tokens.append(draft_tokens[i])
                if len(tokens) - len(prefix) >= max_new_tokens:
                    emitted_bonus = True
                    break
            else:
                tokens.append(resample_on_reject(p, q, rng))
                emitted_bonus = True
                break

        # 4. all accepted → sample one bonus token straight from the target
        if not emitted_bonus and len(tokens) - len(prefix) < max_new_tokens:
            p = target.next_token_probs(tokens)
            tokens.append(int(rng.choice(p.size, p=p)))

    return SpecDecodeResult(tokens=tokens, proposed=proposed, accepted=accepted)
