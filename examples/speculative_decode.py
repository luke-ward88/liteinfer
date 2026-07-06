"""Speculative decoding demo on the bundled toy models.

Shows how the draft temperature trades off against the acceptance rate. A colder
draft agrees with the target more often, so more proposals survive per step.

Run: ``python examples/speculative_decode.py``
"""

import numpy as np

from liteinfer.specdec.ngram import make_draft_target
from liteinfer.specdec.speculative import speculative_decode

VOCAB = 128
GAMMA = 4
TOKENS = 64


def main() -> None:
    print(f"{'draft_temp':>10} {'accept_rate':>12} {'tokens/step':>12}")
    for draft_temp in (1.1, 1.5, 2.0, 3.0):
        draft, target = make_draft_target(VOCAB, order=3, seed=0, draft_temperature=draft_temp)
        rng = np.random.default_rng(0)
        res = speculative_decode(
            [0, 1, 2], draft, target, gamma=GAMMA, max_new_tokens=TOKENS, rng=rng
        )
        steps = res.proposed / GAMMA
        tokens_per_step = (len(res.tokens) - 3) / steps
        print(f"{draft_temp:>10.1f} {res.acceptance_rate:>12.3f} {tokens_per_step:>12.2f}")


if __name__ == "__main__":
    main()
