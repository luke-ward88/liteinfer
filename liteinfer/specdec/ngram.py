"""Build draft/target model pairs for speculative decoding demos.

A realistic speculative-decoding setup needs a fast *draft* whose distribution is
correlated with — but not identical to — the *target*. With the toy n-gram model
we get this for free: share the same table but flatten the draft with a higher
sampling temperature. The draft still mostly agrees with the target (so the
acceptance rate is meaningful) while differing enough that the accept/reject
correction actually does something.
"""

from __future__ import annotations

from liteinfer.models.toy import NGramModel


def make_draft_target(
    vocab_size: int,
    *,
    order: int = 3,
    seed: int = 0,
    draft_temperature: float = 1.5,
) -> tuple[NGramModel, NGramModel]:
    """Return ``(draft, target)`` n-gram models over a shared table.

    The target is the "true" distribution (temperature 1.0); the draft shares the
    same seed and order but is softened, making it a cheaper, slightly-off
    proposer.
    """
    target = NGramModel(vocab_size, order=order, seed=seed, temperature=1.0)
    draft = NGramModel(vocab_size, order=order, seed=seed, temperature=draft_temperature)
    return draft, target
