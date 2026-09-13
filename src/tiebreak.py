"""Tie-aware greedy action selection.

The policy head scores each option from `cat(CLS context, option_features)`, so two options
with byte-identical feature vectors get *exactly* equal logits -- no training can separate
them. Generation samples, which spreads over the tie uniformly; every deploy/eval path used
`argmax`, and `np.argmax`/`torch.argmax` return the FIRST maximum. The engine lists Active
before Bench and bench slots in order, so an argmax agent carries a hard, unlearned prior
("attach to the Active, evolve the leftmost bench slot, promote bench slot 0") on the 26% of
decisions that contain a materially different tie (DECISION_SURFACE_AUDIT C5).

`argmax_tiebreak` keeps greedy behaviour everywhere the model has an opinion and breaks
EXACT ties uniformly at random, so eval measures the same policy generation trained.

The shipped bundle (`submissions/submission_d128_dragkazam/main.py`) is deliberately left
as-is -- changing a submitted agent's behaviour is a separate, deliberate call.
"""

import numpy as np


def argmax_tiebreak(scores, rng=None):
    """Index of the maximum of `scores`, chosen uniformly among EXACT ties.

    `rng` is a random.Random (or anything with .randrange); None = deterministic first-index
    argmax, i.e. the old behaviour, so a caller can opt out without a code change."""
    scores = np.asarray(scores)
    if scores.size == 0:
        raise ValueError("argmax_tiebreak on an empty score vector")
    best = scores.max()
    tied = np.flatnonzero(scores == best)          # exact equality: no tolerance band
    if rng is None or tied.size == 1:
        return int(tied[0])
    return int(tied[rng.randrange(tied.size)])
