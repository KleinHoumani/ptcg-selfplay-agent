"""SEARCH EXTENSION POINT -- interface only, deliberately NOT implemented.

The search tree is being built separately (search_gen.py). When it lands it needs one thing
from this bundle: a batched leaf evaluator that turns determinized states into (priors,
values). This file names that seam so the tree author does not have to reverse-engineer
main.py's forward, and so a reviewer can see exactly what is and is not shipped today.

WHY A STUB AND NOT A NO-OP IMPLEMENTATION. A silently-degrading search client is the worst
outcome: the tree would run, return uniform priors, and cost latency for nothing. `evaluate`
raises, and `main.py` never imports this module (`USE_SEARCH = False`), so raw-policy serving
is complete and self-sufficient and the dead path cannot be entered by accident.

CONTRACT

    client = SearchClient(model, encoder_context)
    priors, values = client.evaluate(states)

    states  : a sequence of leaf states. Each is whatever the tree hands out, but it must
              carry -- or be able to produce -- the four things a v5 forward needs:
                * the observation dict for the seat to move,
                * that seat's CardKnowledge / ActionHistory(extended=True) / InFlightTracker
                  AS OF that leaf (the trackers are path-dependent: the tree must clone them
                  down the branch, not share one set),
                * the select whose options are being scored,
                * the 60-card deck_counts for the seat.
    priors  : [len(states)][n_options_i] float32, softmax over that leaf's option rows -- the
              SAME transform main.py._v5_move applies, so tree priors and root play agree.
    values  : [len(states)] float32 in [-1, 1] from the trunk's value head, seat-relative
              (positive = good for the seat to move), matching train_ppo's convention.

IMPLEMENTATION NOTES FOR WHOEVER FILLS THIS IN
  * Batch across states, not within a select. The board tensors of ONE select are shared
    across its sub-picks (main.py builds them once); across leaves the token counts differ,
    so a real batch needs padding + a padding_mask -- GameStateTransformer.policy_value
    already takes one, and encode_inflight emits ragged [T, 593] arrays.
  * Multi-pick selects are SEQUENTIAL (encode_inflight.resolve_with_v5). A tree that treats one
    select as one node is modelling a different action space from the one the policy trained
    on; expand sub-picks as their own plies or accept the mismatch knowingly.
  * Budget: 2 vCPU, no GPU. The measured single-forward cost is in the build report; multiply
    by the tree's leaf count before assuming a depth.
  * Do NOT add an auto-answer shortcut for cheapness. encode_inflight.forced_answer is the only
    sanctioned non-model answer anywhere in this bundle (owner directive 2026-07-29).
"""


class SearchClient:
    """Batched leaf evaluator for the (not yet shipped) search tree. See module docstring."""

    def __init__(self, model=None, encoder_context=None):
        self.model = model
        self.encoder_context = encoder_context

    def evaluate(self, states):
        """(priors, values) for a batch of leaf states. See the module docstring's contract."""
        raise NotImplementedError(
            "SearchClient is an interface stub: this bundle ships raw-policy serving only. "
            "Implement evaluate() and set USE_SEARCH = True in main.py to enable search.")
