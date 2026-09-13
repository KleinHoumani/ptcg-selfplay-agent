"""v4 action surface: the MODEL decides how many and which options to pick.

WHY THIS EXISTS (owner directive 2026-07-29): every encoding up to v3 answered a whole
class of engine prompts with hand-written code (`_trivial_move`): "take all" whenever
`maxCount >= len(option)`, "the first k in engine order" for every multi-pick, and "[0]"
for every single-option prompt even when `minCount == 0` made declining legal. Those are
real decisions -- which cards to discard, which two to search out, whether to use an
ability at all -- and code must not make them.

v4 = v3 inputs ++ a SEQUENTIAL SELECTION LOOP. The engine takes one index list per
`battle_select`, so the loop is internal to the agent: at every step the model scores the
options it has NOT yet picked plus a synthetic STOP action (offered only once `minCount`
picks are in hand), and the answer is submitted when STOP is chosen or the budget is spent.
Each step is a real forward -- and, in training, its own sample.

THE ONLY ANSWERS THIS MODULE PRODUCES WITHOUT THE MODEL are prompts whose set of legal
answers has exactly ONE member. That is a PREDICATE, not a case list (see `forced_answer`):
with `low = minCount`, `high = min(maxCount, n)`, it is `low == high and (low == 0 or
low == n)`. Everything else reaches the net -- including `minCount == 0` (declining is a
choice) and `minCount == maxCount == k` for `0 < k < n` (exactly k of n).

NEW DIMS (owner-approved, minimal):
  per option (+2, appended after the v3 option vector, so a v3 slice of a v4 option vector
  is exactly its v3 vector):
      0  is_stop_action            1.0 on the synthetic STOP row
      1  same_id_already_chosen    1.0 when an option with the SAME identity (option type +
                                   resolved card id) has already been picked this round
  globals (+3, appended after the v3 globals):
      0  picks_so_far / 10   1  minCount / 10   2  maxCount / 10     (all clipped at 1.0)
  per token: NOTHING -- v4 token width is v3's 589. (A "partial placement" column was built
  and then REMOVED on 2026-07-29: the engine applies each damage counter IMMEDIATELY, so `hp`
  already carries the progress; the audit that showed otherwise had been run against a deck
  with 4 Battle Cage, which PREVENTS the counters. A tracker counting our own picks would have
  asserted damage that does not exist whenever a shield is up. See the build report.)

WHY `same_id_already_chosen` AND NOT A LITERAL "this row is already chosen" FLAG: the
candidate rows ARE the not-yet-chosen options (legality by construction), so a row for an
already-chosen option would have to be masked out of the softmax -- and the policy head
scores each option row INDEPENDENTLY from the board context (`policy_score(cat(context,
option))`, no attention across options in either GameStateTransformer or PolicyNet), so a
masked row cannot influence any other row's logit. A literal flag would therefore be
provably inert. Binding the flag to the identity of what is already in hand is the live
form of the same signal: on the second pick of "discard 2 energy" the model can see that
one copy of that exact card is already going.

THE STOP ROW is defined explicitly: all zeros -- no card, no area, no target, no attached
card -- EXCEPT the v3 SelectContext block (copied verbatim from the real options, which all
share it, so STOP knows *what kind* of prompt it is ending) and its own `is_stop_action`
flag. Zeros are the honest encoding: STOP points at no card, and the board / budget context
it needs lives in the tokens and the three new globals.

v1/v2/v3 are untouched: this module only ADDS, `src/game/encode.py` stays byte-frozen and
`encode_history.py` / `encode_details.py` are composed, not edited.
"""

import numpy as np

from src.game.encode import OPTION_FEATURE_DIM
from src.game.encode_details import (NUM_SELECT_CONTEXTS, OPTION_EXTRA_DIM, _entity_at,
                                encode_option_v3)

# --- the two new option columns ---------------------------------------------------- #
V4_OPTION_EXTRA_DIM = 2
IS_STOP_ACTION = 0
SAME_ID_ALREADY_CHOSEN = 1

# --- the three new globals ---------------------------------------------------------- #
V4_GLOBAL_EXTRA_DIM = 3
G_PICKS_SO_FAR = 0
G_MIN_COUNT = 1
G_MAX_COUNT = 2
PICK_SCALE = 10.0                 # counts are clipped at 10 (a select over 10 picks is rare)

OPTION_FEATURE_DIM_V3 = OPTION_FEATURE_DIM + OPTION_EXTRA_DIM          # 2042
OPTION_FEATURE_DIM_V4 = OPTION_FEATURE_DIM_V3 + V4_OPTION_EXTRA_DIM    # 2044

# The SelectContext one-hot + its validity bit are the first block of `encode_details.option_extra`,
# i.e. they start right after the frozen option vector.
_CONTEXT_OFFSET = OPTION_FEATURE_DIM
_CONTEXT_WIDTH = NUM_SELECT_CONTEXTS + 1

_AREA_HAND = 2
_OPTION_TYPE_PLAY = 7

# --- ORDER-SENSITIVE contexts ------------------------------------------------------- #
# For most selects the answer is a SET, so "you must take all of them" has exactly one legal
# answer. For these contexts the answer is a SEQUENCE -- the engine reads the ORDER of the
# index list -- so `minCount == maxCount == n` has n! legal answers and submitting engine
# order would be code making a real decision. Measured 2026-07-29: SKILL_ORDER (34, the order
# triggered abilities resolve in) always arrives as min == max == n and the engine accepted a
# REVERSED list 5/5 times; 224 such selects over 875 corpus games.
# ANY future order-sensitive context must be added here, or the loop will shortcut it.
ORDER_SENSITIVE_CONTEXTS = frozenset({34})        # cg.api.SelectContext.SKILL_ORDER


def take_bounds(select):
    """(min_take, max_take) for one select, clamped to the engine contract
    (`minCount <= len(answer) <= maxCount`, `maxCount <= len(option)`). Defensive: a
    malformed pair can never produce an illegal answer."""
    count = len(select["option"])
    max_take = max(0, min(int(select["maxCount"]), count))
    min_take = max(0, min(int(select["minCount"]), max_take))
    return min_take, max_take


def forced_answer(select):
    """The prompts with exactly ONE legal answer -> (indices, reason); (None, None) when the
    model must decide. THE ONLY code path in v4 that answers the engine itself.

    Stated as a PREDICATE, not a case list, so a select shape we have never seen is handled
    correctly by construction (the next run is corpus-vs-corpus: every archetype's prompts,
    not just two decks'). The legal answers are the subsets of size k with
    `minCount <= k <= min(maxCount, n)`; that family has exactly one member iff the bounds
    pin k to a single value AND `C(n, k) == 1`, i.e. k == 0 or k == n:

        low == high  and  (low == 0 or low == n)

    which covers -- and is not limited to -- n == 0, minCount == maxCount == 0 (take
    nothing), minCount == maxCount == n (take all) and n == 1 with minCount == 1. Every other
    shape reaches the model, including `minCount == 0` with any n (declining is legal) and
    `minCount == maxCount == k` with 0 < k < n (exactly k of n: a real combinatorial choice,
    C(n,k) > 1).

    ONE CORRECTION to the set-based formula: when the answer is a SEQUENCE
    (`ORDER_SENSITIVE_CONTEXTS`, e.g. SKILL_ORDER), "take all n" has n! legal answers, not
    one, so it is NOT forced -- the loop picks the order instead."""
    count = len(select["option"])
    low, high = take_bounds(select)          # low <= high, both clamped into [0, count]
    if count > 1 and select.get("context") in ORDER_SENSITIVE_CONTEXTS:
        return None, None                     # the ORDER is the answer -> n! of them
    if low != high:
        return None, None
    if low == 0:
        return [], ("no_options" if count == 0 else "must_take_nothing")
    if low == count:
        return list(range(count)), ("single_option_forced" if count == 1
                                    else "must_take_all")
    return None, None                         # exactly k of n, 0 < k < n -> C(n,k) > 1


def option_identity(observation, option):
    """The identity key `same_id_already_chosen` compares: option type + the card the option
    points at (resolved the same way `encode_details.option_extra` resolves its primary entity),
    falling back to the option's own literal ids when it points at no card."""
    me_index = observation["current"]["yourIndex"]
    owner = option.get("playerIndex")
    owner = me_index if owner is None else owner
    area, index = option.get("area"), option.get("index")
    if option["type"] == _OPTION_TYPE_PLAY and index is not None:
        area, owner = _AREA_HAND, me_index
    card, _pokemon = _entity_at(observation, area, index, owner)
    card_id = (card or {}).get("id")
    if card_id:
        return (option["type"], int(card_id))
    return (option["type"], None, option.get("cardId"), option.get("attackId"),
            option.get("number"), option.get("specialConditionType"))


def base_option_matrix(observation, select):
    """The v3 option matrix for one select, encoded ONCE: the board cannot change inside the
    selection loop (no `battle_select` happens until the answer is complete)."""
    return np.stack([encode_option_v3(observation, option, select)
                     for option in select["option"]]).astype(np.float32)


class MultiSelect:
    """One engine select, answered one model pick at a time.

    Legality by construction: `pending()` never returns an index already in `chosen`, the
    loop is `complete()` at `max_take`, and STOP is only `stop_offered()` once `min_take`
    picks are in hand -- so the submitted list has unique in-range indices and a length
    inside [minCount, maxCount] no matter what the model says."""

    __slots__ = ("count", "min_take", "max_take", "chosen", "_keys", "_chosen_keys")

    def __init__(self, observation, select):
        self.count = len(select["option"])
        self.min_take, self.max_take = take_bounds(select)
        self.chosen = []
        self._keys = [option_identity(observation, option) for option in select["option"]]
        self._chosen_keys = set()

    def pending(self):
        """Options still available, in engine order."""
        picked = set(self.chosen)
        return [index for index in range(self.count) if index not in picked]

    def stop_offered(self):
        """Is STOP a legal action right now?"""
        return len(self.chosen) >= self.min_take

    def complete(self):
        """No legal continuation left: the budget is spent, or everything is already in."""
        return len(self.chosen) >= self.max_take or len(self.chosen) >= self.count

    def forced_index(self):
        """The single legal continuation when there is nothing to decide (one option left and
        STOP not yet legal), else None. Never fires when the model has a real choice."""
        pending = self.pending()
        if not self.stop_offered() and len(pending) == 1:
            return pending[0]
        return None

    def take(self, index):
        self.chosen.append(index)
        self._chosen_keys.add(self._keys[index])

    def same_id_flags(self, pending):
        return [1.0 if self._keys[index] in self._chosen_keys else 0.0 for index in pending]

    def answer(self):
        """The list handed to `battle_select`, in the order the model picked."""
        return list(self.chosen)


def candidate_matrix(base_matrix, multiselect, pending, stop_offered):
    """[K, OPTION_FEATURE_DIM_V4] for one sub-pick: the pending options' v3 rows ++ the two
    v4 columns, then the synthetic STOP row (last) when STOP is legal. `base_matrix` is
    `base_option_matrix(...)` for this select."""
    rows = np.zeros((len(pending) + int(bool(stop_offered)), OPTION_FEATURE_DIM_V4),
                    dtype=np.float32)
    if pending:
        rows[:len(pending), :OPTION_FEATURE_DIM_V3] = base_matrix[pending]
        rows[:len(pending), OPTION_FEATURE_DIM_V3 + SAME_ID_ALREADY_CHOSEN] = \
            multiselect.same_id_flags(pending)
    if stop_offered:
        stop = rows[len(pending)]
        # base_matrix always has >= 1 row here: count == 0 is a forced answer.
        stop[_CONTEXT_OFFSET:_CONTEXT_OFFSET + _CONTEXT_WIDTH] = \
            base_matrix[0, _CONTEXT_OFFSET:_CONTEXT_OFFSET + _CONTEXT_WIDTH]
        stop[OPTION_FEATURE_DIM_V3 + IS_STOP_ACTION] = 1.0
    return rows


def global_features(base_globals, multiselect):
    """The v3 globals ++ the three v4 columns (a NEW array; the v3 array is never mutated,
    so a cached encoder's buffers stay intact)."""
    extra = np.zeros(V4_GLOBAL_EXTRA_DIM, dtype=np.float32)
    extra[G_PICKS_SO_FAR] = min(len(multiselect.chosen), PICK_SCALE) / PICK_SCALE
    extra[G_MIN_COUNT] = min(multiselect.min_take, PICK_SCALE) / PICK_SCALE
    extra[G_MAX_COUNT] = min(multiselect.max_take, PICK_SCALE) / PICK_SCALE
    return np.concatenate([base_globals, extra]).astype(np.float32)


def resolve_with(observation, select, choose, stats=None):
    """The v4 selection loop for SYNCHRONOUS callers (probes, agents, fuzz harnesses).

    `choose(option_matrix, globals_extra_state) -> row index` runs one forward over the
    candidate rows; `globals_extra_state` is the MultiSelect, so the caller builds its
    globals with `global_features(base_globals, state)`. Returns the index list for
    `battle_select`.

    train_ppo's generator writes this same loop INLINE (it must `yield` its forward to the
    pipelining driver instead of calling it); the two must stay in step -- any change here
    belongs there too."""
    forced, reason = forced_answer(select)
    if forced is not None:
        if stats is not None:
            stats["forced"] += 1
            stats["forced_" + reason] += 1
        return forced
    state = MultiSelect(observation, select)
    base_matrix = base_option_matrix(observation, select)
    while not state.complete():
        forced_index = state.forced_index()
        if forced_index is not None:
            if stats is not None:
                stats["forced_pick"] += 1
            state.take(forced_index)
            continue
        pending = state.pending()
        stop_offered = state.stop_offered()
        rows = candidate_matrix(base_matrix, state, pending, stop_offered)
        picked = int(choose(rows, state))
        if stats is not None:
            stats["forwards"] += 1
        if stop_offered and picked == len(pending):      # STOP
            if stats is not None:
                stats["stop"] += 1
            break
        state.take(pending[picked])
    if stats is not None:
        stats["selects"] += 1
        stats["picks"] += len(state.chosen)
    return state.answer()
