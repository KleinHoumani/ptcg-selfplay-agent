"""256x1 OUR-TURN determinized PUCT search (deployment).

One determinized world per decision, 256 simulations, and the tree STOPS at the end of our
own turn: opponent moves are never simulated (owner spec 2026-08-02). The moment the seat
to move flips, the node is a LEAF valued by the trunk's value head from the opponent-view
observation, NEGATED into the root player's perspective. That sidesteps the weakest link
of full-game determinized search -- rolling out a guessed opponent hand -- while keeping
what search is best at: sequencing our own turn (energy order, ability chains, which
attack, what to discard).

Conventions deliberately MIRROR the machinery that produced the training targets
(train_ppo.SearchLeafEncoder / search_gen.py):

  * ROOT-FROZEN CONTEXT -- the deciding seat's CardKnowledge / ActionHistory / deck counts
    are captured once at the real decision and applied to every node encode. A search node
    is a hypothetical; it does not get its own trackers.
  * values are root-perspective (negamax at the single seat flip), terminals are exact +-1.
  * manual_coin=True: in-search coin flips surface as SelectType COIN_HEAD (46) YesNo
    selects and are treated as CHANCE nodes -- children sampled uniformly during descent,
    so visit-weighted backprop approximates the 50/50 expectation (the adopted
    manual-coin-expectimax result, +0.028).

In-search decision policy: single-pick selects with >= 2 options are the ONLY branch
points (PUCT, priors = the same softmax the raw agent plays by). Everything else advances
the world the way the agent itself would: forced answers by proof, select==None steps,
multi-pick selects resolved sequentially by the policy (greedy argmax -- the model
decides, never code). Every failure of any kind returns None and the caller plays the raw
policy move: search can only ever be a bonus, never a crash.

Budget: 2 vCPU, no GPU, 600 s/episode bank with no per-move limit. Depth follows the
continuous BANK CURVE (256 sims on a full bank, square-root decline to zero at the 40 s
raw floor) so long games taper gradually into their endgames instead of halving in
cliffs; each decision is capped by its budget-scaled deadline (anytime PUCT: the
most-visited root move so far is played).
"""

import json
import math
import os
import random
import threading
import time
from collections import Counter

# CONTINUOUS BANK CURVE (owner spec 2026-08-15, replacing the stepped ladder): search
# depth is a smooth power curve of the live bank -- SIMS_TOP on a full bank, declining
# SLOWLY at first and steeply only as the bank genuinely runs down (square root of the
# bank above the raw floor), reaching zero exactly at the 40 s reserve where the raw
# policy takes over. Replay tier profiles (4 Kaggle episodes, experiments/
# rule_penalty_repro/) showed stepped ladders both overspending the opening (the old
# 256 top tier) and stranding 110-225 s unspent (the flat 128 ceiling); on this curve
# a typical ~40-searched-move game stays at 130+ sims throughout while genuinely long
# games taper gradually instead of halving in cliffs. remainingOverageTime (per-agent,
# drains with OUR compute only, passed live by the Kaggle runtime) is authoritative;
# the wall-clock estimate is the conservative fallback for local harnesses. Per-move
# deadlines scale linearly with the sims (the old tiers' seconds-per-sim slope).
SIMS_TOP = 256                # simulations at a FULL 600 s bank (owner 2026-08-16:
                              # restored to the proven old_1 256-top budget after the
                              # 160-top curve coincided with a ladder drop; the decay
                              # SHAPE below is unchanged)
SIMS_CURVE_EXPONENT = 0.5     # < 1 = hold depth early, drop off late
RAW_FLOOR_SECONDS = 40.0      # at/below this bank the raw policy plays (timeout reserve)
DEADLINE_BASE = 0.4           # per-move deadline seconds = base + per-sim * sims
DEADLINE_PER_SIM = 0.03359375  # exactly 9.0 s at the 256-sim top (owner 2026-08-16)
C_PUCT = 1.5
EPISODE_BUDGET = 600.0
MIN_SIMULATIONS = 8           # fewer than this completed -> not a search, play raw
SELECT_CONTEXT_COIN_HEAD = 46  # SelectContext (the select's TYPE is YES_NO=9)
MAX_LINE_NODES = 300           # per-simulation node cap (loop guard, see _descend)

# Opt-in action-rule mask for BRANCH POINTS (2026-08-10). main.py sets this to
# state_encoder.counter_option_mask when the checkpoint envelope carries action_rules (the
# model TRAINED under the rule, so search must offer the same menu -- an unmasked branch
# point would explore dead-target placements the policy has never seen). Hook shape:
# (observation_dict, select) -> set of allowed option indices, or None for "no
# restriction". None hook (every rule-less bundle) = behavior bit-identical to before.
option_mask_hook = None

# Opt-in LINE rule (owner 2026-08-12, require_play_supporter_from_meowth_ex): a
# turn-level constraint no single-menu mask can express. main.py sets this to
# state_encoder.line_rule_for(action_rules)'s object when the envelope names one. Each
# simulated line carries a small state: option_labels(observation, select) labels a
# node's options once at creation (None for almost every select), update(state, label)
# folds a taken edge in, and violated(state) is asked at every END-OF-OUR-TURN leaf --
# a violating line is scored -1 (a certain loss) instead of the value estimate, so no
# plan that breaks the rule can win the root comparison. Game-over terminals keep their
# exact result (a line that wins the game owes nobody a Supporter). None = the
# tracking never runs, behavior bit-identical to before.
line_rule = None

# ------------------------------------------------- experimental search structure ------- #
# Owner-ordered search experiments (2026-08-11). BOTH DEFAULT OFF -- behavior stays
# byte-identical to the shipped search until the A/B harness (or a future envelope
# field) flips them on the imported module.

# Card ids whose actions REVEAL RANDOMNESS (draw/shuffle trainers via PLAY, dig/draw
# ability Pokemon via ABILITY; dragapult set: Drakloak 120, Fezandipiti ex 140, Unfair
# Stamp 1080, Lillie's Determination 1227). Taking one of these actions creates a
# SAMPLED CHANCE NODE (owner-approved design 2026-08-12, "IS-MCTS but only at these
# actions"): up to REVEAL_SAMPLE_CAP independent resolutions of the randomness (the
# engine re-randomizes the shuffle/draw each time the edge is re-stepped -- with the
# knowledge-constrained determinize the deck MULTISET is exact, so these are draws from
# the true redraw distribution), each hosting its own fully searched contingency
# subtree, visits rotating to the least-visited outcome so the node's backed-up value
# is the MEAN over sampled outcomes of best conditional play. Standard expectimax-over-
# samples semantics: search below observable chance stays (contingency planning is
# correct decision theory); what dies is betting the line's value on ONE sampled draw.
REVEAL_HORIZON_CARDS = frozenset()
# Outcome fan-out PER 256 SIMS of budget; each search scales it with its own sim
# budget (owner 2026-08-14: "start at 8 and drop it as the bank drains") -- 256 sims ->
# 8 samples, the bank-curve top of 160 -> 5, 64 -> 2, 32 -> 1 (= the pre-sampling
# single-sample behavior). Proportional scaling keeps sims-per-outcome roughly constant
# (~32) as the curve declines instead of starving low-budget contingency planning.
REVEAL_SAMPLE_CAP = 8
# PROGRESSIVE WIDENING (2026-08-14, owner-reported "never plays Lillie's" fix): eager
# widening split a modestly-visited shuffle edge across all cap outcomes at ~2 visits
# each -- eight barely-planned futures averaged against rivals' fully-refined subtrees,
# a measured systematic demotion (paired probe: pick rate 6/24 -> 1/24, mean Q -0.21 ->
# -0.42; the one edge whose outcomes reached ~27 visits each was picked again). A new
# outcome is now sampled only once existing ones average this many visits, so an edge
# stays deep-and-few-sampled until its visit budget EARNS more diversity; by the time
# it can win the root it is both deep and averaged. Coin "chance" nodes (Crushing
# Hammer etc.) are a separate exact mechanism and are untouched.
REVEAL_WIDEN_VISITS = 8

# 0-or-1-pick CARD selects in the deck or discard view (deck-search targets: Ultra
# Ball, Poke Pad; discard recovery: Night Stretcher) become REAL branch points: one
# child per DISTINCT candidate card plus one DECLINE child (v2, owner spec 2026-08-14).
# Identical copies are merged into one branch (fetching copy A vs copy B is the same
# game state; their policy mass sums onto the representative), the FETCH_TOPK strongest
# distinct candidates are kept, and a menu with fewer than 2 distinct cards after the
# merge falls through to greedy -- that is a single-candidate take-or-decline in
# disguise, the first-play-optimism trap the >= 2 guard exists for (08-11 smoke: a
# zero-prior child outvisited real lines by surfing unvisited-child Q seeds; lift only
# with FPU-from-realized-backups). Priors come from the policy's candidate-row scores,
# whose row order is (pending options ++ STOP), so the STOP row prices the decline.
# Off = shipped behavior (greedy-resolved inside the line); enabled per-bundle via the
# checkpoint envelope key "fetch_branch" (main.py sets the module attribute).
FETCH_BRANCH = False
FETCH_TOPK = 4                 # distinct candidate branches kept per fetch node

# Set by main.py to `lambda: _GAME["knowledge"]` (the live CardKnowledge). Determinize
# then samples OUR hidden zones from the tracker's serial-level posterior instead of a
# blind uniform split: sighted cards can never be prized, and after any full deck view
# the prize multiset is EXACT (owner-ordered 2026-08-11 -- "after a search effect we
# know everything in hand+deck+discard+board and can deduce the prizes"). None (or a
# tracker in any inconsistent state) falls back to the shipped uniform split.
knowledge_hook = None

# BATCHED LEAF EVALUATION (owner go 2026-08-14, OPT-IN): several descent threads share
# the tree under a VIRTUAL LOSS (each edge a thread walks is provisionally scored as a
# loss until its real value backs up, so in-flight descents spread instead of piling
# onto one line), and every model forward they issue is routed by main.py's forward
# batcher into single padded batch-N calls -- amortizing the batch-1 dispatch cost that
# dominates deploy wall time (~70%, measured 2026-08-14). OFF by default: the
# sequential loop stays byte-identical until V5_BATCHED_SEARCH=1. When on, this is NOT
# the same search -- visit distributions differ by design, so the ship gate is a
# strength A/B, never a parity drill. Any failure of any kind permanently retires
# batched mode for the process and search continues sequentially.
BATCHED_LEAVES = os.environ.get("V5_BATCHED_SEARCH") == "1"
BATCH_WIDTH = max(2, int(os.environ.get("V5_BATCH_WIDTH", "8") or 8))
VIRTUAL_LOSS = -1.0
forward_batcher = None    # injected by main.py: width -> batcher (enter_worker/serve/..)
_BATCH_STATE = {"broken": False}
_PENDING = object()       # children[i] sentinel: another thread is expanding this edge


def _fold_line_labels(line_state, observation, select, chosen):
    """Fold the line_rule labels of the chosen option indices into `line_state`. Used
    for selects the tree answers WITHOUT branching (forced picks, greedy resolves);
    branched picks fold from the parent node's stored labels in _descend instead."""
    if line_rule is None or not chosen:
        return line_state
    labels = line_rule.option_labels(observation, select)
    if not labels:
        return line_state
    for picked in chosen:
        if 0 <= picked < len(labels) and labels[picked] is not None:
            line_state = line_rule.update(line_state, labels[picked])
    return line_state


def _violation_flags(state):
    """Per-component violated/obligation-open flags for a line_rule state. For every
    shipped rule, violated() evaluated mid-line means exactly "liability outstanding
    now", so it doubles as the obligation-open test the scoped penalty needs."""
    rules = getattr(line_rule, "rules", None)
    if rules is None:
        return (line_rule.violated(state),)
    return tuple(rule.violated(sub_state) for rule, sub_state in zip(rules, state))


def _assign_line_anchors(parent, child, depth):
    """SCOPED PENALTY bookkeeping (owner go 2026-08-14). line_anchors[i] = the path
    position of the edge that OPENED rule component i's still-outstanding obligation
    (None when nothing outstanding; 0 also covers obligations inherited from the real
    turn). Every fold during one expansion belongs to the edge being expanded, so one
    parent-vs-child comparison per expansion is complete. A violated leaf gets
    penalty_anchor = the shallowest opening among its violating components: backup
    charges -1 at and below it, the leaf's honest value above it. Any bookkeeping
    failure degrades to anchor None -> whole-path -1 (the old behavior), never a
    crash."""
    if line_rule is None or child.line_state is None:
        return
    try:
        child_flags = _violation_flags(child.line_state)
        if not any(child_flags):
            child.line_anchors = None
            return
        parent_flags = (_violation_flags(parent.line_state)
                        if parent.line_state is not None
                        else tuple(False for _ in child_flags))
        parent_anchors = parent.line_anchors or tuple(None for _ in child_flags)
        anchors = []
        for component, open_now in enumerate(child_flags):
            if not open_now:
                anchors.append(None)
            elif parent_flags[component]:
                anchors.append(parent_anchors[component]
                               if parent_anchors[component] is not None else 0)
            else:
                anchors.append(depth)
        child.line_anchors = tuple(anchors)
        if child.kind == "leaf" and child.penalty:
            child.penalty_anchor = min(
                (anchor if anchor is not None else 0)
                for component, anchor in enumerate(anchors)
                if child_flags[component])
    except Exception:
        child.line_anchors = None


def _flagged_option_indices(observation, select):
    """Option indices playing/using a REVEAL_HORIZON_CARDS card, or None when off/none."""
    if not REVEAL_HORIZON_CARDS:
        return None
    current = observation["current"]
    players = current["players"]
    mine = players[current["yourIndex"]]
    flagged = set()
    for index, option in enumerate(select.get("option") or []):
        kind = option.get("type")
        if kind == 7:                                  # PLAY from hand
            row = mine.get("hand") or []
        elif kind == 10:                               # ABILITY of a Pokemon in play
            row = mine.get({4: "active", 5: "bench"}.get(option.get("area"), "")) or []
        else:
            continue
        position = option.get("index")
        if position is not None and position < len(row) and row[position] \
                and row[position].get("id") in REVEAL_HORIZON_CARDS:
            flagged.add(index)
    return flagged or None


def _expand_priors(probabilities, n_options, allowed):
    """Masked-select priors arrive in PENDING order (resolve_with_v6 masks the pending
    list before building candidate rows, so the evaluator's softmax has one entry per
    ALLOWED option, ascending) while the node's children are indexed by ORIGINAL option
    index. Scatter them back to full length, masked slots at 0.0. First caught by the
    path-restoration drill as an IndexError (priors len 2, child index 2)."""
    if allowed is None or probabilities is None \
            or len(probabilities) == n_options:
        return probabilities
    full = [0.0] * n_options
    for position, original in enumerate(sorted(allowed)):
        if position < len(probabilities):
            full[original] = float(probabilities[position])
    return full

def _fetch_menu_shape(select, options):
    """True when this select is a branchable fetch menu under FETCH_BRANCH: an
    OPTIONAL take (minCount 0) of at most one card, every option a CARD in the
    deck view (area 1) or the discard pile (area 3)."""
    return (select.get("minCount") == 0 and select.get("maxCount") == 1
            and len(options) >= 2
            and all(o.get("type") == 3 and o.get("area") in (1, 3)
                    for o in options))


def _fetch_groups(observation, options, probabilities, declinable):
    """Dedup + cap the candidates of an optional card-pick menu. Groups options by
    the card id they would take (identical copies differ only by serial -- same game
    state), merges each group's policy mass onto its first option index, keeps the
    FETCH_TOPK strongest groups. Returns (allowed, priors, decline_index) sized for
    a node with len(options) (+1 when declinable) children, or None when fewer than
    2 DISTINCT cards remain -- the caller then resolves greedily, keeping the
    single-candidate guard intact after the merge.
    `probabilities` is in pending order (++ STOP last when declinable); the caller
    has already checked its length."""
    from src.game.encode_details import _entity_at
    groups = {}                                  # card id -> [option indices]
    for index, option in enumerate(options):
        card, _pokemon = _entity_at(observation, option.get("area"),
                                    option.get("index"), option.get("playerIndex"))
        key = card.get("id") if card is not None and card.get("id") is not None \
            else ("unresolved", index)
        groups.setdefault(key, []).append(index)
    if len(groups) < 2:
        return None
    merged = sorted(((sum(float(probabilities[m]) for m in members), members[0])
                     for members in groups.values()), reverse=True)
    decline_index = len(options) if declinable else None
    allowed = {representative for _prior, representative in merged[:FETCH_TOPK]}
    priors = [0.0] * (len(options) + (1 if declinable else 0))
    for prior, representative in merged[:FETCH_TOPK]:
        priors[representative] = prior
    if declinable:
        allowed.add(decline_index)
        priors[decline_index] = float(probabilities[len(options)])
    return allowed, priors, decline_index


ROOT_VETO_Q = -0.9             # a child converged at/below this Q is rule-condemned...
ROOT_VETO_VISITS = 8           # ...once it has at least this many looks


def _pick_root_child(root):
    """Most-visited allowed child, under the ROOT VETO (owner go 2026-08-15): a child
    whose Q has CONVERGED to a certain loss (rule penalties drive Q to -1) cannot win
    on visit count. High-prior moves collect visits EARLY, before their -1s
    accumulate, and most-visited-wins then overrides the verdict -- observed
    on-ladder (episode 93157396: Q(UB) = -1.0, prior 0.69, played anyway). Vetoed
    children are excluded unless EVERY candidate is vetoed: in a genuinely lost
    position (all moves ~-1) the normal pick still answers."""
    candidates = [i for i in range(len(root.children))
                  if root.allowed is None or i in root.allowed]
    viable = [i for i in candidates
              if not (root.visits[i] >= ROOT_VETO_VISITS
                      and root.total[i] / root.visits[i] <= ROOT_VETO_Q)]
    if viable:
        candidates = viable
    return max(candidates,
               key=lambda i: (root.visits[i],
                              (root.total[i] / root.visits[i]) if root.visits[i]
                              else -1e30,
                              root.priors[i]))


_clock = {"episode_start": None, "search_seconds": 0.0}
_rng = random.Random(20260802)


def note_episode_start():
    """Called by main.py at deck selection (a new episode)."""
    _clock["episode_start"] = time.time()
    _clock["search_seconds"] = 0.0


def search_budget(observation_dict=None):
    """(simulations, per_move_deadline_seconds) from the continuous bank curve, or
    None when the bank is at/below the raw floor (the raw policy plays). Simulations
    are quantized to multiples of 8 (smooth steps, legible search_tier stats); the
    curve's sub-16-sim tail (bank under ~44 s) also plays raw -- a search setup is
    not worth that little exploration."""
    remaining = None
    if observation_dict is not None:
        live = observation_dict.get("remainingOverageTime")
        if isinstance(live, (int, float)) and live > 0:
            remaining = float(live)
    if remaining is None:
        if _clock["episode_start"] is None:
            _clock["episode_start"] = time.time()
        remaining = EPISODE_BUDGET - (time.time() - _clock["episode_start"])
    headroom = min(remaining, EPISODE_BUDGET) - RAW_FLOOR_SECONDS
    if headroom <= 0:
        return None
    fraction = headroom / (EPISODE_BUDGET - RAW_FLOOR_SECONDS)
    simulations = int(round(SIMS_TOP * fraction ** SIMS_CURVE_EXPONENT / 8.0)) * 8
    if simulations < 16:
        return None
    return simulations, DEADLINE_BASE + DEADLINE_PER_SIM * simulations


# ------------------------------------------------------------------ engine adapter ---- #
# Dict-in / dict-out wrappers over the bundled engine's search exports. cg.api's own
# functions want dataclass Observations; the agent holds plain dicts, and the tree wants
# plain dicts back (encode_inflight consumes dicts), so we call the ctypes exports directly and
# json.loads the reply. Signatures copied from cg/api.py search_begin/search_step.

# The batched search steps worlds from several threads and the engine's thread safety
# is unverified, so EVERY engine call serializes on this lock (engine time is ~2% of
# search wall -- measured 2026-08-14 -- so the serialization costs nothing). The
# sequential path takes the same uncontended lock, which is free.
_ENGINE_LOCK = threading.Lock()


def _api():
    # The Kaggle loader restores sys.path after main.py loads (on-ladder observation
    # 2026-08-03), so a bare package import here can fail even though the cg/ directory
    # sits right next to this file. Repair the path from OUR OWN location first --
    # this module is loaded by file path, so __file__ is always real.
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    from cg import api
    if not hasattr(api, "agent_ptr"):
        api.agent_ptr = api.lib.AgentStart()
    return api


def _begin(observation_dict, determinization):
    """search_begin from a dict observation. Returns (search_id, observation_dict)."""
    import ctypes
    api = _api()
    sbi = observation_dict["search_begin_input"]
    your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active = determinization
    if observation_dict.get("select") and observation_dict["select"].get("deck") is not None:
        your_deck = []
    with _ENGINE_LOCK:
        reply = api.lib.SearchBegin(
            api.agent_ptr, sbi.encode("ascii"), len(sbi),
            (ctypes.c_int * len(your_deck))(*your_deck),
            (ctypes.c_int * len(your_prize))(*your_prize),
            (ctypes.c_int * len(opp_deck))(*opp_deck),
            (ctypes.c_int * len(opp_prize))(*opp_prize),
            (ctypes.c_int * len(opp_hand))(*opp_hand),
            (ctypes.c_int * len(opp_active))(*opp_active),
            1)                                                 # manual_coin=True always
    result = json.loads(reply)
    if result.get("error"):
        raise RuntimeError(f"SearchBegin error {result['error']}")
    state = result["state"]
    return state["searchId"], state["observation"]


def _step(search_id, move):
    """search_step -> (search_id, observation_dict)."""
    import ctypes
    api = _api()
    with _ENGINE_LOCK:
        reply = api.lib.SearchStep(api.agent_ptr, search_id,
                                   (ctypes.c_int * len(move))(*move), len(move))
    result = json.loads(reply)
    if result.get("error"):
        raise RuntimeError(f"SearchStep error {result['error']}")
    state = result["state"]
    return state["searchId"], state["observation"]


# ------------------------------------------------------------------ determinization ---- #

def _visible_ids(player):
    ids = []
    for pokemon in (player.get("active") or []) + (player.get("bench") or []):
        if pokemon is None:
            continue
        ids.append(pokemon["id"])
        for card in ((pokemon.get("energyCards") or []) + (pokemon.get("tools") or [])
                     + (pokemon.get("preEvolution") or [])):
            ids.append(card["id"])
    for card in (player.get("discard") or []):
        ids.append(card["id"])
    return ids


def determinize(observation_dict, deck_counts, rng):
    """One world. OUR hidden zones (deck order, prizes) are a uniform shuffle of our unseen
    pool -- exactly the split the +6pp ladder search used. The OPPONENT side only has to be
    ENGINE-LEGAL, not accurate: their pieces never move (our-turn-only search), so their
    hidden zones are their revealed cards cycled to the required counts. A wrong guess
    there can only matter through our own cards' rare interactions with their hidden
    zones, which is noise next to not searching at all."""
    current = observation_dict["current"]
    me = current["yourIndex"]
    mine, theirs = current["players"][me], current["players"][1 - me]

    pool = Counter(deck_counts)
    for card_id in _visible_ids(mine):
        if pool.get(card_id, 0) > 0:
            pool[card_id] -= 1
    for card in (mine.get("hand") or []):
        if pool.get(card["id"], 0) > 0:
            pool[card["id"]] -= 1
    prize_count = len(mine.get("prize") or [])
    deck_count = mine.get("deckCount") or 0
    your_prize = None
    knowledge = knowledge_hook() if knowledge_hook else None
    if knowledge is not None and prize_count > 0:
        try:
            # Prizes are a uniform subset of the NEVER-SIGHTED multiset (sighted cards
            # cannot be prized; exact after a full deck view). Sample the prizes from
            # that support and give the deck everything else in the hidden pool -- the
            # deck a search effect reveals then contains only cards that can truly be
            # there. Any arithmetic inconsistency (count drift, tracker corruption)
            # falls back to the blind split rather than risking an illegal world.
            candidates = []
            never_seen = knowledge.never_seen_counts()
            for card_id, copies in never_seen.items():
                candidates.extend([card_id] * min(copies, pool.get(card_id, 0)))
            if len(candidates) >= prize_count:
                rng.shuffle(candidates)
                your_prize = candidates[:prize_count]
                remainder = pool - Counter(your_prize)
                your_deck = [card_id for card_id, count in remainder.items()
                             for _ in range(count)]
                rng.shuffle(your_deck)
                your_deck = your_deck[:deck_count]
                # Position knowledge: known_top (first entry = the very next draw) pins
                # to the front, known_bottom (last entry = lowest card) pins to the back.
                for card_id in reversed(knowledge.known_top_ids()):
                    if card_id in your_deck:
                        your_deck.remove(card_id)
                        your_deck.insert(0, card_id)
                for card_id in knowledge.known_bottom_ids():
                    if card_id in your_deck:
                        your_deck.remove(card_id)
                        your_deck.append(card_id)
                if len(your_prize) != prize_count or len(your_deck) != deck_count:
                    raise ValueError("knowledge split size drift")
        except Exception:
            your_prize = None                          # deploy code never crashes
    if your_prize is None:
        hidden = [card_id for card_id, count in pool.items() for _ in range(count)]
        rng.shuffle(hidden)
        your_prize = hidden[:prize_count]
        your_deck = hidden[prize_count:prize_count + deck_count]

    revealed = _visible_ids(theirs)
    if not revealed:                       # face-down-everything corner: use our ids
        revealed = list(deck_counts.keys())

    def cycle(count):
        return [revealed[i % len(revealed)] for i in range(count)]

    opp_deck = cycle(theirs.get("deckCount") or 0)
    opp_prize = cycle(len(theirs.get("prize") or []))
    opp_hand = cycle(theirs.get("handCount") or 0)
    opp_active = []
    active = theirs.get("active") or []
    if active and active[0] is None:       # face-down active needs a Basic Pokemon guess
        from src.cards import get_card
        basic = next((card_id for card_id in revealed
                      if (get_card(card_id) or {}).get("stage") == 0), None)
        opp_active = [basic if basic is not None else revealed[0]]
    return your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active


# ------------------------------------------------------------------ tree -------------- #

class _Node:
    __slots__ = ("search_id", "kind", "value", "priors", "children", "visits", "total",
                 "allowed", "flagged", "decline_index", "pending_observation",
                 "pending_select", "pinned_multi", "reveal_move", "labels",
                 "line_state", "penalty", "line_anchors", "penalty_anchor")

    def __init__(self, search_id, kind, value, priors=None, n_children=0, allowed=None,
                 flagged=None, decline_index=None):
        self.search_id = search_id
        self.kind = kind               # "decision" | "chance" | "leaf"
        self.value = value             # root-perspective value estimate at creation
        self.priors = priors
        self.children = [None] * n_children
        self.visits = [0] * n_children
        self.total = [0.0] * n_children
        self.allowed = allowed         # option-index set from option_mask_hook, or None
        self.flagged = flagged         # REVEAL_HORIZON option indices, or None
        self.decline_index = decline_index   # FETCH_BRANCH: the "take nothing" child
        self.pending_observation = None      # FETCH_BRANCH multi-pick: state to resolve
        self.pending_select = None
        self.pinned_multi = False            # children = forced FIRST pick, policy rest
        self.reveal_move = None              # kind "reveal": the flagged move to re-step
        self.labels = None             # line_rule option labels for this node's select
        self.line_state = None         # line_rule state on the path INTO this node
        self.penalty = False           # leaf only: line_rule violated at this seat flip
        self.line_anchors = None       # scoped penalty: per-component opening depths
        self.penalty_anchor = None     # penalty leaf: -1 charged from this depth down


class TurnSearch:
    """One search per real decision. `evaluator` is main.py's closure bundle:
    evaluate(observation_dict, select_or_None) -> (probabilities_or_None, value) with the
    root-frozen trackers baked in; `greedy_resolve(observation_dict, select)` resolves a
    multi-pick select with the policy. Both raise on any problem."""

    def __init__(self, evaluator, greedy_resolve, deck_counts, stats):
        self.evaluate = evaluator
        self.greedy_resolve = greedy_resolve
        self.deck_counts = deck_counts
        self.stats = stats
        self.reveal_cap = REVEAL_SAMPLE_CAP    # run() rescales per its sim budget

    # -- world advance: everything that is not a branch point ----------------------- #

    def _advance(self, search_id, observation, root_seat, line_state=None):
        """Step the world until a branch point / seat flip / terminal. Returns a _Node.
        `line_state` is the line_rule state accumulated on the path in; auto-answered
        selects (forced picks, greedy resolves) fold their labels in here so the state
        the created node carries is complete up to its own pending select."""
        for _ in range(200):                                   # hard loop guard
            current = observation["current"]
            if current.get("result", -1) != -1:
                won = current["result"] == root_seat
                return _Node(search_id, "leaf", 1.0 if won else -1.0)
            select = observation.get("select")
            if select is None:
                search_id, observation = _step(search_id, [])
                continue
            if current["yourIndex"] != root_seat:
                # seat flipped: END OF OUR TURN. A line that violates the opt-in line
                # rule is a forbidden plan. SCOPED PENALTY (owner go 2026-08-14; fixes
                # the measured all-lines--1 poisoning, see experiments/
                # rule_penalty_repro/): the leaf is marked `penalty` and ALSO carries
                # the honest value estimate -- backup charges -1 only to the edges at
                # or below the obligation-opening edge (the item play / Meowth bench),
                # and the honest value to the innocent edges above it, so a root move
                # is no longer damned by a rule-breaking sub-plan the search can
                # simply route around. The anchor is computed at expansion time in
                # _descend from line_anchors bookkeeping.
                if line_rule is not None and line_rule.violated(line_state):
                    self.stats["search_line_penalties"] += 1
                    _probabilities, value = self.evaluate(observation,
                                                          observation.get("select"))
                    node = _Node(search_id, "leaf", -float(value))
                    node.penalty = True
                    node.line_state = line_state
                    return node
                # Leaf value = -(V from the opponent's
                # view), root perspective, root-frozen context (mirrors training).
                # Evaluated THROUGH the leaf's own pending select -- the exact state
                # shape every training forward used (states always carry the mover's
                # select; SearchLeafEncoder does the same). select=None only for the
                # no-menu corner, which the null-pick guard in main.py handles.
                _probabilities, value = self.evaluate(observation,
                                                      observation.get("select"))
                return _Node(search_id, "leaf", -float(value))
            # COIN detection FIXED 2026-08-09: COIN_HEAD (46) is a SelectCONTEXT; the
            # select's TYPE is YES_NO (9). The old `type == 46` never matched, so coin
            # flips fell through to the decision branch below -- the search CHOSE its own
            # outcomes, PUCT steered to all-heads (better value), flip-until-tails
            # attacks became unbounded self-dealt heads chains, and every flip line was
            # valued as if each coin were guaranteed won. Context 46 only ever appears
            # in-search (manual_coin=1); real play flips inside the engine.
            if select.get("context") == SELECT_CONTEXT_COIN_HEAD:
                node = _Node(search_id, "chance", 0.0,
                             n_children=len(select["option"]))
                node.line_state = line_state       # coin outcomes carry no labels
                return node
            from src.game.encode_inflight import forced_answer
            forced, _reason = forced_answer(select)
            if forced is not None:
                line_state = _fold_line_labels(line_state, observation, select, forced)
                search_id, observation = _step(search_id, forced)
                continue
            options = select.get("option") or []
            if select.get("maxCount") == 1 and select.get("minCount") == 1 \
                    and len(options) >= 2:
                probabilities, value = self.evaluate(observation, select)
                allowed = option_mask_hook(observation, select) \
                    if option_mask_hook else None
                node = _Node(search_id, "decision", float(value),
                             priors=_expand_priors(probabilities, len(options), allowed),
                             n_children=len(options), allowed=allowed,
                             flagged=_flagged_option_indices(observation, select))
                if line_rule is not None:
                    node.labels = line_rule.option_labels(observation, select)
                    node.line_state = line_state
                return node
            if FETCH_BRANCH and _fetch_menu_shape(select, options):
                # Fetch target choice becomes a branch point: one child per DISTINCT
                # candidate card (copies merged, FETCH_TOPK strongest kept -- see the
                # flag's block comment) plus the DECLINE child, whose prior is the
                # STOP row. This engine presents even take-up-to-2 searches as
                # SEQUENTIAL 0/1 selects, so each take with >= 2 distinct candidates
                # branches here. Menus that collapse to ONE distinct card after the
                # merge fall through to greedy (the single-candidate guard). Shape
                # surprises fall through to greedy.
                probabilities, value = self.evaluate(observation, select)
                if probabilities is not None and len(probabilities) == len(options) + 1:
                    grouped = _fetch_groups(observation, options, probabilities, True)
                    if grouped is not None:
                        allowed, priors, decline_index = grouped
                        self.stats["search_fetch_nodes"] += 1
                        node = _Node(search_id, "decision", float(value),
                                     priors=priors,
                                     n_children=len(options) + 1,
                                     allowed=allowed, decline_index=decline_index)
                        if line_rule is not None:
                            node.labels = line_rule.option_labels(observation, select)
                            node.line_state = line_state
                        return node
            if FETCH_BRANCH and select.get("maxCount", 0) > 1 and len(options) >= 2 \
                    and getattr(self, "greedy_resolve_pinned", None) is not None \
                    and all(o.get("type") == 3 and o.get("area") in (1, 3)
                            for o in options):
                # Multi-target search (Poffin's take-up-to-2): branch the FIRST
                # target -- one child per DISTINCT candidate (copies merged, capped,
                # + decline when minCount 0) -- and the policy fills the remaining
                # picks conditioned on it. Full subset enumeration would be C(N,k)
                # children; the first pick carries the strategic weight. First-sub-
                # pick probabilities are the priors (rows = pending options ++
                # STOP-when-declinable).
                probabilities, value = self.evaluate(observation, select)
                declinable = select.get("minCount", 0) == 0
                expected = len(options) + (1 if declinable else 0)
                if probabilities is not None and len(probabilities) == expected:
                    grouped = _fetch_groups(observation, options, probabilities,
                                            declinable)
                    if grouped is not None:
                        allowed, priors, decline_index = grouped
                        self.stats["search_fetch_multi_nodes"] += 1
                        node = _Node(search_id, "decision", float(value),
                                     priors=priors, n_children=expected,
                                     allowed=allowed, decline_index=decline_index)
                        node.pending_observation = observation
                        node.pending_select = select
                        node.pinned_multi = True
                        if line_rule is not None:
                            node.labels = line_rule.option_labels(observation, select)
                            node.line_state = line_state
                        return node
            # multi-pick (or 0/1-pick with STOP): the policy resolves it greedily and the
            # world moves on -- these are not branch points (matches training search).
            move = self.greedy_resolve(observation, select)
            line_state = _fold_line_labels(line_state, observation, select, move)
            search_id, observation = _step(search_id, move)
        raise RuntimeError("advance loop guard tripped")

    def _descend(self, node, root_seat):
        """One simulation from `node`: descend to a seat-flip/terminal leaf, back the
        LEAF value up every edge of the path, return it.

        LEAF-ONLY BACKUP (2026-08-09 fix): only leaf values enter Q. The old code backed
        up a freshly expanded node's own creation value -- a MID-TURN value-head estimate,
        which runs ~+0.2-0.4 above the same position at end of turn (mover bias), so any
        root move opening a long menu chain collected optimistic backups while ATTACK/END
        (instant seat flip) were scored honestly. Measured on-ladder: RETREAT (prior
        0.001, 5-select chain) outvisited a winning ATTACK (prior 0.82) 122-61. node.value
        survives solely as the (uniformly optimistic) first-play Q, which guarantees each
        option one look.

        ITERATIVE, not recursive (2026-08-09): descend paths can be hundreds of nodes
        (the tree deepens across sims; chain-heavy turns), and the model forward at the
        expansion tip stacks torch frames on top -- the recursive version blew Python's
        stack in the path-restoration drill. MAX_LINE_NODES is the loop guard: a
        pathological line backs up the current node's estimate instead (rare, documented
        bias beats a crash)."""
        path = []                                      # (node, child index) edges walked
        penalty_anchor = None                          # scoped penalty: -1 from here down
        while True:
            if node.kind == "leaf":
                value = node.value
                if node.penalty:
                    # Violated leaf: honest value above the anchor, -1 at/below it.
                    # Anchor None (bookkeeping failed / inherited) = whole path, the
                    # pre-fix behavior.
                    penalty_anchor = node.penalty_anchor \
                        if node.penalty_anchor is not None else 0
                break
            if len(path) >= MAX_LINE_NODES:
                value = node.value                     # guard fallback: mid-turn estimate
                break
            if node.kind == "reveal":
                # Sampled chance node: PROGRESSIVELY widen toward reveal_cap
                # (tier-scaled) outcome subtrees -- a new sample only once existing
                # ones average REVEAL_WIDEN_VISITS -- then rotate visits to the
                # least-visited outcome; the aggregate converges to the MEAN over
                # outcomes of best conditional play.
                if len(node.children) < self.reveal_cap \
                        and sum(node.visits) >= REVEAL_WIDEN_VISITS * len(node.children):
                    node.children.append(None)
                    node.visits.append(0)
                    node.total.append(0.0)
                    index = len(node.children) - 1
                else:
                    index = min(range(len(node.children)), key=lambda i: node.visits[i])
                child = node.children[index]
                if child is None:
                    self.stats["search_reveal_samples"] += 1
                    search_id, observation = _step(node.search_id, node.reveal_move)
                    child = self._advance(search_id, observation, root_seat,
                                          node.line_state)
                    _assign_line_anchors(node, child, len(path))
                    node.children[index] = child
                path.append((node, index))
                node = child
                continue
            if node.kind == "chance":
                index = _rng.randrange(len(node.children))     # uniform coin
            else:
                sqrt_total = math.sqrt(1 + sum(node.visits))
                best, best_score = None, -1e30
                for index in range(len(node.children)):
                    if node.allowed is not None and index not in node.allowed:
                        continue                       # action-rule-masked option
                    q = (node.total[index] / node.visits[index]) if node.visits[index] \
                        else node.value
                    u = C_PUCT * node.priors[index] * sqrt_total / (1 + node.visits[index])
                    score = q + u
                    if score > best_score:
                        best, best_score = index, score
                index = best if best is not None else 0
            child = node.children[index]
            if child is None:
                if node.flagged is not None and index in node.flagged:
                    # Flagged randomness: the child is a sampled CHANCE node anchored at
                    # THIS node's state; each of its outcome children re-steps the same
                    # move for a fresh resolution.
                    self.stats["search_reveal_nodes"] += 1
                    child = _Node(node.search_id, "reveal", node.value)
                    child.reveal_move = [index]
                    child.line_state = node.line_state
                    if line_rule is not None and node.labels \
                            and node.labels[index] is not None:
                        child.line_state = line_rule.update(node.line_state,
                                                            node.labels[index])
                    _assign_line_anchors(node, child, len(path))
                    node.children[index] = child
                    path.append((node, index))
                    node = child
                    continue
                if node.pinned_multi and node.decline_index != index:
                    move = self.greedy_resolve_pinned(node.pending_observation,
                                                      node.pending_select, index)
                else:
                    move = [] if node.decline_index == index else [index]
                line_state = node.line_state
                if line_rule is not None and node.labels:
                    # Branched pick(s): fold the taken options' labels (pinned_multi
                    # moves list ORIGINAL option indices, so labels line up).
                    for picked in move:
                        if 0 <= picked < len(node.labels) \
                                and node.labels[picked] is not None:
                            line_state = line_rule.update(line_state,
                                                          node.labels[picked])
                search_id, observation = _step(node.search_id, move)
                child = self._advance(search_id, observation, root_seat, line_state)
                _assign_line_anchors(node, child, len(path))
                node.children[index] = child
            path.append((node, index))
            node = child
        for position, (parent, index) in enumerate(path):
            parent.visits[index] += 1
            parent.total[index] += (-1.0 if penalty_anchor is not None
                                    and position >= penalty_anchor else value)
        return value

    # -- batched descent (V5_BATCHED_SEARCH=1): threads + virtual loss + shared tree - #

    @staticmethod
    def _finalize_batched(path, value):
        """Convert the path's virtual losses into the real backup. The visit counts were
        already taken when the loss was applied, so only the totals move."""
        for parent, index in path:
            parent.total[index] += value - VIRTUAL_LOSS

    @staticmethod
    def _rollback_batched(path):
        """Abandon a descent completely: remove the virtual losses AND the visits."""
        for parent, index in path:
            parent.visits[index] -= 1
            parent.total[index] -= VIRTUAL_LOSS

    def _descend_batched(self, root, root_seat, condition, deadline):
        """One simulation on the SHARED tree. Selection and all pure bookkeeping run
        under `condition`'s lock with a virtual loss applied to every edge walked, so
        concurrent descents spread instead of piling onto one line. Expensive expansion
        (engine steps + model forwards, which the batcher coalesces across threads) runs
        with the lock RELEASED while the claimed edge holds a _PENDING sentinel; other
        threads treat pending edges as unavailable. Returns True when a value was backed
        up, False when the deadline forced an abandon (bookkeeping fully rolled back).
        Semantics per node kind mirror _descend exactly -- only the concurrency
        scaffolding is new."""
        path = []
        node = root
        while True:
            pending = None
            value = None
            with condition:
                while True:
                    if node.kind == "leaf":
                        # Batched mode keeps the PRE-scoping semantics: a penalty
                        # leaf backs up -1 along the whole path (threading the
                        # scoped anchor through virtual losses isn't worth it for a
                        # retired mode -- ships dark, never enabled).
                        value = -1.0 if node.penalty else node.value
                        break
                    if len(path) >= MAX_LINE_NODES:
                        value = node.value             # guard fallback (see _descend)
                        break
                    if node.kind == "reveal":
                        if len(node.children) < self.reveal_cap \
                                and sum(node.visits) >= REVEAL_WIDEN_VISITS \
                                * len(node.children):
                            node.children.append(None)
                            node.visits.append(0)
                            node.total.append(0.0)
                            index = len(node.children) - 1
                        else:
                            open_children = [i for i in range(len(node.children))
                                             if node.children[i] is not _PENDING]
                            if not open_children:
                                if time.time() >= deadline:
                                    self._rollback_batched(path)
                                    condition.notify_all()
                                    return False
                                condition.wait(0.02)
                                continue
                            index = min(open_children, key=lambda i: node.visits[i])
                    elif node.kind == "chance":
                        open_children = [i for i in range(len(node.children))
                                         if node.children[i] is not _PENDING]
                        if not open_children:
                            if time.time() >= deadline:
                                self._rollback_batched(path)
                                condition.notify_all()
                                return False
                            condition.wait(0.02)
                            continue
                        index = open_children[_rng.randrange(len(open_children))]
                    else:
                        sqrt_total = math.sqrt(1 + sum(node.visits))
                        best, best_score = None, -1e30
                        for index in range(len(node.children)):
                            if node.allowed is not None and index not in node.allowed:
                                continue
                            if node.children[index] is _PENDING:
                                continue       # another thread is expanding this edge
                            q = (node.total[index] / node.visits[index]) \
                                if node.visits[index] else node.value
                            u = C_PUCT * node.priors[index] * sqrt_total \
                                / (1 + node.visits[index])
                            score = q + u
                            if score > best_score:
                                best, best_score = index, score
                        if best is None:
                            if time.time() >= deadline:
                                self._rollback_batched(path)
                                condition.notify_all()
                                return False
                            condition.wait(0.02)
                            continue
                        index = best
                    child = node.children[index]
                    if child is None and node.kind == "decision" \
                            and node.flagged is not None and index in node.flagged:
                        # Flagged randomness: pure bookkeeping (no engine or model
                        # work), so the reveal node is created inline under the lock,
                        # exactly as the sequential descent creates it.
                        self.stats["search_reveal_nodes"] += 1
                        child = _Node(node.search_id, "reveal", node.value)
                        child.reveal_move = [index]
                        child.line_state = node.line_state
                        if line_rule is not None and node.labels \
                                and node.labels[index] is not None:
                            child.line_state = line_rule.update(node.line_state,
                                                                node.labels[index])
                        node.children[index] = child
                    node.visits[index] += 1            # virtual loss: visit now ...
                    node.total[index] += VIRTUAL_LOSS  # ... provisional loss until real
                    path.append((node, index))
                    if child is None:
                        node.children[index] = _PENDING
                        pending = (node, index)
                        break
                    node = child
                if value is not None:
                    self._finalize_batched(path, value)
                    condition.notify_all()
                    return True
            # Lock released: expand the claimed edge. Engine steps serialize on
            # _ENGINE_LOCK; model forwards coalesce in the batcher with every other
            # thread parked at this same point.
            parent, index = pending
            try:
                if parent.kind == "reveal":
                    self.stats["search_reveal_samples"] += 1
                    search_id, observation = _step(parent.search_id, parent.reveal_move)
                    child = self._advance(search_id, observation, root_seat,
                                          parent.line_state)
                else:
                    if parent.pinned_multi and parent.decline_index != index \
                            and getattr(self, "greedy_resolve_pinned", None) is not None:
                        child_move = self.greedy_resolve_pinned(
                            parent.pending_observation, parent.pending_select, index)
                    else:
                        child_move = [] if parent.decline_index == index else [index]
                    line_state = parent.line_state
                    if line_rule is not None and parent.labels:
                        for picked in child_move:
                            if 0 <= picked < len(parent.labels) \
                                    and parent.labels[picked] is not None:
                                line_state = line_rule.update(line_state,
                                                              parent.labels[picked])
                    search_id, observation = _step(parent.search_id, child_move)
                    child = self._advance(search_id, observation, root_seat, line_state)
            except BaseException:
                with condition:
                    parent.children[index] = None      # un-claim the edge
                    self._rollback_batched(path)
                    condition.notify_all()
                raise
            with condition:
                parent.children[index] = child
                condition.notify_all()
            node = child

    def _run_batched(self, root, root_seat, deadline, simulations_budget):
        """The batched simulation loop: descent worker threads plus the calling thread
        serving their coalesced forwards. The width scales down with the tier so the
        virtual-loss distortion stays a small fraction of the budget (256 sims -> 8
        wide, 32 sims -> 4 wide). Returns completed simulations; raises on any worker
        failure (the caller then retires batched mode for the process)."""
        width = max(2, min(BATCH_WIDTH, simulations_budget // 8))
        condition = threading.Condition()
        counter_lock = threading.Lock()
        counters = {"started": 0, "done": 0}
        errors = []
        batcher = forward_batcher(width)

        def worker():
            batcher.enter_worker()
            try:
                while True:
                    with counter_lock:
                        if counters["started"] >= simulations_budget or errors \
                                or time.time() >= deadline:
                            return
                        counters["started"] += 1
                    completed = self._descend_batched(root, root_seat, condition,
                                                      deadline)
                    with counter_lock:
                        if completed:
                            counters["done"] += 1
                        else:
                            counters["started"] -= 1
            except BaseException as error:     # a worker must never die silently
                errors.append(error)
                with condition:
                    condition.notify_all()
            finally:
                batcher.exit_worker()

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(width)]
        for thread in threads:
            thread.start()
        try:
            batcher.serve(lambda: any(thread.is_alive() for thread in threads))
        finally:
            # Close BEFORE joining: close releases any submitter still blocked on the
            # batcher (they raise, roll back their descent and exit), so the join can
            # never hang on a worker waiting for a forward that will not come.
            batcher.close()
            for thread in threads:
                thread.join()
        if errors:
            raise errors[0]
        return counters["done"]

    # -- entry ----------------------------------------------------------------------- #

    def run(self, observation_dict, select, deadline, simulations_budget=256):
        """Search the CURRENT select with `simulations_budget` sims (the bank curve
        picks it). Returns the chosen option index, or None (caller plays the raw
        policy move). Two root shapes: the classic mandatory single pick, and --
        under FETCH_BRANCH -- an optional fetch menu, whose extra DECLINE child is
        reported as index len(options); the caller maps it to the empty move."""
        root_seat = observation_dict["current"]["yourIndex"]
        options = select.get("option") or []
        # Tier-scaled outcome fan-out: full budget -> REVEAL_SAMPLE_CAP samples per
        # reveal node, lower bank tiers proportionally fewer (floor 1 = single-sample).
        self.reveal_cap = max(1, REVEAL_SAMPLE_CAP * simulations_budget // 256)
        world = determinize(observation_dict, self.deck_counts, _rng)
        search_id, root_observation = _begin(observation_dict, world)
        probabilities, value = self.evaluate(root_observation, select)
        if FETCH_BRANCH and _fetch_menu_shape(select, options):
            # REAL fetch decision (owner spec 2026-08-14): searched with the same
            # dedup + cap + decline machinery as in-tree fetch nodes. Any shape
            # surprise (< 2 distinct candidates, unexpected row count) -> None, and
            # the caller plays the raw policy exactly as before.
            if probabilities is None or len(probabilities) != len(options) + 1:
                return None
            grouped = _fetch_groups(root_observation, options, probabilities, True)
            if grouped is None:
                return None
            allowed, priors, decline_index = grouped
            self.stats["search_fetch_roots"] += 1
            root = _Node(search_id, "decision", float(value), priors=priors,
                         n_children=len(options) + 1, allowed=allowed,
                         decline_index=decline_index,
                         flagged=_flagged_option_indices(root_observation, select))
        else:
            # Mask from the SAME observation the evaluator masked its pending list
            # with (the in-search root view), so the prior expansion cannot disagree
            # with it.
            root_allowed = option_mask_hook(root_observation, select) \
                if option_mask_hook else None
            root = _Node(search_id, "decision", float(value),
                         priors=_expand_priors(probabilities, len(select["option"]),
                                               root_allowed),
                         n_children=len(select["option"]), allowed=root_allowed,
                         flagged=_flagged_option_indices(root_observation, select))
        if line_rule is not None:
            root.labels = line_rule.option_labels(root_observation, select)
            # root_state, not initial_state: the REAL turn may already carry an
            # outstanding obligation from an earlier decision (main.py feeds every real
            # move through line_rule.observe_real).
            root.line_state = line_rule.root_state()
            try:
                # Scoped-penalty seed: an obligation already outstanding at the root
                # (an inherited due card) anchors at 0 -- every edge of a line that
                # fails it is charged, since every line can still choose to comply.
                root.line_anchors = tuple(
                    0 if flag else None
                    for flag in _violation_flags(root.line_state))
            except Exception:
                root.line_anchors = None
        simulations = 0
        if BATCHED_LEAVES and forward_batcher is not None \
                and not _BATCH_STATE["broken"]:
            try:
                simulations = self._run_batched(root, root_seat, deadline,
                                                simulations_budget)
                self.stats["search_batched_decisions"] += 1
            except Exception:
                # Retire batched mode for the whole process; the SEQUENTIAL search
                # keeps serving from the next decision on. This decision plays the raw
                # policy: the tree may hold _PENDING claims from dead workers, so it
                # must not be walked again.
                _BATCH_STATE["broken"] = True
                self.stats["search_batch_broken"] += 1
                import sys
                import traceback
                print("[search] batched mode failed (sequential serves from here):\n"
                      + traceback.format_exc(), file=sys.stderr, flush=True)
                return None
        while simulations < simulations_budget and time.time() < deadline:
            self._descend(root, root_seat)
            simulations += 1
        self.stats["search_sims"] += simulations
        if simulations < MIN_SIMULATIONS:
            return None
        return _pick_root_child(root)
