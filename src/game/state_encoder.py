"""v6 encoding = v5 ++ two CHAIN-PROGRESS option columns (the batched-effect fix).

Why (chain-freeze audit, 2026-08-03): some engine effects apply their picks in one batch
at resolution -- Phantom Dive's six counter placements all see the IDENTICAL observation
(proved by probe: full mutable state frozen across steps 2..6), while other cards stream
their updates live. During a batched chain the model literally cannot know where its
earlier counters went: the select-context countdown (remainDamageCounter) says how many
remain, `same_id_already_chosen` only works within ONE multi-pick select (the chain is
six separate selects), and the board never moves. Observed consequence on-ladder: all six
counters dumped on one target, straight past 0 HP.

The fix: InFlightTracker (v5's chain-lifecycle memory, whose reset/observe/record
contract already flows through every driver) now accumulates accepted picks per target
serial within the effect chain (`chain_picks`); v6 appends per raw option:

  V6_CHAIN_PICKS    picks already accepted on this option's target THIS chain, /8
  V6_PROJECTED_HP   the HP this target would have if THIS option takes one more counter,
                    /340 -- minus 10 in damage-counter contexts (13 DAMAGE_COUNTER /
                    14 ANY / 15 DAMAGE), plus 10 in 16 (REMOVE_DAMAGE_COUNTER); other
                    contexts: raw HP when the option targets a Pokemon, 0 otherwise.

CORRECTION 2026-08-05 (owner-reported overkill; audit in scratchpad/confirm_bug2.py).
The premise above -- that batched chains freeze the board -- does NOT hold for Phantom
Dive on either cg.dll or build_v23. Measured over 50 confound-free consecutive pairs
(one chain, one target, recording the exact option answered):
  * 45/50 the engine STREAMS: the accepted counter is applied immediately and the next
    select already shows hp 10 lower;
  * 5/50 the counter was PREVENTED (Team Rocket's Articuno's Repelling Veil -- placement
    is an attack EFFECT) and hp correctly did not move.
In BOTH branches the observed `hp` is the truth at encode time, so the original
`hp - 10*assigned` subtracted every accepted pick a SECOND time (error 10*picks, growing
through the chain), and the floor reported "already dead" while the target still had HP
-- for Articuno it claimed 80 HP on a Pokemon sitting at 120 and immune. Fixed: the
column is the live HP one counter forward, unfloored. V6_CHAIN_PICKS was always correct
and is unchanged.

STOP rows carry zeros, like every v5 pointer column.

Token / global encodes are v5's UNCHANGED (re-exported); only the option matrix widens:
OPTION_FEATURE_DIM_V6 = OPTION_FEATURE_DIM_V5 + 2 = 2048. A v5 checkpoint warm-starts a
v6 model by zero-padding the option projection's input weights -- with zero weights on
the new columns the v6 forward is exactly the v5 forward, so nothing is forgotten.
"""

import numpy as np

from src.cards import CARDS, get_card
from src.game.encode_details import _entity_at
from src.game.encode_selection import MultiSelect, candidate_matrix
from src.game.encode_inflight import (InFlightTracker, OPTION_FEATURE_DIM_V4,
                                OPTION_FEATURE_DIM_V5, base_option_matrix_v5,
                                encode_observation_v5, forced_answer,
                                global_features_v5)

__all__ = ["InFlightTracker", "OPTION_FEATURE_DIM_V6", "base_option_matrix_v6",
           "candidate_matrix_v6", "chain_columns", "encode_observation_v6",
           "forced_answer", "global_features_v6", "resolve_with_v6"]

V6_OPTION_EXTRA_DIM = 2
OPTION_FEATURE_DIM_V6 = OPTION_FEATURE_DIM_V5 + V6_OPTION_EXTRA_DIM     # 2048
V6_CHAIN_PICKS = 0                     # column offsets within the v6 extra block
V6_PROJECTED_HP = 1

_AREA_ACTIVE, _AREA_BENCH = 4, 5
_DAMAGE_MINUS_CONTEXTS = frozenset((13, 14, 15))    # counters/damage being placed
_DAMAGE_PLUS_CONTEXTS = frozenset((16,))            # counters being removed (healing)
_PICKS_SCALE = 8.0
_HP_SCALE = 340.0

# Tokens and globals are v5's, verbatim.
encode_observation_v6 = encode_observation_v5
global_features_v6 = global_features_v5


def chain_columns(observation, select, in_flight, engine_hp=None):
    """[N, 2] chain-progress columns for the select's RAW options.

    `engine_hp`: optional {option index: post-placement HP} from
    src.game.engine_projection.engine_projected_hp -- the ENGINE's own answer, with
    immunity/prevention/modifiers already resolved. Where it has a value it WINS; the
    hand-computed `hp -/+ 10` below is only the fallback for options the engine could not
    be asked about (see engine_projection's SCOPE). The hand formula disagrees with the
    engine on 39% of placement options, so prefer the engine wherever it is available."""
    options = select["option"]
    columns = np.zeros((len(options), V6_OPTION_EXTRA_DIM), dtype=np.float32)
    picks = getattr(in_flight, "chain_picks", None) or {}
    context = select.get("context")
    me_index = observation["current"]["yourIndex"]
    for position, option in enumerate(options):
        area = option.get("area")
        if area not in (_AREA_ACTIVE, _AREA_BENCH):
            continue
        owner = option.get("playerIndex")
        owner = me_index if owner is None else owner
        _card, pokemon = _entity_at(observation, area, option.get("index"), owner)
        if pokemon is None:
            continue
        assigned = picks.get(pokemon.get("serial") or 0, 0)
        columns[position, V6_CHAIN_PICKS] = min(assigned, _PICKS_SCALE) / _PICKS_SCALE
        # `hp` is ALREADY true at encode time -- see the 2026-08-05 audit note above. It
        # must NOT be adjusted for `assigned`; doing so counted every accepted pick a
        # second time. What the option row wants is the HP that results from taking THIS
        # option once more, which is one counter off the live value.
        # NOTHING about damage is computed here. Either the ENGINE tells us the HP this
        # option leads to (engine_hp, from engine_projection.engine_projected_hp), or we
        # report the engine's CURRENT HP for the target. The old `hp -/+ 10` guessed at the
        # result -- it assumed a counter is worth 10 and that it lands, and both are false
        # for immune/prevented targets. Guessing is what this column is no longer allowed
        # to do, so there is no arithmetic fallback: an unknown result is reported as the
        # present state, never as an invented one.
        hp = pokemon.get("hp") or 0
        if engine_hp is not None and position in engine_hp:
            hp = engine_hp[position]
        # No floor: overkill reads negative and must stay distinguishable from exactly
        # lethal. The old max(hp, 0) reported 0 while the target still had HP.
        columns[position, V6_PROJECTED_HP] = min(hp, _HP_SCALE) / _HP_SCALE
    return columns


def base_option_matrix_v6(observation, select, in_flight, engine_hp=None):
    """(v3 rows, v6 extra rows): the v5 extras ++ the two chain columns, one encode per
    select -- chain state cannot change until the answer is sent.

    `engine_hp` is passed straight through to chain_columns; None keeps the pre-engine
    behaviour, so every existing caller is byte-identical."""
    v3_matrix, v5_extra = base_option_matrix_v5(observation, select, in_flight)
    v6_extra = np.concatenate(
        [v5_extra, chain_columns(observation, select, in_flight, engine_hp)], axis=1)
    return v3_matrix, v6_extra


def candidate_matrix_v6(v3_matrix, v6_extra, multiselect, pending, stop_offered):
    """[K, OPTION_FEATURE_DIM_V6]: v4 candidate rows ++ v5 columns ++ chain columns
    (zeros on the STOP row, which points at nothing)."""
    v4_rows = candidate_matrix(v3_matrix, multiselect, pending, stop_offered)
    rows = np.zeros((v4_rows.shape[0], OPTION_FEATURE_DIM_V6), dtype=np.float32)
    rows[:, :OPTION_FEATURE_DIM_V4] = v4_rows
    if pending:
        rows[:len(pending), OPTION_FEATURE_DIM_V4:] = v6_extra[pending]
    return rows


# cg SelectContext ids for damage-counter placement menus (Phantom Dive's effect for the
# dragapult deck; card-agnostic by design, like the rest of the rule vocabulary).
COUNTER_PLACEMENT_CONTEXTS = (13, 14)          # DAMAGE_COUNTER, DAMAGE_COUNTER_ANY

# ACTION RULES ARE OPT-IN (owner spec 2026-08-09): every rule is OFF unless named in the
# CG_ACTION_RULES env var (comma-separated, read at import so spawned workers inherit it)
# or enabled programmatically. Default behavior of every existing checkpoint, bundle and
# harness is bit-for-bit unchanged.
import os as _os

_ACTION_RULES = {rule.strip() for rule in
                 _os.environ.get("CG_ACTION_RULES", "").split(",") if rule.strip()}


def enable_action_rules(*rules):
    _ACTION_RULES.update(rules)


def action_rule_enabled(rule):
    return rule in _ACTION_RULES


def counter_option_mask(observation, select):
    """OWNER RULE (2026-08-09, explicit exception to no-code-decisions): damage counters
    may not be placed onto a target that is already at 0 HP while a live target is
    offered. The engine allows those placements but they are dominated by ARITHMETIC
    (the counters change nothing), not by strategy -- measured 2026-08-08: the policy
    wasted 1 in 7 placements this way with the alternative fully encoded in its input,
    and no learning signal (PPO win/loss, matchup fine-tune, aux heads) moved it.

    Returns the set of allowed ORIGINAL option indices, or None for "no restriction"
    (rule disabled, wrong context, no dead target offered, or every target dead -- an
    all-dead menu is the engine forcing a no-op and is left alone). Unknown-shaped
    options are always allowed. Applies at TRAINING and DEPLOY both -- never one without
    the other. OPT-IN: rule name "counter_cap" (see _ACTION_RULES above)."""
    if not action_rule_enabled("counter_cap"):
        return None
    if select.get("context") not in COUNTER_PLACEMENT_CONTEXTS:
        return None
    options = select.get("option") or []
    players = (observation.get("current") or {}).get("players") or []
    allowed, restricted = [], False
    for index, option in enumerate(options):
        hp = None
        try:
            player = players[option["playerIndex"]]
            area = option.get("area")
            if area == 4:                                        # active
                pokemon = (player.get("active") or [None])[option.get("index", 0)]
            elif area == 5:                                      # bench
                pokemon = (player.get("bench") or [])[option.get("index", 0)]
            else:
                pokemon = None
            if pokemon:
                hp = int(pokemon.get("hp", 0))
        except Exception:
            hp = None
        if hp is not None and hp <= 0:
            restricted = True
        else:
            allowed.append(index)
    if restricted and allowed:
        return set(allowed)
    return None


# Special Energies whose engine effect is NoEffectEnemyAttack (CardImpl.h): the holder
# ignores effects of attacks used by the OTHER player's Pokemon, which includes damage
# counters placed by an attack effect (State.h blocks the placement when the attacker's
# playerIndex differs from the target's). The engine still OFFERS the protected target
# in the placement menu -- picking it just wastes the counters.
MIST_ENERGY_CARD_ID = 11              # protects any holder
ROCK_FIGHTING_ENERGY_CARD_ID = 20     # protects only a {F} Fighting holder
FIGHTING_ENERGY_TYPE = 6              # cg EnergyType.FIGHTING (cards.json "energyType")


def _effect_shielded(pokemon):
    """True when the engine would prevent our attack's effect on this Pokemon via an
    attached shield energy. Rock Fighting is gated on the holder's PRINTED type (the
    engine checks the live type, which the observation does not carry; no card in the
    pool changes a Pokemon's own type, so printed == live in practice)."""
    for energy_card in pokemon.get("energyCards") or []:
        card_id = energy_card.get("id")
        if card_id == MIST_ENERGY_CARD_ID:
            return True
        if card_id == ROCK_FIGHTING_ENERGY_CARD_ID:
            holder = get_card(pokemon.get("id"))
            if holder and holder.get("energyType") == FIGHTING_ENERGY_TYPE:
                return True
    return False


def shielded_counter_mask(observation, select):
    """OWNER RULE (2026-08-12): damage counters from an attack effect (Phantom Dive's
    placement menu) may not be placed onto an opponent Pokemon holding Mist Energy or
    Rock Fighting Energy -- the shield prevents the effect and the counters vanish, so
    the pick is dominated by ARITHMETIC, like counter_cap's dead targets. Own-side
    targets are never masked (the shield only blocks the OPPONENT's attacks, and the
    decider at these menus is the attacker). Same return contract as
    counter_option_mask: allowed ORIGINAL option indices, or None for "no restriction"
    (rule disabled, wrong context, nothing shielded, or everything shielded -- an
    all-blocked menu is the engine forcing a no-op and is left alone). OPT-IN: rule
    name "counter_shield"."""
    if not action_rule_enabled("counter_shield"):
        return None
    # ctx 14 (DamageCounterAny, the attack-effect "as you like" menus) ONLY -- ctx 13
    # is ALSO Adrena-Brain's target menu (CreateCard.h maps DamageCounterRemoved to
    # SelectContext::DamageCounter), and State.h gates the Mist/Rock shields on
    # onAttackEffect(): ability-moved counters STICK on a shielded holder, so masking
    # them at ctx 13 would deny valid Adrena-Brain targets (latent over-mask found
    # 2026-08-15 while building damage_solver).
    if select.get("context") != 14:
        return None
    options = select.get("option") or []
    current = observation.get("current") or {}
    players = current.get("players") or []
    my_index = current.get("yourIndex")
    allowed, restricted = [], False
    for index, option in enumerate(options):
        shielded = False
        try:
            if option["playerIndex"] != my_index:
                player = players[option["playerIndex"]]
                area = option.get("area")
                if area == 4:                                        # active
                    pokemon = (player.get("active") or [None])[option.get("index", 0)]
                elif area == 5:                                      # bench
                    pokemon = (player.get("bench") or [])[option.get("index", 0)]
                else:
                    pokemon = None
                if pokemon:
                    shielded = _effect_shielded(pokemon)
        except Exception:
            shielded = False
        if shielded:
            restricted = True
        else:
            allowed.append(index)
    if restricted and allowed:
        return set(allowed)
    return None


# ---- require_play_supporter_from_meowth_ex (owner rule 2026-08-12) ------------------- #
# Meowth ex (card 1071) is benched for exactly one reason: Last-Ditch Catch fetches a
# Supporter from the deck when it is played from hand. The owner-identified bad play is
# benching it (a 2-prize vanilla body otherwise) without playing the fetched Supporter
# that same turn. ONE rule name, two enforcement surfaces:
#   * a mask (meowth_supporter_gate_mask, in OPTION_MASK_RULES): playing Meowth ex from
#     hand while this turn's Supporter is already spent is a CERTAIN violation -- the
#     fetched card can never be played "later in the turn" -- so the option is hidden
#     wherever the combined mask applies (raw picks, greedy resolves, branch points).
#     This is the only guard raw-policy play gets.
#   * a search line rule (MeowthSupporterLineRule, wired by main.py into
#     turn_search.line_rule): a line that benches Meowth and reaches its end-of-turn
#     leaf without playing THE FETCHED CARD (matched by serial) is scored -1 instead of
#     the value estimate, so no wasted-Meowth plan can win the root comparison and the
#     line is never started. Declining the fetch, or benching with no Supporter left in
#     the deck, leaves the obligation unmet -- same -1.

RULE_MEOWTH_SUPPORTER = "require_play_supporter_from_meowth_ex"
MEOWTH_EX_CARD_ID = 1071
_MEOWTH_EX_NAME = (get_card(MEOWTH_EX_CARD_ID) or {}).get("name")
SUPPORTER_CARD_TYPE = 3                # cg CardType.SUPPORTER
OPTION_TYPE_PLAY = 7                   # cg OptionType.PLAY (play a card from hand)
OPTION_TYPE_CARD = 3                   # cg OptionType.CARD
AREA_DECK, AREA_LOOKING = 1, 12        # cg AreaType (deck view / shared reveal area)
SELECT_CONTEXT_TO_HAND = 7             # cg SelectContext.TO_HAND


def _played_hand_card(observation, option):
    """The hand card a PLAY option would play, or None for any other option shape."""
    if option.get("type") != OPTION_TYPE_PLAY:
        return None
    current = observation.get("current") or {}
    players = current.get("players") or []
    try:
        hand = (players[current.get("yourIndex")] or {}).get("hand") or []
        position = option.get("index")
        if position is not None and 0 <= position < len(hand):
            return hand[position]
    except Exception:
        pass
    return None


def _our_hand_serials(observation):
    """Serials currently in OUR hand (empty set on any shape surprise)."""
    current = observation.get("current") or {}
    players = current.get("players") or []
    try:
        hand = (players[current.get("yourIndex")] or {}).get("hand") or []
    except Exception:
        return set()
    return {card.get("serial") for card in hand
            if card is not None and card.get("serial") is not None}


def _to_hand_menu_areas(select):
    """Areas of the CARD options on a TO_HAND select (empty set for any other
    select). The sunk-cost cleanup uses it to ask "is the pending fetch menu still
    the one on the table?" -- while it is, the obligation is live, not sunk."""
    if not select or select.get("context") != SELECT_CONTEXT_TO_HAND:
        return set()
    return {option.get("area") for option in (select.get("option") or [])
            if option.get("type") == OPTION_TYPE_CARD}


def meowth_supporter_gate_mask(observation, select):
    """The mask surfaces of require_play_supporter_from_meowth_ex. Same return
    contract as the other mask rules; OPT-IN via the rule name. Two surfaces:
    (a) no PLAYING Meowth ex from hand once this turn's Supporter is spent
        (supporterPlayed is the DECIDER's turn flag on the State itself, and this
        mask is only ever consulted for our own pending selects);
    (b) GOING-FIRST TURN 1 (owner rule 2026-08-15): no FETCHING Meowth ex out of
        the deck on turn 1. The engine blocks Supporter plays only on turn 1 --
        the going-first player's turn (GameProc.h: `state.turn <= 1 &&
        !canPlayFirstTurn`; the second player's first turn is turn 2,
        unrestricted) -- so a turn-1 fetched Meowth is provably unusable: benching
        it cannot cash the fetched Supporter, holding it violates the fetched-card
        rule. Masking the fetch pick makes 'save the Ultra Ball for next turn'
        fall out of the normal obligation scoring (observed misplay: turn-1 UB ->
        Meowth -> held). EXCEPTION (owner rule 2026-08-15 evening): opponent known
        dragapult -> the turn-1 fetch is allowed, and _fetch_hold_waived covers the
        hold, so 'UB out Meowth turn 1, bench it turn 2' becomes a legal plan in
        the mirror."""
    if not action_rule_enabled(RULE_MEOWTH_SUPPORTER):
        return None
    current = observation.get("current") or {}
    supporter_spent = bool(current.get("supporterPlayed"))
    first_turn = current.get("turn") == 1
    if not supporter_spent and not first_turn:
        return None
    options = select.get("option") or []
    to_hand = select.get("context") == SELECT_CONTEXT_TO_HAND
    allowed, restricted = [], False
    for index, option in enumerate(options):
        blocked = False
        if supporter_spent:
            card = _played_hand_card(observation, option)
            if card is not None and card.get("id") == MEOWTH_EX_CARD_ID:
                blocked = True
        if not blocked and first_turn and not _DRAGAPULT_OPPONENT and to_hand \
                and option.get("type") == OPTION_TYPE_CARD \
                and option.get("area") in (AREA_DECK, AREA_LOOKING):
            fetched, _pokemon = _entity_at(observation, option.get("area"),
                                           option.get("index"),
                                           option.get("playerIndex"))
            if fetched is not None and fetched.get("id") == MEOWTH_EX_CARD_ID:
                blocked = True
        if blocked:
            restricted = True
        else:
            allowed.append(index)
    if restricted and allowed:
        return set(allowed)
    return None


# REAL-side pointer to the Meowth-fetched supporter still owed this turn (its serial,
# or None). Written ONLY by MeowthSupporterLineRule.observe_real / reset_episode --
# main.py's pre-search pass runs observe_real before every decision, so the mask below
# always reads the current turn's truth.
_MEOWTH_PENDING_SERIAL = None


def _set_meowth_pending(serial):
    global _MEOWTH_PENDING_SERIAL
    _MEOWTH_PENDING_SERIAL = serial


def meowth_obligation_exit_mask(observation, select):
    """The commitment surface of require_play_supporter_from_meowth_ex (owner go
    2026-08-15 night, from the live Crispin-over-fetched-Lillie's defection): while
    the supporter Meowth's ability fetched is IN OUR HAND and this turn's Supporter
    is unspent, the turn cannot close around it and no rival supporter can steal
    the slot -- OTHER supporter plays, ATTACK and END are masked at MAIN menus.
    Items / attaches / evolves / abilities / retreat stay open (ordering is the
    model's call); the obligated play itself is never masked, so the menu cannot
    mask to empty. Because the pending state is REAL-side, this binds at raw picks
    and at every in-tree node of searches run while the obligation is live (the
    defection point); obligations first opened INSIDE a simulated line remain the
    line rule's -1 to enforce. Unmeetable states (slot spent / card gone) are
    pruned by observe_real before this mask ever fires, so it cannot lock a dead
    obligation."""
    if not action_rule_enabled(RULE_MEOWTH_SUPPORTER):
        return None
    if _MEOWTH_PENDING_SERIAL is None or select.get("context") != 0:
        return None
    current = observation.get("current") or {}
    if current.get("supporterPlayed"):
        return None                    # slot already spent: the prune handles it
    if _MEOWTH_PENDING_SERIAL not in _our_hand_serials(observation):
        return None                    # left our hand: the prune handles it
    allowed, restricted = [], False
    for index, option in enumerate(select.get("option") or []):
        kind = option.get("type")
        blocked = kind in (OPTION_TYPE_ATTACK, OPTION_TYPE_END)
        if not blocked and kind == OPTION_TYPE_PLAY:
            card = _played_hand_card(observation, option)
            if card is not None and card.get("serial") != _MEOWTH_PENDING_SERIAL:
                details = get_card(card.get("id"))
                blocked = bool(details
                               and details.get("cardType") == SUPPORTER_CARD_TYPE)
        if blocked:
            restricted = True
        else:
            allowed.append(index)
    if restricted and allowed:
        return set(allowed)
    return None


class MeowthSupporterLineRule:
    """The search surface of require_play_supporter_from_meowth_ex. turn_search treats
    this object as opaque: root_state() seeds each search root, option_labels() is
    computed once per node's select (None for the vast majority of selects -- cheap),
    update() folds the label of a taken edge into the line's state, violated() is asked
    at each end-of-turn leaf. State tuple: (benched, awaiting_fetch, fetched_serial,
    satisfied).

    Semantics: every Meowth bench opens a fresh obligation to play the Supporter its
    own ability fetches (serial match -- a duplicate copy already in hand does not
    count). Since only one Supporter can be played per turn, a second bench in the same
    line can never be followed through and kills the line, as does declining the fetch
    or whiffing on an empty deck.

    THE OBLIGATION SURVIVES ACROSS REAL DECISIONS: each real decision runs its own
    fresh search, so if the previous search already benched Meowth for real, a fresh
    root_state would forget the outstanding debt and happily end the turn. main.py
    therefore feeds every REAL decision through observe_real(), and root_state() seeds
    each new search with the real turn's accumulated state (reset when current["turn"]
    changes, and via reset_episode() on a new game)."""

    name = RULE_MEOWTH_SUPPORTER

    def __init__(self):
        self._real_turn = None
        self._real_state = self.initial_state()

    @staticmethod
    def initial_state():
        return (False, False, None, False)

    def reset_episode(self):
        self._real_turn = None
        self._real_state = self.initial_state()
        _set_meowth_pending(None)

    def observe_real(self, observation, select, chosen):
        """Fold one REAL decision (the answer main.py is about to send) into the real
        turn's state. Never raises: the rule must not be able to crash the agent."""
        try:
            current = observation.get("current") or {}
            turn = current.get("turn")
            if turn != self._real_turn:
                self._real_turn = turn
                self._real_state = self.initial_state()
            # SUNK-COST CLEANUP (owner go 2026-08-14): an obligation that can no
            # longer be met this turn must not keep scoring every future search line
            # -1 (measured poisoning: experiments/rule_penalty_repro/). Runs BEFORE
            # this decision's labels fold, so a just-created obligation is never
            # touched. In-search semantics are unchanged -- a line that opens a NEW
            # obligation still answers for it.
            benched, awaiting, fetched_serial, satisfied = self._real_state
            if benched and not satisfied:
                if awaiting:
                    if not (_to_hand_menu_areas(select)
                            & {AREA_DECK, AREA_LOOKING}) \
                            and (select or {}).get("context") == 0:
                        # Back at a MAIN menu with the supporter-fetch window passed
                        # unfulfilled (declined or whiffed for real): unmeetable,
                        # drop it. Non-main selects still belong to the resolution
                        # chain (2026-08-15 fix, same shape as the UB cost-menu bug).
                        self._real_state = self.initial_state()
                elif fetched_serial is not None and (
                        fetched_serial not in _our_hand_serials(observation)
                        or current.get("supporterPlayed")):
                    # The fetched supporter left our hand without being played, OR
                    # the turn's Supporter slot was spent on ANOTHER card (observed
                    # live 2026-08-15: Crispin played over the fetched Lillie's) --
                    # either way unmeetable, drop it so the rest of the turn's
                    # searches are not all-condemned. (The satisfied case never
                    # reaches here: the fold marks it before supporterPlayed shows.)
                    self._real_state = self.initial_state()
            if not chosen or select is None:
                return
            labels = self.option_labels(observation, select)
            if not labels:
                return
            state = self._real_state
            for picked in chosen:
                if 0 <= picked < len(labels) and labels[picked] is not None:
                    state = self.update(state, labels[picked])
            self._real_state = state
        except Exception:
            pass
        finally:
            try:
                benched, _awaiting, serial, satisfied = self._real_state
                _set_meowth_pending(serial if benched and not satisfied
                                    and serial is not None else None)
            except Exception:
                _set_meowth_pending(None)

    def root_state(self):
        return self._real_state

    @staticmethod
    def option_labels(observation, select):
        options = select.get("option") or []
        if not options:
            return None
        labels = None
        to_hand = select.get("context") == SELECT_CONTEXT_TO_HAND
        for index, option in enumerate(options):
            label = None
            card = _played_hand_card(observation, option)
            if card is not None:
                if card.get("id") == MEOWTH_EX_CARD_ID:
                    label = ("meowth_benched", None)
                else:
                    details = get_card(card.get("id"))
                    if details and details.get("cardType") == SUPPORTER_CARD_TYPE:
                        label = ("supporter_played", card.get("serial"))
            elif to_hand and option.get("type") == OPTION_TYPE_CARD \
                    and option.get("area") in (AREA_DECK, AREA_LOOKING):
                fetched, _pokemon = _entity_at(observation, option.get("area"),
                                               option.get("index"),
                                               option.get("playerIndex"))
                if fetched:
                    details = get_card(fetched.get("id"))
                    if details and details.get("cardType") == SUPPORTER_CARD_TYPE:
                        label = ("supporter_fetched", fetched.get("serial"))
            if label is not None:
                if labels is None:
                    labels = [None] * len(options)
                labels[index] = label
        return labels

    @staticmethod
    def update(state, label):
        benched, awaiting, fetched_serial, satisfied = state
        kind, serial = label
        if kind == "meowth_benched":
            return (True, True, None, False)
        if kind == "supporter_fetched" and awaiting:
            return (benched, False, serial, satisfied)
        if kind == "supporter_played" and serial is not None \
                and serial == fetched_serial:
            return (benched, awaiting, fetched_serial, True)
        return state

    @staticmethod
    def violated(state):
        benched, _awaiting, _fetched_serial, satisfied = state
        return benched and not satisfied


# ---- require_play_fetched_card (owner rule 2026-08-13) ------------------------------- #
# The owner-identified bad play: the model fires its search/recovery Items (Poke Pad,
# Ultra Ball, Night Stretcher) just because it holds them, fetching a card it then sits
# on. The rule: a card fetched by one of these Items must LEAVE OUR HAND INTO PLAY
# (bench / evolve / attach) before the end of the turn, or the Item should never have
# been played. Same enforcement shape as the Meowth line rule: a violating search line
# scores -1 at its end-of-turn leaf, so the Item-play edge loses the root comparison and
# the whole line is never started. Whiffing the fetch (declining, or empty search) is a
# violation too, tracked as its own state field so a future whiff-specific rule can hook
# the same machinery (owner 2026-08-13).
#
# THE DRAGAPULT EXCEPTION (owner 2026-08-13): dragapult decks run Budew, whose attack
# locks Items -- against them you WANT to fire search Items early and stock evolutions
# in hand while you still can. When the opponent is known to be dragapult (a revealed
# Budew, or the archetype router's posterior crossing its own tau_enter -- REAL
# knowledge only, a predicted Budew in a determinized search world must never count),
# holding a fetched card is allowed iff it is an evolution whose DIRECT pre-evolution is
# in our play, capped by count: copies of that evolution in hand AFTER the fetch may not
# exceed copies of its pre-evolution in play (1 Dreepy in play + 1 Drakloak in hand ->
# fetching a second Drakloak is not covered; 3 Dreepy cover 3 Drakloak). Both counts are
# read LIVE from the observation at the fetch menu, so a Dreepy already evolved earlier
# in the line no longer counts as an evolver, and by-NAME matching keeps the check
# print-blind. Basics and Energy are never covered by the exception.
#
# THE MASK SURFACE (owner go 2026-08-15, same ONE-name-TWO-surfaces shape as the Meowth
# rule): when EVERY card an Item could possibly fetch is provably unplayable this turn
# (and not covered by the dragapult exception), no line through that Item play can ever
# satisfy the rule, so dead_fetch_item_mask removes the play from the menu outright.
# Diagnosed need: episode 93157396 (experiments/rule_penalty_repro/) -- bench full,
# only hidden Pokemon two unplayable Basics, every in-search Ultra Ball line correctly
# scored -1, and the root's max-VISITS pick played it anyway off the 0.69 policy prior
# because the value field sat at -0.9 where the -1 penalty has no contrast. A mask is
# immune to that failure mode and, unlike the line rule, also protects raw-policy picks
# (the no-search bank tier). The mask is strictly conservative: it needs the decklist
# registered (note_our_decklist), the hidden-zone accounting to close EXACTLY against
# deckCount + prize count, and every candidate to fail the playability checks -- any
# uncertainty stands down, so it can only ever under-mask.

RULE_FETCHED_CARD = "require_play_fetched_card"
AREA_HAND, AREA_DISCARD = 2, 3         # cg AreaType
OPTION_TYPE_ATTACH = 8                 # cg OptionType.ATTACH (attach a card from hand)
OPTION_TYPE_EVOLVE = 9                 # cg OptionType.EVOLVE (evolve with a hand card)
POKEMON_CARD_TYPE = 0                  # cg CardType.POKEMON
BASIC_ENERGY_CARD_TYPE = 5             # cg CardType.BASIC_ENERGY
# Card-select contexts that move a hand card into play/attached (satisfy an obligation).
HAND_TO_PLAY_CONTEXTS = (4, 5, 6, 22)  # TO_ACTIVE, TO_BENCH, TO_FIELD, ATTACH_TO
# The tracked Items, mapped to where their fetch menu picks from. Deck searchers present
# their targets in the DECK view or the shared LOOKING area; Night Stretcher picks from
# our discard pile. The fetch menu is attributed to a pending Item only when the area
# matches, and only for card types the Item can actually fetch (Pokemon for the deck
# searchers, Pokemon or Basic Energy for Night Stretcher) -- so Meowth ex's Supporter
# fetch can never be misattributed even right after a whiff.
POKE_PAD_CARD_ID, ULTRA_BALL_CARD_ID, NIGHT_STRETCHER_CARD_ID = 1152, 1121, 1097
SEARCH_ITEM_FETCH_AREAS = {POKE_PAD_CARD_ID: AREA_DECK,
                           ULTRA_BALL_CARD_ID: AREA_DECK,
                           NIGHT_STRETCHER_CARD_ID: AREA_DISCARD}
BUDEW_CARD_NAME = "Budew"

# Opponent context for the dragapult exception, module-level so every surface (the line
# rule's labels in-search, a router main.py, future rules) reads the same flag. Sticky
# once true -- a deck revealed as dragapult does not stop being one -- and reset per
# episode via the line rule's reset_episode(). Fed ONLY from real knowledge: the Budew
# scan runs on real observations (observe_real), the posterior comes from the router.
_DRAGAPULT_OPPONENT = False


def note_dragapult_opponent():
    global _DRAGAPULT_OPPONENT
    _DRAGAPULT_OPPONENT = True


def dragapult_opponent_known():
    return _DRAGAPULT_OPPONENT


def reset_opponent_context():
    global _DRAGAPULT_OPPONENT, _SAFEGUARD_OPPONENT, _SOLVER_MATCHUP
    _DRAGAPULT_OPPONENT = False
    _SAFEGUARD_OPPONENT = False
    _SOLVER_MATCHUP = None


def _opponent_shows_budew(observation):
    """True when the OPPONENT's visible zones hold a Budew. Our own Budew (dragapult_v1
    runs one) must never trigger the exception, so only their side is scanned."""
    current = observation.get("current") or {}
    players = current.get("players") or []
    my_index = current.get("yourIndex")
    if my_index is None or len(players) < 2:
        return False
    opponent = players[1 - my_index] or {}
    for zone in ("active", "bench", "discard"):
        for entry in opponent.get(zone) or []:
            if entry is None:
                continue
            details = get_card(entry.get("id"))
            if details and details.get("name") == BUDEW_CARD_NAME:
                return True
    return False


def _hand_card_at(observation, position):
    """Our hand card at `position`, or None."""
    current = observation.get("current") or {}
    players = current.get("players") or []
    try:
        hand = (players[current.get("yourIndex")] or {}).get("hand") or []
        if position is not None and 0 <= position < len(hand):
            return hand[position]
    except Exception:
        pass
    return None


def _fetch_hold_waived(observation, details):
    """True when HOLDING this fetched card is covered by a dragapult exception (see
    the block comment above). Two clauses, both requiring the opponent known
    dragapult: (1) evolutions -- direct pre-evolution in our play and the in-hand
    copy count stays within the evolver count after the fetch (at the fetch menu the
    card is still in the deck/discard view, so hand counts are pre-fetch and the cap
    is `in_hand + 1 <= evolvers`); (2) MEOWTH EX ON OUR GOING-FIRST TURN 1 (owner
    rule 2026-08-15 evening): in the mirror the agent may Ultra Ball out Meowth ex
    turn 1 and hold it -- benching waits for turn 2 where the fetched Supporter can
    actually be played. This waiver also un-deadens the Item for
    dead_fetch_item_mask and lifts meowth_supporter_gate_mask's turn-1 fetch block
    (both consult it / the same flag)."""
    if not _DRAGAPULT_OPPONENT:
        return False
    if details.get("name") == _MEOWTH_EX_NAME \
            and (observation.get("current") or {}).get("turn") == 1:
        return True
    evolves_from = details.get("evolvesFrom")
    if not evolves_from:
        return False
    current = observation.get("current") or {}
    players = current.get("players") or []
    try:
        me = players[current.get("yourIndex")] or {}
    except Exception:
        return False
    evolvers = 0
    for zone in ("active", "bench"):
        for pokemon in me.get(zone) or []:
            if pokemon is None:
                continue
            holder = get_card(pokemon.get("id"))
            if holder and holder.get("name") == evolves_from:
                evolvers += 1
    if not evolvers:
        return False
    fetched_name = details.get("name")
    in_hand = 0
    for card in me.get("hand") or []:
        held = get_card(card.get("id"))
        if held and held.get("name") == fetched_name:
            in_hand += 1
    return in_hand + 1 <= evolvers


# ---- pokepad_supporter_gate (owner rule 2026-08-16, the SYLVEON deck) ---------------- #
# TARGET CORRECTED 2026-08-16 night (owner, from a live replay of a turn-1 play the
# gate missed): the rule was always MEANT for POKEGEAR 3.0 (1122, "look at the top 7,
# take a Supporter") -- the original spec said "pokepad" and was implemented against
# the card literally named Poke Pad (1152, a Pokemon search the rule has no business
# gating). The envelope rule NAME stays pokepad_supporter_gate (already stamped in
# the sylveon checkpoints); only the targeted card id changed.
# Pokegear may not be played on a turn that cannot follow it with a Supporter: once
# this turn's Supporter is spent, or on our going-first turn 1 (the engine blocks
# Supporters only there -- GameProc.h `state.turn <= 1`). EXCEPTION (owner): opponent
# known dragapult, or a revealed Budew anywhere visible on their side -- the item lock
# is coming, so fire items while they are still playable. OPT-IN by name; the sylveon
# router is the intended consumer (its models trained under counter_cap only -- this
# is a deploy-only mask like the dragapult bundle's later rules).
RULE_POKEPAD_GATE = "pokepad_supporter_gate"
POKEGEAR_CARD_ID = 1122


def pokepad_supporter_gate_mask(observation, select):
    """Mask surface of pokepad_supporter_gate (gates POKEGEAR 3.0 -- see the block
    comment). Same return contract as the other mask rules: allowed ORIGINAL
    option indices, or None for no restriction."""
    if not action_rule_enabled(RULE_POKEPAD_GATE):
        return None
    if select.get("context") != 0:
        return None
    current = observation.get("current") or {}
    supporter_spent = bool(current.get("supporterPlayed"))
    first_turn = current.get("turn") == 1
    if not (supporter_spent or first_turn):
        return None
    if _DRAGAPULT_OPPONENT or _opponent_shows_budew(observation):
        return None
    allowed, restricted = [], False
    for index, option in enumerate(select.get("option") or []):
        card = _played_hand_card(observation, option)
        if card is not None and card.get("id") == POKEGEAR_CARD_ID:
            restricted = True
        else:
            allowed.append(index)
    if restricted and allowed:
        return set(allowed)
    return None


# Our submitted 60-card decklist (id -> count), registered once at load by main.py's
# deploy templates. The dead-fetch mask needs it to bound what a deck search could
# offer; unregistered (e.g. training, where seat/deck pairing is ambiguous and a wrong
# list could over-mask the opponent seat's rollout menus) the mask stands down.
_OUR_DECKLIST = None


def note_our_decklist(card_counts):
    global _OUR_DECKLIST
    _OUR_DECKLIST = dict(card_counts) if card_counts else None


def _our_hidden_pool(observation):
    """Multiset (id -> count) of OUR hidden cards -- deck plus remaining face-down
    prizes -- computed as the registered decklist minus every visible card of ours.
    Returns None (mask stands down) when the decklist is unregistered or the count
    does not close EXACTLY against deckCount + prize length: any drift means a zone
    was missed and trusting it could over-mask."""
    if not _OUR_DECKLIST:
        return None
    current = observation.get("current") or {}
    players = current.get("players") or []
    my_index = current.get("yourIndex")
    try:
        me = players[my_index]
    except Exception:
        return None
    if not me:
        return None
    visible = {}

    def count_card(card_id):
        if card_id is not None:
            visible[card_id] = visible.get(card_id, 0) + 1

    def count_pokemon(entry):
        if not isinstance(entry, dict):
            return
        count_card(entry.get("id"))
        for attached in (entry.get("energyCards") or []) + (entry.get("tools") or []):
            if isinstance(attached, dict):
                count_card(attached.get("id"))
        for under in entry.get("preEvolution") or []:
            if isinstance(under, dict):
                count_card(under.get("id"))

    for entry in me.get("active") or []:
        count_pokemon(entry)
    for entry in me.get("bench") or []:
        count_pokemon(entry)
    for card in (me.get("hand") or []) + (me.get("discard") or []):
        if isinstance(card, dict):
            count_card(card.get("id"))
    for entry in current.get("stadium") or []:
        if isinstance(entry, dict) and entry.get("playerIndex") == my_index:
            count_card(entry.get("id"))
    hidden = {}
    for card_id, count in _OUR_DECKLIST.items():
        remaining = count - visible.get(card_id, 0)
        if remaining < 0:
            return None
        if remaining:
            hidden[card_id] = remaining
    if sum(hidden.values()) != (me.get("deckCount") or 0) + len(me.get("prize") or []):
        return None
    return hidden


def _fetchable_card_playable(observation, details):
    """Could this card POSSIBLY enter play this turn from our hand? Unknown shapes
    answer True (the safe direction: True means the Item is not masked). Basic ->
    bench space (benchMax from the observation -- stadiums can raise it). Evolution ->
    a name-matching holder of ours in play with appearThisTurn False (the engine's own
    evolve-legality flag; first-turn rules make this answer True where the engine
    still forbids the evolve, which only under-masks). Basic Energy (Night Stretcher)
    -> always True: already-attached-this-turn is not reliably observable."""
    card_type = details.get("cardType")
    if card_type != POKEMON_CARD_TYPE:
        return True
    current = observation.get("current") or {}
    players = current.get("players") or []
    try:
        me = players[current.get("yourIndex")] or {}
    except Exception:
        return True
    evolves_from = details.get("evolvesFrom")
    if not evolves_from:
        bench_max = me.get("benchMax")
        if not isinstance(bench_max, int):
            return True
        occupied = len([entry for entry in me.get("bench") or [] if entry])
        return occupied < bench_max
    for zone in ("active", "bench"):
        for pokemon in me.get(zone) or []:
            if not isinstance(pokemon, dict) or pokemon.get("appearThisTurn"):
                continue
            holder = get_card(pokemon.get("id"))
            if holder and holder.get("name") == evolves_from:
                return True
    return False


def _discard_fetch_candidates(observation):
    """Card details Night Stretcher could take from OUR discard pile (exact: the
    discard is fully visible)."""
    current = observation.get("current") or {}
    players = current.get("players") or []
    try:
        me = players[current.get("yourIndex")] or {}
    except Exception:
        return []
    candidates = []
    for card in me.get("discard") or []:
        details = get_card(card.get("id")) if isinstance(card, dict) else None
        if details and details.get("cardType") in (POKEMON_CARD_TYPE,
                                                   BASIC_ENERGY_CARD_TYPE):
            candidates.append(details)
    return candidates


def dead_fetch_item_mask(observation, select):
    """The mask surface of require_play_fetched_card (see block comment): remove a
    Poke Pad / Ultra Ball / Night Stretcher play whose every possible fetch target is
    provably unplayable this turn and not covered by the dragapult holding exception.
    Same return contract as the other mask rules; conservative throughout -- any
    uncertainty leaves the option on the menu."""
    if not action_rule_enabled(RULE_FETCHED_CARD):
        return None
    options = select.get("option") or []
    item_plays = {}
    for index, option in enumerate(options):
        card = _played_hand_card(observation, option)
        if card is not None and card.get("id") in SEARCH_ITEM_FETCH_AREAS:
            item_plays[index] = card.get("id")
    if not item_plays:
        return None
    try:
        hidden_pool = None
        dead = set()
        for index, item_id in item_plays.items():
            if SEARCH_ITEM_FETCH_AREAS[item_id] == AREA_DISCARD:
                candidates = _discard_fetch_candidates(observation)
            else:
                if hidden_pool is None:
                    hidden_pool = _our_hidden_pool(observation)
                if hidden_pool is None:
                    continue                     # accounting unavailable: stand down
                candidates = [details for details in
                              (get_card(card_id) for card_id in hidden_pool)
                              if details
                              and details.get("cardType") == POKEMON_CARD_TYPE]
                if item_id == POKE_PAD_CARD_ID:
                    # Poke Pad fetches only Pokemon WITHOUT a Rule Box. Only exclude
                    # certain rule-box flags; missing one merely under-masks.
                    candidates = [details for details in candidates
                                  if not (details.get("ex")
                                          or details.get("megaEx"))]
            if not any(_fetchable_card_playable(observation, details)
                       or _fetch_hold_waived(observation, details)
                       for details in candidates):
                dead.add(index)
        if not dead:
            return None
        allowed = {index for index in range(len(options)) if index not in dead}
        return allowed or None
    except Exception:
        return None


class FetchedCardLineRule:
    """The line rule of require_play_fetched_card. Same opaque interface as
    MeowthSupporterLineRule (root_state / option_labels / update / violated /
    observe_real / reset_episode). State tuple: (awaiting_item, due_serials, whiffed).

    awaiting_item: the card id of a played search Item whose fetch menu has not resolved
    yet (None otherwise). A leaf reached with it still armed means the fetch came and
    went empty -- the whiff violation. due_serials: frozenset of fetched serials that
    still must leave our hand into play. whiffed: a previous Item's fetch resolved
    empty earlier in the turn (kept as its own field, not folded into due_serials, so a
    future whiff rule can distinguish the two failure shapes)."""

    name = RULE_FETCHED_CARD

    def __init__(self):
        self._real_turn = None
        self._real_state = self.initial_state()

    @staticmethod
    def initial_state():
        return (None, frozenset(), False)

    def reset_episode(self):
        self._real_turn = None
        self._real_state = self.initial_state()
        reset_opponent_context()

    def observe_real(self, observation, select, chosen):
        """Fold one REAL decision into the turn state, and refresh the real-knowledge
        opponent context (the Budew scan) -- main.py also calls this with chosen=None
        BEFORE each search so the context lands ahead of the root. Never raises."""
        try:
            if not _DRAGAPULT_OPPONENT and _opponent_shows_budew(observation):
                note_dragapult_opponent()
            current = observation.get("current") or {}
            turn = current.get("turn")
            if turn != self._real_turn:
                self._real_turn = turn
                self._real_state = self.initial_state()
            # SUNK-COST CLEANUP (owner go 2026-08-14): unmeetable liabilities are
            # dropped from the REAL state so they cannot poison every subsequent
            # search this turn (measured: a real whiff/decline turned ALL leaves -1,
            # experiments/rule_penalty_repro/). Runs BEFORE this decision's labels
            # fold. In-search lines still answer for obligations THEY open.
            awaiting, due, whiffed = self._real_state
            pruned = False
            if awaiting is not None:
                expected_area = SEARCH_ITEM_FETCH_AREAS.get(awaiting)
                live_areas = ({AREA_DISCARD} if expected_area == AREA_DISCARD
                              else {AREA_DECK, AREA_LOOKING})
                # The window has passed only when we are BACK AT A MAIN MENU without
                # the fetch having resolved (fix 2026-08-15, overnight-flagged bug:
                # Ultra Ball's discard-COST menu arrives between the play and the
                # fetch menu, and treating it as "window passed" killed the real
                # obligation mid-chain). Any non-main select still belongs to the
                # item's own resolution chain, so the obligation stays live there.
                if not (_to_hand_menu_areas(select) & live_areas) \
                        and (select or {}).get("context") == 0:
                    awaiting, pruned = None, True   # fetch window passed: sunk
            if whiffed:
                whiffed, pruned = False, True       # a real whiff is sunk
            if due:
                in_hand = _our_hand_serials(observation)
                still_due = frozenset(serial for serial in due
                                      if serial in in_hand)
                if still_due != due:
                    due, pruned = still_due, True   # left our hand for real: sunk
            if pruned:
                self._real_state = (awaiting, due, whiffed)
            if not chosen or select is None:
                return
            labels = self.option_labels(observation, select)
            if not labels:
                return
            state = self._real_state
            for picked in chosen:
                if 0 <= picked < len(labels) and labels[picked] is not None:
                    state = self.update(state, labels[picked])
            self._real_state = state
        except Exception:
            pass

    def root_state(self):
        return self._real_state

    @staticmethod
    def option_labels(observation, select):
        options = select.get("option") or []
        if not options:
            return None
        context = select.get("context")
        labels = None
        for index, option in enumerate(options):
            label = None
            option_type = option.get("type")
            if option_type == OPTION_TYPE_PLAY:
                card = _played_hand_card(observation, option)
                if card is not None:
                    if card.get("id") in SEARCH_ITEM_FETCH_AREAS:
                        label = ("search_item_played", card.get("id"))
                    elif card.get("serial") is not None:
                        label = ("hand_card_used", card.get("serial"))
            elif option_type in (OPTION_TYPE_ATTACH, OPTION_TYPE_EVOLVE) \
                    and option.get("area") == AREA_HAND:
                card = _hand_card_at(observation, option.get("index"))
                if card is not None and card.get("serial") is not None:
                    label = ("hand_card_used", card.get("serial"))
            elif option_type == OPTION_TYPE_CARD:
                area = option.get("area")
                if context == SELECT_CONTEXT_TO_HAND \
                        and area in (AREA_DECK, AREA_LOOKING, AREA_DISCARD):
                    fetched, _pokemon = _entity_at(observation, area,
                                                   option.get("index"),
                                                   option.get("playerIndex"))
                    details = get_card(fetched.get("id")) if fetched else None
                    fetchable = (POKEMON_CARD_TYPE, BASIC_ENERGY_CARD_TYPE) \
                        if area == AREA_DISCARD else (POKEMON_CARD_TYPE,)
                    if details and details.get("cardType") in fetchable \
                            and fetched.get("serial") is not None:
                        label = ("card_fetched",
                                 (fetched.get("serial"), area,
                                  _fetch_hold_waived(observation, details)))
                elif context in HAND_TO_PLAY_CONTEXTS and area == AREA_HAND:
                    card, _pokemon = _entity_at(observation, area,
                                                option.get("index"),
                                                option.get("playerIndex"))
                    if card is not None and card.get("serial") is not None:
                        label = ("hand_card_used", card.get("serial"))
            if label is not None:
                if labels is None:
                    labels = [None] * len(options)
                labels[index] = label
        return labels

    @staticmethod
    def update(state, label):
        awaiting, due, whiffed = state
        kind, payload = label
        if kind == "search_item_played":
            # A still-armed fetch here means the previous Item's search resolved empty.
            return (payload, due, whiffed or awaiting is not None)
        if kind == "card_fetched" and awaiting is not None:
            serial, area, waived = payload
            if SEARCH_ITEM_FETCH_AREAS.get(awaiting) == AREA_DISCARD:
                matches = area == AREA_DISCARD
            else:
                matches = area in (AREA_DECK, AREA_LOOKING)
            if not matches:
                return state
            if waived:
                return (None, due, whiffed)
            return (None, due | {serial}, whiffed)
        if kind == "hand_card_used" and payload in due:
            return (awaiting, due - {payload}, whiffed)
        return state

    @staticmethod
    def violated(state):
        awaiting, due, whiffed = state
        return bool(due) or whiffed or awaiting is not None


# ---- require_evolve_before_shuffle_draw (owner rule 2026-08-13) ---------------------- #
# The owner-identified bad play: shuffle-draw effects (Lillie's Determination, Judge,
# Unfair Stamp -- all "shuffle your hand into your deck, then draw") played BEFORE using
# an evolution sitting in hand, shuffling the evolution away (observed: Judge played
# with Dragapult ex in hand and a legal evolve on the same menu). Root cause in-search:
# each simulated shuffle line sees ONE random redraw, so lucky-redraw contingencies
# ("Judge, draw the Dragapult back, evolve later") value the shuffle-first plan as if
# the gamble were free. The rule kills exactly that ordering: a line that plays a
# shuffle-draw while an evolve was offered ON THAT SAME MENU and then evolves one of
# those SAME evolutions later in the line is scored -1 at its end-of-turn leaf. Evolve-
# then-shuffle is untouched, and so is shuffling the evolution away WITHOUT evolving
# (the deliberate hold -- e.g. keeping Drakloak's draw ability -- stays the model's
# strategic call, owner 2026-08-13). Matching is by evolution NAME (the redrawn copy is
# a different serial; print-blind like everything else). Evolve legality cannot newly
# appear mid-turn (benched-this-turn / evolved-this-turn Pokemon stay ineligible), so
# the same-menu snapshot has no false-negative window.
#
# IN-LINE ONLY, deliberately: root_state() is always fresh and observe_real records
# nothing. After a REAL shuffle the redraw has actually happened -- if the evolution
# came back, evolving it is now the best play, and carried-over state would refuse it
# and compound the mistake.

RULE_EVOLVE_BEFORE_SHUFFLE = "require_evolve_before_shuffle_draw"
# Lillie's Determination, Judge, Unfair Stamp -- every hand-shuffle-then-draw effect in
# the dragapult_v1 pool.
SHUFFLE_DRAW_CARD_IDS = frozenset((1227, 1213, 1080))


class EvolveBeforeShuffleLineRule:
    """The line rule of require_evolve_before_shuffle_draw. Same opaque interface as the
    other line rules. State tuple: (shuffled_away_names, violated) -- the union of
    evolution names that were evolvable on some shuffle-draw's menu earlier in this
    line, and whether one of them was evolved after the fact."""

    name = RULE_EVOLVE_BEFORE_SHUFFLE

    @staticmethod
    def initial_state():
        return (frozenset(), False)

    def reset_episode(self):
        pass

    def observe_real(self, observation, select, chosen):
        pass                            # in-line only -- see the block comment

    def root_state(self):
        return self.initial_state()

    @staticmethod
    def option_labels(observation, select):
        options = select.get("option") or []
        if not options:
            return None
        # One pass for the menu's evolvable-from-hand names; they both label the EVOLVE
        # options and are the snapshot a co-offered shuffle-draw play freezes.
        evolve_names = {}
        for index, option in enumerate(options):
            if option.get("type") == OPTION_TYPE_EVOLVE \
                    and option.get("area") == AREA_HAND:
                card = _hand_card_at(observation, option.get("index"))
                details = get_card(card.get("id")) if card else None
                if details and details.get("name"):
                    evolve_names[index] = details["name"]
        offered = frozenset(evolve_names.values())
        labels = None
        for index, option in enumerate(options):
            label = None
            if index in evolve_names:
                label = ("evolved", evolve_names[index])
            elif offered and option.get("type") == OPTION_TYPE_PLAY:
                card = _played_hand_card(observation, option)
                if card is not None and card.get("id") in SHUFFLE_DRAW_CARD_IDS:
                    label = ("shuffle_draw_played", offered)
            if label is not None:
                if labels is None:
                    labels = [None] * len(options)
                labels[index] = label
        return labels

    @staticmethod
    def update(state, label):
        shuffled_away, violated = state
        kind, payload = label
        if kind == "shuffle_draw_played":
            return (shuffled_away | payload, violated)
        if kind == "evolved" and payload in shuffled_away:
            return (shuffled_away, True)
        return state

    @staticmethod
    def violated(state):
        return state[1]


# ---- force_evolve_before_shuffle_draw (owner rule 2026-08-14) ------------------------ #
# Replaces require_evolve_before_shuffle_draw in the dragapult bundles with an owner
# play-script, enforced at SUBMISSION time (owner-specified mechanism): menus are never
# masked -- when the model submits a shuffle-draw play (Lillie's / Judge / Unfair Stamp)
# while a forcible dragapult-line evolution sits on the same menu, main.py sends the
# EVOLUTION instead (or the evolving Drakloak's unused Recon Directive first, so the
# forced evolve never wastes the ability) and the model re-decides from the post-evolve
# state; repeated substitution walks multi-evolve turns one step at a time. Owner
# exceptions protect the free Petty Grudge (10 damage) finisher lines and the Safeguard
# matchups:
#   * Dreepy -> Drakloak is forced for BENCHED Dreepy only -- a Dreepy in the ACTIVE
#     spot is never forced (owner 2026-08-14: the model's call). It is also NOT forced
#     onto a Dreepy lacking Fire+Psychic energy while the opponent's Active is at 10 HP
#     (the bare Dreepy's Petty Grudge IS the KO), and not forced at all while a
#     Munkidori KO line is live: >= 1 dark-attached Munkidori able to move 3 counters
#     with the Active at <= 40 HP, or 2 able to move 3 each with the Active at <= 70
#     (Adrena-Brain 30/60 + Petty Grudge 10 = 40/70).
#   * Drakloak -> Dragapult ex is forced only while NO full-health Dragapult ex is in
#     play.
#   * Against a recognized Crustle or Sylveon deck (Safeguard walls: the ex line cannot
#     damage them, non-ex Drakloak is the attacker) NOTHING is forced. Real knowledge
#     only, sticky per episode: any revealed opponent card named Crustle/Sylveon, or
#     the router's crustle posterior (sylveon has no pool archetype -- reveals only).

RULE_FORCE_EVOLVE = "force_evolve_before_shuffle_draw"
FIRE_ENERGY_TYPE, PSYCHIC_ENERGY_TYPE, DARK_ENERGY_TYPE = 2, 5, 7
DREEPY_NAME, DRAKLOAK_NAME, DRAGAPULT_NAME = "Dreepy", "Drakloak", "Dragapult ex"
MUNKIDORI_NAME = "Munkidori"
SAFEGUARD_DECK_NAMES = ("Crustle", "Sylveon")
OPTION_TYPE_ABILITY = 10               # cg OptionType.ABILITY

_SAFEGUARD_OPPONENT = False


def note_safeguard_opponent():
    global _SAFEGUARD_OPPONENT
    _SAFEGUARD_OPPONENT = True


def safeguard_opponent_known():
    return _SAFEGUARD_OPPONENT


def _opponent_shows_safeguard_deck(observation):
    """True when the OPPONENT's visible zones hold any card named Crustle/Sylveon."""
    current = observation.get("current") or {}
    players = current.get("players") or []
    my_index = current.get("yourIndex")
    if my_index is None or len(players) < 2:
        return False
    opponent = players[1 - my_index] or {}
    for zone in ("active", "bench", "discard"):
        for entry in opponent.get(zone) or []:
            if entry is None:
                continue
            details = get_card(entry.get("id"))
            name = details.get("name") if details else ""
            if name and any(marker in name for marker in SAFEGUARD_DECK_NAMES):
                return True
    return False


def _our_side(observation):
    current = observation.get("current") or {}
    players = current.get("players") or []
    try:
        return players[current.get("yourIndex")] or {}
    except Exception:
        return {}


def _attached_energy_types(pokemon):
    types = set()
    for card in pokemon.get("energyCards") or []:
        details = get_card(card.get("id"))
        if details is not None:
            types.add(details.get("energyType"))
    return types


def _damage_counters_on(pokemon):
    max_hp = pokemon.get("maxHp") or 0
    hp = pokemon.get("hp") or 0
    return max(0, (max_hp - hp) // 10)


def _munkidori_takes(observation):
    """Per-use counter moves available to our dark-attached Munkidori, greedy best
    source first (each Adrena-Brain use moves up to 3 counters from ONE of our
    Pokemon, any of them including a Munkidori itself)."""
    me = _our_side(observation)
    in_play = [pokemon for zone in ("active", "bench")
               for pokemon in (me.get(zone) or []) if pokemon]
    uses = 0
    for pokemon in in_play:
        details = get_card(pokemon.get("id"))
        if details and details.get("name") == MUNKIDORI_NAME \
                and DARK_ENERGY_TYPE in _attached_energy_types(pokemon):
            uses += 1
    pools = [_damage_counters_on(pokemon) for pokemon in in_play]
    takes = []
    for _ in range(uses):
        pools.sort(reverse=True)
        take = min(3, pools[0]) if pools else 0
        takes.append(take)
        if pools:
            pools[0] -= take
    return takes


def _opponent_active_hp(observation):
    current = observation.get("current") or {}
    players = current.get("players") or []
    my_index = current.get("yourIndex")
    try:
        active = (players[1 - my_index] or {}).get("active") or []
        if active and active[0]:
            return active[0].get("hp")
    except Exception:
        pass
    return None


def shuffle_play_indices(observation, select):
    """Option indices that PLAY a shuffle-draw card from hand (the interception
    trigger of force_evolve_before_shuffle_draw)."""
    indices = set()
    for index, option in enumerate(select.get("option") or []):
        if option.get("type") == OPTION_TYPE_PLAY:
            card = _played_hand_card(observation, option)
            if card is not None and card.get("id") in SHUFFLE_DRAW_CARD_IDS:
                indices.add(index)
    return indices


def forced_evolution_options(observation, select):
    """Option indices main.py may SUBSTITUTE for a submitted shuffle-draw play under
    force_evolve_before_shuffle_draw: forcible EVOLVE options -- for a Drakloak ->
    Dragapult ex whose target still has its Recon Directive available on this menu,
    that ABILITY option rides in its place (use it first; the evolve is forced on a
    later interception). Empty when the rule is off, nothing is forcible, or an
    owner exception holds."""
    if not action_rule_enabled(RULE_FORCE_EVOLVE):
        return []
    if _SAFEGUARD_OPPONENT:
        return []
    options = select.get("option") or []
    if not options:
        return []
    me = _our_side(observation)
    zones = {4: me.get("active") or [], 5: me.get("bench") or []}
    active_hp = _opponent_active_hp(observation)
    takes = _munkidori_takes(observation)
    munkidori_ko_line = active_hp is not None and (
        (len(takes) >= 1 and takes[0] >= 3 and active_hp <= 40)
        or (len(takes) >= 2 and takes[0] >= 3 and takes[1] >= 3 and active_hp <= 70))
    healthy_dragapult = False
    for zone in zones.values():
        for pokemon in zone:
            if pokemon is None:
                continue
            details = get_card(pokemon.get("id"))
            if details and details.get("name") == DRAGAPULT_NAME \
                    and (pokemon.get("hp") or 0) >= (pokemon.get("maxHp") or 0):
                healthy_dragapult = True
    candidates = []
    for index, option in enumerate(options):
        if option.get("type") != OPTION_TYPE_EVOLVE \
                or option.get("area") != AREA_HAND:
            continue
        hand_card = _hand_card_at(observation, option.get("index"))
        details = get_card(hand_card.get("id")) if hand_card else None
        target_zone = zones.get(option.get("inPlayArea"))
        target_index = option.get("inPlayIndex")
        target = None
        if target_zone is not None and target_index is not None \
                and 0 <= target_index < len(target_zone):
            target = target_zone[target_index]
        if details is None or target is None:
            continue
        target_details = get_card(target.get("id"))
        target_name = target_details.get("name") if target_details else None
        if details.get("name") == DRAKLOAK_NAME and target_name == DREEPY_NAME:
            if option.get("inPlayArea") == _AREA_ACTIVE:
                continue               # ACTIVE Dreepy: never forced (owner 2026-08-14)
            if munkidori_ko_line:
                continue
            if active_hp is not None and active_hp <= 10 \
                    and not {FIRE_ENERGY_TYPE, PSYCHIC_ENERGY_TYPE} \
                    <= _attached_energy_types(target):
                continue               # the bare Dreepy's Petty Grudge IS the KO
            candidates.append(index)
        elif details.get("name") == DRAGAPULT_NAME and target_name == DRAKLOAK_NAME:
            if healthy_dragapult:
                continue
            recon = None
            for other_index, other in enumerate(options):
                if other.get("type") == OPTION_TYPE_ABILITY \
                        and other.get("area") == option.get("inPlayArea") \
                        and other.get("index") == option.get("inPlayIndex"):
                    recon = other_index
                    break
            candidates.append(recon if recon is not None else index)
    return candidates


class ForceEvolveContextTracker:
    """The context surface of force_evolve_before_shuffle_draw, riding the line-rule
    wiring (episode reset + the real-observation Safeguard scan through main.py's
    observe_real calls, which run BEFORE each decision's search). It is NOT a line
    rule: labels and violations are inert -- the substitution itself happens in
    main.py at move-submission time via shuffle_play_indices +
    forced_evolution_options."""

    name = RULE_FORCE_EVOLVE

    def reset_episode(self):
        reset_opponent_context()

    def observe_real(self, observation, select, chosen):
        try:
            if not _SAFEGUARD_OPPONENT and _opponent_shows_safeguard_deck(observation):
                note_safeguard_opponent()
        except Exception:
            pass

    def root_state(self):
        return None

    @staticmethod
    def option_labels(observation, select):
        return None

    @staticmethod
    def update(state, label):
        return state

    @staticmethod
    def violated(state):
        return False


# ---- stadium_discipline (owner rule 2026-08-15) -------------------------------------- #
# Matchup-conditional stadium discipline for the dragapult deck (owner spec, verbatim):
#   * vs grimmsnarl / lucario / team_rocket / archaludon / starmie / cynthia_garchomp:
#     never PLAY Team Rocket's Watchtower unless it removes an OPPONENT'S stadium --
#     EXCEPT when our Meowth ex can no longer be blanked by it: a Meowth already in
#     play (its bench-fetch has fired), every Meowth AND every Night Stretcher in our
#     discard (it can never return), or Meowth PROVABLY prized (prizes are fixed at
#     game start, so once a full deck view proves the never-seen pool IS the prize
#     pool, a never-seen Meowth is prized; main.py feeds note_meowth_prize_proven).
#   * vs lucario additionally: never Watchtower while Jamming Tower is in play
#     (either side), no exceptions.
#   * vs festival_lead: never play ANY of our stadiums unless it removes an
#     opponent's stadium.
#   * vs alakazam: playing Jamming Tower while a Team Rocket's Watchtower is in play
#     (either side's -- removing it un-blanks abilities we want blanked) is a SEARCH
#     LINE violation unless the line then benches Meowth ex and plays the supporter
#     it fetches (the deliberate unblank-Meowth line). LINE RULE ONLY (owner
#     2026-08-15: no raw-tier mask -- below the search floor the model decides).
# Matchup flags come from the archetype router's posterior (note_matchup, sticky per
# episode); without a router (training, non-router bundles) no flag is ever set and
# every surface stands down. Stadium OWNERSHIP is tracked from our own real plays
# (StadiumDisciplineLineRule.observe_real records the serial of every stadium WE
# play); an in-play stadium whose serial was never recorded is the opponent's.

RULE_STADIUM_DISCIPLINE = "stadium_discipline"
TEAM_ROCKETS_WATCHTOWER_ID = 1256
JAMMING_TOWER_ID = 1246
NIGHT_STRETCHER_ID = 1097
STADIUM_CARD_TYPE = 4                  # cg CardType.STADIUM
WATCHTOWER_RESTRICTED_MATCHUPS = ("grimmsnarl", "lucario", "team_rocket",
                                  "archaludon", "starmie", "cynthia_garchomp")
STADIUM_RULE_MATCHUPS = WATCHTOWER_RESTRICTED_MATCHUPS + ("festival_lead", "alakazam")

_MATCHUP_FLAGS = set()                 # router-confirmed opponent archetypes
_MEOWTH_PRIZE_PROVEN = False
_OUR_STADIUM_SERIALS = set()           # serials of stadium cards WE played


def note_matchup(archetype):
    _MATCHUP_FLAGS.add(archetype)


def set_matchup_flags(archetypes):
    """REPLACE the flag set with the router's current >=tau archetypes (owner fix
    2026-08-15): flags follow live belief, so an early lock the evidence later
    dethrones lifts its restrictions instead of accumulating as a union."""
    _MATCHUP_FLAGS.clear()
    _MATCHUP_FLAGS.update(archetypes)


def matchup_known(archetype):
    return archetype in _MATCHUP_FLAGS


def note_meowth_prize_proven():
    global _MEOWTH_PRIZE_PROVEN
    _MEOWTH_PRIZE_PROVEN = True


def _note_our_stadium(serial):
    if serial is not None:
        _OUR_STADIUM_SERIALS.add(serial)


def _stadium_in_play(observation):
    """The in-play stadium card dict, or None."""
    stadium = (observation.get("current") or {}).get("stadium") or []
    return stadium[0] if stadium and stadium[0] else None


def _opponent_stadium_in_play(observation):
    """True when a stadium is in play and it is NOT one we played (a play that
    replaces it removes the OPPONENT'S stadium)."""
    stadium = _stadium_in_play(observation)
    return stadium is not None and stadium.get("serial") not in _OUR_STADIUM_SERIALS


def _watchtower_in_play(observation):
    stadium = _stadium_in_play(observation)
    return stadium is not None and stadium.get("id") == TEAM_ROCKETS_WATCHTOWER_ID


def _jamming_tower_in_play(observation):
    stadium = _stadium_in_play(observation)
    return stadium is not None and stadium.get("id") == JAMMING_TOWER_ID


def _meowth_watchtower_exception(observation):
    """True when our Watchtower can no longer blank a future Meowth ex bench-fetch:
    Meowth already in play, provably prized, or gone for good (every Meowth and every
    Night Stretcher in our discard -- with the recovery outlets spent it cannot
    return). Discard exhaustion needs the registered decklist for the copy counts;
    unregistered, that branch conservatively never fires."""
    current = observation.get("current") or {}
    players = current.get("players") or []
    try:
        me = players[current.get("yourIndex")] or {}
    except Exception:
        return False
    for zone in ("active", "bench"):
        for pokemon in me.get(zone) or []:
            if pokemon is not None and pokemon.get("id") == MEOWTH_EX_CARD_ID:
                return True                    # its fetch already fired
    if _MEOWTH_PRIZE_PROVEN:
        # The proof is a statement about where Meowth WAS -- prizes are fixed at game
        # start, but a taken prize can put it in our hand (and from there the
        # discard). Once it is visible in a zone we control, "prized" no longer means
        # "unavailable": in hand it is benchable (Watchtower would blank the fetch),
        # in the discard the Night-Stretcher branch below is the authority.
        visible = any(card is not None and card.get("id") == MEOWTH_EX_CARD_ID
                      for card in (me.get("hand") or []) + (me.get("discard") or []))
        if not visible:
            return True
    if _OUR_DECKLIST:
        discard = me.get("discard") or []
        meowth_discarded = sum(1 for card in discard
                               if card.get("id") == MEOWTH_EX_CARD_ID)
        stretchers_discarded = sum(1 for card in discard
                                   if card.get("id") == NIGHT_STRETCHER_ID)
        if meowth_discarded >= _OUR_DECKLIST.get(MEOWTH_EX_CARD_ID, 0) > 0 \
                and stretchers_discarded >= _OUR_DECKLIST.get(NIGHT_STRETCHER_ID, 0):
            return True
    return False


def _their_meowth_in_play(observation):
    """True when the OPPONENT'S Meowth ex is currently in play (its bench-fetch
    already fired, so an in-play Watchtower no longer threatens them). A face-down
    hidden active reads as not-Meowth, erring toward holding the Watchtower."""
    current = observation.get("current") or {}
    players = current.get("players") or []
    my_index = current.get("yourIndex")
    try:
        them = players[1 - my_index] or {}
    except Exception:
        return False
    for zone in ("active", "bench"):
        for pokemon in them.get(zone) or []:
            if pokemon is not None and pokemon.get("id") == MEOWTH_EX_CARD_ID:
                return True
    return False


def stadium_discipline_mask(observation, select):
    """The mask surfaces of stadium_discipline (the Watchtower-restricted matchups,
    the festival_lead stadium freeze, and the dragapult-MIRROR Watchtower hold; the
    alakazam Jamming clause is line-rule only). Same return contract as every mask
    rule: allowed ORIGINAL option indices, or None for no restriction.

    MIRROR WATCHTOWER HOLD (owner rule 2026-08-16, from a live misplay: Jamming
    Tower bumped an in-play Watchtower with our Meowth already benched and theirs
    unfired): vs known dragapult (_DRAGAPULT_OPPONENT -- the mirror runs on the
    Budew/posterior flag, NOT the stadium matchup flags, which never carry
    dragapult), while a Watchtower is in play (either side's), our own Meowth fetch
    can no longer be blanked (_meowth_watchtower_exception: in play, provably
    prized, or unrecoverable), and THEIR Meowth ex is not in play, every stadium
    play from hand is masked -- the Watchtower is one-sidedly ours to keep."""
    if not action_rule_enabled(RULE_STADIUM_DISCIPLINE):
        return None
    mirror_hold = (_DRAGAPULT_OPPONENT
                   and _watchtower_in_play(observation)
                   and _meowth_watchtower_exception(observation)
                   and not _their_meowth_in_play(observation))
    festival = matchup_known("festival_lead")
    watchtower_restricted = any(matchup_known(archetype)
                                for archetype in WATCHTOWER_RESTRICTED_MATCHUPS)
    if not festival and not watchtower_restricted and not mirror_hold:
        return None
    options = select.get("option") or []
    allowed, restricted = [], False
    for index, option in enumerate(options):
        card = _played_hand_card(observation, option)
        blocked = False
        if card is not None:
            details = get_card(card.get("id"))
            if details and details.get("cardType") == STADIUM_CARD_TYPE:
                bumps_theirs = _opponent_stadium_in_play(observation)
                if mirror_hold:
                    blocked = True     # ANY stadium play bumps the held Watchtower
                if not blocked and festival and not bumps_theirs:
                    blocked = True
                if not blocked and watchtower_restricted \
                        and card.get("id") == TEAM_ROCKETS_WATCHTOWER_ID:
                    if matchup_known("lucario") and _jamming_tower_in_play(observation):
                        blocked = True         # lucario: never over Jamming Tower
                    elif not bumps_theirs \
                            and not _meowth_watchtower_exception(observation):
                        blocked = True
        if blocked:
            restricted = True
        else:
            allowed.append(index)
    if restricted and allowed:
        return set(allowed)
    return None


# Battle Cage clause (owner rule 2026-08-15, matchup-INDEPENDENT -- no router flag):
# the opponent's Battle Cage blanks placed damage counters, so using Adrena-Brain or
# attacking with Phantom Dive while it is up AND one of our stadiums is playable on
# the same menu throws the counters away for nothing -- bumping first strictly
# dominates. Enforced at SUBMISSION time (the force_evolve mechanism): main.py swaps
# the chosen counter action for the stadium play and the model re-decides post-bump.
# An interception, not a mask, deliberately: under Battle Cage Adrena-Brain still
# HEALS our side, so banning it outright would cost legitimate pure-heal uses when
# the model declines to play the stadium; the swap only fires once the model has
# already committed to the counter action this turn.
BATTLE_CAGE_ID = 1264
PHANTOM_DIVE_ATTACK_ID = 154
OPTION_TYPE_ATTACK = 13                # cg OptionType.ATTACK (option carries attackId)


def battle_cage_blocked_action_indices(observation, select):
    """Option indices whose placed counters the opponent's Battle Cage would blank:
    Adrena-Brain uses (ABILITY options on OUR dark-attached Munkidori) and the
    Phantom Dive attack. Empty when the rule is off or Battle Cage is not the
    in-play stadium."""
    if not action_rule_enabled(RULE_STADIUM_DISCIPLINE):
        return set()
    stadium = _stadium_in_play(observation)
    if stadium is None or stadium.get("id") != BATTLE_CAGE_ID:
        return set()
    me = _our_side(observation)
    zones = {4: me.get("active") or [], 5: me.get("bench") or []}
    indices = set()
    for index, option in enumerate(select.get("option") or []):
        kind = option.get("type")
        if kind == OPTION_TYPE_ATTACK \
                and option.get("attackId") == PHANTOM_DIVE_ATTACK_ID:
            indices.add(index)
        elif kind == OPTION_TYPE_ABILITY:
            zone = zones.get(option.get("area"))
            position = option.get("index")
            pokemon = zone[position] if zone is not None and position is not None \
                and 0 <= position < len(zone) else None
            if pokemon is not None:
                details = get_card(pokemon.get("id"))
                if details and details.get("name") == MUNKIDORI_NAME \
                        and DARK_ENERGY_TYPE in _attached_energy_types(pokemon):
                    indices.add(index)
    return indices


def stadium_bump_candidates(observation, select):
    """PLAY-option indices putting one of OUR stadiums into play (the bump that
    removes Battle Cage). Empty when the rule is off."""
    if not action_rule_enabled(RULE_STADIUM_DISCIPLINE):
        return []
    candidates = []
    for index, option in enumerate(select.get("option") or []):
        card = _played_hand_card(observation, option)
        if card is not None:
            details = get_card(card.get("id"))
            if details and details.get("cardType") == STADIUM_CARD_TYPE:
                candidates.append(index)
    return candidates


# ---- damage_solver (owner rule 2026-08-15) ------------------------------------------- #
# Deterministic damage-counter placement for Phantom Dive and Adrena-Brain. Measured
# root cause: end-of-turn value is FLAT across placements (search-multipick memo), so
# neither raw policy nor search can rank them and counters were dumped on one target.
# Engine facts encoded here (CardImpl.h / State.h / CreateCard.h, read 2026-08-15):
#   * Phantom Dive (attack 154) = DamageCounterAny eVal(6) targetBench -> ctx-14
#     menus, ONE counter per select, select.remainDamageCounter counting down.
#   * Adrena-Brain (Munkidori 112; engine offers it only with {D} attached AND some
#     of OUR Pokemon damaged) = source pick (ctx 16, left to the model per the
#     owner), count pick (ctx 40, NUMBER rows -- FORCED to the max), enemy target
#     pick (ctx 13, ANY of their Pokemon incl. the active).
#   * Mist/Rock shields blank ATTACK-effect counters only (State.h isPreventEffect is
#     gated on onAttackEffect()) -- ability-moved counters stick, hence the solver's
#     shield exclusion applies at ctx 14 and never at ctx 13.
#   * Battle Cage blanks both attack and ability counters on their bench -- handled
#     UPSTREAM by stadium_discipline's bump interception.
# Solver semantics (owner rework 2026-08-16): re-derived at EVERY menu from the live
# observation (self-correcting if any effect altered how a counter landed -- one pick
# is the most an unmodeled interaction can cost). Tiers: game-winning combination >
# KO sweep in the archetype PRIORITY LIST's order (tiebreak: most setup = energy >
# tool > Hero's Cape, EXCEPT when a cheaper same-entry target preserves an additional
# KO) > the UNIVERSAL staging ladder (prize-gated sections, our-Munkidori "n" budget,
# their-healer-aware thresholds, evolution-aware sub-checks) > the model's own
# choice. Prize yields per engine State.h getPrizeCount: Ex = 2, MegaEx = 3. Picks
# only ever come from offered rows.

RULE_DAMAGE_SOLVER = "damage_solver"
SELECT_CONTEXT_DAMAGE_COUNTER = 13         # DamageCounter (Adrena-Brain target menu)
SELECT_CONTEXT_DAMAGE_COUNTER_ANY = 14     # DamageCounterAny (Phantom Dive menus)
SELECT_CONTEXT_REMOVE_COUNTER_COUNT = 40   # Count menu ("up to 3")
OPTION_TYPE_NUMBER = 0
OPTION_TYPE_RETREAT = 12
OPTION_TYPE_END = 14
DRAGAPULT_EX_CARD_ID = 121
MUNKIDORI_CARD_ID = 112

# ---- solver priority lists (owner-authored 2026-08-16, VERBATIM order) --------------- #
# One list per archetype pool. Every entry is resolved from a CARD ID, never a typed
# name (curly-apostrophe trap: "Lillie's Clefairy ex" / "N's Zorua" are curly, the
# Team Rocket's / Cynthia's / Marnie's families are straight). Matching is BY NAME,
# so multi-variant cards (Torchic 324/410, Dipplin x3, Applin x4, Snorunt, Abra,
# Duraludon, ...) are all covered by one entry automatically.
HEROS_CAPE_TOOL_ID = 1159
HYDRAPPLE_EX_CARD_ID = 150


def _card_name(card_id):
    return (get_card(card_id) or {}).get("name") or ""


GRIMMSNARL_EX_NAME = _card_name(648)      # Marnie's Grimmsnarl ex
MORGREM_NAME = _card_name(647)            # Marnie's Morgrem
IMPIDIMP_NAME = _card_name(646)           # Marnie's Impidimp
FROSLASS_CARD_ID = 104                    # THE Freezing Shroud printing
FROSLASS_NAME = _card_name(104)           # the chip's name-based exemption
RELLOR_NAME = _card_name(73)              # festival_lead staging priority
CRUSTLE_NAME = _card_name(345)            # crustle staging priority (240/120/70)


def _entry(card_id, condition=None):
    """Priority entry: (name resolved from the card id, optional condition).
    Conditions: "energy>=2" (per-target filter), "their_count==1"/"their_count>=2"
    (how many of that name they have in play), "cynthia_bench>=200" (total damage
    on their benched Cynthia's Pokemon)."""
    return (_card_name(card_id), condition)


_ANYTHING_ELSE = (None, None)

SOLVER_PRIORITY = {
    "dragapult": (
        _entry(121), _entry(120), _entry(119),         # Dragapult ex line
        _entry(133), _entry(132), _entry(131),         # Dusknoir line
        _entry(326), _entry(325), _entry(324),         # Blaziken ex line
        _entry(112), _entry(140), _entry(1071),        # Munkidori/Fezandipiti/Meowth
        _entry(235), _ANYTHING_ELSE),                  # Budew
    "grimmsnarl": (
        _entry(648), _entry(647), _entry(646),         # Marnie's Grimmsnarl line
        _entry(104), _entry(103),                      # Froslass line
        _entry(112), _entry(689), _entry(235),         # Munkidori/Yveltal/Budew
        _ANYTHING_ELSE),
    "alakazam": (
        _entry(245), _entry(742), _entry(741),         # Alakazam line
        _entry(140), _entry(65), _entry(66),           # Fez/Dunsparce/Dudunsparce
        _ANYTHING_ELSE),
    "slowking": (
        _entry(163), _entry(162),                      # Slowking line
        _entry(434), _entry(272), _entry(756),         # TR Mimikyu/Clefairy/M-Kanga
        _entry(140), _entry(184), _entry(1071),        # Fez/Latias ex/Meowth
        _entry(183), _ANYTHING_ELSE),                  # Smoochum
    "lucario": (
        _entry(678), _entry(333),                      # Mega Lucario ex / Riolu
        _entry(674, "energy>=2"), _entry(673),         # loaded Hariyama / Makuhita
        _entry(676), _entry(674), _entry(140),         # Solrock / any Hariyama / Fez
        _entry(675, "their_count==1"), _entry(117),    # lone Lunatone / Cornerstone
        _entry(1071), _entry(675, "their_count>=2"),   # Meowth / spare Lunatone
        _entry(65), _ANYTHING_ELSE),                   # Dunsparce
    "hydrapple": (
        _entry(710), _entry(150),                      # Meganium / Hydrapple ex
        _entry(709), _entry(708),                      # Bayleef / Chikorita
        _entry(93), _entry(42),                        # Dipplin / Applin
        _entry(95), _entry(920), _entry(140),          # Teal Ogerpon/Tapu Bulu/Fez
        _entry(1071), _entry(655), _ANYTHING_ELSE),    # Meowth / Celebi
    "lopunny": (
        _entry(861), _entry(849), _entry(306),         # M-Froslass/M-Lopunny/Dudun ex
        _entry(103), _entry(758),                      # Snorunt / Buneary
        _entry(66), _entry(65), _entry(174),           # Dudunsparce/Dunsparce/Fan Rotom
        _ANYTHING_ELSE),
    "festival_lead": (
        _entry(74), _entry(73),                        # Rabsca / Rellor
        _entry(93), _entry(42),                        # Dipplin / Applin
        _entry(90), _entry(89), _entry(91),            # Thwackey/Grookey/Rillaboom
        _entry(322), _entry(321),                      # Lilligant / Petilil
        _entry(240), _entry(100), _ANYTHING_ELSE),     # Seaking / Goldeen
    "crustle": (
        _entry(345), _entry(344), _entry(756),         # Crustle/Dwebble/M-Kangaskhan
        _ANYTHING_ELSE),
    "basic_box": (
        _entry(272), _entry(756), _entry(63),          # Clefairy/M-Kanga/Raging Bolt
        _entry(108), _entry(96), _entry(140),          # Wellspring/Teal ex/Fez
        _entry(979), _entry(184), _entry(1071),        # Koraidon ex/Latias ex/Meowth
        _entry(978), _ANYTHING_ELSE),                  # Passimian
    "starmie": (
        _entry(861), _entry(1031),                     # M-Froslass ex / M-Starmie ex
        _entry(104), _entry(103), _entry(1030),        # Froslass/Snorunt/Staryu
        _entry(112), _entry(140), _entry(1071),        # Munkidori/Fez/Meowth
        _entry(65), _ANYTHING_ELSE),                   # Dunsparce
    "archaludon": (
        _entry(190), _entry(169), _entry(57),          # Archaludon ex/Duraludon/Relicanth
        _entry(140), _ANYTHING_ELSE),                  # Fezandipiti ex
    "cynthia_garchomp": (
        _entry(381), _entry(380), _entry(342),         # Garchomp ex/Gabite/Roserade
        _entry(387, "cynthia_bench>=200"),             # Spiritomb (bench loaded)
        _entry(379), _entry(341), _entry(387),         # Gible/Roselia/Spiritomb
        _ANYTHING_ELSE),
    "team_rocket": (
        _entry(272), _entry(434), _entry(431),         # Clefairy/TR Mimikyu/TR Mewtwo
        _entry(401), _entry(400),                      # TR Spidops / TR Tarountula
        _entry(414), _entry(463), _ANYTHING_ELSE),     # TR Articuno / TR Murkrow
    "ns_zoroark": (
        _entry(293), _entry(292), _entry(112),         # N's Zoroark ex/N's Zorua/Munki
        _entry(140), _entry(122), _entry(141),         # Fez/Tatsugiri/Pecharunt ex
        _entry(1071), _entry(689), _entry(235),        # Meowth/Yveltal/Budew
        _ANYTHING_ELSE),
}

# Evolution-aware staging (owner rule 2026-08-16): every "put to X" check also runs
# at the level of the target's DIRECT evolutions -- e.g. a benched Dreepy (70) can be
# staged to 10 hp so a future Drakloak (90) sits at 30. ONE evolution step only (the
# owner's dreepy -> drakloak example); with several possible evolutions the LARGEST
# evolved HP that leaves the pre-evolution alive (>= 10 hp) is used, so a Snorunt
# stages for Froslass (90) when Mega Froslass ex (310) is out of reach. Built from
# the card table's evolvesFrom names: base name -> distinct evolved HPs, descending.
_EVOLUTION_HPS = {}
for _card in CARDS.values():
    if _card.get("cardType") == 0 and _card.get("evolvesFrom") and _card.get("hp"):
        _EVOLUTION_HPS.setdefault(_card["evolvesFrom"], set()).add(int(_card["hp"]))
_EVOLUTION_HPS = {_base: tuple(sorted(_hps, reverse=True))
                  for _base, _hps in _EVOLUTION_HPS.items()}

# Which archetype list serves, fed each decision from the router posterior
# (REPLACED, non-sticky -- same semantics as the stadium flags); the Budew scan's
# _DRAGAPULT_OPPONENT is the mirror fallback when no posterior is available.
SOLVER_TABLE_MATCHUPS = tuple(SOLVER_PRIORITY)
_SOLVER_MATCHUP = None


def set_solver_matchup(archetype):
    global _SOLVER_MATCHUP
    _SOLVER_MATCHUP = archetype if archetype in SOLVER_PRIORITY else None


def _has_cape(pokemon):
    return any(isinstance(tool, dict) and tool.get("id") == HEROS_CAPE_TOOL_ID
               for tool in pokemon.get("tools") or [])


def _our_dark_munkidori_count(observation):
    """How many of OUR in-play Munkidori have a Dark energy attached -- the owner's
    "n" budget for the staging ladder's put-to-30 steps."""
    count = 0
    me = _our_side(observation)
    for zone in ("active", "bench"):
        for pokemon in me.get(zone) or []:
            if pokemon:
                details = get_card(pokemon.get("id"))
                if details and details.get("name") == MUNKIDORI_NAME \
                        and DARK_ENERGY_TYPE in _attached_energy_types(pokemon):
                    count += 1
    return count


def _their_healer_count(observation):
    """Their board healing sources (owner 2026-08-16): dark-attached Munkidori plus
    in-play Hydrapple ex (moves-3-counters and heals-30 treated the same). The
    Grimmsnarl assumption: with a Marnie's Impidimp or Morgrem in play, EVERY one
    of their Munkidori counts as dark-attached."""
    them = _their_side(observation)
    munki = munki_dark = hydrapple = 0
    grimmsnarl_line = False
    for zone in ("active", "bench"):
        for pokemon in them.get(zone) or []:
            if not pokemon:
                continue
            details = get_card(pokemon.get("id")) or {}
            name = details.get("name")
            if name == MUNKIDORI_NAME:
                munki += 1
                if DARK_ENERGY_TYPE in _attached_energy_types(pokemon):
                    munki_dark += 1
            elif pokemon.get("id") == HYDRAPPLE_EX_CARD_ID:
                hydrapple += 1
            elif name in (IMPIDIMP_NAME, MORGREM_NAME):
                grimmsnarl_line = True
    return (munki if grimmsnarl_line else munki_dark) + hydrapple


def _evolved_stage_target(pokemon, threshold):
    """The hp this pre-evolution must be brought DOWN to so that, once evolved, the
    evolution sits at `threshold` (damage carries across evolution). Largest
    evolved HP that keeps the pre-evolution alive (>= 10 hp); None when the card
    has no evolution or none is reachable."""
    name = (get_card(pokemon.get("id")) or {}).get("name")
    max_hp = int(pokemon.get("maxHp") or 0)
    if not name or max_hp <= 0:
        return None
    for evolved_hp in _EVOLUTION_HPS.get(name, ()):
        hp_target = max_hp - (evolved_hp - threshold)
        if hp_target >= 10:
            return hp_target
    return None


def _their_staged_to_thirty_count(observation):
    """How many of their live Pokemon already sit at a put-to-30 level -- at <= 30
    hp, or evolution-staged so the evolved form would be at <= 30. Board-derived,
    so the "n" budget stays consistent across re-derived menus."""
    count = 0
    them = _their_side(observation)
    for zone in ("active", "bench"):
        for pokemon in them.get(zone) or []:
            if not pokemon:
                continue
            hp = int(pokemon.get("hp") or 0)
            if hp <= 0:
                continue
            if hp <= 30:
                count += 1
                continue
            name = (get_card(pokemon.get("id")) or {}).get("name")
            max_hp = int(pokemon.get("maxHp") or 0)
            damage = max(0, max_hp - hp)
            if any(evolved_hp - damage <= 30
                   for evolved_hp in _EVOLUTION_HPS.get(name, ())):
                count += 1
    return count


def _entry_condition_matches(condition, pokemon, their_names, observation):
    if not condition:
        return True
    if condition == "energy>=2":
        return len(pokemon.get("energyCards") or []) >= 2
    name = (get_card(pokemon.get("id")) or {}).get("name")
    if condition == "their_count==1":
        return their_names.count(name) == 1
    if condition == "their_count>=2":
        return their_names.count(name) >= 2
    if condition == "cynthia_bench>=200":
        total = 0
        for benched in _their_side(observation).get("bench") or []:
            if benched:
                benched_name = (get_card(benched.get("id")) or {}).get("name") or ""
                if benched_name.startswith("Cynthia"):
                    total += max(0, int(benched.get("maxHp") or 0)
                                 - max(0, int(benched.get("hp") or 0)))
        return total >= 200
    return False                       # unknown condition: fail closed


def _priority_position(pokemon, entries, their_names, observation):
    """The target's rank in the archetype priority list (first matching entry). With
    no list (unknown deck) rank = prize class, biggest first. Every list ends with
    the anything-else entry, so the fallthrough is unreachable in practice."""
    if entries is None:
        return 3 - _prize_yield(pokemon)
    name = (get_card(pokemon.get("id")) or {}).get("name")
    for position, (entry_name, condition) in enumerate(entries):
        if entry_name is not None and name != entry_name:
            continue
        if _entry_condition_matches(condition, pokemon, their_names, observation):
            return position
    return len(entries)


def _max_additional_kos(needs, budget):
    """How many KOs a counter budget can buy from a list of per-target needs
    (cheapest-first greedy is optimal for identical-value counting)."""
    count = 0
    for need in sorted(needs):
        if need > budget:
            break
        budget -= need
        count += 1
    return count


def _rank_matches(matches):
    """The owner's tiebreak chain (2026-08-16) over interchangeable (index, pokemon,
    need) rows: most setup = more energy > has a tool > Hero's Cape among tools;
    still tied = the MODEL chooses (the tied index set is returned)."""
    def sort_key(row):
        _index, pokemon, _need = row
        energy = len(pokemon.get("energyCards") or [])
        return (-energy, 0 if pokemon.get("tools") else 1,
                0 if _has_cape(pokemon) else 1)
    best = min(sort_key(row) for row in matches)
    tied = {index for index, pokemon, need in matches
            if sort_key((index, pokemon, need)) == best}
    return tied


def _their_side(observation):
    current = observation.get("current") or {}
    players = current.get("players") or []
    my_index = current.get("yourIndex")
    if my_index is None or len(players) < 2:
        return {}
    return players[1 - my_index] or {}


def _prize_yield(pokemon):
    details = get_card(pokemon.get("id")) or {}
    if details.get("megaEx"):
        return 3       # engine State.h getPrizeCount: MegaEx = 3, Ex = 2
    return 2 if details.get("ex") else 1


def _pending_prizes(observation):
    """Prizes we collect from THEIR already-dead board (hp <= 0 -- at a ctx-14 menu
    the attack's 200 to the active has already been applied, so a doomed active
    shows up here without any attack-damage prediction)."""
    pending = 0
    them = _their_side(observation)
    for zone in ("active", "bench"):
        for pokemon in them.get(zone) or []:
            if pokemon and int(pokemon.get("hp") or 0) <= 0:
                pending += _prize_yield(pokemon)
    return pending


def _our_phantom_dive_ready(observation):
    """Our active is a Dragapult ex with the attack's 2 energy attached (count only;
    type coverage is the model's problem -- this gates one conservative exclusion)."""
    active = (_our_side(observation).get("active") or [None])[0]
    return bool(active and active.get("id") == DRAGAPULT_EX_CARD_ID
                and len(active.get("energyCards") or []) >= 2)


def _solver_rows(observation, select):
    """Offered ENEMY-side CARD rows at a counter menu: (index, pokemon, hp, area)."""
    current = observation.get("current") or {}
    players = current.get("players") or []
    my_index = current.get("yourIndex")
    rows = []
    for index, option in enumerate(select.get("option") or []):
        if option.get("type") != OPTION_TYPE_CARD \
                or option.get("playerIndex") == my_index:
            continue
        try:
            player = players[option["playerIndex"]] or {}
            area = option.get("area")
            zone = player.get("active") if area == 4 else (
                player.get("bench") if area == 5 else None)
            pokemon = zone[option["index"]] if zone is not None else None
        except Exception:
            pokemon = None
        if pokemon:
            rows.append((index, pokemon, max(0, int(pokemon.get("hp") or 0)),
                         option.get("area")))
    return rows


def _in_play_froslass_count(observation):
    """In-play copies of THE Freezing Shroud Froslass (exact card id 104, owner
    2026-08-16: check the right printing, not the name family) on BOTH sides --
    each one chips every ability Pokemon at checkup. The exemption below stays
    name-based because that is the engine's own semantics (CardImpl 104
    targetNotMyName(): only Pokemon named exactly "Froslass" are spared; a Mega
    Froslass ex IS chipped)."""
    count = 0
    current = observation.get("current") or {}
    for player in current.get("players") or []:
        for zone in ("active", "bench"):
            for pokemon in (player or {}).get(zone) or []:
                if pokemon and pokemon.get("id") == FROSLASS_CARD_ID:
                    count += 1
    return count


def _ko_need(pokemon, hp, froslass_chip):
    """Counters to a knockout. With Freezing Shroud in play (ANY matchup, owner
    2026-08-16: the trigger is the card being in play, not the archetype), an
    ability target only needs to reach 10*froslass hp -- the checkup finishes it;
    Froslass-named targets are exempt per the card text. Need 0 = already dying
    between turns, place nothing."""
    if froslass_chip:
        details = get_card(pokemon.get("id")) or {}
        if details.get("skills") and details.get("name") != FROSLASS_NAME:
            return max(0, (hp - 10 * froslass_chip + 9) // 10)
    return (hp + 9) // 10


def _ko_pick(live, entries, budget, their_names, observation, combo_rows=()):
    """Tier-1 KO sweep: the priority list's order (or biggest-prize-first with no
    list). Within an entry's matches the owner's most-setup tiebreak applies,
    EXCEPT when a cheaper same-entry target preserves an additional knockout with
    the saved counters (owner 2026-08-16) -- the KO-count-optimal candidates win,
    most-setup breaks ties among them. With the Freezing Shroud Froslass in play
    (any matchup), chip damage discounts every ability target's need (a
    between-turns finish counts). `combo_rows` (owner rule 2026-08-16 evening,
    Adrena ctx-13 with the dive armed): dive-combo SETUPS compete in the SAME
    walk -- a setup on a higher-priority target beats a lower direct kill; on the
    same entry the direct KO wins (certain now beats setup); the extra-KO
    exception stays a direct-KO affair (one Adrena = one setup)."""
    froslass_chip = _in_play_froslass_count(observation)
    ko_able = []
    for index, pokemon, hp in live:
        need = _ko_need(pokemon, hp, froslass_chip)
        if 0 < need <= budget:
            ko_able.append((index, pokemon, need))
    if not ko_able and not combo_rows:
        return None
    all_needs = [need for _index, _pokemon, need in ko_able]

    def choose(matches):
        totals = []
        for _index, _pokemon, need in matches:
            rest = list(all_needs)
            rest.remove(need)
            totals.append(1 + _max_additional_kos(rest, budget - need))
        top = max(totals)
        return _rank_matches([row for row, total in zip(matches, totals)
                              if total == top])

    if entries is None:
        for prize_class in (3, 2, 1):
            matches = [row for row in ko_able
                       if _prize_yield(row[1]) == prize_class]
            if matches:
                return choose(matches)
            setups = [row for row in combo_rows
                      if _prize_yield(row[1]) == prize_class]
            if setups:
                return _rank_matches(setups)
        return None

    def entry_rows(rows, name, condition):
        return [row for row in rows
                if (name is None
                    or (get_card(row[1].get("id")) or {}).get("name") == name)
                and _entry_condition_matches(condition, row[1], their_names,
                                             observation)]

    for name, condition in entries:
        matches = entry_rows(ko_able, name, condition)
        if matches:
            return choose(matches)
        setups = entry_rows(combo_rows, name, condition)
        if setups:
            return _rank_matches(setups)
    return None


def _staging_pick(live, entries, budget, observation, our_prizes, pending,
                  their_names, matchup, bench_indices=frozenset()):
    """Tier-2 UNIVERSAL staging ladder (owner rework 2026-08-16). Sections: at <= 3
    prizes left after this attack, 3-prize targets (Mega ex) first; at <= 2, the
    2-prize targets next; then every target, all walked in priority-list order.
    Steps per section: [up to "n" -> 30] (n = our dark Munkidori minus targets
    already staged to 30, a GLOBAL budget), [30 if they heal else 60],
    [170 if they heal else 200], and for the all-targets section additionally
    [30], [230 if they heal else 260], [260], [230], [290], [400] (the 400 tier
    reaches Cape-boosted giants like a caped Mega Lucario). Every "put to X" runs
    a self-HP pass and then an evolution-HP pass (_evolved_stage_target). Only
    placements the remaining budget can COMPLETE are started.
    Matchup clauses (owner 2026-08-16): festival_lead -- before everything, a
    Rellor in play is staged to 30 then to 10 (the pre-loaded Rabsca kill);
    crustle -- BENCHED Crustle are the top staging priority: a Cape-boosted one
    (Hero's Cape, the +100 tool -- the owner's "giant cape") to 240 first, then
    any to 120, then to 70. (The archaludon 230-damage cap was REMOVED same day
    -- owner: the rule was wrong.)"""
    prizes_after = our_prizes - pending
    healed = _their_healer_count(observation) >= 1
    n_left = max(0, _our_dark_munkidori_count(observation)
                 - _their_staged_to_thirty_count(observation))
    rows = sorted(((_priority_position(pokemon, entries, their_names, observation),
                    index, pokemon, hp) for index, pokemon, hp in live),
                  key=lambda row: row[0])
    if matchup == "festival_lead":
        for threshold in (30, 10):
            matches = []
            for _position, index, pokemon, hp in rows:
                if (get_card(pokemon.get("id")) or {}).get("name") != RELLOR_NAME:
                    continue
                need = (hp - threshold + 9) // 10
                if 0 < need <= budget:
                    matches.append((index, pokemon, need))
            if matches:
                return _rank_matches(matches)
    if matchup == "crustle":
        for threshold, cape_required in ((240, True), (120, False), (70, False)):
            matches = []
            for _position, index, pokemon, hp in rows:
                if index not in bench_indices:
                    continue
                if (get_card(pokemon.get("id")) or {}).get("name") != CRUSTLE_NAME:
                    continue
                if cape_required and not _has_cape(pokemon):
                    continue
                need = (hp - threshold + 9) // 10
                if 0 < need <= budget:
                    matches.append((index, pokemon, need))
            if matches:
                return _rank_matches(matches)
    base_steps = ("n30", 30 if healed else 60, 170 if healed else 200)
    deep_steps = base_steps + (30, 230 if healed else 260, 260, 230, 290, 400)
    sections = []
    if prizes_after <= 3:
        sections.append((3, base_steps))
    if prizes_after <= 2:
        sections.append((2, base_steps))
    sections.append((None, deep_steps))
    for prize_class, steps in sections:
        section_rows = [row for row in rows
                        if prize_class is None
                        or _prize_yield(row[2]) == prize_class]
        if not section_rows:
            continue
        for step in steps:
            if step == "n30":
                if n_left <= 0:
                    continue
                threshold = 30
            else:
                threshold = step
            for level in ("self", "evolved"):
                best_position, matches = None, []
                for position, index, pokemon, hp in section_rows:
                    if best_position is not None and position > best_position:
                        break          # rows are position-sorted
                    if level == "self":
                        hp_target = threshold
                    else:
                        hp_target = _evolved_stage_target(pokemon, threshold)
                        if hp_target is None:
                            continue
                    need = (hp - hp_target + 9) // 10
                    if not 0 < need <= budget:
                        continue
                    best_position = position
                    matches.append((index, pokemon, need))
                if matches:
                    return _rank_matches(matches)
    return None


def _solver_target_pick(observation, eligible, budget, single_target, matchup):
    """The pick for this counter/move: an option index set (usually one index; a
    larger set = the owner's tiebreak chain ended tied and the MODEL chooses), or
    None (waterfall exhausted). eligible = live, unshielded enemy rows; budget =
    counters still to land in this resolution; single_target = ctx-13 (one
    Adrena-Brain use hits ONE Pokemon, so the win tier considers single-target
    finishes only); matchup = a SOLVER_PRIORITY key or None (unknown deck: KO by
    max prizes, staging over a biggest-prize-first ordering -- owner answer 6,
    same prize sections and Munkidori factors)."""
    live = [(index, pokemon, hp) for index, pokemon, hp, _area in eligible if hp > 0]
    if not live:
        return None
    our_prizes = len(_our_side(observation).get("prize") or [])
    pending = _pending_prizes(observation)
    their_names = [(get_card(pokemon.get("id")) or {}).get("name")
                   for zone in ("active", "bench")
                   for pokemon in _their_side(observation).get(zone) or [] if pokemon]
    # Dive-combo setups (owner rule 2026-08-16 evening, ctx 13 with our Phantom
    # Dive armed): targets this Adrena use converts into THIS-TURN kills -- the
    # active into the dive's 200 (window 201..200+10*budget) or a bench target
    # into the dive's 6 counters (window 61..60+10*budget). A Mist/Rock-shielded
    # bench target is no setup (the dive's ATTACK counters would be blanked);
    # the active window needs no shield check (the 200 is damage, not an effect).
    # Windows start strictly above what the dive kills unaided, so every row here
    # is a kill that exists only because of this use.
    combo_rows = []
    if single_target and _our_phantom_dive_ready(observation):
        for index, pokemon, hp, area in eligible:
            if hp <= 0:
                continue
            if area == 4 and 200 < hp <= 200 + 10 * budget:
                combo_rows.append((index, pokemon, (hp - 200 + 9) // 10))
            elif area == 5 and 60 < hp <= 60 + 10 * budget \
                    and not _effect_shielded(pokemon):
                combo_rows.append((index, pokemon, (hp - 60 + 9) // 10))
    # -- tier 0: game-winning combination (owner: checked before everything else).
    # At ctx 13 the finishers are the direct single-target kills PLUS the dive-
    # combo setups (a mega setup at <=3 prizes or an ex setup at <=2 IS the win).
    if our_prizes and pending < our_prizes:
        if single_target:
            best = None
            for index, pokemon, need in (
                    [(index, pokemon, (hp + 9) // 10)
                     for index, pokemon, hp in live] + combo_rows):
                if need > budget:
                    continue
                if pending + _prize_yield(pokemon) >= our_prizes \
                        and (best is None or need < best[0]):
                    best = (need, index)
            if best is not None:
                return {best[1]}
        else:
            from itertools import combinations
            best = None
            for size in range(1, len(live) + 1):
                for combo in combinations(live, size):
                    cost = sum((hp + 9) // 10 for _i, _p, hp in combo)
                    if cost > budget:
                        continue
                    prizes = sum(_prize_yield(pokemon)
                                 for _i, pokemon, _hp in combo)
                    if pending + prizes >= our_prizes \
                            and (best is None or cost < best[0]):
                        best = (cost, combo[0][0])
            if best is not None:
                return {best[1]}
    entries = SOLVER_PRIORITY.get(matchup)
    pick = _ko_pick(live, entries, budget, their_names, observation, combo_rows)
    if pick:
        return pick
    bench_indices = frozenset(index for index, _pokemon, hp, area in eligible
                              if hp > 0 and area == 5)
    return _staging_pick(live, entries, budget, observation, our_prizes, pending,
                         their_names, matchup, bench_indices)


def damage_solver_mask(observation, select):
    """The pick surface of damage_solver: at the three menus the solver owns it
    returns a SINGLE allowed index (a forced pick through the ordinary mask
    machinery -- raw play, greedy resolves and in-tree branch points all obey it),
    None everywhere else or when the waterfall is exhausted (owner: the model picks
    the rest). OPT-IN via the rule name."""
    if not action_rule_enabled(RULE_DAMAGE_SOLVER):
        return None
    context = select.get("context")
    options = select.get("option") or []
    if context == SELECT_CONTEXT_REMOVE_COUNTER_COUNT:
        # Adrena-Brain's "up to 3": ALWAYS move the max (owner rule).
        numbered = [(option.get("number"), index)
                    for index, option in enumerate(options)
                    if option.get("type") == OPTION_TYPE_NUMBER
                    and isinstance(option.get("number"), int)]
        if len(numbered) > 1:
            return {max(numbered)[1]}
        return None
    if context not in (SELECT_CONTEXT_DAMAGE_COUNTER,
                       SELECT_CONTEXT_DAMAGE_COUNTER_ANY):
        return None
    ability = context == SELECT_CONTEXT_DAMAGE_COUNTER
    if ability:
        # ctx 13 is shared by every plain-DamageCounter effect; the select's own
        # `effect` field (replay-verified 2026-08-15: {id, playerIndex, serial} of
        # the source card) proves this one is OUR Adrena-Brain. Unknown source ->
        # stand down entirely rather than apply Adrena logic to a foreign effect.
        effect = select.get("effect") or {}
        my_index = (observation.get("current") or {}).get("yourIndex")
        if effect and not (effect.get("id") == MUNKIDORI_CARD_ID
                           and effect.get("playerIndex") == my_index):
            return None
    budget = select.get("remainDamageCounter")
    if not isinstance(budget, int) or budget <= 0:
        budget = 3 if ability else 6
    matchup = _SOLVER_MATCHUP or ("dragapult" if _DRAGAPULT_OPPONENT else None)
    eligible, excluded_any = [], False
    for index, pokemon, hp, area in _solver_rows(observation, select):
        if not ability and _effect_shielded(pokemon):
            excluded_any = True   # Mist/Rock shield blanks attack-effect counters
            continue
        if ability and area == 4 and hp <= 200 \
                and _our_phantom_dive_ready(observation):
            excluded_any = True   # owner rule: never load the active our attack kills
            continue
        eligible.append((index, pokemon, hp, area))
    if not eligible:
        return None
    pick = _solver_target_pick(observation, eligible, budget,
                               single_target=ability, matchup=matchup)
    if pick:
        return set(pick)               # one index, or a tied set the model resolves
    if excluded_any:
        # The waterfall has no opinion but the exclusions still bind: mask the
        # shielded / doomed rows and let the model pick among the rest (audit fix
        # 2026-08-15 -- returning None here let the model dump onto excluded rows).
        return {index for index, _p, _hp, _a in eligible}
    return None


def _adrena_option_indices(observation, select):
    """MAIN-menu ABILITY option indices on OUR dark-attached Munkidori (the engine
    offers Adrena-Brain only while unused AND usable, so presence = availability)."""
    me = _our_side(observation)
    zones = {4: me.get("active") or [], 5: me.get("bench") or []}
    indices = set()
    for index, option in enumerate(select.get("option") or []):
        if option.get("type") != OPTION_TYPE_ABILITY:
            continue
        zone = zones.get(option.get("area"))
        position = option.get("index")
        pokemon = zone[position] if zone is not None and position is not None \
            and 0 <= position < len(zone) else None
        if pokemon is not None:
            details = get_card(pokemon.get("id"))
            if details and details.get("name") == MUNKIDORI_NAME \
                    and DARK_ENERGY_TYPE in _attached_energy_types(pokemon):
                indices.add(index)
    return indices


# NO HOLD MASK (owner decision 2026-08-15 night, replacing the earlier hold-until-
# pre-attack mask): mid-turn Adrena-Brain use is the MODEL's call -- in-search the
# ability lines are freely explorable, so the tree prices the ability naturally
# instead of attack lines undervaluing the real (interception-forced) use. The
# guarantees that remain: targeting via damage_solver_mask, the forced max count,
# the EXIT MASK below, and adrena_first_index as a redundant real-side net.


def adrena_exit_mask(observation, select):
    """The never-waste surface of damage_solver (owner ask 2026-08-15 night): while
    an unused, usable Adrena-Brain option sits on a MAIN menu, the turn's EXITS are
    closed -- ATTACK options, END, and (when the active is the dark Munkidori
    itself, whose retreat cost could strip the dark) RETREAT. Everything else stays
    open, so ordering is still the model's call; the ability option itself is never
    masked, so the menu can never mask to empty, and after the use resolves the
    option is gone and the exits reopen (stateless). Because this rides the
    ordinary mask machinery it binds IDENTICALLY in-tree and at raw picks: every
    searched future now includes the use the real turn is guaranteed to make.
    Correctness note: attacking (or ending) with Adrena-Brain unused is a
    guaranteed waste -- the attack ends our turn -- so no legitimate line is ever
    removed."""
    if not action_rule_enabled(RULE_DAMAGE_SOLVER):
        return None
    if select.get("context") != 0:
        return None
    if not _adrena_option_indices(observation, select):
        return None
    active = (_our_side(observation).get("active") or [None])[0]
    munki_active = False
    if active is not None:
        details = get_card(active.get("id"))
        munki_active = bool(details and details.get("name") == MUNKIDORI_NAME
                            and DARK_ENERGY_TYPE in _attached_energy_types(active))
    allowed, restricted = [], False
    for index, option in enumerate(select.get("option") or []):
        kind = option.get("type")
        if kind in (OPTION_TYPE_ATTACK, OPTION_TYPE_END) \
                or (munki_active and kind == OPTION_TYPE_RETREAT):
            restricted = True
        else:
            allowed.append(index)
    if restricted and allowed:
        return set(allowed)
    return None


def adrena_first_index(observation, select, chosen):
    """The Munkidori ABILITY index to fire BEFORE the chosen move, or None. Triggers
    on a chosen ATTACK or END (owner: never let a turn end with an unused
    Adrena-Brain -- the engine's existDamaged condition means the option's presence
    already implies the heal is real, and the owner rule is 'use it even with no
    good target'; the END trigger closes the observed turn-5 miss of episode
    'did not use adrena brain': a NON-attacking turn otherwise ends with the
    ability suppressed by the hold mask and never forced) and on a chosen RETREAT
    of the dark Munkidori itself (the retreat cost can strip the dark that powers
    the ability). Stateless: after the real use the option is gone, so no
    interception loop is possible. main.py runs this BEFORE the Battle Cage
    substitute, so a forced use under Battle Cage still becomes the stadium bump
    first."""
    if not action_rule_enabled(RULE_DAMAGE_SOLVER):
        return None
    if select.get("context") != 0 or len(chosen) != 1:
        return None
    options = select.get("option") or []
    position = chosen[0]
    if not (0 <= position < len(options)):
        return None
    kind = options[position].get("type")
    trigger = kind in (OPTION_TYPE_ATTACK, OPTION_TYPE_END)
    if not trigger and kind == OPTION_TYPE_RETREAT:
        active = (_our_side(observation).get("active") or [None])[0]
        details = get_card(active.get("id")) if active else None
        trigger = bool(details and details.get("name") == MUNKIDORI_NAME
                       and DARK_ENERGY_TYPE in _attached_energy_types(active))
    if not trigger:
        return None
    ability_rows = _adrena_option_indices(observation, select)
    if not ability_rows:
        return None
    return min(ability_rows)


class StadiumDisciplineLineRule:
    """The line-rule surface of stadium_discipline: the alakazam Jamming-over-
    Watchtower clause, plus the rule's context tracking (stadium ownership from our
    real plays; episode reset). State tuple:
    (jammed, awaiting_fetch, fetched_serial, satisfied) -- `jammed` arms when the
    line plays Jamming Tower over an in-play Watchtower with the alakazam flag set;
    the obligation is then the Meowth sequence AFTER the jam: bench Meowth ex, play
    the supporter its ability fetches (serial match, same machinery as the Meowth
    rule). violated() = jammed and not satisfied."""

    name = RULE_STADIUM_DISCIPLINE

    def __init__(self):
        self._real_turn = None
        self._real_state = self.initial_state()

    @staticmethod
    def initial_state():
        return (False, False, None, False)

    def reset_episode(self):
        self._real_turn = None
        self._real_state = self.initial_state()
        _MATCHUP_FLAGS.clear()
        _OUR_STADIUM_SERIALS.clear()
        global _MEOWTH_PRIZE_PROVEN
        _MEOWTH_PRIZE_PROVEN = False

    def observe_real(self, observation, select, chosen):
        """Track our real stadium plays (ownership), keep the real turn state, and
        prune the sunk jam obligation exactly like the Meowth rule: once its fetch
        window has passed or the fetched supporter is gone, the real jam is sunk and
        must not keep poisoning the turn's searches. Never raises."""
        try:
            current = observation.get("current") or {}
            turn = current.get("turn")
            if turn != self._real_turn:
                self._real_turn = turn
                self._real_state = self.initial_state()
            jammed, awaiting, fetched_serial, satisfied = self._real_state
            if jammed and not satisfied:
                if awaiting:
                    if not (_to_hand_menu_areas(select)
                            & {AREA_DECK, AREA_LOOKING}) \
                            and (select or {}).get("context") == 0:
                        # Main menu only (2026-08-15, the UB cost-menu bug shape).
                        self._real_state = self.initial_state()
                elif fetched_serial is not None and fetched_serial \
                        not in _our_hand_serials(observation):
                    self._real_state = self.initial_state()
            if not chosen or select is None:
                return
            # Ownership: record the serial of any stadium WE actually play.
            for picked in chosen:
                options = select.get("option") or []
                if 0 <= picked < len(options):
                    card = _played_hand_card(observation, options[picked])
                    if card is not None:
                        details = get_card(card.get("id"))
                        if details and details.get("cardType") == STADIUM_CARD_TYPE:
                            _note_our_stadium(card.get("serial"))
            labels = self.option_labels(observation, select)
            if not labels:
                return
            state = self._real_state
            for picked in chosen:
                if 0 <= picked < len(labels) and labels[picked] is not None:
                    state = self.update(state, labels[picked])
            self._real_state = state
        except Exception:
            pass

    def root_state(self):
        return self._real_state

    @staticmethod
    def option_labels(observation, select):
        if not action_rule_enabled(RULE_STADIUM_DISCIPLINE) \
                or not matchup_known("alakazam"):
            return None
        options = select.get("option") or []
        if not options:
            return None
        labels = None
        to_hand = select.get("context") == SELECT_CONTEXT_TO_HAND
        watchtower_up = _watchtower_in_play(observation)
        for index, option in enumerate(options):
            label = None
            card = _played_hand_card(observation, option)
            if card is not None:
                if card.get("id") == JAMMING_TOWER_ID and watchtower_up:
                    label = ("jam_over_watchtower", None)
                elif card.get("id") == MEOWTH_EX_CARD_ID:
                    label = ("meowth_benched", None)
                else:
                    details = get_card(card.get("id"))
                    if details and details.get("cardType") == SUPPORTER_CARD_TYPE:
                        label = ("supporter_played", card.get("serial"))
            elif to_hand and option.get("type") == OPTION_TYPE_CARD \
                    and option.get("area") in (AREA_DECK, AREA_LOOKING):
                fetched, _pokemon = _entity_at(observation, option.get("area"),
                                               option.get("index"),
                                               option.get("playerIndex"))
                if fetched:
                    details = get_card(fetched.get("id"))
                    if details and details.get("cardType") == SUPPORTER_CARD_TYPE:
                        label = ("supporter_fetched", fetched.get("serial"))
            if label is not None:
                if labels is None:
                    labels = [None] * len(options)
                labels[index] = label
        return labels

    @staticmethod
    def update(state, label):
        jammed, awaiting, fetched_serial, satisfied = state
        kind, serial = label
        if kind == "jam_over_watchtower":
            return (True, False, None, False)
        if not jammed:
            return state                      # the sequence only matters post-jam
        if kind == "meowth_benched":
            return (True, True, None, False)
        if kind == "supporter_fetched" and awaiting:
            return (True, False, serial, satisfied)
        if kind == "supporter_played" and serial is not None \
                and serial == fetched_serial:
            return (True, awaiting, fetched_serial, True)
        return state

    @staticmethod
    def violated(state):
        jammed, _awaiting, _fetched_serial, satisfied = state
        return jammed and not satisfied


# ---- prevent_deck_out (owner rule 2026-08-16) ---------------------------------------- #
# With OUR deck at 0 and THEIRS at 1+, ending the turn loses at our next mandatory
# draw. If Lillie's Determination is in hand, the Supporter unspent, and the hand
# (excluding one Lillie's) exceeds what it draws back (8 at exactly 6 prizes remaining,
# else 6 -- engine-verified CardImpl 1227), playing it leaves >= 1 card in deck and
# survives. Two surfaces: a mask that closes END at main menus while the save is
# available (the turn cannot simply end past it), and a line rule scoring -1 at any
# end-of-turn leaf that neither played Lillie's nor WON -- game-over terminals keep
# their exact result (framework convention), so a Boss line that takes the winning KO
# is exempt automatically, and the SEARCH is the win-checker (owner design decision:
# no hand-written lethal arithmetic). Between-turns poison/burn wins need no
# exception: checkup happens after our turn regardless, so the forced Lillie's can
# never cost that win. INERT whenever their deck is 0 (they draw first and deck out
# -- including the both-at-0 case, owner 2026-08-16: we may need the turn to heal or
# otherwise survive until their draw).
# EXTENSION deckout_judge_stamp (owner 2026-08-16 evening, DRAGAPULT envelopes only):
# Judge (1213, supporter, draws us 4) and Unfair Stamp (1080, ace-spec item, draws
# us 5, legal only when the engine offers its play) also count as saves, preferred
# Lillie's > Judge > Stamp -- with a better save armed, worse save plays are masked.
RULE_DECKOUT_GUARD = "prevent_deck_out"
RULE_DECKOUT_EXTENDED = "deckout_judge_stamp"
LILLIES_DETERMINATION_CARD_ID = 1227
JUDGE_CARD_ID = 1213
UNFAIR_STAMP_CARD_ID = 1080
DECKOUT_SAVE_CARD_IDS = (LILLIES_DETERMINATION_CARD_ID, JUDGE_CARD_ID,
                         UNFAIR_STAMP_CARD_ID)


def _deckout_save_ids(observation, select):
    """Preference-ordered card ids of the deck-out saves armed THIS decision --
    Lillie's > Judge > Unfair Stamp (owner order 2026-08-16 evening). Base need:
    their deck >= 1, ours exactly 0. Judge (supporter, draws us 4) and Unfair
    Stamp (item, draws us 5) arm only under the DRAGAPULT-ONLY extension rule
    deckout_judge_stamp -- absent from the sylveon envelopes, whose behavior
    stays bit-identical to the Lillie's-only original. Unfair Stamp's ace-spec
    timing legality comes from the engine actually OFFERING its play on this
    menu, never from hand presence."""
    current = observation.get("current") or {}
    players = current.get("players") or []
    my_index = current.get("yourIndex")
    if my_index is None or len(players) < 2:
        return []
    me = players[my_index] or {}
    them = players[1 - my_index] or {}
    if (them.get("deckCount") or 0) < 1 or (me.get("deckCount") or 0) != 0:
        return []
    hand = [card for card in me.get("hand") or [] if isinstance(card, dict)]
    hand_ids = [card.get("id") for card in hand]
    supporter_free = not current.get("supporterPlayed")
    saves = []
    draw_back = 8 if len(me.get("prize") or []) == 6 else 6
    if supporter_free and LILLIES_DETERMINATION_CARD_ID in hand_ids \
            and len(hand) - 1 > draw_back:
        saves.append(LILLIES_DETERMINATION_CARD_ID)
    if action_rule_enabled(RULE_DECKOUT_EXTENDED):
        if supporter_free and JUDGE_CARD_ID in hand_ids and len(hand) - 1 > 4:
            saves.append(JUDGE_CARD_ID)
        if len(hand) - 1 > 5:
            for option in (select or {}).get("option") or []:
                card = _played_hand_card(observation, option)
                if card is not None and card.get("id") == UNFAIR_STAMP_CARD_ID:
                    saves.append(UNFAIR_STAMP_CARD_ID)
                    break
    return saves


def deckout_guard_mask(observation, select):
    """Mask surface of prevent_deck_out: while a save is available, END is closed
    at MAIN menus. Everything else stays open -- attacks and rival supporters keep
    their win-attempt lines reachable (the line rule condemns the non-winning
    ones), and the preferred save's play is never masked, so no empty menus. With
    the extension enabled and several saves armed, the LOWER-preference save
    plays are masked too (owner: Lillie's > Judge > Unfair Stamp)."""
    if not action_rule_enabled(RULE_DECKOUT_GUARD):
        return None
    if select.get("context") != 0:
        return None
    saves = _deckout_save_ids(observation, select)
    if not saves:
        return None
    preferred = saves[0]
    extended = action_rule_enabled(RULE_DECKOUT_EXTENDED)
    allowed, restricted = [], False
    for index, option in enumerate(select.get("option") or []):
        blocked = option.get("type") == OPTION_TYPE_END
        if not blocked and extended:
            card = _played_hand_card(observation, option)
            if card is not None and card.get("id") in DECKOUT_SAVE_CARD_IDS \
                    and card.get("id") != preferred:
                blocked = True         # a worse save while a better one is armed
        if blocked:
            restricted = True
        else:
            allowed.append(index)
    if restricted and allowed:
        return set(allowed)
    return None


class DeckoutGuardLineRule:
    """Line-rule surface of prevent_deck_out. State: (armed, saved) -- armed is
    recomputed from the REAL observation each decision (the condition is fully
    observable, no cross-decision memory), saved flips when the line plays
    Lillie's. violated() = armed and not saved; end-of-turn leaves then score -1,
    while game-over terminals keep their exact result, exempting genuine wins."""

    name = RULE_DECKOUT_GUARD

    def __init__(self):
        self._real_state = (False, False)

    @staticmethod
    def initial_state():
        return (False, False)

    def reset_episode(self):
        self._real_state = (False, False)

    def observe_real(self, observation, select, chosen):
        try:
            self._real_state = (bool(_deckout_save_ids(observation, select)),
                                False)
        except Exception:
            self._real_state = (False, False)

    def root_state(self):
        return self._real_state

    @staticmethod
    def option_labels(observation, select):
        options = select.get("option") or []
        if not options:
            return None
        save_ids = DECKOUT_SAVE_CARD_IDS \
            if action_rule_enabled(RULE_DECKOUT_EXTENDED) \
            else (LILLIES_DETERMINATION_CARD_ID,)
        labels = None
        for index, option in enumerate(options):
            card = _played_hand_card(observation, option)
            if card is not None and card.get("id") in save_ids:
                if labels is None:
                    labels = [None] * len(options)
                labels[index] = ("deckout_save_played", None)
        return labels

    @staticmethod
    def update(state, label):
        armed, saved = state
        if label[0] == "deckout_save_played":
            return (armed, True)
        return state

    @staticmethod
    def violated(state):
        armed, saved = state
        return armed and not saved


# ---- academy_at_night_gate (owner rule 2026-08-16, dragapult AND sylveon) ------------ #
# Academy at Night 1248 ("Once during each player's turn, that player may put a card
# from their hand on top of their deck"): returning a hand card behind our next draw
# is tempo-negative in the normal case, so the stadium-USE option (OptionType.ABILITY
# in AreaType.STADIUM) is masked whenever OUR deck has >= 1 card. At deck 0 the mask
# lifts entirely and the use is the model's call (the returned card is exactly our
# next draw, averting deck-out). Fires on the in-play stadium regardless of who
# played it (stadium effects are shared); PLAYING the card from hand is untouched.
# The mask binds identically in-tree, so a search line whose simulated deck empties
# mid-turn sees the option open up inside that line.
RULE_ACADEMY_GATE = "academy_at_night_gate"
ACADEMY_AT_NIGHT_ID = 1248
_AREA_STADIUM = 7                      # cg AreaType.STADIUM


def academy_at_night_gate_mask(observation, select):
    """Mask surface of academy_at_night_gate. Same return contract as the other mask
    rules: allowed ORIGINAL option indices, or None for no restriction."""
    if not action_rule_enabled(RULE_ACADEMY_GATE):
        return None
    stadium = _stadium_in_play(observation)
    if stadium is None or stadium.get("id") != ACADEMY_AT_NIGHT_ID:
        return None
    current = observation.get("current") or {}
    players = current.get("players") or []
    my_index = current.get("yourIndex")
    try:
        me = players[my_index] or {}
    except Exception:
        return None
    if (me.get("deckCount") or 0) == 0:
        return None                    # deck at 0: the save is the model's to take
    allowed, restricted = [], False
    for index, option in enumerate(select.get("option") or []):
        if option.get("type") == OPTION_TYPE_ABILITY \
                and option.get("area") == _AREA_STADIUM:
            restricted = True
        else:
            allowed.append(index)
    if restricted and allowed:
        return set(allowed)
    return None


class CompositeLineRule:
    """Two or more enabled line rules behind the exact single-object interface
    turn_search wires: state is a tuple of sub-states, an option's label is a tuple
    with one slot per rule (None where that rule is silent), and a line violates when
    ANY member rule does. Rule order is fixed at construction, so states and labels
    always line up."""

    def __init__(self, rules):
        self.rules = tuple(rules)
        self.name = "+".join(rule.name for rule in self.rules)

    def reset_episode(self):
        for rule in self.rules:
            rule.reset_episode()

    def observe_real(self, observation, select, chosen):
        for rule in self.rules:
            rule.observe_real(observation, select, chosen)

    def root_state(self):
        return tuple(rule.root_state() for rule in self.rules)

    def option_labels(self, observation, select):
        per_rule = [rule.option_labels(observation, select) for rule in self.rules]
        if all(labels is None for labels in per_rule):
            return None
        count = len(select.get("option") or [])
        merged = []
        for index in range(count):
            entry = tuple(labels[index] if labels is not None else None
                          for labels in per_rule)
            merged.append(entry if any(sub is not None for sub in entry) else None)
        return merged

    def update(self, state, label):
        return tuple(rule.update(sub_state, sub_label)
                     if sub_label is not None else sub_state
                     for rule, sub_state, sub_label
                     in zip(self.rules, state, label))

    def violated(self, state):
        return any(rule.violated(sub_state)
                   for rule, sub_state in zip(self.rules, state))


def line_rule_for(action_rules):
    """The search line-rule object for an envelope's action_rules list, or None when no
    enabled rule needs line-level tracking. Wired by the bundle main.py into
    turn_search.line_rule (capability injection -- older turn_search copies without the
    attribute are simply never wired). Multiple enabled line rules come back as one
    CompositeLineRule behind the same interface."""
    rules = []
    if RULE_MEOWTH_SUPPORTER in (action_rules or ()):
        rules.append(MeowthSupporterLineRule())
    if RULE_FETCHED_CARD in (action_rules or ()):
        rules.append(FetchedCardLineRule())
    if RULE_EVOLVE_BEFORE_SHUFFLE in (action_rules or ()):
        rules.append(EvolveBeforeShuffleLineRule())
    if RULE_FORCE_EVOLVE in (action_rules or ()):
        rules.append(ForceEvolveContextTracker())
    if RULE_STADIUM_DISCIPLINE in (action_rules or ()):
        rules.append(StadiumDisciplineLineRule())
    if RULE_DECKOUT_GUARD in (action_rules or ()):
        rules.append(DeckoutGuardLineRule())
    if not rules:
        return None
    if len(rules) == 1:
        return rules[0]
    return CompositeLineRule(rules)


# Every opt-in action rule, in one place. combined_option_mask is the single mask
# surface: resolve_with_v6 (raw picks), train_ppo's mirrored rollout loop, and the
# bundles' search branch points (option_mask_hook) must all call it so the menus can
# never drift apart. Each rule checks its own enabled flag and returns None when off,
# so with no rules enabled this is bit-identical to no masking at all.
OPTION_MASK_RULES = (counter_option_mask, shielded_counter_mask,
                     meowth_supporter_gate_mask, meowth_obligation_exit_mask,
                     pokepad_supporter_gate_mask, dead_fetch_item_mask,
                     stadium_discipline_mask, damage_solver_mask,
                     adrena_exit_mask, deckout_guard_mask,
                     academy_at_night_gate_mask)


def combined_option_mask(observation, select):
    allowed = None
    for rule_mask in OPTION_MASK_RULES:
        mask = rule_mask(observation, select)
        if mask is None:
            continue
        allowed = mask if allowed is None else (allowed & mask)
    if allowed is not None and not allowed:
        # The enabled rules only agree on an empty menu when every option is a no-op
        # (e.g. every live target shielded, every unshielded target dead); that is the
        # engine forcing a wasted placement, so stand down like a single rule would.
        return None
    return allowed


def resolve_with_v6(observation, select, choose, in_flight, stats=None, engine_hp=None):
    """encode_inflight.resolve_with_v5 with v6 candidate rows. The loops must stay in step."""
    forced, reason = forced_answer(select)
    if forced is not None:
        if stats is not None:
            stats["forced"] += 1
            stats["forced_" + reason] += 1
        return forced
    state = MultiSelect(observation, select)
    v3_matrix, v6_extra = base_option_matrix_v6(observation, select, in_flight, engine_hp)
    counter_mask = combined_option_mask(observation, select)
    while not state.complete():
        forced_index = state.forced_index()
        if forced_index is not None:
            if stats is not None:
                stats["forced_pick"] += 1
            state.take(forced_index)
            continue
        pending = state.pending()
        if counter_mask is not None:
            masked_pending = [index for index in pending if index in counter_mask]
            if masked_pending:                    # never mask into an empty menu
                if stats is not None and len(masked_pending) < len(pending):
                    stats["counter_masked"] += len(pending) - len(masked_pending)
                pending = masked_pending
        stop_offered = state.stop_offered()
        rows = candidate_matrix_v6(v3_matrix, v6_extra, state, pending, stop_offered)
        picked = int(choose(rows, state))
        if stats is not None:
            stats["forwards"] += 1
        if stop_offered and picked == len(pending):
            if stats is not None:
                stats["stop"] += 1
            break
        state.take(pending[picked])
    if stats is not None:
        stats["selects"] += 1
        stats["picks"] += len(state.chosen)
    return state.answer()
