"""v2.3 decision-adjacent hindsight labels + heads (AUX_V23_DESIGN.md).

v23 = v22 + four head groups, behind `--heads v23`. Everything here is NEW code beside the
v21/v22 machinery in train_ppo.py, which is untouched: with any other --heads not one function
in this module is reached, and the observation encoder is not touched at all (the heads read
the embeddings the option scorer and the trunk already produce).

Label groups -- all hindsight-FACTUAL; no judgment, no heuristics, and no card names or
card-specific branches anywhere: everything keys on engine mechanisms (resolved DumpState
fields, the engine's own log events, the mechanically extracted attach-energy condition table,
and the HP_CHANGE putDamageCounter flag).

  L1 attachment_need      per ATTACH option row: beyond_need / already_covered      (BCE)
                          -- the manual hand/discard attach AND every EFFECT attach
                             (ATTACH_FROM prompts; see effect_attach_energy)
  L2 placement_conversion per damage-counter target row: ko_delay (CE) / decisive   (BCE)
  L3 bench_contributed    per PLAY-into-play row: attacked / evolved / used_ability /
                          donated / fetched_card_used (BCE) + prizes_donated        (CE)
  T1 ability_blocked      per in-play token: the engine's resolved noAbility bit    (BCE)
  T2 effective_cost_delta per in-play token x attack slot: effective minus printed  (CE)

L1/L2/L3 are OPTION-pathway heads (input = the option row the scorer reads: the same
[board context | option features] concatenation policy_score consumes). T1/T2 are
token-pathway heads (the v22 recipe).

CONVENTION (owner spec, L3: "labels exist only for plays actually taken -- a Pokemon that
never entered play has no hindsight"): an option-row label about the CONSEQUENCE of taking
that option exists only for the option actually TAKEN. So L1, L3 and L2's `decisive` are
labelled on the chosen row and masked everywhere else. L2's `ko_delay` is a property of the
TARGET rather than of the choice, so it is labelled on every candidate row naming a board
Pokemon.

The engine's per-attack effective energy requirement rides in on the DumpState patch
(engine_src, `attackCost` / `energyOrder`): per in-play Pokemon, the PREFIX of shortfalls as
its attached energy cards are added one at a time in the engine's own attach order. Energy
card k filled a slot at that attack exactly when prefix[k+1] < prefix[k] -- the engine's own
typed-slots-before-colorless fill, so this module does ZERO cost arithmetic. Without that
engine build the L1 attack evidence and T2 are MASKED (and counted in `stats`), never guessed.
"""

import bisect
import itertools
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CONDITION_TABLE = ROOT / "data" / "cards" / "attach_energy_conditions.json"

# ---- shapes (cross-checked against train_ppo's own constants at import there) ------- #
V23_MAX_OPTION_ROWS = 64         # option-row label width; rows beyond this are masked.
                                 # A MAIN menu (every hand play + attach + ability + attack
                                 # + retreat + end) is the widest select the labelled groups
                                 # live in; 64 covers it with room, and the heads only ever
                                 # compute over min(actual option width, this).
V23_MAX_BOARD_TOKENS = 18        # == train_ppo.MAX_BOARD_TOKENS
V23_ATTACK_SLOTS = 2             # == train_ppo.ATTACK_SLOTS
V23_KO_DELAY_CLASSES = 5         # {this turn, next, 2, 3+, never}
V23_PRIZE_CLASSES = 4            # prizes donated: {0, 1, 2, 3}
V23_DELTA_CLASSES = 5            # effective - printed, clipped to [-2, +2]
V23_DELTA_ZERO = 2               # ...offset, so class = clip(delta, -2, 2) + 2
V23_L1_BITS = 2                  # beyond_need, already_covered
V23_L3_BITS = 5                  # attacked, evolved, used_ability, donated, fetched_card_used
V23_L3_BIT_NAMES = ("attacked", "evolved", "used_ability", "donated", "fetched_card_used")
assert len(V23_L3_BIT_NAMES) == V23_L3_BITS
V23_FETCH_BIT = V23_L3_BIT_NAMES.index("fetched_card_used")
V23_DONATED_BIT = V23_L3_BIT_NAMES.index("donated")
V23_FETCH_DELAY_CLASSES = 5      # {this turn, next own turn, 2, 3+, never} -- OWN turns:
                                 # we can only play our own cards on our own turns, so an
                                 # engine-turn count leaves the odd classes structurally
                                 # empty (the ko_delay bucketing lesson, at design time).
V23_FETCH_NEVER = V23_FETCH_DELAY_CLASSES - 1
V23_FETCH_RUNWAY = 3             # own turns of runway required before "never played" is
                                 # evidence rather than a cut-short window (masked).
V23_KO_JOIN_MOVES = 4            # == train_ppo.KO_JOIN_MOVES (asserted at import there):
                                 # a board departure is a KO iff the OTHER side takes a
                                 # prize within this many moves of it.

# option-row descriptor kinds (worker side)
ROW_OTHER, ROW_ATTACH, ROW_PLAY, ROW_TARGET = 0, 1, 2, 3

# cg.api ints, inlined (this module must import with no cg on the path)
AREA_HAND, AREA_DISCARD, AREA_ACTIVE, AREA_BENCH = 2, 3, 4, 5
OPTION_TYPE_CARD, OPTION_TYPE_PLAY, OPTION_TYPE_ATTACH = 3, 7, 8
SELECT_CONTEXT_DAMAGE_COUNTER = 13
SELECT_CONTEXT_DAMAGE_COUNTER_ANY = 14
SELECT_CONTEXT_ATTACH_FROM = 21          # "select the Pokemon to attach the card to"
COUNTER_CONTEXTS = (SELECT_CONTEXT_DAMAGE_COUNTER, SELECT_CONTEXT_DAMAGE_COUNTER_ANY)
CARD_TYPE_BASIC_ENERGY, CARD_TYPE_SPECIAL_ENERGY = 5, 6
LOG_DRAW, LOG_MOVE_CARD, LOG_PLAY, LOG_EVOLVE = 4, 6, 10, 12

V23_COMPONENTS = ("attach_need", "attach_covered", "ko_delay", "decisive", "bench_bits",
                  "prizes_donated", "ability_blocked", "cost_delta")

# Which components each --heads value SUPERVISES; see train_ppo.V22_DISABLED_BY_HEADS for
# the convention (module, labels and checkpoint slot all retained, only `add()` skipped).
#   ability_blocked  LABEL LEAK -- the label IS encode_rich.py:109's input bit on the same
#                    token. Loss 1.374e-05 -> 1.636e-21, recall 0.984. Never re-enable
#                    without removing that input or repointing the label.
#   cost_delta       no supervision to learn from: the certification harness reports
#                    "T2 non-trivial (delta != 0) label instances: 0" -- the channel is
#                    ~98% the zero class and the pool never exercises the rest.
#   prizes_donated   RE-ENABLED for v24 (2026-08-06). It was switched off on 08-04 for
#                    "no separation from base after 320 iters", but the cause was
#                    _ko_after: 175 of 176 bench rows returned "never KO'd", so the head
#                    had almost no labels at all. The prize-join KO test fixes that, and
#                    deleting the head would have hidden the defect.
V23_DISABLED_BY_HEADS = {
    "v23": frozenset({"prizes_donated", "ability_blocked", "cost_delta"}),
    "v24": frozenset({"ability_blocked", "cost_delta"}),
    # v25 = v24 + the fetch_delay head (turns until a fetched card is played), which
    # REPLACES the binary fetched_card_used bit in the pooled bench loss: that bit's
    # "ever later played" is ~68% yes regardless of whether the fetch was needed NOW,
    # so it cannot express the timing judgment (it blanket-fired at chance for all of
    # d128_uniform). The bit's LABEL is still built (v23/v24 compat) and its per-bit
    # metric still prints.
    "v25": frozenset({"ability_blocked", "cost_delta"}),
    # v26 = v25 MINUS the fetch_delay head (2026-08-07 verdict, 700 iters of d256_uniform):
    # acc 0.24-0.27 vs a majority-class bar of ~0.40 for the WHOLE run, dead flat -- and the
    # binary fetched_card_used bit independently fails the same way (prec 0.39 vs a 0.44
    # positive rate). "Will this fetched card get used" depends on the policy's own future
    # rollout; it is not predictable at decision time. The bit stays masked out of the
    # pooled bench loss (both forms fail), so under v26 the fetch channel is unsupervised
    # everywhere. wasted_counters stays (recall 0.71 vs a declining 0.59-0.64 base).
    "v26": frozenset({"ability_blocked", "cost_delta"}),
}


# ======================================================================================= #
# The mechanically extracted condition table
# (scripts/extract_attach_energy_conditions.py -- generated, never hand-edited)
# ======================================================================================= #

_TABLE = None


def _table():
    """({card id: energy type} for ABILITY hosts, {attack id: energy type} for ATTACK hosts).

    The engine declares "if this Pokemon has any {X} Energy attached" with one builder,
    `.conditionAttachEnergyMe(X)`. An ABILITY log/event names its HOST CARD (the option type
    carries no skill identity), so the ability side is re-keyed by card id; the attack side
    keys by attack id, which the ATTACK log carries directly.

    A missing file yields empty tables: the credit rule then never fires, which can only cost
    recall, never invent waste."""
    global _TABLE
    if _TABLE is None:
        by_card, by_attack = {}, {}
        try:
            blob = json.loads(CONDITION_TABLE.read_text(encoding="utf-8"))
            for entry in blob.get("entries") or ():
                if entry["host"] == "ability":
                    by_card[int(entry["cardId"])] = int(entry["energyType"])
                else:
                    by_attack[int(entry["hostId"])] = int(entry["energyType"])
        except Exception:
            by_card, by_attack = {}, {}
        _TABLE = (by_card, by_attack)
    return _TABLE


_ENERGY_TYPE, _PRINTED_COST, _ATTACK_SLOT_IDS, _IS_ENERGY = {}, {}, {}, {}
_PRINTED_MULTISET = {}

# cg EnergyType ORDINAL -> the engine's provided-type BITMASK (build_v25 `energyTypes`).
# The engine's own condition/slot test is (provided & wanted) != 0; Colorless is mask 0.
_ORDINAL_TO_MASK = {0: 0, 1: 1, 2: 2, 3: 4, 4: 8, 5: 16, 6: 32, 7: 64, 8: 128, 9: 256,
                    10: 511,          # RAINBOW: every type
                    11: 16 | 64}      # TEAM_ROCKET: Psychic and Darkness


def _is_energy_card(card_id):
    """Does this card id name an Energy card (basic or special)? Card-TABLE lookup, not a
    card list: the effect-attach recognizer uses it to ignore the tool/other attaches that
    share the ATTACH_FROM prompt."""
    if card_id not in _IS_ENERGY:
        from src.cards import get_card
        card = get_card(int(card_id)) or {}
        _IS_ENERGY[card_id] = card.get("cardType") in (CARD_TYPE_BASIC_ENERGY,
                                                       CARD_TYPE_SPECIAL_ENERGY)
    return _IS_ENERGY[card_id]


def _energy_card_type(card_id):
    """The energy type a card PROVIDES, as the observation's EnergyType ordinal, or None when
    the card table does not pin it down (special energies). None is treated as able to satisfy
    any declared condition, so the credit rule never manufactures waste."""
    if card_id not in _ENERGY_TYPE:
        from src.cards import get_card
        card = get_card(int(card_id)) or {}
        _ENERGY_TYPE[card_id] = (card.get("energyType") or 0) if card.get("cardType") == 5 \
            else None
    return _ENERGY_TYPE[card_id]


def _printed_cost(attack_id):
    if attack_id not in _PRINTED_COST:
        from src.cards import get_attack
        attack = get_attack(int(attack_id)) or {}
        _PRINTED_COST[attack_id] = len(attack.get("energies") or ())
    return _PRINTED_COST[attack_id]


def _printed_cost_multiset(attack_id):
    """The printed cost as a tuple of EnergyType ORDINALS (0 = a Colorless slot), or None
    when the card table does not know the attack. Only ever used behind the prefix[0]
    tripwire (printed total == engine effective total), so a live cost modifier can never
    smuggle printed arithmetic into a label -- T2 measured ZERO non-trivial modifiers in
    the pool, which is what makes this table usable at all."""
    if attack_id not in _PRINTED_MULTISET:
        from src.cards import get_attack
        attack = get_attack(int(attack_id)) or {}
        energies = attack.get("energies")
        _PRINTED_MULTISET[attack_id] = None if energies is None \
            else tuple(int(energy) for energy in energies)
    return _PRINTED_MULTISET[attack_id]


def attack_slot_ids(card_id):
    """The card's attack ids in card-table order -- the same slot convention the payability
    head uses, so an attack COLUMN always means the same attack."""
    if card_id not in _ATTACK_SLOT_IDS:
        from src.cards import get_card
        card = get_card(int(card_id)) or {}
        _ATTACK_SLOT_IDS[card_id] = tuple(
            attack_id for attack_id in (card.get("attacks") or ())[:V23_ATTACK_SLOTS]
            if attack_id)
    return _ATTACK_SLOT_IDS[card_id]


# ======================================================================================= #
# Worker side: what one decision records
# ======================================================================================= #

def bottom_serial(pokemon):
    """The stack's BOTTOM serial -- stable across evolution, unlike the top. The same key the
    EventTracker's per-energy diff uses, so holders line up across an evolution.

    `preEvolution` is OLDEST-FIRST (engine State.h walks it in push_back order and CardMove.h
    appends at each evolve; measured 6892/6892 observations), so the Basic is [0], NOT [-1].
    Using [-1] (2026-08-06 audit) returned the STAGE 1 of a Stage-2 stack, and the key
    therefore CHANGED at the Stage-1 -> Stage-2 evolution -- silently, because the re-key
    makes `prev_energy_serials.get(bottom)` None and the diff's `previous is None: continue`
    branch swallows it. Relabelling 40 recorded focus games with [0] moved 21/646 chosen
    ATTACH rows from MASKED to labelled and flipped `attacked` 0 -> 1 on 10/307 bench rows:
    a WRONG label, not a missing one, for every Stage-2 attacker -- i.e. for Dragapult."""
    stack = [underneath["serial"] for underneath in (pokemon.get("preEvolution") or ())
             if isinstance(underneath, dict) and "serial" in underneath]
    return stack[0] if stack else pokemon["serial"]


def _board_lookup(observation):
    """(area, player index, slot) -> the Pokemon dict sitting there."""
    lookup = {}
    for player_index, player in enumerate(observation["current"]["players"]):
        for area, slots in ((AREA_ACTIVE, player.get("active") or ()),
                            (AREA_BENCH, player.get("bench") or ())):
            for slot, pokemon in enumerate(slots):
                if pokemon is not None:
                    lookup[(area, player_index, slot)] = pokemon
    return lookup


def effect_attach_energy(select):
    """(energy serial, energy card id) when THIS select is an effect attach asking which
    Pokemon receives an energy the engine has already named -- else None.

    SelectContext.ATTACH_FROM is the engine's single funnel for "attach this card to a
    Pokemon", and it names the card in flight in `contextCard`. Every card that attaches
    energy by EFFECT rather than by the once-per-turn manual attach comes through here --
    deck searches (Crispin, Marnie's Grimmsnarl ex), discard recursion (Blaziken ex, Mega
    Lucario ex), items (Wondrous Patch, Glass Trumpet): 37 such cards in the engine, 7 of
    them in the current ladder pools. Recognition is by MECHANISM only (context +
    contextCard + a board target the mover owns): no card ids, no effect names, so a card
    the pool has never played is covered by construction.

    TWO PROMPT SHAPES exist, and this covers only the first (re-measured 2026-08-05 over 250
    basic-box games vs the corpus -- 676 ATTACH_FROM prompts):

      ENERGY-FIRST (587/676, 87%) -- the engine names the card in `contextCard` and asks
        which Pokemon receives it. Crispin, Marnie's Grimmsnarl ex, Mega Lucario ex ...
        Covered here; contextCard was an energy in every one of those 587.

      TARGET-FIRST (89/676, 13%) -- `contextCard` is None because the energy has not been
        chosen yet: the prompt picks the TARGET(S) and the engine draws the energy
        afterwards. All of them are Glass Trumpet ("attach a Basic Energy from your discard
        to EACH of up to 2 Benched {C} Pokemon"), Rosa's Encouragement ("attach up to 2
        Basic Energy from your discard to 1 Stage 2") and Iono's Bellibolt ex. `looking` is
        empty at that moment too, so the energy is nameable NOWHERE at decision time.
        This function returns None for them and the row carries no L1 label -- an ABSENT
        label, never a wrong one.

    An earlier version of this docstring claimed "101/101 prompts carried a contextCard";
    that was a narrower pool and is not a general invariant. Recovering the target-first
    shape needs the energy resolved at LABEL time from context["attach"] rather than at
    decision time -- see AUX_V24_PLAN.md, and note Rosa's attaches TWO energies to ONE
    target, so (move, holder) does not uniquely name one."""
    if select.get("context") != SELECT_CONTEXT_ATTACH_FROM:
        return None
    card = select.get("contextCard")
    if not isinstance(card, dict):
        return None
    serial, card_id = card.get("serial"), card.get("id")
    if not serial or not card_id or not _is_energy_card(card_id):
        return None
    return serial, card_id


def option_descriptors(observation, select):
    """One 4-tuple per ENGINE option: (kind, primary serial, secondary serial, card id).

        ATTACH  -> (ROW_ATTACH, attached card serial, target holder BOTTOM serial, card id)
        PLAY    -> (ROW_PLAY,   hand card serial,     0,                            card id)
        CARD naming a board Pokemon
                -> (ROW_TARGET, that stack's BOTTOM serial, 0, its card id)
        anything else -> zeros.

    An ATTACH_FROM option naming one of the MOVER's own Pokemon is also a ROW_ATTACH --
    same tuple, energy serial from the select's `contextCard` (see effect_attach_energy).
    Deliberately OUR side only: an ATTACH_FROM option naming the OPPONENT's board falls
    through to ROW_TARGET and carries no L1 label. L1 asks "was this energy WE attached
    needed", which is a question about our own board; and `attachment_need` is called with
    owner=mover, so an opponent-side holder would be scored against the wrong player's
    event stream. Rare in the pools (1-22 options per 250 games) and a missing label, never
    a wrong one.
    Without this, every effect-driven attach was invisible to L1: OptionType.ATTACH covers
    only the manual hand/discard attach, so the whole Crispin / Grimmsnarl / Wondrous Patch
    family produced ROW_OTHER + ROW_TARGET and no attachment_need label at all. Those rows
    carried no other label either (ATTACH_FROM is not a COUNTER_CONTEXT, so the ROW_TARGET
    branch never fired for them), which is what keeps this change purely additive.

    Pure lookup off the observation the model was shown; no engine call, no card knowledge."""
    current = observation["current"]
    mover = current["yourIndex"]
    player = current["players"][mover]
    hand = player.get("hand") or []
    lookup = _board_lookup(observation)
    in_flight = effect_attach_energy(select)
    attach_from = select.get("context") == SELECT_CONTEXT_ATTACH_FROM
    rows = []
    for option in (select.get("option") or ()):
        option_type, index = option.get("type"), option.get("index")
        index = -1 if index is None else index
        if option_type == OPTION_TYPE_ATTACH:
            source = {AREA_HAND: hand,
                      AREA_DISCARD: player.get("discard") or []}.get(
                          option.get("area") or AREA_HAND) or []
            target = lookup.get((option.get("inPlayArea"), mover, option.get("inPlayIndex")))
            if 0 <= index < len(source) and target is not None \
                    and _is_energy_card(source[index]["id"]):
                # ENERGY only. OptionType.ATTACH also carries POKEMON TOOLS (Hero's Cape,
                # Handheld Fan, ...), and L1 is entirely about energy: cost slots, declared
                # attach-energy conditions, and consumption. A tool has none of those, so
                # `beyond_need` could only ever come back MASKED for one -- but
                # `already_covered` was still being labelled on those rows, teaching the
                # head an energy-coverage answer to a question about a tool. Measured
                # 15/192 chosen ATTACH rows (7.8%) in random play; it is also part of why
                # `v23_attach_masked` looked so high. The effect-attach recognizer
                # (effect_attach_energy) has always applied exactly this test -- the manual
                # path simply never did (2026-08-06 audit).
                card = source[index]
                rows.append((ROW_ATTACH, card["serial"], bottom_serial(target), card["id"]))
                continue
        elif option_type == OPTION_TYPE_PLAY:
            if 0 <= index < len(hand):
                card = hand[index]
                rows.append((ROW_PLAY, card["serial"], 0, card["id"]))
                continue
        elif option_type == OPTION_TYPE_CARD:
            target = lookup.get((option.get("area"), option.get("playerIndex"), index))
            if target is not None:
                if in_flight is not None and option.get("playerIndex") == mover:
                    # ENERGY-FIRST: the energy is already chosen; this option picks WHICH of
                    # our Pokemon it lands on -- the same (energy, holder) pair a manual
                    # attach names.
                    rows.append((ROW_ATTACH, in_flight[0], bottom_serial(target),
                                 in_flight[1]))
                elif attach_from and option.get("playerIndex") == mover:
                    # TARGET-FIRST: an ATTACH_FROM prompt whose energy the engine has not
                    # drawn yet (Glass Trumpet / Rosa's / Iono's Bellibolt ex -- contextCard
                    # None, `looking` empty). Serial 0 is the sentinel that says "resolve
                    # this from the engine's attach events at LABEL time"; see
                    # energies_attached_by. Recording the row is what makes the label
                    # possible at all -- before this it fell to ROW_TARGET and was lost.
                    rows.append((ROW_ATTACH, 0, bottom_serial(target), 0))
                else:
                    rows.append((ROW_TARGET, bottom_serial(target), 0, target["id"]))
                continue
        rows.append((ROW_OTHER, 0, 0, 0))
    return tuple(rows)


def decision_state(rich_cards, observation, mover, main_select):
    """This decision's engine-state facts for v23, or None when no dump was decoded.

    Returns (token_state, holder_state):
      token_state   ((serial, noAbility, ((attack id, effective cost), ...)), ...) over the
                    in-play Pokemon of BOTH sides, in meta["board"] emission order -- the
                    T1 / T2 labels. `effective cost` is -1 where the engine build predates
                    the attackCost dump patch (-> that column is masked, never guessed).
      holder_state  {bottom serial: (energy card serials in the engine's attach order,
                    {attack id: prefix shortfall tuple})} for the MOVER's own board, recorded
                    at MAIN selects only (the selects attacks are declared at). None when this
                    is not a MAIN select or the engine emitted no attackCost at all.
    """
    if not rich_cards:
        return None
    current = observation["current"]
    token_state, holder_state, saw_cost = [], ({} if main_select else None), False
    for player_index in (mover, 1 - mover):
        player = current["players"][player_index]
        for pokemon in (list(player.get("active") or ()) + list(player.get("bench") or ())):
            if pokemon is None:
                continue
            rich = rich_cards.get(pokemon["serial"]) or {}
            prefixes = {int(entry["attackId"]): tuple(entry.get("prefix") or ())
                        for entry in (rich.get("attackCost") or ())}
            saw_cost = saw_cost or bool(prefixes)
            slots = tuple(
                (attack_id, int(prefixes[attack_id][0]) if prefixes.get(attack_id) else -1)
                for attack_id in attack_slot_ids(pokemon["id"]))
            token_state.append((pokemon["serial"], int(rich.get("noAbility") or 0), slots))
            if holder_state is not None and player_index == mover and prefixes:
                # Third element (build_v25 dump; empty tuple on older engines): the
                # RESOLVED provided-type (mask, units) per card, index-parallel to
                # energyOrder -- the engine truth the condition credit and the joint
                # fill consume. Additive: every consumer indexes, never unpacks.
                holder_state[bottom_serial(pokemon)] = (
                    tuple(rich.get("energyOrder") or ()), prefixes,
                    tuple((int(entry.get("type") or 0), int(entry.get("count") or 1))
                          for entry in (rich.get("energyTypes") or ())))
    return (tuple(token_state), holder_state if saw_cost else None)


# ======================================================================================= #
# Label side: per-game context
# ======================================================================================= #

def build_context(record):
    """Per-GAME index of the streams the v23 labels read, built once per game."""
    context = {
        "attach": {},          # energy serial -> [(move, turn, holder, owner, card id)]
        "lost": {},            # energy serial -> [(move, turn, holder, owner)]
        "attack": [],          # (move, turn, player, holder serial, attack id)
        "ability": [],         # (move, turn, player, holder serial, card id)
        "left": {},            # serial -> [(move, turn, hp, player)]
        "entered": {},         # serial -> (move, turn, player)
        "prize": [],           # (move, turn, player, count)
        "hp": {},              # target serial -> [(move, turn, value, counter, hp before)]
        "hand_add": [],        # (move, turn, player, serial)
        "ko_records": {},      # bottom serial -> [(move, turn, prizes, taker, victim)]
                               # -- the build_v25 engine KO record (EXACT; empty on
                               # older-engine games, where the prize join is the fallback)
        "hand_lost": {},       # serial -> [move, ...] -- left a hand WITHOUT being played
                               # (opponent disruption, shuffle-away, our own discard cost)
        "played_at": {},       # serial -> [(move, turn), ...] first-to-last
        "evolution_parent": {},  # evolving card serial -> the serial it evolved onto
        "evolved_from": {},    # evolving target serial -> [move, ...]
        "action_moves": [],
    }
    for event in record.get("events") or ():
        kind, move, turn = event["kind"], event["move"], event["turn"]
        player = event.get("player")
        if kind == "attach_seen":
            context["attach"].setdefault(event["serial"], []).append(
                (move, turn, event["holder"], player, event.get("card")))
        elif kind == "energy_lost":
            context["lost"].setdefault(event["serial"], []).append(
                (move, turn, event["holder"], player))
        elif kind == "attack" and event.get("serial"):
            context["attack"].append((move, turn, player, event["serial"],
                                      event.get("attack") or 0))
        elif kind == "ability" and event.get("serial"):
            context["ability"].append((move, turn, player, event["serial"],
                                       event.get("card") or 0))
        elif kind == "left":
            context["left"].setdefault(event["serial"], []).append(
                (move, turn, event.get("hp", 0), player))
        elif kind == "entered":
            context["entered"].setdefault(event["serial"], (move, turn, player))
        elif kind == "prize":
            context["prize"].append((move, turn, player, event.get("count", 1)))
        elif kind == "hp":
            context["hp"].setdefault(event["serial"], []).append(
                (move, turn, event["value"], bool(event.get("counter")),
                 event.get("before")))
        elif kind == "hand_add":
            context["hand_add"].append((move, turn, player, event["serial"]))
        elif kind == "ko":
            context["ko_records"].setdefault(event["serial"], []).append(
                (move, turn, event.get("prizes", 0), event.get("taker"), player))
        elif kind == "hand_lost":
            context["hand_lost"].setdefault(event["serial"], []).append(move)
        elif kind == "used_serial":
            if event.get("log") == LOG_PLAY and event.get("serial"):
                context["played_at"].setdefault(event["serial"], []).append((move, turn))
            elif event.get("log") == LOG_EVOLVE and event.get("target"):
                if event.get("serial"):
                    # Stack membership for the prizes_donated same-stack exclusion: a KO'd
                    # evolved stack emits one departure per member at the same move, and
                    # without this map the row would mask ITSELF as a "simultaneous" KO.
                    context["evolution_parent"][event["serial"]] = event["target"]
                # WHEN, not just whether: `evolved` is a consequence of a bench play, so an
                # evolution that happened BEFORE the play must not count for it. A set had
                # no time and the bit was the only one of the three without a `> move`
                # filter (2026-08-06 audit; 0/508 today, but it fires the moment L3 covers
                # EVOLVE rows).
                context["evolved_from"].setdefault(event["target"], []).append(move)
        if kind in ("play", "ability", "declare_attack", "retreat"):
            context["action_moves"].append(move)
    context["action_moves"].sort()
    scan_high = record.get("v23_ko_scan_high")
    context["ko_scan_high"] = int(scan_high) if scan_high is not None else -1
    context["ko"] = _ko_index(context)
    context["turn_owner"] = {int(turn): int(seat) for turn, seat
                             in (record.get("turn_owner") or {}).items()}
    context["energy_moves"], context["energy_holders"] = {0: [], 1: []}, {0: [], 1: []}
    for move, mover, holders in (record.get("v23_energy_state") or ()):
        context["energy_moves"][mover].append(move)
        context["energy_holders"][mover].append(holders)
    return context


def _ko_index(context):
    """{serial: [(move, turn), ...]} -- the departures that were KNOCK-OUTS, in move order.

    THE KO TEST IS THE PRIZE JOIN, not the recorded HP (2026-08-06 audit). `left.hp` is the
    target's HP at the LAST BOARD SCAN BEFORE the lethal hit, so a Pokemon killed by an
    attack still reads 70/90/110 there and `hp <= 0` is False -- the old test only ever
    caught damage-counter kills on BENCHED Pokemon, which sit at 0 until removal. Measured
    over four independent samples it found 11-25% of real KOs and shipped class-4 "never"
    for the rest, inverting `ko_delay`, `decisive`, `donated` and `prizes_donated`.

    The join ("the OTHER side took a prize within KO_JOIN_MOVES moves of this departure")
    is the same primitive `train_ppo._is_ko` already uses for the v21/v22 KO clocks, which
    is why those heads were never affected. A pure HP-ledger test agreed with it on
    231/231 and 253/253 departures, and no `hp <= 0` departure lacked a prize join, so the
    old test was a strict subset with no compensating path.

    ENGINE RECORDS FIRST (build_v25): where the dump's cumulative KO record was scanned
    (`ko_scan_high` covers the departure), the record IS the answer -- the join's known
    false-positive class (a bounce within the window of an unrelated same-side prize take)
    cannot occur. The join remains the fallback for departures past the last scan (the
    terminal scan normally leaves none) and for whole games on older engines."""
    knockouts = {}
    for serial, records in context["ko_records"].items():
        for move, turn, _prizes, _taker, _victim in records:
            knockouts.setdefault(serial, []).append((move, turn))
    scan_high = context.get("ko_scan_high", -1)
    prize_moves = {0: [], 1: []}
    for move, _turn, player, _count in context["prize"]:
        prize_moves[player].append(move)
    for moves in prize_moves.values():
        moves.sort()
    for serial, departures in context["left"].items():
        for move, turn, _hp, player in departures:
            if move <= scan_high:
                continue                     # engine truth covered this departure
            opponent = prize_moves[1 - player]
            if bisect.bisect_right(opponent, move + V23_KO_JOIN_MOVES) > \
                    bisect.bisect_left(opponent, move - V23_KO_JOIN_MOVES):
                knockouts.setdefault(serial, []).append((move, turn))
    for entries in knockouts.values():
        entries.sort()
    return knockouts


def _energy_state(context, mover, move):
    """The mover's engine energy state as recorded at its latest MAIN select at or before
    `move`; None when none preceded it (or the dump carried no attackCost)."""
    moves = context["energy_moves"].get(mover) or ()
    position = bisect.bisect_right(moves, move) - 1
    return context["energy_holders"][mover][position] if position >= 0 else None


def _resolution_end(context, move):
    """(move, end] -- everything the engine resolved for the action taken at `move`, i.e. up to
    the next deliberate action of either seat. Mechanical: no fixed move budget."""
    moves = context["action_moves"]
    position = bisect.bisect_right(moves, move)
    return moves[position] if position < len(moves) else float("inf")


# ======================================================================================= #
# L1 -- attachment need
# ======================================================================================= #

def _spans(context, energy_serial, owner, from_move):
    """[(holder bottom serial, attached at, detached at)] for this energy from `from_move` on.
    Credit follows the SERIAL, so an energy moved between our Pokemon keeps accruing."""
    spans = []
    for move, _turn, holder, event_owner, _card in context["attach"].get(energy_serial, ()):
        if event_owner != owner or move < from_move:
            continue
        end = float("inf")
        for lost_move, _lost_turn, lost_holder, _lost_owner in context["lost"].get(
                energy_serial, ()):
            if lost_holder == holder and lost_move > move:
                end = lost_move
                break
        spans.append((holder, move, end))
    return spans


def _satisfies(context, energy_serial, energy_type, entry=None):
    """Can this energy card satisfy a declared "any {X} Energy attached" condition?

    ENGINE-RESOLVED TYPES FIRST (build_v25 `energyTypes`, index-parallel to energyOrder):
    the engine's own test is (provided & wanted) != 0, and a colorless-only special is
    mask 0 -- which never satisfies a colored condition. This kills the wildcard class
    (2026-08-06 energy audit F4): a Double Turbo attached before the real Dark on
    Munkidori stole the Adrena-Brain credit and the real Dark could label WASTE. The
    card-table path remains the fallback for older-engine records, where special
    energies stay permissive (the rule may only ever cost recall, never invent waste)."""
    if entry is not None and len(entry) > 2 and entry[2]:
        order = entry[0]
        if energy_serial in order:
            position = order.index(energy_serial)
            if position < len(entry[2]):
                wanted = _ORDINAL_TO_MASK.get(energy_type)
                if wanted:                   # a colored condition with a known mask
                    return bool(entry[2][position][0] & wanted)
    card_id = next((event[4] for event in context["attach"].get(energy_serial, ())
                    if event[4]), None)
    if card_id is None:
        return True
    provided = _energy_card_type(card_id)
    return provided is None or provided == energy_type


def _fills_condition(context, energy_serial, energy_type, holders, holder):
    """Does THIS energy occupy the ONE slot a declared "any {X} Energy attached" condition
    consumes?

    The condition requires one energy of the declared type, so exactly one attached card
    earns the credit -- the engine's own first matching card in `energyOrder`. Crediting
    every matching card (2026-08-06 audit) let a second and third Dark on Munkidori ride
    Adrena-Brain's condition and read as needed: 17/71 condition uses over-credited.

    Falls back to the plain type test when no engine energy order was recorded, because the
    credit rule may only ever cost recall -- never manufacture waste."""
    entry = (holders or {}).get(holder)
    if not _satisfies(context, energy_serial, energy_type, entry):
        return False
    order = entry[0] if entry else None
    if not order or energy_serial not in order:
        return True
    for candidate in order:
        if _satisfies(context, candidate, energy_type, entry):
            return candidate == energy_serial
    return True


def _used_attacks(context, owner, holder):
    """Every attack id this holder's stack used, whole game."""
    return {attack_id for _move, _turn, player, serial, attack_id in context["attack"]
            if player == owner and serial == holder}


def _supports(subset, types, cost):
    """Can these cards' resolved units pay this one attack's printed cost? Typed slots
    first (descending mask), colorless (mask 0) last accepting any unit; backtracking over
    the tiny unit/slot counts (<= ~8 units, <= ~5 slots)."""
    units = []
    for position in subset:
        mask, count = types[position]
        units.extend([mask] * max(1, count))
    slots = sorted((_ORDINAL_TO_MASK.get(ordinal, 0) for ordinal in cost), reverse=True)
    if len(units) < len(slots):
        return False

    def place(index, remaining):
        if index == len(slots):
            return True
        wanted = slots[index]
        tried = set()
        for k, mask in enumerate(remaining):
            if mask in tried:
                continue
            tried.add(mask)
            if wanted == 0 or (mask & wanted):
                if place(index + 1, remaining[:k] + remaining[k + 1:]):
                    return True
        return False

    return place(0, tuple(units))


def _minimal_fill(order, types, costs):
    """The positions of the SMALLEST sub-multiset of the attached cards supporting EVERY
    cost simultaneously (attacks overlap -- they do not spend -- so their costs max
    rather than add), earliest-attached preferred among equals (the design doc's fill:
    "energy #1 needed, #2 flagged"). None when even the full set cannot support them all
    (cannot happen for costs pre-filtered to payable)."""
    for size in range(len(order) + 1):
        for subset in itertools.combinations(range(len(order)), size):
            if all(_supports(subset, types, cost) for cost in costs):
                return set(subset)
    return None


def _attack_fill_credited(context, owner, holder, entry, energy_serial, declared_prefix,
                          stats):
    """Was this energy in the fill at this exercised attack event? -- AUX_V23_DESIGN's
    smallest-multiset semantics (2026-08-06 energy audit F3).

    The per-event MARGINAL test (the engine prefix shortfall dropping at this card) cannot
    see cross-attack double duty: with attacks {P} and {C} both used and [Fire, Psychic]
    attached, Fire's marginal at the {C} event credits it although Psychic alone supports
    both. The joint fill runs over every used attack PAYABLE from the current attachments
    (engine fact: prefix[-1] == 0; the declared attack is always among them). Costs are
    PRINTED, guarded per attack by the T2 identity (printed unit total == engine
    prefix[0]; the pool exercises zero cost modifiers) -- a mismatch, a missing card-table
    cost, or a record without resolved energyTypes all fall back to the engine's own
    marginal test, counted."""
    order, prefixes = entry[0], entry[1]
    position = order.index(energy_serial)
    marginal = (position + 1 < len(declared_prefix)
                and declared_prefix[position + 1] < declared_prefix[position])
    types = entry[2] if len(entry) > 2 else ()
    if not types or len(types) != len(order):
        return marginal                     # pre-v25 engine: the certified marginal path
    costs = []
    for attack_id in _used_attacks(context, owner, holder):
        prefix = prefixes.get(attack_id)
        if not prefix or prefix[-1] != 0:
            continue                        # not payable from the CURRENT attachments
        cost = _printed_cost_multiset(attack_id)
        if cost is None or len(cost) != prefix[0]:
            stats["v23_cost_tripwire"] += 1
            return marginal
        costs.append(cost)
    if not costs:
        return marginal
    fill = _minimal_fill(order, types, costs)
    if fill is None:
        stats["v23_joint_fill_infeasible"] += 1
        return marginal
    return position in fill


def _relocated(context, energy_serial, move):
    """Was this energy's departure at `move` a RELOCATION rather than consumption?

    `energy_lost` comes from the board diff and carries no destination, so a card MOVED to
    another of our Pokemon looked exactly like one discarded as a cost -- and the own-turn
    rule then credited the source holder for an energy that is still in play. The move is
    identifiable without any new event: the same serial re-appears as an `attach_seen` at
    the same moment on a different holder. Mechanism-keyed, no card knowledge."""
    return any(attach_move == move
               for attach_move, _turn, _holder, _owner, _card
               in context["attach"].get(energy_serial, ()))


def energies_attached_by(context, holder_bottom, owner, move):
    """Energy serials this decision put on `holder_bottom` -- the TARGET-FIRST resolution.

    Crispin-style prompts name the energy up front (`contextCard`), so option_descriptors
    can record its serial at decision time. Glass Trumpet / Rosa's Encouragement / Iono's
    Bellibolt ex prompt the other way round: they ask which Pokemon receives, and the engine
    draws the energy from the discard AFTERWARDS. At decision time the card does not exist
    yet -- `contextCard` is None and `looking` is empty -- so the serial can only be read
    back from the engine's own attach events once they have happened.

    Matched on (holder, owner, move): the attach the engine records at or after the move we
    answered, before any later move. Returns every energy that landed, because a single such
    option can attach more than one (Rosa's: "up to 2 Basic Energy to 1 Pokemon").
    """
    # `> move`, not `>=`: `move` is the index this decision was ASKED at, and the engine
    # stamps the logs it produces in response with move + 1, so an attach recorded AT `move`
    # belongs to the PREVIOUS action. Bounded above by the resolution of this action, so a
    # later manual attach onto the same holder can never be stolen (2026-08-06 audit).
    end = _resolution_end(context, move)
    found = []
    for energy_serial, events in context["attach"].items():
        for event_move, _turn, holder, event_owner, _card in events:
            if holder == holder_bottom and event_owner == owner \
                    and move < event_move <= end:
                found.append((event_move, energy_serial))
                break
    if not found:
        return ()
    first = min(event_move for event_move, _serial in found)
    return tuple(serial for event_move, serial in found if event_move == first)


def attachment_need_of(context, energy_serials, owner, attach_move, stats):
    """`beyond_need` for the ATTACHMENT an option performed -- the three-way rule lifted
    over the set of energies it attached, or None (MASKED).

        CREDITED (0) -- ANY of them was credited
        WASTE    (1) -- none credited and at least one was exercised
        MASKED (None) -- none exercised

    For the one-energy case (every manual attach and every energy-first effect attach) a
    lift over a singleton IS that singleton, so this is byte-identical to calling
    attachment_need directly -- it generalises rather than special-cases. The row is an
    OPTION row and the option's consequence is every energy it lands, so labelling the
    attachment rather than one card is the question the head is actually being asked.
    """
    exercised_any, credited_any = False, False
    for energy_serial in energy_serials:
        need = attachment_need(context, energy_serial, owner, attach_move, stats)
        if need is None:
            continue                       # that serial saw no exercised event
        exercised_any = True
        if need == 0.0:
            credited_any = True
    if not exercised_any:
        return None
    return 0.0 if credited_any else 1.0


def attachment_need(context, energy_serial, owner, attach_move, stats):
    """`beyond_need` for one attachment, or None (MASKED).

    The three-way evidence rule per serial (AUX_V23_DESIGN.md L1), destination-agnostic:
      CREDITED (0) -- it filled a cost slot at some exercised event of its holder (the
                      ENGINE's own attach-order fill: prefix[k+1] < prefix[k]), satisfied a
                      declared attach-energy condition at a use of that skill/attack, or was
                      consumed by our OWN retreat / effect / attack discard.
      WASTE    (1) -- present-but-unneeded at >= 1 exercised event and never credited.
      MASKED (None) -- present at ZERO exercised events: the window was cut short (opponent
                      removal, KO, game end) before any evidence could accrue. How it left
                      play is irrelevant -- a KO does NOT rescue a proven surplus.
    """
    by_card, by_attack = _table()
    credited, exercised = False, 0
    previous_end = None
    for holder, start, end in _spans(context, energy_serial, owner, attach_move):
        if previous_end is not None and start != previous_end:
            # F6 (2026-08-06 energy audit): a re-attach AFTER a removal is a NEW
            # attachment decision -- its evidence must neither rescue nor convict THIS
            # row (a cut-short window is MASKED, per the evidence rule). A same-move
            # relocation (Energy Switch class) continues the chain.
            break
        previous_end = end
        for move, _turn, player, serial, attack_id in context["attack"]:
            if player != owner or serial != holder or not start < move <= end:
                continue
            holders = _energy_state(context, owner, move - 1)
            entry = (holders or {}).get(holder)
            prefix = (entry[1].get(attack_id) if entry else None)
            if not prefix or energy_serial not in entry[0]:
                stats["v23_attack_no_engine_state"] += 1
                continue                       # no engine evidence for this event: not counted
            exercised += 1
            if _attack_fill_credited(context, owner, holder, entry, energy_serial,
                                     prefix, stats):
                credited = True
            condition = by_attack.get(attack_id)
            if condition is not None and _fills_condition(context, energy_serial, condition,
                                                          holders, holder):
                credited = True
        for move, _turn, player, serial, card_id in context["ability"]:
            if player != owner or serial != holder or not start < move <= end:
                continue
            condition = by_card.get(card_id)
            if condition is None:
                continue
            # Same presence guard as the attack branch (2026-08-06 energy audit, F5):
            # without it a recovered-and-replayed holder could exercise -- or via
            # _fills_condition's permissive missing-state path even CREDIT -- an energy
            # that left play with its first incarnation. No engine evidence = not counted,
            # which fails toward MASK, never toward WASTE.
            holders = _energy_state(context, owner, move - 1)
            entry = (holders or {}).get(holder)
            if entry is None or energy_serial not in entry[0]:
                stats["v23_ability_no_engine_state"] += 1
                continue
            exercised += 1
            if _fills_condition(context, energy_serial, condition, holders, holder):
                credited = True
        for move, turn, lost_holder, lost_owner in context["lost"].get(energy_serial, ()):
            if lost_holder != holder or lost_owner != owner or not start < move <= end:
                continue
            if _relocated(context, energy_serial, move):
                continue          # moved to another of our Pokemon: still in play, not spent
            # OUR OWN consumption (retreat / effect / attack discard) is credit; the opponent
            # removing it destroys FUTURE evidence but creates none. A turn with no recorded
            # owner (setup / the final turn: 36/853, no interior gaps) is NOT evidence either
            # way, so it is skipped rather than read as the opponent's doing.
            if context["turn_owner"].get(turn) == owner:
                credited, exercised = True, exercised + 1
    if credited:
        return 0.0
    return None if exercised == 0 else 1.0


def already_covered(context, holder_serial, owner, move):
    """1 if the costs of every attack the holder ACTUALLY WENT ON TO USE were already payable
    BEFORE this attach -- read straight off the engine shortfall recorded AT this very attach
    select, so nothing is reconstructed. -1 when no engine state was recorded there.

    Hindsight-consistent: attaching toward a bigger attack that WAS later used is not
    penalised, because only the attacks actually used enter the test.

    MASKED (-1) when the holder never used an attack afterwards (owner 2026-08-03). The
    vacuous truth -- "every attack it used was payable" over an empty set -- is not evidence
    about this attach; it is the same window-cut-short situation `beyond_need` masks, and
    labelling it 1 taught the head that a never-exercised attach is a covered attach."""
    entry = (_energy_state(context, owner, move) or {}).get(holder_serial)
    if entry is None:
        return -1.0
    used = {attack_id for event_move, _turn, player, serial, attack_id
            in context["attack"]
            if player == owner and serial == holder_serial and event_move > move}
    if not used:
        return -1.0
    covered = 1.0
    for attack_id in used:
        prefix = entry[1].get(attack_id)
        if not prefix:
            return -1.0
        if prefix[-1] > 0:
            covered = 0.0
    return covered


# ======================================================================================= #
# L2 -- placement conversion
# ======================================================================================= #

def _ko_after(context, serial, move):
    """(move, turn) of the first KNOCK-OUT of this stack after `move`, or None.

    Two fixes over the original (2026-08-06 audit), both in these three lines:
      * the KO test is the prize join (see _ko_index), not the stale `left.hp`;
      * a non-KO departure is SKIPPED rather than ending the scan. `context["left"]` is a
        list per serial precisely because a serial can leave and return (a bounce, then a
        replay and a real KO); returning None at the first bounce lost that KO."""
    for ko_move, ko_turn in context["ko"].get(serial, ()):
        if ko_move > move:
            return (ko_move, ko_turn)
    return None


def ko_delay(context, serial, move, turn):
    """{0 this round, 1 next round, 2, 3 (3+ rounds), 4 never} until this target's stack
    was KO'd. A property of the TARGET, so every candidate row carries it.

    ROUNDS (`// 2`), not raw engine turns (2026-08-06 audit): turns increment per PLAYER
    turn and the placer's targets die on the placer's own turns, so raw-turn deltas are
    almost always even -- class 1 was structurally near-empty and class 3 absorbed
    everything past 1.5 rounds. Width stays 5, so checkpoints load; the head retrains its
    class semantics on restart."""
    knocked_out = _ko_after(context, serial, move)
    if knocked_out is None:
        return V23_KO_DELAY_CLASSES - 1
    return min(max(0, knocked_out[1] - turn) // 2, V23_KO_DELAY_CLASSES - 2)


def decisive(context, serial, move):
    """1 if THIS decision's damage counters converted the eventual KO-ing hit from a non-KO
    into a KO: at the target's KO, finishing damage D < h + c, where h is the target's HP just
    before the finishing hit and c is this decision's counter contribution. Pure HP-ledger
    arithmetic on observed quantities -- a single-intervention counterfactual; opponent
    adaptation is not computable from one trajectory and stays out of it.

    -1 (masked) when this decision placed no counters on this target, or the ledger does not
    close (no finishing hit recorded, or no observed pre-hit HP)."""
    # THIS decision's counters only. The engine STREAMS a placement (measured 2026-08-05:
    # 45/50 chains apply the accepted counter immediately and the next select already shows
    # hp 10 lower), and answering the select at `move` stamps its logs with `move + 1` --
    # so this decision's contribution is exactly the counters at move + 1. The old window
    # ran to the next DELIBERATE action, which for a six-select chain is past the whole
    # chain: every select in it was credited with all six counters, inflating the earliest
    # rows ~6x (2026-08-06 audit; 3/87 contribution-bearing rows today, and it grows as a
    # trained policy concentrates counters). Counters that arrive later than move + 1 now
    # yield contribution 0 -> MASKED, which is the safe failure: absent evidence, not
    # invented evidence.
    end = min(move + 1, _resolution_end(context, move))
    ledger = context["hp"].get(serial) or ()
    contribution = sum(abs(value) for hp_move, _turn, value, counter, _before in ledger
                       if counter and move < hp_move <= end)
    if contribution <= 0:
        return -1.0
    knocked_out = _ko_after(context, serial, move)
    if knocked_out is None:
        return 0.0                       # it never died: these counters converted nothing
    # DAMAGE only: HP_CHANGE is negative for damage and POSITIVE for heals, so an
    # unsigned test let a heal masquerade as the finishing hit.
    finishing = [entry for entry in ledger
                 if entry[0] <= knocked_out[0] and entry[2] and entry[2] < 0
                 and entry[0] > move]
    if not finishing:
        return -1.0
    damage, before = abs(finishing[-1][2]), finishing[-1][4]
    if before is None:
        return -1.0
    return float(damage < before + contribution)


def wasted_counters(context, serial, move):
    """1 if THIS decision's counters were SURPLUS to the target's eventual KO: the
    overkill s at death (how far below zero the ledger closes) covers the whole
    contribution c, so removing these counters changes nothing -- which includes the
    observed placed-onto-an-already-dead-target case, where every counter lands in the
    negatives. 0 when the KO consumed them (s < c: without them the kill fails or
    shrinks). -1 masked: no counters placed by this decision, the target never died, or
    the ledger does not close on the death.

    decisive's sibling, same machinery, same single-intervention convention (remove ONE
    decision's counters, everything else realized as played): decisive marks the
    conversion, this marks the excess -- the pair is the full cost/benefit of a
    placement. Pure realized arithmetic; no one's choices are counterfactualed."""
    end = min(move + 1, _resolution_end(context, move))
    ledger = context["hp"].get(serial) or ()
    contribution = sum(abs(value) for hp_move, _turn, value, counter, _before in ledger
                      if counter and move < hp_move <= end)
    if contribution <= 0:
        return -1.0
    knocked_out = _ko_after(context, serial, move)
    if knocked_out is None:
        return -1.0
    closing = [entry for entry in ledger if entry[0] <= knocked_out[0]]
    if not closing or closing[-1][4] is None:
        return -1.0
    final_hp = closing[-1][4] + (closing[-1][2] or 0)
    if final_hp > 0:
        return -1.0                       # a death the ledger cannot account for: mask
    return float(-float(final_hp) >= contribution)


# ======================================================================================= #
# L3 -- bench contribution
# ======================================================================================= #

def _stack_root(context, serial):
    """The bottom serial of the evolution stack this card belongs to (itself if unevolved)."""
    parent = context["evolution_parent"]
    seen = set()
    while serial in parent and serial not in seen:
        seen.add(serial)
        serial = parent[serial]
    return serial


def _own_turns_between(context, owner, from_turn, to_turn):
    """How many of `owner`'s turns lie in (from_turn, to_turn]."""
    return sum(1 for turn in range(from_turn + 1, to_turn + 1)
               if context["turn_owner"].get(turn) == owner)


def fetch_delay(context, fetched, owner, turn):
    """{0..2, 3 (3+), 4 never} OWN turns from this decision until any fetched card was
    played; -1 masked. The v25 replacement for the binary fetched_card_used bit, which is
    ~68% "yes, eventually" and cannot express "needed the fetch NOW" (the Meowth case).

    Masked when: nothing was fetched; a never-played fetch LEFT the hand unplayed
    (opponent disruption or our own discard cost -- either way the rot evidence is
    destroyed); or the game ended inside the runway (a "never" with no opportunity is a
    cut-short window, the L1 masking convention, not evidence)."""
    if not fetched:
        return -1
    played_turns = [play_turn
                    for fetch_move, fetched_serial in fetched
                    for play_move, play_turn in context["played_at"].get(fetched_serial, ())
                    if play_move > fetch_move]
    if played_turns:
        return min(_own_turns_between(context, owner, turn, min(played_turns)),
                   V23_FETCH_NEVER - 1)
    if any(lost_move > fetch_move
           for fetch_move, fetched_serial in fetched
           for lost_move in context["hand_lost"].get(fetched_serial, ())):
        return -1
    last_turn = max(context["turn_owner"], default=None)
    if last_turn is None \
            or _own_turns_between(context, owner, turn, last_turn) < V23_FETCH_RUNWAY:
        return -1
    return V23_FETCH_NEVER


def bench_contribution(context, serial, owner, move, turn):
    """(the 5 bits, prizes_donated, fetch_delay) for a Pokemon this decision put into play.

    The channels are SPLIT rather than OR'd (owner review): an ability-fetcher that fires once
    and then dies for 2 prizes without its fetch ever being played reads out as the generic
    signature (used_ability=1, fetched_card_used=0, prizes_donated=2) with no card named."""
    attacked = any(player == owner and event_serial == serial and event_move > move
                   for event_move, _turn, player, event_serial, _attack in context["attack"])
    ability_moves = [event_move for event_move, _turn, player, event_serial, _card
                     in context["ability"]
                     if player == owner and event_serial == serial and event_move > move]
    used_ability = bool(ability_moves)
    evolved = any(event_move > move
                  for event_move in context["evolved_from"].get(serial, ()))
    knocked_out = _ko_after(context, serial, move)
    donated = knocked_out is not None and not (attacked or evolved or used_ability)
    # The fetch window is the resolution of the PLAY *and* of every ability this Pokemon
    # itself went on to use. AUX_V23_DESIGN's own motivating case is Meowth ex -- benched,
    # fires its ability once, dies for 2 prizes with the fetched supporter still in hand --
    # and that fetch happens in the ABILITY's window, not the play's. Reading only the play
    # window left the bit masked on 506/506 bench rows and it never once emitted a metric
    # in 676 iterations (2026-08-06 audit).
    windows = [(move, _resolution_end(context, move))]
    windows += [(ability_move, _resolution_end(context, ability_move))
                for ability_move in ability_moves]
    fetched = [(entry[0], entry[3]) for entry in context["hand_add"]
               if entry[2] == owner
               and any(low < entry[0] <= high for low, high in windows)]
    bits = np.array([int(attacked), int(evolved), int(used_ability), int(donated),
                     int(any(serial_ in context["played_at"] for _move, serial_ in fetched))
                     if fetched else -1], dtype=np.int8)
    prizes = -1
    if knocked_out is not None:
        # ENGINE RECORD FIRST (build_v25): per-victim prize attribution is exact, so the
        # merged-pile-diff problem and the simultaneous-KO mask below cannot arise -- each
        # victim of a double-KO carries its own count. `prizeCount` is the AWARD at the KO
        # site (0 under no-prize effects -- which finally makes class 0 reachable); the
        # physical take can be smaller only when the taker's pile ran out, i.e. the game
        # ended on this KO, where the award is still the right cost semantics.
        records = [entry for entry in context["ko_records"].get(serial, ())
                   if entry[0] == knocked_out[0]]
        if records:
            prizes = min(records[0][2], V23_PRIZE_CLASSES - 1)
        else:
            taken = [count for prize_move, _turn, player, count in context["prize"]
                     if player != owner and abs(prize_move - knocked_out[0]) <= 2]
            # JOIN FALLBACK (older engines / past the last scan). ANOTHER of our stacks
            # KO'd at the same move makes the pile-diff count unattributable -> mask. The
            # KO test is the prize join (context["ko"]), NEVER the recorded departure hp:
            # that hp is the target's HP at the last board scan BEFORE the lethal hit, so
            # an attack-killed Pokemon reads 70/90/110 there and the old
            # `departure[2] <= 0` guard missed it -- a Phantom Dive double-KO labelled the
            # counter-killed 1-prize basic with the MERGED count. And a KO'd evolved stack
            # emits one departure per member at the same move, so without the same-stack
            # exclusion the row masks ITSELF on every evolved victim.
            root = _stack_root(context, serial)
            simultaneous = sum(
                1 for other, knockouts in context["ko"].items()
                if _stack_root(context, other) != root
                and any(ko_move == knocked_out[0] for ko_move, _ko_turn in knockouts)
                and any(departure[3] == owner and departure[0] == knocked_out[0]
                        for departure in context["left"].get(other, ())))
            if len(taken) == 1 and simultaneous == 0:
                prizes = min(taken[0], V23_PRIZE_CLASSES - 1)
    return bits, prizes, fetch_delay(context, fetched, owner, turn)


# ======================================================================================= #
# T1 / T2 -- per-token engine facts read at the decision itself
# ======================================================================================= #

def token_labels(meta):
    """(blocked [T], cost delta [T, ATTACK_SLOTS]) in meta['board'] emission order. -1 masked.

    Not hindsight: both are the engine's resolved CURRENT state, which the observation does not
    expose -- `noAbility` (T1) and the effective-minus-printed attack cost (T2)."""
    # int8 throughout: every v23 label is a bit, a small class index, or -1 (masked).
    # Same call the v22 grids make -- the losses cast only the masked slice they use.
    blocked = np.full(V23_MAX_BOARD_TOKENS, -1, dtype=np.int8)
    delta = np.full((V23_MAX_BOARD_TOKENS, V23_ATTACK_SLOTS), -1, dtype=np.int8)
    state = meta.get("v23_state")
    if not state:
        return blocked, delta
    by_serial = {entry[0]: entry for entry in state}
    for position, (serial, _owner, _hp) in enumerate(
            (meta.get("board") or ())[:V23_MAX_BOARD_TOKENS]):
        entry = by_serial.get(serial)
        if entry is None:
            continue
        blocked[position] = int(entry[1])
        for slot, (attack_id, effective) in enumerate(entry[2][:V23_ATTACK_SLOTS]):
            if not attack_id or effective < 0:
                continue
            printed = _printed_cost(attack_id)
            delta[position, slot] = int(
                np.clip(effective - printed, -V23_DELTA_ZERO, V23_DELTA_ZERO)
                + V23_DELTA_ZERO)
    return blocked, delta


# ======================================================================================= #
# Per-decision label block
# ======================================================================================= #

def labels(meta, chosen, context, mover, stats):
    """The v23 label block of ONE decision, keyed the way `collate_v23` reads it."""
    rows = meta.get("v23_rows") or ()
    width = min(len(rows), V23_MAX_OPTION_ROWS)
    move, turn = meta["move"], meta["turn"]
    counter_select = meta.get("v23_context") in COUNTER_CONTEXTS

    attach = np.full((V23_MAX_OPTION_ROWS, V23_L1_BITS), -1, dtype=np.int8)
    delay = np.full(V23_MAX_OPTION_ROWS, -1, dtype=np.int8)
    converted = np.full(V23_MAX_OPTION_ROWS, -1, dtype=np.int8)
    bench = np.full((V23_MAX_OPTION_ROWS, V23_L3_BITS), -1, dtype=np.int8)
    prizes = np.full(V23_MAX_OPTION_ROWS, -1, dtype=np.int8)
    fetch = np.full(V23_MAX_OPTION_ROWS, -1, dtype=np.int8)
    wasted = np.full(V23_MAX_OPTION_ROWS, -1, dtype=np.int8)

    if counter_select:
        for row in range(width):
            kind, primary, _secondary, _card_id = rows[row]
            if kind == ROW_TARGET:
                delay[row] = ko_delay(context, primary, move, turn)
                stats["v23_ko_delay_rows"] += 1
    if chosen is not None and 0 <= chosen < width:
        kind, primary, secondary, _card_id = rows[chosen]
        if kind == ROW_ATTACH:
            # primary == 0 is the TARGET-FIRST sentinel option_descriptors emits when the
            # engine had not chosen the energy yet; resolve it from the engine's own attach
            # events now that they exist. Everything else already names its energy.
            if primary:
                serials = (primary,)
            else:
                serials = energies_attached_by(context, secondary, mover, move)
                stats["v23_attach_deferred"] += 1
                stats["v23_attach_deferred_unresolved"] += int(not serials)
                stats["v23_attach_deferred_multi"] += int(len(serials) > 1)
            need = attachment_need_of(context, serials, mover, move, stats)
            if need is None:
                stats["v23_attach_masked"] += 1
            else:
                attach[chosen, 0] = int(need)
                stats["v23_attach_waste"] += int(need > 0)
                stats["v23_attach_rows"] += 1
            covered = already_covered(context, secondary, mover, move)
            attach[chosen, 1] = int(covered)
            stats["v23_covered_masked"] += int(covered < 0)
        elif kind == ROW_PLAY:
            entered = context["entered"].get(primary)
            if entered is not None and entered[0] >= move:
                bench[chosen], prizes[chosen], fetch[chosen] = bench_contribution(
                    context, primary, mover, move, turn)
                stats["v23_bench_rows"] += 1
                stats["v23_fetch_rows"] += int(fetch[chosen] >= 0)
                # How many of those bench rows the KO test actually resolved. Without it,
                # prizes_donated being empty is indistinguishable from prizes_donated being
                # uninformative -- which is exactly the confusion that nearly got the head
                # deleted on 2026-08-04 (its real cause was _ko_after).
                stats["v23_prize_rows"] += int(prizes[chosen] >= 0)
                stats["v23_bench_ko_rows"] += int(
                    _ko_after(context, primary, move) is not None)
        elif kind == ROW_TARGET and counter_select:
            converted[chosen] = int(decisive(context, primary, move))
            stats["v23_decisive_rows"] += int(converted[chosen] >= 0)
            wasted[chosen] = int(wasted_counters(context, primary, move))
            stats["v23_wasted_rows"] += int(wasted[chosen] >= 0)

    blocked, cost_delta = token_labels(meta)
    stats["v23_blocked_known"] += int(bool((blocked >= 0).any()))
    stats["v23_delta_known"] += int(bool((cost_delta >= 0).any()))
    return {"v23_attach": attach, "v23_ko_delay": delay, "v23_decisive": converted,
            "v23_bench": bench, "v23_prizes": prizes, "v23_fetch": fetch,
            "v23_wasted": wasted, "v23_row_mask": _row_mask(width),
            "v23_blocked": blocked, "v23_delta": cost_delta}


def _row_mask(width):
    mask = np.zeros(V23_MAX_OPTION_ROWS, dtype=bool)
    mask[:width] = True
    return mask


# ======================================================================================= #
# Spec layer 3: sampled per-decision cross-checks (the v21 / v22 pattern)
# ======================================================================================= #

def _departed_by_ko(context, serial, move):
    """Did this stack leave the board as a KO after `move`, re-derived from the RAW streams?

    Deliberately NOT a call to _ko_after / context["ko"]: this is the independent side of a
    cross-check, so it walks `ko_records` / `left` / `prize` itself, mirroring the index's
    engine-first-join-fallback split with its own linear scans. If it ever disagrees with
    the index, the index's CONSTRUCTION is wrong -- which is the whole point."""
    for record_move, _turn, _prizes, _taker, _victim in context["ko_records"].get(serial, ()):
        if record_move > move:
            return True
    scan_high = context.get("ko_scan_high", -1)
    for left_move, _turn, _hp, player in context["left"].get(serial, ()):
        if left_move <= move or left_move <= scan_high:
            continue
        for prize_move, _prize_turn, prize_player, _count in context["prize"]:
            if prize_player == 1 - player \
                    and abs(prize_move - left_move) <= V23_KO_JOIN_MOVES:
                return True
    return False


def _join_departed(context, serial, move):
    """The PURE prize-join answer, engine records ignored -- only for the informational
    join-vs-engine divergence counter (each count = a game state where the old inference
    would have mislabelled and the engine record corrected it)."""
    for left_move, _turn, _hp, player in context["left"].get(serial, ()):
        if left_move <= move:
            continue
        for prize_move, _prize_turn, prize_player, _count in context["prize"]:
            if prize_player == 1 - player \
                    and abs(prize_move - left_move) <= V23_KO_JOIN_MOVES:
                return True
    return False


def cross_checks(meta, context, mover, entry, stats, chosen=None):
    """Relations between a decision's v23 labels and the raw event stream, on a sampled
    stride. Counters only -- a violation is COUNTED (`v23_bad_*`), never silently corrected,
    so a defect shows up in the iteration line instead of in the weights."""
    rows = meta.get("v23_rows") or ()
    move = meta["move"]
    for row, (kind, primary, _secondary, _card_id) in enumerate(rows[:V23_MAX_OPTION_ROWS]):
        if entry["v23_ko_delay"][row] >= 0:
            stats["v23_check_ko_delay"] += 1
            # Re-derived from the RAW departure + prize streams, not from _ko_after: this
            # tripwire used to call the very function it was meant to police, which made it
            # a tautology that could never fire (2026-08-06 audit).
            never = entry["v23_ko_delay"][row] == V23_KO_DELAY_CLASSES - 1
            if never != (not _departed_by_ko(context, primary, move)):
                stats["v23_bad_ko_delay"] += 1
            if context.get("ko_scan_high", -1) >= 0 \
                    and _join_departed(context, primary, move) != _departed_by_ko(
                        context, primary, move):
                # informational, not bad_: the join inference disagreeing with engine
                # truth is the false-positive class the engine record exists to kill
                stats["v23_join_vs_engine"] += 1
        if entry["v23_bench"][row, 0] >= 0:
            stats["v23_check_bench"] += 1
            # donated excludes every contribution channel, by construction
            if entry["v23_bench"][row, 3] > 0 and float(entry["v23_bench"][row, :3].max()) > 0:
                stats["v23_bad_bench_donated"] += 1
            if kind != ROW_PLAY:
                stats["v23_bad_bench_kind"] += 1
        if entry["v23_attach"][row, 0] >= 0:
            stats["v23_check_attach"] += 1
            if kind != ROW_ATTACH:
                stats["v23_bad_attach_kind"] += 1
        # Descriptor integrity on the row that was actually TAKEN, whether or not a label
        # survived on it. Under the old `>= 0` gate this could not catch what it documents
        # -- a label exists only if the serial had attach spans, so "labelled but never
        # attached" was unreachable (2026-08-06 audit). It must stay scoped to the CHOSEN
        # row: an ATTACH candidate the model did NOT pick names an energy that correctly
        # never attaches, so checking every row just counts roads not taken. The
        # target-first sentinel (primary == 0, resolved later from the engine's own attach
        # events) is skipped -- it names no serial by construction.
        if row == chosen and kind == ROW_ATTACH and primary:
            stats["v23_check_attach_event"] += 1
            if primary not in context["attach"]:
                stats["v23_bad_attach_event_missing"] += 1
    blocked = entry["v23_blocked"]
    if bool((blocked >= 0).any()):
        stats["v23_check_blocked"] += 1
        if not bool(((blocked < 0) | (blocked == 0) | (blocked == 1)).all()):
            stats["v23_bad_blocked_bit"] += 1


# ======================================================================================= #
# Collate + batch checks (spec layer 2)
# ======================================================================================= #

def _stack(batch, key):
    return torch.from_numpy(np.stack([decision[key] for decision in batch]))


def collate_v23(out, batch):
    """The v23 target tensors, on top of whatever the v22 collate already built."""
    for key in ("v23_attach", "v23_ko_delay", "v23_decisive", "v23_bench", "v23_prizes",
                "v23_fetch", "v23_wasted", "v23_row_mask", "v23_blocked", "v23_delta"):
        out[key] = _stack(batch, key)
    return out


def batch_checks(batch):
    """Collated-batch invariants, vectorized and always on under --v21-checks."""
    if "v23_ko_delay" not in batch:
        return
    delay = batch["v23_ko_delay"]
    assert bool(((delay >= -1) & (delay < V23_KO_DELAY_CLASSES)).all()), \
        "v23 ko_delay class out of range"
    prizes = batch["v23_prizes"]
    assert bool(((prizes >= -1) & (prizes < V23_PRIZE_CLASSES)).all()), \
        "v23 prizes_donated class out of range"
    fetch = batch["v23_fetch"]
    assert bool(((fetch >= -1) & (fetch < V23_FETCH_DELAY_CLASSES)).all()), \
        "v23 fetch_delay class out of range"
    assert not bool((fetch[~batch["v23_row_mask"]] >= 0).any()), \
        "v23 fetch_delay label off-mask"
    delta = batch["v23_delta"]
    assert bool(((delta >= -1) & (delta < V23_DELTA_CLASSES)).all()), \
        "v23 cost delta class out of range"
    for key in ("v23_attach", "v23_decisive", "v23_bench", "v23_blocked", "v23_wasted"):
        values = batch[key]
        assert bool((((values == 0) | (values == 1)) | (values < 0)).all()), \
            f"{key} label is not a bit"
    mask = batch["v23_row_mask"]
    # no label may sit on a row that does not exist
    assert not bool((batch["v23_attach"][~mask] >= 0).any()), "v23 attach label off-mask"
    assert not bool((batch["v23_ko_delay"][~mask] >= 0).any()), "v23 ko_delay label off-mask"
    assert not bool((batch["v23_bench"][~mask] >= 0).any()), "v23 bench label off-mask"
    # decisive only ever rides a row that also carries a ko_delay (same placement select)
    assert not bool(((batch["v23_decisive"] >= 0)
                     & (batch["v23_ko_delay"] < 0)).any()), \
        "v23 decisive label without a placement target"
    assert not bool(((batch["v23_wasted"] >= 0)
                     & (batch["v23_ko_delay"] < 0)).any()), \
        "v23 wasted_counters label without a placement target"


# ======================================================================================= #
# Heads (training only -- never shipped in a bundle)
# ======================================================================================= #

class AuxHeadsV23(torch.nn.Module):
    """The v2.3 additions. L1 / L2 / L3 read the OPTION ROW representation -- the very tensor
    the policy scorer consumes, `cat([board context, option features])` -- so the concepts are
    explicit at the CHOICE, not only in the board summary. T1 / T2 read the per-token trunk
    embeddings (the v22 pathway).

    Saved as aux_v23_state, a SEPARATE key: the trunk state_dict and the v22 aux_state are
    untouched, so every existing checkpoint loads and only these heads start fresh."""

    def __init__(self, d_model=128, option_dim=0, suite="v23"):
        super().__init__()
        row_dim = d_model + option_dim
        self.components = V23_COMPONENTS
        # Not a buffer and not in the state_dict: which channels are supervised is a
        # property of the RUN, not of the weights, so a checkpoint stays loadable under any
        # --heads value.
        self.disabled = V23_DISABLED_BY_HEADS.get(suite, V23_DISABLED_BY_HEADS["v23"])
        self.row_decoder = torch.nn.Sequential(torch.nn.Linear(row_dim, d_model),
                                               torch.nn.GELU())
        self.attach = torch.nn.Linear(d_model, V23_L1_BITS)
        self.ko_delay = torch.nn.Linear(d_model, V23_KO_DELAY_CLASSES)
        self.decisive = torch.nn.Linear(d_model, 1)
        self.bench = torch.nn.Linear(d_model, V23_L3_BITS)
        self.prizes = torch.nn.Linear(d_model, V23_PRIZE_CLASSES)
        self.token_decoder = torch.nn.Sequential(torch.nn.Linear(d_model, d_model),
                                                 torch.nn.GELU())
        self.blocked = torch.nn.Linear(d_model, 1)
        self.cost_delta = torch.nn.Linear(d_model, V23_ATTACK_SLOTS * V23_DELTA_CLASSES)
        # v25's fetch_delay head is constructed ONLY under v25, so the state_dict of every
        # v23/v24 checkpoint is unchanged and still strict-loads. Its normalizer lives in
        # its OWN buffers: widening loss_scale would resize a saved buffer and break every
        # existing checkpoint's strict load (the reason V23_COMPONENTS never shrinks/grows).
        if suite == "v25":
            self.fetch_delay = torch.nn.Linear(d_model, V23_FETCH_DELAY_CLASSES)
            self.register_buffer("fetch_scale", torch.ones(()))
            self.register_buffer("fetch_seen", torch.zeros(()))
        else:
            # v26 retired fetch_delay (see V23_DISABLED_BY_HEADS) -- a v26 state_dict has
            # the wasted keys but none of the fetch keys.
            self.fetch_delay = None
        if suite in ("v25", "v26"):
            # wasted_counters: decisive's sibling on the same placement rows (owner-
            # approved 08-07) -- "these counters were surplus to the KO", the direct
            # supervision for the observed overkill/negative-counter placements.
            self.wasted = torch.nn.Linear(d_model, 1)
            self.register_buffer("wasted_scale", torch.ones(()))
            self.register_buffer("wasted_seen", torch.zeros(()))
        else:
            self.wasted = None
        # Both fetch-family forms fail their bars (fetch_delay AND the binary bit), so any
        # suite past v24 keeps fetched_card_used/donated masked out of the pooled bench
        # loss whether or not a fetch head exists to supervise the channel instead.
        self.mask_failed_bench_bits = suite in ("v25", "v26")
        self.register_buffer("loss_scale", torch.ones(len(V23_COMPONENTS)))
        self.register_buffer("loss_seen", torch.zeros(len(V23_COMPONENTS)))

    def v25_parameters(self):
        """The v25-only params -- their OWN optimizer group, so resuming a v24 checkpoint
        under v25 keeps every pre-existing Adam moment and only these start fresh."""
        parameters = [] if self.fetch_delay is None else list(self.fetch_delay.parameters())
        if self.wasted is not None:
            parameters += list(self.wasted.parameters())
        return parameters

    def base_parameters(self):
        """Every param except the v25 additions, in the same order v23/v24 built their
        optimizer group -- positional moment matching across --heads upgrades depends on it."""
        exclude = {id(parameter) for parameter in self.v25_parameters()}
        return [parameter for parameter in self.parameters() if id(parameter) not in exclude]

    def normalized(self, name, raw):
        """raw loss -> ~unit scale, dividing by its own running mean. Verbatim the v22
        normalizer (device-side, no host sync)."""
        index = self.components.index(name)
        with torch.no_grad():
            value = raw.detach().float()
            if bool(value.numel()):
                update = torch.where(self.loss_seen[index] > 0,
                                     0.99 * self.loss_scale[index] + 0.01 * value, value)
                keep = value > 0
                self.loss_scale[index] = torch.where(keep, update, self.loss_scale[index])
                self.loss_seen[index] = torch.where(keep,
                                                    torch.ones_like(self.loss_seen[index]),
                                                    self.loss_seen[index])
        return raw / self.loss_scale[index].clamp(min=1e-3)

    def normalized_fetch(self, raw):
        """`normalized` for the v25 fetch_delay head, against its own buffers (see __init__
        for why it cannot share loss_scale)."""
        with torch.no_grad():
            value = raw.detach().float()
            if bool(value.numel()):
                update = torch.where(self.fetch_seen > 0,
                                     0.99 * self.fetch_scale + 0.01 * value, value)
                keep = value > 0
                self.fetch_scale.copy_(torch.where(keep, update, self.fetch_scale))
                self.fetch_seen.copy_(torch.where(keep, torch.ones_like(self.fetch_seen),
                                                  self.fetch_seen))
        return raw / self.fetch_scale.clamp(min=1e-3)

    def normalized_wasted(self, raw):
        """`normalized` for the v25 wasted_counters head, against its own buffers (same
        reasoning as normalized_fetch)."""
        with torch.no_grad():
            value = raw.detach().float()
            if bool(value.numel()):
                update = torch.where(self.wasted_seen > 0,
                                     0.99 * self.wasted_scale + 0.01 * value, value)
                keep = value > 0
                self.wasted_scale.copy_(torch.where(keep, update, self.wasted_scale))
                self.wasted_seen.copy_(torch.where(keep,
                                                   torch.ones_like(self.wasted_seen),
                                                   self.wasted_seen))
        return raw / self.wasted_scale.clamp(min=1e-3)


def _bce(logits, target, stats, name, n):
    """Masked BCE over the entries whose label is a real bit. `_recall` on the positives and
    `_base` (the positive rate) next to it -- no loss-only heads."""
    mask = target >= 0
    if not bool(mask.any()):
        return torch.zeros((), device=logits.device)
    logits, target = logits[mask], target[mask].float()   # int8 -> float, masked only
    positives = target > 0
    # `_n` counts only the minibatches that CARRIED a label for this head. Everything else
    # here is accumulated * n and divided by the run's total minibatch count, so a head
    # whose label is present in 11 of 17 minibatches had every metric silently scaled by
    # 0.65 -- `decisive`'s "recall 0.428" is really ~0.51 (2026-08-06 audit).
    stats[f"aux_{name}_n"] += n
    stats[f"aux_{name}_base"] += float(target.mean()) * n
    predicted = logits.detach() > 0
    stats[f"aux_{name}_fire"] += float(predicted.float().mean()) * n
    if bool(predicted.any()):
        stats[f"aux_{name}_prec"] += float(
            (predicted & positives).float().sum() / predicted.float().sum()) * n
    if bool(positives.any()):
        stats[f"aux_{name}_recall"] += float(
            (predicted & positives).float().sum() / positives.float().sum()) * n
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, target)


def _ce(logits, labels, stats, name, n, metric_exclude=None):
    """Masked CE + accuracy against the majority-class base rate.

    `metric_exclude`: a class the METRIC ignores (the LOSS is untouched). cost_delta's
    labels are ~98% "no modifier", which pins both acc and base at 0.98 and hides whether
    the head has ever learned a real modifier -- so its pair is reported over the rows
    where a modifier is actually present."""
    mask = labels >= 0
    if not bool(mask.any()):
        return torch.zeros((), device=logits.device)
    selected, flat = labels[mask].long(), logits[mask]     # int8 -> long, masked only
    predicted = flat.detach().argmax(dim=-1)
    scored, scored_predictions = selected, predicted
    if metric_exclude is not None:
        interesting = selected != metric_exclude
        scored = selected[interesting]
        scored_predictions = predicted[interesting]
    if scored.numel():
        stats[f"aux_{name}_n"] += n          # see the note in _bce
        stats[f"aux_{name}_acc"] += float(
            (scored_predictions == scored).float().mean()) * n
        counts = torch.bincount(scored, minlength=flat.shape[-1])
        stats[f"aux_{name}_base"] += float(counts.max().float() / scored.numel()) * n
    return torch.nn.functional.cross_entropy(flat, selected)


def aux_losses_v23(aux, context, token_embeddings, option_features, batch, stats, n):
    """The v2.3 loss: every component normalized to ~unit scale and AVERAGED, so --aux-weight
    stays the weight of the whole v23 suite (it is added NEXT TO the v22 suite's own term)."""
    rows = min(option_features.shape[1], V23_MAX_OPTION_ROWS)
    broadcast = context.unsqueeze(1).expand(-1, rows, -1)
    # exactly what policy_score reads, so the heads sharpen the representation at the CHOICE
    row = aux.row_decoder(torch.cat([broadcast, option_features[:, :rows]], dim=-1))
    width = min(token_embeddings.shape[1], V23_MAX_BOARD_TOKENS)
    terms = []

    def live(name):
        """Is this channel supervised under the run's --heads value?

        A disabled channel must not be COMPUTED either: `_bce` / `_ce` write their metrics
        as a side effect, so evaluating one and then dropping it would keep printing a
        report line for a head that is receiving no gradient -- the exact way `deckout`,
        `ability_blocked` and `cost_delta` kept appearing as live in the head report card
        after being switched off (2026-08-06 audit)."""
        return name not in aux.disabled

    def add(name, raw):
        stats[f"aux_{name}"] += float(raw.detach()) * n
        terms.append(aux.normalized(name, raw))

    attach_logits = aux.attach(row)
    add("attach_need", _bce(attach_logits[:, :, 0], batch["v23_attach"][:, :rows, 0],
                            stats, "attach_need", n))
    add("attach_covered", _bce(attach_logits[:, :, 1], batch["v23_attach"][:, :rows, 1],
                               stats, "attach_covered", n))
    add("ko_delay", _ce(aux.ko_delay(row).reshape(-1, V23_KO_DELAY_CLASSES),
                        batch["v23_ko_delay"][:, :rows].reshape(-1), stats, "ko_delay", n))
    add("decisive", _bce(aux.decisive(row).squeeze(-1), batch["v23_decisive"][:, :rows],
                         stats, "decisive", n))
    bench_logits = aux.bench(row)
    bench_target = batch["v23_bench"][:, :rows]
    if aux.mask_failed_bench_bits:
        # The binary fetch bit is masked out of the pooled loss (its per-bit METRIC still
        # prints in the loop underneath): under v25 fetch_delay supervised the channel
        # instead; under v26 both fetch forms are retired as unlearnable.
        # `donated` is masked too: recall 4% at fire 1.2%, flat for 500 iterations of
        # d128_uniform, and its signal is subsumed by prizes_donated (donated == KO'd with
        # no contribution == prizes_donated > 0 on a contribution-free row) -- in the
        # pooled loss it is noise on the three bits that DO learn.
        bench_target = bench_target.clone()
        bench_target[:, :, V23_FETCH_BIT] = -1
        bench_target[:, :, V23_DONATED_BIT] = -1
    add("bench_bits", _bce(bench_logits, bench_target, stats, "bench_bits", n))
    # PER-BIT VISIBILITY (metrics only -- the pooled loss above is unchanged, and these
    # names are deliberately NOT in V23_COMPONENTS because that tuple sizes the loss_scale
    # buffer and widening it would break every existing v23 checkpoint's strict load).
    # The pooled recall averages five very different bits, so a bit that is never learned
    # is invisible: `used_ability` is the one that says whether a play's ability actually
    # fired, i.e. the Meowth-under-Watchtower question.
    for _index, _bit in enumerate(V23_L3_BIT_NAMES):
        _bce(bench_logits[:, :, _index], batch["v23_bench"][:, :rows, _index],
             stats, f"benchbit_{_bit}", n)
    # The three OPTIONAL channels -- V23_DISABLED_BY_HEADS holds each one's evidence.
    # Leaving a disabled one out of `terms` is what removes the DILUTION: the suite is a
    # MEAN, so a dead head contributed ~0 to the numerator and 1 to the denominator and
    # scaled down every head that WAS learning.
    if live("prizes_donated"):
        add("prizes_donated", _ce(aux.prizes(row).reshape(-1, V23_PRIZE_CLASSES),
                                  batch["v23_prizes"][:, :rows].reshape(-1), stats,
                                  "prizes_donated", n))
    if aux.fetch_delay is not None:
        # Outside `add()`: normalization runs against the head's own buffers, not
        # loss_scale (see AuxHeadsV23.__init__).
        raw = _ce(aux.fetch_delay(row).reshape(-1, V23_FETCH_DELAY_CLASSES),
                  batch["v23_fetch"][:, :rows].reshape(-1), stats, "fetch_delay", n)
        stats["aux_fetch_delay"] += float(raw.detach()) * n
        terms.append(aux.normalized_fetch(raw))
    if aux.wasted is not None:
        raw = _bce(aux.wasted(row).squeeze(-1), batch["v23_wasted"][:, :rows],
                   stats, "wasted_counters", n)
        stats["aux_wasted_counters"] += float(raw.detach()) * n
        terms.append(aux.normalized_wasted(raw))
    if live("ability_blocked") or live("cost_delta"):
        # The token decoder feeds ONLY these two. It was being forward-computed every
        # minibatch with both of them off -- a [B, 18, d] x [d, d] matmul thrown away.
        tokens = aux.token_decoder(token_embeddings[:, :width])
        if live("ability_blocked"):
            add("ability_blocked", _bce(aux.blocked(tokens).squeeze(-1),
                                        batch["v23_blocked"][:, :width], stats,
                                        "ability_blocked", n))
        if live("cost_delta"):
            add("cost_delta", _ce(
                aux.cost_delta(tokens).reshape(-1, V23_DELTA_CLASSES),
                batch["v23_delta"][:, :width].reshape(-1), stats, "cost_delta", n,
                metric_exclude=V23_DELTA_ZERO))
    total = torch.stack(terms).mean()
    stats["aux_v23_normalized"] += float(total.detach()) * n
    # How many terms the mean was actually taken over. Without it the suite's own scale
    # steps silently whenever a channel is switched on or off (it jumped +50% at iteration
    # 641 of d256_v6) and every cross-iteration trend read is wrong at that seam.
    stats["aux_v23_terms"] += len(terms) * n
    return total
