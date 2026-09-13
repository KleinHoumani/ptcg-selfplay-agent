"""v2 encoding: the full-detail state PLUS the inputs the v1 models never had.

Composes `encode_observation_full` (itself composing the FROZEN base encoder), then appends,
all as new feature COLUMNS and new token ROWS so nothing existing changes shape or value:

  1. per-card v2 block  -- exact prize / deck-position knowledge for MY hidden library
     (CardKnowledge: every serial ever sighted is not-prized forever; cards we ourselves put
     on the bottom stay there until a shuffle). Design doc 2.5 + 2.6.
  2. history tokens     -- one row per meaningful game event (src/game/action_history.py),
     ordered by feature-encoded turn coordinates because the trunk is a SET transformer.
     Design doc 2.2.
  3. card ids           -- a per-row card id array so the model can add a LEARNED identity
     embedding alongside the frozen text embedding. Design doc 2.4.

Deliberately NOT included: any hand-designed summary of the action stream. An earlier draft
wired in src/decks/history.py's 88-dim behavioural digest ("pass-like turn", "turns since a
supporter", capped per-turn counts); the owner rejected it -- the model gets the raw events
and works out the tells itself, per the standing no-hand-engineered-features rule. That
module stays in the tree, still unused.

Every v1 checkpoint keeps working: this module is additive and opt-in, `encode.py` /
`encode_full.py` / `encode_rich.py` are untouched, and models built for the old dims never
see these arrays.

Token layout (one shared width, disjoint slices -- the zone embedding says which kind a row
is): [ full+rich card block | v2 card block | history block ]. Card rows carry zeros in the
history slice, history rows carry zeros in the card slices.

Row-alignment contract: `token_identities` mirrors encode_game's + encode_full's emission
order exactly (the same discipline encode_rich.py follows). It is VERIFIABLE, not just
asserted: a card row's leading static block is `_card_static_features(card_id)`, so
`verify_identities` re-derives every claimed id from the encoded features themselves. The
harness runs it; drift cannot pass silently.
"""

from collections import Counter

import numpy as np

from src.cards import get_attack, get_card
from src.game.action_history import (FINGERPRINT_DIM, KIND_ATTACK, NUM_AREAS, NUM_KINDS,
                                     NUM_STATUS_TYPES)
from src.game.encode import _card_static_features, TOKEN_FEATURE_DIM
from src.game.encode_full import (GLOBAL_FEATURE_DIM_FULL, NUM_ZONES_FULL, SELECT_CONTEXT_DIM,
                                  TOKEN_FEATURE_DIM_FULL_RICH,
                                  ZONE_CONTEXT_CARD, ZONE_DECK_TOP, ZONE_DECK_VIEW,
                                  ZONE_EFFECT_SOURCE, ZONE_ENERGY_CARD, ZONE_FACEDOWN,
                                  ZONE_LOOKING, ZONE_OPPONENT_SEEN,
                                  ZONE_PRE_EVOLUTION, ZONE_PRIZE, ZONE_STADIUM, ZONE_TOOL,
                                  encode_observation_full)
from src.game.encode_rich import RICH_GLOBAL_DIM, rich_state
from src.game.encode_details import (V3_GLOBAL_EXTRA_DIM, V3_TOKEN_EXTRA_DIM, facedown_counts,
                                global_extra, history_extra_row, token_extra_rows)
from src.segments import (OWNER_ME, OWNER_NEUTRAL, OWNER_OPPONENT, ZONE_ABILITY, ZONE_ACTIVE,
                          ZONE_ATTACK, ZONE_BENCH, ZONE_DECK, ZONE_DISCARD, ZONE_HAND,
                          ZONE_PREDICTED)

# One new zone for history rows, appended after the full set (old models keep their table).
ZONE_HISTORY = NUM_ZONES_FULL
NUM_ZONES_V2 = NUM_ZONES_FULL + 1
# v3 reuses everything above and adds encode_full's two v3 zones (face-down slot, deck view).
NUM_ZONES_V3 = NUM_ZONES_FULL + 3

MAX_COPIES = 4.0
# Most-recent events kept as tokens (owner call 2026-07-21). Measured: 32 events covers ~7
# turns and costs ~1.4x run6's token count. None = keep the whole game. Events older than
# the window are simply gone (the digest that once summarised them was removed; accepted).
DEFAULT_HISTORY_CAP = 32
# OWNER CALL 2026-07-28: the 32-event clip is DELIBERATE (token cost; the 07-21 decision
# stands) -- v3 keeps it. The audit's M5 widening to 224 was built, tried for ~1h of
# training, and REVERTED on the owner's instruction: encoder changes require advance
# owner notice, and this one nearly doubled tokens/state (120 -> 218).
V3_HISTORY_CAP = DEFAULT_HISTORY_CAP

# --- per-card v2 block: what we KNOW about a card id's hidden copies -------------------- #
# (prizes are fixed at game start, so a sighted serial is not-prized forever; position
# knowledge comes only from our own to-top/to-bottom selections and dies on a shuffle)
V2_CARD_EXTRA_DIM = 8
_ZERO_CARD_EXTRA = np.zeros(V2_CARD_EXTRA_DIM, dtype=np.float32)
_ZERO_CARD_EXTRA.flags.writeable = False

# --- history token block ---------------------------------------------------------------- #
HISTORY_SCALAR_DIM = 8
HISTORY_FEATURE_DIM = (3 + NUM_KINDS + NUM_STATUS_TYPES + 2 * NUM_AREAS
                       + HISTORY_SCALAR_DIM + FINGERPRINT_DIM)
_ZERO_HISTORY = np.zeros(HISTORY_FEATURE_DIM, dtype=np.float32)
_ZERO_HISTORY.flags.writeable = False

TOKEN_FEATURE_DIM_V2 = (TOKEN_FEATURE_DIM_FULL_RICH + V2_CARD_EXTRA_DIM + HISTORY_FEATURE_DIM)

# --- globals ----------------------------------------------------------------------------- #
# No solver block (owner call 2026-07-23): the 35 board aggregates duplicate token/global
# facts and the 14-dim prize-race ledger is a greedy projection, not game state -- v2 keeps
# factual inputs only and the model does its own aggregation.
V2_GLOBAL_EXTRA_DIM = 6
GLOBAL_FEATURE_DIM_V2 = (GLOBAL_FEATURE_DIM_FULL + SELECT_CONTEXT_DIM + RICH_GLOBAL_DIM
                         + V2_GLOBAL_EXTRA_DIM)

# v3 = v2 ++ the audit-closing columns (both appended LAST, so a v2 model's slice of a v3
# vector is bit-for-bit its own v2 vector).
TOKEN_FEATURE_DIM_V3 = TOKEN_FEATURE_DIM_V2 + V3_TOKEN_EXTRA_DIM
GLOBAL_FEATURE_DIM_V3 = GLOBAL_FEATURE_DIM_V2 + V3_GLOBAL_EXTRA_DIM

CARD_VOCAB = 1268             # card ids are dense 1..1267; index 0 = "no card" (history rows)

_STATIC_LEN = None            # lazily measured from a real card (verification helper)


# =========================================================================== #
# Row identities: mirror the emission order of encode_game + encode_full
# =========================================================================== #

def _in_play(player):
    """encode.py's None filtering, in its order: active then bench."""
    return ([pokemon for pokemon in (player["active"] or []) if pokemon is not None]
            + [pokemon for pokemon in (player["bench"] or []) if pokemon is not None])


def token_identities(observation, deck_counts=None, my_unseen=None, opponent_belief=None,
                     belief_top_k=None, my_prize_belief=None, opponent_seen_hidden=None,
                     my_known_top=None, include_select_context=False,
                     include_v3_zones=False):
    """(card_ids, zone_ids) for the FULL token layout, in emission order. `card_ids` feeds the
    learned identity embedding; capability tokens report their HOST card's id (their feature
    block leads with the attack/ability embedding, not the card's). Must be called with the
    same arguments `encode_observation_v2` passes to `encode_observation_full`."""
    current = observation["current"]
    me_index = current["yourIndex"]
    me, opponent = current["players"][me_index], current["players"][1 - me_index]
    card_ids, zones = [], []

    def emit(card_id, zone):
        card_ids.append(int(card_id or 0))
        zones.append(zone)

    # 1. board: me active+bench, then opponent active+bench
    for player, zone_owner in ((me, None), (opponent, None)):
        for zone_id, key in ((ZONE_ACTIVE, "active"), (ZONE_BENCH, "bench")):
            for pokemon in [p for p in (player[key] or []) if p is not None]:
                emit(pokemon["id"], zone_id)
    # 2. my hand
    for card in (me["hand"] or []):
        emit(card["id"], ZONE_HAND)
    # 3. discards (both sides, count-weighted, one row per distinct id) -- always on in full
    for player in (me, opponent):
        for card_id in Counter(card["id"] for card in (player["discard"] or [])):
            emit(card_id, ZONE_DISCARD)
    # 4. my hidden library
    if my_unseen:
        for card_id in my_unseen:
            emit(card_id, ZONE_DECK)
    # 5. opponent belief
    if opponent_belief:
        predicted = sorted(opponent_belief.items(), key=lambda item: -item[1])
        if belief_top_k is not None:
            predicted = predicted[:belief_top_k]
        for card_id, _expected in predicted:
            emit(card_id, ZONE_PREDICTED)
    # 6. capability tokens (attacks then abilities, per in-play Pokemon, both sides)
    for player in (me, opponent):
        for pokemon in _in_play(player):
            card = get_card(pokemon["id"])
            for _attack_id in card["attacks"]:
                emit(pokemon["id"], ZONE_ATTACK)
            for _skill_index in range(len(card["skills"])):
                emit(pokemon["id"], ZONE_ABILITY)
    # 7. encode_full's attached-entity tokens: tools, energy cards, pre-evolutions per
    #    in-play Pokemon (both sides), then stadium, then looking cards
    for player in (me, opponent):
        for pokemon in _in_play(player):
            for tool in (pokemon.get("tools") or []):
                emit(tool["id"], ZONE_TOOL)
            for energy_card in (pokemon.get("energyCards") or []):
                emit(energy_card["id"], ZONE_ENERGY_CARD)
            for pre_evolution in (pokemon.get("preEvolution") or []):
                emit(pre_evolution["id"], ZONE_PRE_EVOLUTION)
    for stadium in (current.get("stadium") or []):
        if stadium is not None:
            emit(stadium["id"], ZONE_STADIUM)
    for looked in (current.get("looking") or []):
        if looked is not None:
            emit(looked["id"], ZONE_LOOKING)
    # 8. knowledge tokens, in encode_full's order: prize belief, opponent-seen, known top
    if my_prize_belief:
        for card_id, _expected in sorted(my_prize_belief.items(), key=lambda item: -item[1]):
            emit(card_id, ZONE_PRIZE)
    if opponent_seen_hidden:
        for card_id, _copies in sorted(opponent_seen_hidden.items(), key=lambda item: -item[1]):
            emit(card_id, ZONE_OPPONENT_SEEN)
    if my_known_top:
        for card_id in my_known_top:
            emit(card_id, ZONE_DECK_TOP)
    # 9. select-context cards
    if include_select_context:
        select = observation.get("select") or {}
        for card, zone in ((select.get("effect"), ZONE_EFFECT_SOURCE),
                           (select.get("contextCard"), ZONE_CONTEXT_CARD)):
            if card is None or not card.get("id"):
                continue
            emit(card["id"], zone)
    # 10/11. v3 tokens: face-down in-play slots (no id -> 0), then the deck search view
    if include_v3_zones:
        for player in (me, opponent):
            for slots in ((player["active"] or []), (player["bench"] or [])):
                for pokemon in slots:
                    if pokemon is None:
                        emit(0, ZONE_FACEDOWN)
        for card in ((observation.get("select") or {}).get("deck") or []):
            if card is not None and card.get("id"):
                emit(card["id"], ZONE_DECK_VIEW)
    return np.array(card_ids, dtype=np.int64), np.array(zones, dtype=np.int64)


def verify_identities(token_features, card_ids, zone_ids):
    """Re-derive each PLAIN card row's id from the encoded features themselves and compare
    with the claimed id: a card token's leading block IS `_card_static_features(card_id)`.
    Returns (checked, mismatches). Capability rows lead with an attack/ability embedding
    instead, so they are skipped here (their host id is checked by the harness separately)."""
    global _STATIC_LEN
    if _STATIC_LEN is None:
        _STATIC_LEN = len(_card_static_features(int(card_ids[0]) if len(card_ids) else 1))
    skip = {ZONE_ATTACK, ZONE_ABILITY}
    checked = mismatches = 0
    for row, (card_id, zone) in enumerate(zip(card_ids, zone_ids)):
        if int(zone) in skip or int(card_id) <= 0 or int(zone) == ZONE_HISTORY:
            continue
        expected = _card_static_features(int(card_id))
        if not np.array_equal(token_features[row, :_STATIC_LEN], expected):
            mismatches += 1
        checked += 1
    return checked, mismatches


# =========================================================================== #
# Feature blocks
# =========================================================================== #

def _hypergeometric_at_least_one(copies, unseen_total, prizes_left):
    """P(at least one of `copies` specific cards lands in a uniform `prizes_left`-subset of
    `unseen_total` cards). Closed form: 1 - C(unseen-copies, prizes)/C(unseen, prizes)."""
    if copies <= 0 or prizes_left <= 0 or unseen_total <= 0:
        return 0.0
    if unseen_total - copies < prizes_left:
        return 1.0
    probability_none = 1.0
    for step in range(prizes_left):
        probability_none *= (unseen_total - copies - step) / (unseen_total - step)
    return 1.0 - probability_none


def knowledge_summary(knowledge):
    """The per-STATE knowledge facts card_knowledge_block reads -- computed once per encode
    call instead of per row (never_seen_counts alone is a Counter subtraction; at ~150 rows
    x thousands of decisions the per-row recompute was ~15% of encode time)."""
    if knowledge is None:
        return None
    never_seen = knowledge.never_seen_counts()
    return (never_seen, sum(never_seen.values()), knowledge.prizes_left(),
            Counter(knowledge.known_bottom_ids()), knowledge.known_top_ids())


def card_knowledge_block(summary, card_id, zone_id, deck_count):
    """V2_CARD_EXTRA_DIM features about a card id's HIDDEN copies. Only meaningful for my
    hidden-library / prize rows; every other row (board, hand, discard, opponent) is zeros --
    the knowledge is about what I cannot see, not about what is on the table.
    `summary` = knowledge_summary(knowledge), hoisted out of the per-row loop."""
    if summary is None or zone_id not in (ZONE_DECK, ZONE_PRIZE, ZONE_DECK_TOP):
        return _ZERO_CARD_EXTRA
    never_seen, unseen_total, prizes_left, bottom_ids, top_ids = summary
    hidden_copies = never_seen.get(card_id, 0)
    expected_prized = (hidden_copies * min(1.0, prizes_left / unseen_total)
                       if unseen_total else 0.0)
    top_position = next((index for index, top_id in enumerate(top_ids)
                         if top_id == card_id), None)
    return np.array([
        min(expected_prized, MAX_COPIES) / MAX_COPIES,
        _hypergeometric_at_least_one(hidden_copies, unseen_total, prizes_left),
        float(hidden_copies == 0),                       # every copy sighted -> none prized
        min(bottom_ids.get(card_id, 0), MAX_COPIES) / MAX_COPIES,
        min(top_ids.count(card_id), MAX_COPIES) / MAX_COPIES,
        0.0 if top_position is None else 1.0 / (1.0 + top_position),
        # a bottomed card is unreachable until the deck is drawn through: proximity in [0,1]
        1.0 - min(deck_count, 60) / 60.0 if bottom_ids.get(card_id, 0) else 0.0,
        min(hidden_copies, MAX_COPIES) / MAX_COPIES,
    ], dtype=np.float32)


_HISTORY_SCALAR_OFFSET = 3 + NUM_KINDS + NUM_STATUS_TYPES + 2 * NUM_AREAS


def _history_static_base(event):
    """The event-static columns of a history row: one-hots, event scalars, fingerprint --
    everything except the 3 turn-RELATIVE scalars (turns-ago, recency, this-turn flag)."""
    features = np.zeros(HISTORY_FEATURE_DIM, dtype=np.float32)
    offset = 0
    actor = min(max(event["actor"], 0), 2)
    features[offset + actor] = 1.0
    offset += 3
    features[offset + event["kind"]] = 1.0
    offset += NUM_KINDS
    if event["status"] >= 0:
        features[offset + min(event["status"], NUM_STATUS_TYPES - 1)] = 1.0
    offset += NUM_STATUS_TYPES
    if 0 <= event["area_from"] < NUM_AREAS:
        features[offset + event["area_from"]] = 1.0
    offset += NUM_AREAS
    if 0 <= event["area_to"] < NUM_AREAS:
        features[offset + event["area_to"]] = 1.0
    offset += NUM_AREAS
    attack_damage = 0.0
    if event["kind"] == KIND_ATTACK and event["attack_id"]:
        attack = get_attack(event["attack_id"])
        if attack:
            attack_damage = (attack.get("damage") or 0) / 100.0
    features[offset + 0] = np.clip(event["value"] / 100.0, -4.0, 4.0)
    features[offset + 1] = float(event["flag"])
    features[offset + 2] = float(bool(event["attack_id"]))
    features[offset + 3] = attack_damage
    features[offset + 6] = min(event["index_in_turn"], 20) / 20.0
    features[offset + HISTORY_SCALAR_DIM:
             offset + HISTORY_SCALAR_DIM + FINGERPRINT_DIM] = event["fingerprint"]
    return features


def history_block(event, current_turn):
    """One history event -> HISTORY_FEATURE_DIM features. Ordering is FEATURE-encoded
    (turns-ago + index-in-turn), because the trunk is a set transformer with no positional
    signal: without these two, 'Rare Candy then Gardevoir' and its reverse look identical.
    An event is re-encoded at every later decision (~30-60x per game) but only the 3
    turn-relative scalars change -- the static columns are cached ON the event dict."""
    base = event.get("_static_row")
    if base is None:
        base = _history_static_base(event)
        event["_static_row"] = base
    features = base.copy()
    turns_ago = max(0, current_turn - event["turn"])
    offset = _HISTORY_SCALAR_OFFSET
    features[offset + 4] = min(turns_ago, 20) / 20.0
    features[offset + 5] = 1.0 / (1.0 + turns_ago)      # recency, sharp on the recent past
    features[offset + 7] = float(turns_ago == 0)
    return features


def _v2_global_block(knowledge, history, observation):
    events = len(history.events) if history is not None else 0
    if knowledge is None:
        return np.zeros(V2_GLOBAL_EXTRA_DIM, dtype=np.float32)
    never_seen_total = sum(knowledge.never_seen_counts().values())
    return np.array([
        min(events, 200) / 200.0,
        min(history.turns_seen if history is not None else 0, 40) / 40.0,
        min(len(knowledge.known_top), 5) / 5.0,
        min(len(knowledge.known_bottom), 10) / 10.0,
        min(never_seen_total, 20) / 20.0,
        float(knowledge.prizes_exact() is not None),
    ], dtype=np.float32)


# =========================================================================== #
# The encoder
# =========================================================================== #

def encode_observation_v2(observation, deck_counts=None, knowledge=None, history=None,
                          opponent_belief=None, belief_top_k=None,
                          history_cap=DEFAULT_HISTORY_CAP, rich_override=None, v3=False):
    """observation -> v2 arrays: {token_features [T, TOKEN_FEATURE_DIM_V2], owner_ids [T],
    zone_ids [T], global_features [GLOBAL_FEATURE_DIM_V2], card_ids [T]}.

    knowledge: a CardKnowledge for MY side (exact prize deduction + deck-position knowledge).
    history:   an ActionHistory for THIS seat (visible-only event stream).
    history_cap: keep only the most recent N events as tokens (None = all); older events
      are simply outside the model's view.
    rich_override: (rich_cards, rich_players) to apply instead of decoding this observation's
      own engine state blob -- how the SEARCH side keeps the rich block live (search states
      carry no blob; the root decision's effect state is re-bound by serial). None = decode
      this observation, which is what training does.
    v3: append the v3 input surface (encode_details.py) -- two extra token kinds, the
      V3_TOKEN_EXTRA_DIM per-row columns and the V3_GLOBAL_EXTRA_DIM globals. OFF by
      default: with v3=False every array is byte-identical to what v2 checkpoints trained
      on. `encode_observation_v3` is the convenience wrapper.
    """
    from src.game.encode import unseen_my_counts             # local: keeps import graph flat

    my_unseen = unseen_my_counts(observation, deck_counts) if deck_counts else None
    if v3 and rich_override is None:
        # Decode the engine state ONCE and hand the same pair to every consumer below
        # (encode_full decodes it twice today). Identical result, one fewer decode.
        rich_override = rich_state(observation)
    knowledge_kwargs = {}
    if knowledge is not None:
        knowledge_kwargs = {
            "my_prize_belief": knowledge.prize_belief(),
            "my_prize_certainty": knowledge.prize_certainty(),
            "opponent_seen_hidden": knowledge.opponent_not_prized_counts(observation),
            "my_known_top": knowledge.known_top_ids(),
        }
    encoded = encode_observation_full(
        observation, deck_counts=deck_counts, opponent_belief=opponent_belief,
        belief_top_k=belief_top_k, include_rich=True, include_select_context=True,
        rich_override=rich_override, include_v3_zones=v3, **knowledge_kwargs)

    card_ids, identity_zones = token_identities(
        observation, deck_counts=deck_counts, my_unseen=my_unseen,
        opponent_belief=opponent_belief, belief_top_k=belief_top_k,
        my_prize_belief=knowledge_kwargs.get("my_prize_belief"),
        opponent_seen_hidden=knowledge_kwargs.get("opponent_seen_hidden"),
        my_known_top=knowledge_kwargs.get("my_known_top"),
        include_select_context=True, include_v3_zones=v3)
    token_count = encoded["token_features"].shape[0]
    assert len(card_ids) == token_count, (
        f"v2 identity mirror drifted from the encoder: {len(card_ids)} ids vs "
        f"{token_count} tokens -- token_identities must track encode_full's emission order")

    # --- card rows: [full+rich | v2 card block | zero history block] --- #
    my_deck_count = (observation["current"]["players"][observation["current"]["yourIndex"]]
                     .get("deckCount") or 0)
    summary = knowledge_summary(knowledge)
    card_extra = np.stack([card_knowledge_block(summary, int(card_id), int(zone),
                                                my_deck_count)
                           for card_id, zone in zip(card_ids, encoded["zone_ids"])]) \
        if token_count else np.zeros((0, V2_CARD_EXTRA_DIM), dtype=np.float32)
    card_block_stack = [encoded["token_features"], card_extra.astype(np.float32),
                        np.zeros((token_count, HISTORY_FEATURE_DIM), dtype=np.float32)]
    if v3:
        v3_rows = token_extra_rows(
            observation, my_unseen=my_unseen, opponent_belief=opponent_belief,
            belief_top_k=belief_top_k,
            my_prize_belief=knowledge_kwargs.get("my_prize_belief"),
            opponent_seen_hidden=knowledge_kwargs.get("opponent_seen_hidden"),
            my_known_top=knowledge_kwargs.get("my_known_top"),
            rich_cards=(rich_override or (None, None))[0])
        assert v3_rows.shape[0] == token_count, (
            f"v3 row mirror drifted from the encoder: {v3_rows.shape[0]} rows vs "
            f"{token_count} tokens -- token_extra_rows must track encode_full's order")
        card_block_stack.append(v3_rows)
    card_rows = np.concatenate(card_block_stack, axis=1)

    # --- history rows: [zero card blocks | history block] --- #
    events = history.events if history is not None else []
    dropped_events = 0
    if history_cap is not None:
        dropped_events = max(0, len(events) - history_cap)
        events = events[-history_cap:]
    if events:
        current_turn = observation["current"].get("turn", 0) or 0
        history_features = np.stack([history_block(event, current_turn) for event in events])
        history_stack = [
            np.zeros((len(events), TOKEN_FEATURE_DIM_FULL_RICH + V2_CARD_EXTRA_DIM),
                     dtype=np.float32),
            history_features.astype(np.float32)]
        if v3:
            history_stack.append(
                np.stack([history_extra_row(event) for event in events]).astype(np.float32))
        history_rows = np.concatenate(history_stack, axis=1)
        history_owners = np.array([OWNER_ME if event["actor"] == 0
                                   else (OWNER_OPPONENT if event["actor"] == 1
                                         else OWNER_NEUTRAL) for event in events],
                                  dtype=np.int64)
        history_zones = np.full(len(events), ZONE_HISTORY, dtype=np.int64)
        history_card_ids = np.array([event["card_id"] if 0 < event["card_id"] < CARD_VOCAB
                                     else 0 for event in events], dtype=np.int64)
        token_features = np.concatenate([card_rows, history_rows])
        owner_ids = np.concatenate([encoded["owner_ids"], history_owners])
        zone_ids = np.concatenate([encoded["zone_ids"], history_zones])
        all_card_ids = np.concatenate([np.clip(card_ids, 0, CARD_VOCAB - 1), history_card_ids])
    else:
        token_features = card_rows
        owner_ids = encoded["owner_ids"]
        zone_ids = encoded["zone_ids"]
        all_card_ids = np.clip(card_ids, 0, CARD_VOCAB - 1)

    global_stack = [encoded["global_features"],
                    _v2_global_block(knowledge, history, observation)]
    if v3:
        deck_view = [card for card in ((observation.get("select") or {}).get("deck") or [])
                     if card is not None and card.get("id")]
        global_stack.append(global_extra(observation, facedown_counts(observation),
                                         len(deck_view), dropped_events))
    global_features = np.concatenate(global_stack)
    return {"token_features": token_features.astype(np.float32),
            "owner_ids": owner_ids, "zone_ids": zone_ids,
            "global_features": global_features.astype(np.float32),
            "card_ids": all_card_ids}


def encode_observation_v3(observation, deck_counts=None, knowledge=None, history=None,
                          opponent_belief=None, belief_top_k=None,
                          history_cap=V3_HISTORY_CAP, rich_override=None):
    """v3 = v2 ++ the audit-closing input surface (see src/game/encode_details.py). Same output
    schema; widths are TOKEN_FEATURE_DIM_V3 / GLOBAL_FEATURE_DIM_V3, zones < NUM_ZONES_V3."""
    return encode_observation_v2(observation, deck_counts=deck_counts, knowledge=knowledge,
                                 history=history, opponent_belief=opponent_belief,
                                 belief_top_k=belief_top_k, history_cap=history_cap,
                                 rich_override=rich_override, v3=True)


# --------------------------------------------------------------------------- #
# Self-test: encode real states, prove the identity mirror against the encoded
# features, and prove the v1 path is untouched.
#   ./.venv/Scripts/python.exe -m src.game.encode_history
# --------------------------------------------------------------------------- #

def _self_test():
    import random

    from cg import game

    from src.decks.card_knowledge import CardKnowledge
    from src.game.action_history import ActionHistory
    from src.game.encode import encode_observation

    deck = [int(line) for line in
            open("submissions/submission_alakazam/deck.csv").read().split() if line.strip()]
    deck_counts = dict(Counter(deck))
    rng = random.Random(11)

    def random_legal(observation):
        select = observation["select"]
        count = len(select["option"])
        take = max(min(select["maxCount"], count), select["minCount"])
        return sorted(rng.sample(range(count), take)) if count else []

    knowledge = CardKnowledge(Counter(deck))
    history = ActionHistory()

    observation, _ = game.battle_start(list(deck), list(deck), seed=77)
    moves = checked = total_mismatch = 0
    widths, token_counts = set(), []
    while observation["current"]["result"] == -1 and moves < 320:
        if observation["current"]["yourIndex"] == 0:
            knowledge.update(observation)
            history.update(observation)
            if moves and moves % 40 == 0:
                encoded = encode_observation_v2(observation, deck_counts=deck_counts,
                                                knowledge=knowledge, history=history)
                widths.add(encoded["token_features"].shape[1])
                token_counts.append(encoded["token_features"].shape[0])
                assert encoded["global_features"].shape[0] == GLOBAL_FEATURE_DIM_V2
                assert encoded["card_ids"].shape[0] == encoded["token_features"].shape[0]
                assert int(encoded["zone_ids"].max()) < NUM_ZONES_V2
                count, mismatches = verify_identities(
                    encoded["token_features"], encoded["card_ids"], encoded["zone_ids"])
                checked += count
                total_mismatch += mismatches
                # the frozen v1 path must be untouched by anything this module does
                base = encode_observation(observation)
                assert base["token_features"].shape[1] == TOKEN_FEATURE_DIM
                assert base["global_features"].shape[0] == 22
        select = observation.get("select")
        observation = game.battle_select(random_legal(observation) if select else [])
        moves += 1
    game.battle_finish()

    assert widths == {TOKEN_FEATURE_DIM_V2}, widths
    assert total_mismatch == 0, f"{total_mismatch}/{checked} identity rows MISALIGNED"
    print(f"OK: width {TOKEN_FEATURE_DIM_V2} (card {TOKEN_FEATURE_DIM_FULL_RICH} + v2 "
          f"{V2_CARD_EXTRA_DIM} + history {HISTORY_FEATURE_DIM}), "
          f"globals {GLOBAL_FEATURE_DIM_V2}, zones {NUM_ZONES_V2}")
    print(f"    identity rows verified against encoded features: {checked}, mismatches 0")
    print(f"    tokens/state: mean {sum(token_counts) / len(token_counts):.0f} "
          f"max {max(token_counts)} (history cap {DEFAULT_HISTORY_CAP})")


if __name__ == "__main__":
    _self_test()
