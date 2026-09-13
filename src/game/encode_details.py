"""v3 input surface: the blocks that close the 2026-07-28 audit findings.

This module holds ONLY the new feature blocks. It is composed by:
  * `encode_history.encode_observation_v3` (state side -- extra token rows/columns + globals),
  * `encode_option_v3` here (option side -- the anti-aliasing extension).

Nothing in `src/game/encode.py` is touched (it is byte-frozen), and every v3 addition is
gated off by default in `encode_full.py` / `encode_history.py`, so v1 and v2 checkpoints keep
producing byte-identical arrays.

WHAT IT FIXES (see experiments/audit_2026_07_28/):
  DECISION_SURFACE C1  AreaType.LOOKING (12) is resolved, so reveal-and-pick options carry
                       the real card instead of an all-zero block.
  DECISION_SURFACE C2  energyIndex / toolIndex are resolved to the ATTACHED card, with its
                       full static block, the energy unit count and the resolved EnergyType.
  DECISION_SURFACE C3  every option carries owner / area / slot for its primary AND its
                       target, plus the target's live instance state (damage, hp ratio,
                       absolute hp/maxHp, energy count + per-type composition, attached tool
                       identity, evolution depth, played-this-turn, is-active).
  DECISION_SURFACE C4  the 49-value SelectContext one-hot rides on every option vector AND
                       on the global select-context block.
  DECISION_SURFACE M1  SKILL options carry their `cardId` identity embedding; SPECIAL_CONDITION
                       options carry their `specialConditionType` one-hot.
  DECISION_SURFACE M3  in-play targets are resolved with the option's own `playerIndex`
                       (encode.py:578 hardcodes `your_index`); the corrected identity rides
                       in the target entity block.
  STATE M2/M10         the three dropped DumpState card keys.
  STATE M4             hidden-copy counts un-clamped (a second, wide-scale pair of columns).
  STATE M5             history rows gain the event's target-card and attack identity.
  STATE M6             absolute maxHp / hp / damage as explicit columns.
  STATE M7/M8          face-down Pokemon and the `select.deck` view become tokens.
  STATE C1             board tokens gain a slot index + is-active, so an option's slot
                       one-hot has something to bind to.

Layout discipline: every v3 block is APPENDED after the existing v2 blocks, so all v2
offsets are unchanged and a v2 checkpoint's slice of a v3 vector is exactly its v2 vector.
"""

import numpy as np

from src.cards import get_attack, get_card
from src.embeddings import DIMENSION as EMBEDDING_DIM, attack_embedding, card_embedding
from src.game.encode import (MAX_HP, NUM_ENERGY_TYPES, STATIC_FEATURE_DIM, _card_static_features,
                             _one_hot, encode_option)

# ---------------------------------------------------------------------------------- #
# Vocabularies
# ---------------------------------------------------------------------------------- #
NUM_SELECT_CONTEXTS = 49          # cg.api.SelectContext (0..48); the enum may grow -> clipped
AREA_VOCAB = 16                   # cg.api.AreaType 1..12 + the two undocumented (13/14); 0 = absent
OPTION_SLOT_VOCAB = 16            # exact slot identity for indices 0..15 (hand can exceed -> clipped)
BOARD_SLOT_VOCAB = 9              # active + up to 8 bench (benchMax reaches 8)
NUM_SPECIAL_CONDITIONS = 5        # cg.api.SpecialConditionType

MAX_ATTACHED_ENERGY = 5.0
MAX_SLOT_SCALE = 40.0
WIDE_COPIES = 12.0                # un-clamped copy scale (decks run 8-12 basic energy)
DECK_SIZE = 60.0

_AREA_ACTIVE, _AREA_BENCH = 4, 5


# ---------------------------------------------------------------------------------- #
# Option extension
# ---------------------------------------------------------------------------------- #
# One entity (the card/slot an option coordinate points at):
#   valid, owner(mine/theirs), area one-hot, slot one-hot, slot scalar,
#   in-play flag, is-active, hp ratio, hp/400, maxHp/400, damage/400,
#   energy count, per-EnergyType counts, tool count, evolution depth, played-this-turn,
#   attached-tool identity embedding, the entity card's own identity embedding.
ENTITY_STRUCT_DIM = (1 + 2 + AREA_VOCAB + OPTION_SLOT_VOCAB + 1
                     + 1 + 1 + 4 + 1 + NUM_ENERGY_TYPES + 3
                     + EMBEDDING_DIM + EMBEDDING_DIM)

# The picked ATTACHED card (energyIndex / toolIndex): validity, its full static block, the
# EnergyType it resolves to, and the index/count scalars.
ATTACHED_DIM = 1 + STATIC_FEATURE_DIM + NUM_ENERGY_TYPES + 5

# Everything else the base encoder drops: specialConditionType, SKILL cardId, playerIndex.
MISC_DIM = NUM_SPECIAL_CONDITIONS + 1 + EMBEDDING_DIM + 1 + 1 + 1 + 1

# The primary additionally carries a full v3-RESOLVED static card block: that is what makes
# an AreaType.LOOKING option (all zeros in the frozen encoder) identifiable. The target does
# not repeat the 310-wide block -- the frozen encoder already emits one for it and the v3
# entity block carries the correctly-player-resolved identity embedding on top.
OPTION_EXTRA_DIM = (NUM_SELECT_CONTEXTS + 1
                    + ENTITY_STRUCT_DIM + STATIC_FEATURE_DIM
                    + ENTITY_STRUCT_DIM
                    + ATTACHED_DIM + MISC_DIM)

_ZERO_STATIC = np.zeros(STATIC_FEATURE_DIM, dtype=np.float32)
_ZERO_STATIC.flags.writeable = False
_ZERO_EMBEDDING = np.zeros(EMBEDDING_DIM, dtype=np.float32)
_ZERO_EMBEDDING.flags.writeable = False


def _zone_sequence(observation, area, player_index):
    """The list an (area, index) coordinate indexes into, or None when it is not addressable
    from the observation alone. Unlike encode.py's `_AREA_ZONE` this covers AreaType.LOOKING
    (12) -- the shared mid-effect reveal area -- which is the C1 blind spot."""
    current = observation["current"]
    if area == 1:                                     # DECK -> the select's deck view
        select = observation.get("select") or {}
        return select.get("deck")
    if area == 7:                                     # STADIUM
        return current.get("stadium")
    if area == 12:                                    # LOOKING (shared reveal area)
        return current.get("looking")
    zone = {2: "hand", 3: "discard", 4: "active", 5: "bench", 6: "prize"}.get(area)
    if zone is None:
        return None
    player = current["players"][player_index]
    return player.get(zone)


def _entity_at(observation, area, index, player_index):
    """(card dict or None, pokemon dict or None) for one option coordinate."""
    if area is None or index is None:
        return None, None
    sequence = _zone_sequence(observation, area, player_index)
    if sequence is None or not 0 <= index < len(sequence):
        return None, None
    entry = sequence[index]
    if entry is None:
        return None, None
    if area in (_AREA_ACTIVE, _AREA_BENCH):
        return entry, entry
    return entry, None


def _entity_block(observation, area, index, player_index, me_index):
    """ENTITY_STRUCT_DIM features for one option coordinate: WHERE it points (owner, area,
    slot) and, when it points at a Pokemon in play, WHAT STATE that Pokemon is in."""
    block = np.zeros(ENTITY_STRUCT_DIM, dtype=np.float32)
    if area is None or index is None:
        return block
    offset = 0
    block[offset] = 1.0                                             # coordinate present
    offset += 1
    block[offset] = float(player_index == me_index)
    block[offset + 1] = float(player_index != me_index)
    offset += 2
    block[offset:offset + AREA_VOCAB] = _one_hot(area, AREA_VOCAB)
    offset += AREA_VOCAB
    block[offset:offset + OPTION_SLOT_VOCAB] = _one_hot(min(index, OPTION_SLOT_VOCAB - 1),
                                                        OPTION_SLOT_VOCAB)
    offset += OPTION_SLOT_VOCAB
    block[offset] = min(index, MAX_SLOT_SCALE) / MAX_SLOT_SCALE
    offset += 1

    card, pokemon = _entity_at(observation, area, index, player_index)
    if pokemon is not None:
        max_hp = pokemon.get("maxHp") or 1
        hp = pokemon.get("hp") or 0
        block[offset] = 1.0                                         # is an in-play Pokemon
        block[offset + 1] = float(area == _AREA_ACTIVE)
        block[offset + 2] = hp / max_hp
        block[offset + 3] = hp / MAX_HP
        block[offset + 4] = max_hp / MAX_HP
        block[offset + 5] = max(0, max_hp - hp) / MAX_HP             # damage counters on it
        energies = pokemon.get("energies") or []
        block[offset + 6] = min(len(energies), MAX_ATTACHED_ENERGY) / MAX_ATTACHED_ENERGY
        for energy_type in energies:
            if 0 <= energy_type < NUM_ENERGY_TYPES:
                block[offset + 7 + energy_type] += 1.0
        tools = pokemon.get("tools") or []
        block[offset + 7 + NUM_ENERGY_TYPES] = len(tools) / 2.0
        block[offset + 8 + NUM_ENERGY_TYPES] = len(pokemon.get("preEvolution") or []) / 2.0
        block[offset + 9 + NUM_ENERGY_TYPES] = float(pokemon.get("appearThisTurn") or False)
        if tools:
            tool_start = offset + 10 + NUM_ENERGY_TYPES
            block[tool_start:tool_start + EMBEDDING_DIM] = card_embedding(tools[0]["id"])
    offset += 2 + 4 + 1 + NUM_ENERGY_TYPES + 3 + EMBEDDING_DIM
    if card is not None and (card.get("id") or 0) > 0:
        block[offset:offset + EMBEDDING_DIM] = card_embedding(card["id"])
    return block


def _attached_block(observation, option, me_index):
    """ATTACHED_DIM features for the energy/tool card an ENERGY / ENERGY_CARD / TOOL_CARD
    option actually picks -- `area`/`index` address the HOST Pokemon, so without this the
    engine's `energyIndex` / `toolIndex` (which card) is dropped entirely (C2)."""
    block = np.zeros(ATTACHED_DIM, dtype=np.float32)
    energy_index, tool_index = option.get("energyIndex"), option.get("toolIndex")
    tail = 1 + STATIC_FEATURE_DIM + NUM_ENERGY_TYPES
    block[tail + 0] = min(option.get("count") or 0, 4) / 4.0
    block[tail + 1] = float(energy_index is not None)
    block[tail + 2] = min(energy_index or 0, 8) / 8.0 if energy_index is not None else 0.0
    block[tail + 3] = float(tool_index is not None)
    block[tail + 4] = min(tool_index or 0, 2) / 2.0 if tool_index is not None else 0.0
    if energy_index is None and tool_index is None:
        return block
    owner = option.get("playerIndex")
    owner = me_index if owner is None else owner
    _card, pokemon = _entity_at(observation, option.get("area"), option.get("index"), owner)
    if pokemon is None:
        return block
    if energy_index is not None:
        cards = pokemon.get("energyCards") or []
        if 0 <= energy_index < len(cards):
            block[0] = 1.0
            block[1:1 + STATIC_FEATURE_DIM] = _card_static_features(cards[energy_index]["id"])
        energies = pokemon.get("energies") or []
        if 0 <= energy_index < len(energies):
            block[1 + STATIC_FEATURE_DIM:tail] = _one_hot(energies[energy_index],
                                                          NUM_ENERGY_TYPES)
    else:
        cards = pokemon.get("tools") or []
        if 0 <= tool_index < len(cards):
            block[0] = 1.0
            block[1:1 + STATIC_FEATURE_DIM] = _card_static_features(cards[tool_index]["id"])
    return block


def _misc_block(option, me_index):
    """The remaining Option fields the frozen encoder never reads."""
    block = np.zeros(MISC_DIM, dtype=np.float32)
    condition = option.get("specialConditionType")
    if condition is not None:
        block[:NUM_SPECIAL_CONDITIONS] = _one_hot(condition, NUM_SPECIAL_CONDITIONS)
        block[NUM_SPECIAL_CONDITIONS] = 1.0
    offset = NUM_SPECIAL_CONDITIONS + 1
    card_id = option.get("cardId") or 0                 # SKILL options: whose skill this is
    if card_id > 0:
        block[offset:offset + EMBEDDING_DIM] = card_embedding(card_id)
        block[offset + EMBEDDING_DIM] = 1.0
    offset += EMBEDDING_DIM + 1
    player_index = option.get("playerIndex")
    block[offset] = float(player_index is not None)
    block[offset + 1] = float(player_index == me_index)
    block[offset + 2] = float(option.get("number") is not None)
    return block


def option_extra(observation, option, select=None):
    """OPTION_EXTRA_DIM features appended to `encode.encode_option`'s output."""
    current = observation["current"]
    me_index = current["yourIndex"]
    if select is None:
        select = observation.get("select") or {}

    context_block = np.zeros(NUM_SELECT_CONTEXTS + 1, dtype=np.float32)
    context = select.get("context")
    if context is not None:
        context_block[:NUM_SELECT_CONTEXTS] = _one_hot(context, NUM_SELECT_CONTEXTS)
        context_block[NUM_SELECT_CONTEXTS] = 1.0

    owner = option.get("playerIndex")
    owner = me_index if owner is None else owner
    area, index = option.get("area"), option.get("index")
    if option["type"] == 7 and index is not None:       # PLAY -> hand[index], area implied
        area, owner = 2, me_index
    primary = _entity_block(observation, area, index, owner, me_index)
    primary_card, _pokemon = _entity_at(observation, area, index, owner)
    primary_static = (_card_static_features(primary_card["id"])
                      if primary_card is not None and (primary_card.get("id") or 0) > 0
                      else _ZERO_STATIC)

    # encode.py:578 resolves inPlay targets with `your_index` unconditionally; here the
    # option's own playerIndex wins (M3), so an effect that ever touches the opponent's
    # board describes the RIGHT card instead of a same-slot card of ours.
    target = _entity_block(observation, option.get("inPlayArea"), option.get("inPlayIndex"),
                           owner, me_index)

    return np.concatenate([
        context_block,
        primary, primary_static,
        target,
        _attached_block(observation, option, me_index),
        _misc_block(option, me_index),
    ]).astype(np.float32)


def encode_option_v3(observation, option, select=None):
    """The frozen option vector ++ the v3 extension. `select` defaults to
    observation['select'] (pass it explicitly to save the dict lookup in a hot loop)."""
    return np.concatenate([encode_option(observation, option),
                           option_extra(observation, option, select)]).astype(np.float32)


# ---------------------------------------------------------------------------------- #
# State extension: per-token extra columns
# ---------------------------------------------------------------------------------- #
# Column map (documented so the report's field-disposition table can point at offsets).
X_NO_DAMAGE_COUNTER = 0          # DumpState noDamageCounterEnemyAttackAbility (Flower Curtain)
X_BENCH_TO_ACTIVE = 1            # DumpState benchToActive
X_NO_WEAKNESS_NEXT = 2           # DumpState noWeaknessNextEnemyTurn
X_MAX_HP = 3                     # absolute live maxHp / 400
X_HP = 4                         # absolute current hp / 400
X_DAMAGE = 5                     # absolute damage / 400
X_SLOT_SCALAR = 6                # slot index / 8
X_IS_ACTIVE = 7
X_SLOT_ONEHOT = 8                # + BOARD_SLOT_VOCAB
X_COPIES_WIDE = X_SLOT_ONEHOT + BOARD_SLOT_VOCAB      # min(copies, 12) / 12
X_COPIES_DECK = X_COPIES_WIDE + 1                     # min(copies, 60) / 60
X_FACEDOWN = X_COPIES_DECK + 1
X_DECK_VIEW = X_FACEDOWN + 1
X_HISTORY_TARGET = X_DECK_VIEW + 1                    # + EMBEDDING_DIM
X_HISTORY_ATTACK = X_HISTORY_TARGET + EMBEDDING_DIM   # + EMBEDDING_DIM
X_HISTORY_HAS_TARGET = X_HISTORY_ATTACK + EMBEDDING_DIM
X_HISTORY_HAS_ATTACK = X_HISTORY_HAS_TARGET + 1
V3_TOKEN_EXTRA_DIM = X_HISTORY_HAS_ATTACK + 1

_ZERO_TOKEN_EXTRA = np.zeros(V3_TOKEN_EXTRA_DIM, dtype=np.float32)
_ZERO_TOKEN_EXTRA.flags.writeable = False


def instance_row(pokemon, slot, is_active, rich):
    """The extra columns for a row bound to one in-play Pokemon (its own token, its
    capability tokens, and its attached-entity tokens all share the host's state)."""
    row = np.zeros(V3_TOKEN_EXTRA_DIM, dtype=np.float32)
    if rich:
        row[X_NO_DAMAGE_COUNTER] = float(bool(rich.get("noDamageCounterEnemyAttackAbility")))
        row[X_BENCH_TO_ACTIVE] = float(bool(rich.get("benchToActive")))
        row[X_NO_WEAKNESS_NEXT] = float(bool(rich.get("noWeaknessNextEnemyTurn")))
    max_hp = pokemon.get("maxHp") or 0
    hp = pokemon.get("hp") or 0
    row[X_MAX_HP] = max_hp / MAX_HP
    row[X_HP] = hp / MAX_HP
    row[X_DAMAGE] = max(0, max_hp - hp) / MAX_HP
    row[X_SLOT_SCALAR] = min(slot, BOARD_SLOT_VOCAB - 1) / 8.0
    row[X_IS_ACTIVE] = float(is_active)
    row[X_SLOT_ONEHOT + min(slot, BOARD_SLOT_VOCAB - 1)] = 1.0
    return row


def count_row(copies):
    """The extra columns for a count-weighted belief/library/discard row: the TRUE remaining
    count, un-clamped past the frozen encoder's MAX_COPIES = 4 saturation (M4)."""
    row = np.zeros(V3_TOKEN_EXTRA_DIM, dtype=np.float32)
    row[X_COPIES_WIDE] = min(copies, WIDE_COPIES) / WIDE_COPIES
    row[X_COPIES_DECK] = min(copies, DECK_SIZE) / DECK_SIZE
    return row


def facedown_row(slot, is_active):
    row = np.zeros(V3_TOKEN_EXTRA_DIM, dtype=np.float32)
    row[X_FACEDOWN] = 1.0
    row[X_SLOT_SCALAR] = min(slot, BOARD_SLOT_VOCAB - 1) / 8.0
    row[X_IS_ACTIVE] = float(is_active)
    row[X_SLOT_ONEHOT + min(slot, BOARD_SLOT_VOCAB - 1)] = 1.0
    return row


def deck_view_row():
    row = np.zeros(V3_TOKEN_EXTRA_DIM, dtype=np.float32)
    row[X_DECK_VIEW] = 1.0
    return row


def history_extra_row(event):
    """History rows gain the event's TARGET card and the ATTACK's identity -- the frozen v2
    history block keeps only `bool(attack_id)` + printed damage and drops `target_id`
    entirely, so 'they attached to the bench, not the active' was unreadable (M5)."""
    row = np.zeros(V3_TOKEN_EXTRA_DIM, dtype=np.float32)
    target_id = event.get("target_id") or 0
    if target_id > 0:
        row[X_HISTORY_TARGET:X_HISTORY_TARGET + EMBEDDING_DIM] = card_embedding(target_id)
        row[X_HISTORY_HAS_TARGET] = 1.0
    attack_id = event.get("attack_id") or 0
    if attack_id > 0:
        row[X_HISTORY_ATTACK:X_HISTORY_ATTACK + EMBEDDING_DIM] = attack_embedding(attack_id)
        row[X_HISTORY_HAS_ATTACK] = 1.0
    return row


# ---------------------------------------------------------------------------------- #
# State extension: extra global scalars
# ---------------------------------------------------------------------------------- #
V3_GLOBAL_EXTRA_DIM = NUM_SELECT_CONTEXTS + 1 + 1 + 1 + 1 + 2 + 1


def global_extra(observation, facedown_counts, deck_view_size, dropped_events):
    """The 49-value SelectContext one-hot (C4/M-C2) + the facts the frozen globals leave
    derivable-but-uncomputed: whose turn it is (M9), the deck view size, how many face-down
    Pokemon each side has (M7) and how much history fell outside the token window."""
    current = observation["current"]
    me_index = current["yourIndex"]
    features = np.zeros(V3_GLOBAL_EXTRA_DIM, dtype=np.float32)
    select = observation.get("select")
    if select is not None:
        context = select.get("context")
        if context is not None:
            features[:NUM_SELECT_CONTEXTS] = _one_hot(context, NUM_SELECT_CONTEXTS)
            features[NUM_SELECT_CONTEXTS] = 1.0
    offset = NUM_SELECT_CONTEXTS + 1
    turn = current.get("turn", 0) or 0
    first_player = current.get("firstPlayer", -1)
    features[offset] = float(turn % 2)                          # raw parity
    if first_player is not None and first_player >= 0 and turn > 0:
        turn_player = first_player if turn % 2 == 1 else 1 - first_player
        features[offset + 1] = float(turn_player == me_index)   # it is MY turn
    features[offset + 2] = min(deck_view_size, DECK_SIZE) / DECK_SIZE
    features[offset + 3] = min(facedown_counts[0], 9) / 9.0
    features[offset + 4] = min(facedown_counts[1], 9) / 9.0
    features[offset + 5] = min(dropped_events, 200) / 200.0
    return features


# ---------------------------------------------------------------------------------- #
# The row walk: mirrors encode_game + encode_full's emission order exactly
# ---------------------------------------------------------------------------------- #

def _in_play(player):
    return ([p for p in (player["active"] or []) if p is not None]
            + [p for p in (player["bench"] or []) if p is not None])


def _slots(player):
    """(pokemon, slot, is_active) for the in-play Pokemon, in encode.py's emission order.
    Slot 0 = the Active, 1..N = bench positions (the bench list index + 1), so the number
    an option's `inPlayIndex` carries is recoverable from the token."""
    rows = []
    for index, pokemon in enumerate(player["active"] or []):
        if pokemon is not None:
            rows.append((pokemon, 0, True))
    for index, pokemon in enumerate(player["bench"] or []):
        if pokemon is not None:
            rows.append((pokemon, index + 1, False))
    return rows


def _facedown_slots(player):
    rows = []
    for index, pokemon in enumerate(player["active"] or []):
        if pokemon is None:
            rows.append((0, True))
    for index, pokemon in enumerate(player["bench"] or []):
        if pokemon is None:
            rows.append((index + 1, False))
    return rows


def token_extra_rows(observation, my_unseen=None, opponent_belief=None, belief_top_k=None,
                     my_prize_belief=None, opponent_seen_hidden=None, my_known_top=None,
                     rich_cards=None):
    """[T_card, V3_TOKEN_EXTRA_DIM] aligned with the card rows `encode_observation_full`
    emits (v3 zones on). Mirrors the same walk `encode_history.token_identities` does; the two
    are cross-checked against each other and against the encoder's own token count."""
    from collections import Counter

    current = observation["current"]
    me_index = current["yourIndex"]
    me, opponent = current["players"][me_index], current["players"][1 - me_index]
    rich_cards = rich_cards or {}
    rows = []

    def host_row(pokemon, slot, is_active):
        return instance_row(pokemon, slot, is_active,
                            rich_cards.get(pokemon.get("serial")))

    # 1. board (both sides, active then bench)
    for player in (me, opponent):
        for pokemon, slot, is_active in _slots(player):
            rows.append(host_row(pokemon, slot, is_active))
    # 2. my hand
    rows.extend(_ZERO_TOKEN_EXTRA for _ in (me["hand"] or []))
    # 3. discards (count-weighted, one row per distinct id) -- always on under `full`
    for player in (me, opponent):
        for _card_id, copies in Counter(card["id"] for card in (player["discard"] or [])).items():
            rows.append(count_row(copies))
    # 4. my hidden library
    if my_unseen:
        for _card_id, copies in my_unseen.items():
            rows.append(count_row(copies))
    # 5. opponent belief
    if opponent_belief:
        predicted = sorted(opponent_belief.items(), key=lambda item: -item[1])
        if belief_top_k is not None:
            predicted = predicted[:belief_top_k]
        for _card_id, expected in predicted:
            rows.append(count_row(expected))
    # 6. capability tokens (attacks then abilities, per in-play Pokemon, both sides)
    for player in (me, opponent):
        for pokemon, slot, is_active in _slots(player):
            card = get_card(pokemon["id"])
            row = host_row(pokemon, slot, is_active)
            rows.extend(row for _ in card["attacks"])
            rows.extend(row for _ in card["skills"])
    # 7. attached entities (tools, energy cards, pre-evolutions) then stadium then looking
    for player in (me, opponent):
        for pokemon, slot, is_active in _slots(player):
            row = host_row(pokemon, slot, is_active)
            attached = (len(pokemon.get("tools") or [])
                        + len(pokemon.get("energyCards") or [])
                        + len(pokemon.get("preEvolution") or []))
            rows.extend(row for _ in range(attached))
    rows.extend(_ZERO_TOKEN_EXTRA for stadium in (current.get("stadium") or [])
                if stadium is not None)
    rows.extend(_ZERO_TOKEN_EXTRA for looked in (current.get("looking") or [])
                if looked is not None)
    # 8. knowledge tokens: prize belief, opponent-seen, known top
    if my_prize_belief:
        for _card_id, expected in sorted(my_prize_belief.items(), key=lambda item: -item[1]):
            rows.append(count_row(expected))
    if opponent_seen_hidden:
        for _card_id, copies in sorted(opponent_seen_hidden.items(),
                                       key=lambda item: -item[1]):
            rows.append(count_row(copies))
    if my_known_top:
        rows.extend(count_row(1) for _ in my_known_top)
    # 9. select-context cards (effect / contextCard) -- always on under v2/v3
    select = observation.get("select") or {}
    for card in (select.get("effect"), select.get("contextCard")):
        if card is not None and card.get("id"):
            rows.append(_ZERO_TOKEN_EXTRA)
    # 10. v3: face-down Pokemon slots (M7)
    for player in (me, opponent):
        for slot, is_active in _facedown_slots(player):
            rows.append(facedown_row(slot, is_active))
    # 11. v3: the deck view being searched (M8)
    rows.extend(deck_view_row() for card in (select.get("deck") or [])
                if card is not None and card.get("id"))

    return (np.stack(rows).astype(np.float32) if rows
            else np.zeros((0, V3_TOKEN_EXTRA_DIM), dtype=np.float32))


def facedown_counts(observation):
    current = observation["current"]
    me_index = current["yourIndex"]
    return (len(_facedown_slots(current["players"][me_index])),
            len(_facedown_slots(current["players"][1 - me_index])))
