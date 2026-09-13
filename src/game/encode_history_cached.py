"""Vectorised front-end for `encode_observation_v2` -- SAME BYTES, ~3x cheaper.

Why this exists (Workstream B2, experiments/native_rollout/PROFILE_REPORT.md): the v2 encoder
is 71% of a self-play worker's CPU-side cost and is per-row Python-loop bound
(`encode_card_token` 71x, `_dynamic_block` 87x, `card_knowledge_block` 87x, ~1,900 `dict.get`
calls per encode). Nothing under `src/game/` is modified -- this module walks the same
emission order and then assembles the token matrix with a handful of whole-matrix numpy
operations instead of one `np.concatenate` per row.

The trick is that every token has ONE layout:

    [ static 310 | dynamic 16 | belief 1 | attack block 13 | rich 16 | v2 knowledge 8 | history 74 ]
      ^cached by card/attack/ability id      ^cached by attack id

so the two wide, expensive blocks are pure lookups. This module keeps process-global tables of
those cached blocks -- filled by calling the FROZEN builders `encode._card_static_features` /
`_attack_static_features` / `_ability_static_features` / `_attack_block` -- and gathers whole
columns with fancy indexing. The per-decision blocks (dynamic state, engine effect state) are
built once per in-play Pokemon with the frozen `encode._dynamic_block` / `encode_rich.card_block`
and gathered the same way. A gathered row is bit-identical to a concatenated one by
construction: the same float32 bytes are copied, never recomputed.

History rows are kept incrementally (only 3 of their 74 columns depend on the current turn) and
patched with a vectorised update when the turn advances.

Correctness rules followed throughout:
  - every wide block comes from the frozen builder, so bytes cannot drift;
  - the emission ORDER is the only thing restated, and `encode_history.token_identities` already
    mirrors it independently -- `self_check()` compares both encoders bit-for-bit, and the
    level-1 harness (experiments/native_rollout) byte-checks 200 model-driven games.

v3 (2026-07-28): the same walk, one layout parameter. `CachedV2Encoder(v3=True)` (alias
`CachedV3Encoder`) appends the two v3 token kinds (ZONE_FACEDOWN / ZONE_DECK_VIEW), the
`V3_TOKEN_EXTRA_DIM` per-row columns and the `V3_GLOBAL_EXTRA_DIM` globals -- i.e. exactly
what `encode_observation_v3` returns. The extra columns are the same three shapes the walk
already gathers: a per-HOST instance row (cached per in-play Pokemon, gathered by the same
`host_slots` index the dynamic/rich blocks use), a per-row count scalar (vectorised), and a
per-EVENT static row (kept incrementally alongside the history rows). With `v3=False` not a
single byte of the v2 path moves -- the v3 branches are all behind `if self.v3`.

Usage: ONE instance per (game, seat) -- the history state is per-seat. Call `encode(...)`
exactly where `encode_observation_v2` would be called, with the same arguments.

Scope: the TRAINING call shape only -- `opponent_belief=None`, `belief_top_k=None`,
`rich_override=None`, `history_cap=DEFAULT_HISTORY_CAP`. Anything else raises, so a caller that
needs those falls back to the real encoder instead of silently getting different rows.
"""

from collections import Counter

import numpy as np

from src.cards import get_card
from src.game import encode as _encode
from src.game import encode_full as _full
from src.game import encode_rich as _rich
from src.game import encode_history as _v2
from src.game import encode_details as _v3
from src.segments import (OWNER_ME, OWNER_NEUTRAL, OWNER_OPPONENT, ZONE_ABILITY, ZONE_ACTIVE,
                          ZONE_ATTACK, ZONE_BENCH, ZONE_DECK, ZONE_DISCARD, ZONE_HAND)

# --- the one token layout, derived from encode.py (never hardcoded) --------------------- #
_STATIC = _encode.STATIC_FEATURE_DIM                       # 310  card / attack / ability block
_DYNAMIC = 4 + _encode.NUM_ENERGY_TYPES                    # 16   live host state
_BELIEF_COLUMN = _STATIC + _DYNAMIC                        # 326
_ATTACK_START = _BELIEF_COLUMN + 1                         # 327
_TOKEN = _encode.TOKEN_FEATURE_DIM                         # 340
_RICH_START = _TOKEN
_BASE = _full.TOKEN_FEATURE_DIM_FULL_RICH                  # 356  = 340 + rich 16
_EXTRA_END = _BASE + _v2.V2_CARD_EXTRA_DIM                 # 364  = + v2 card-knowledge block
_WIDTH = _v2.TOKEN_FEATURE_DIM_V2                          # 438  = + history block
_HISTORY_DIM = _v2.HISTORY_FEATURE_DIM
_SCALAR_OFFSET = _v2._HISTORY_SCALAR_OFFSET
_MAX_COPIES = _encode.MAX_COPIES
assert _ATTACK_START + _encode.ATTACK_BLOCK_DIM == _TOKEN, "encode.py token layout changed"
assert _EXTRA_END + _HISTORY_DIM == _WIDTH, "encode_history token layout changed"

# --- the v3 extension, appended after the v2 width (encode_details.py) --------------------- #
_V3_EXTRA = _v3.V3_TOKEN_EXTRA_DIM                         # 151
_WIDTH_V3 = _v2.TOKEN_FEATURE_DIM_V3                       # 589 = 438 + 151
assert _WIDTH + _V3_EXTRA == _WIDTH_V3, "encode_details token layout changed"


# =========================================================================== #
# Process-global lookup tables of the frozen, cached blocks.
# Bounded by the card pool (~1.3k cards, ~2.5k attacks/abilities), so a few MB
# shared by every game and every seat in the process.
# =========================================================================== #

class _Table:
    """Growable float32 matrix + key -> row index, filled from a frozen builder."""

    def __init__(self, width, zero_row=True):
        self.width = width
        self.matrix = np.zeros((64, width), dtype=np.float32)
        self.count = 1 if zero_row else 0          # row 0 stays all-zero ("no such block")
        self.index = {}

    def slot(self, key, builder):
        row = self.index.get(key)
        if row is not None:
            return row
        block = builder()
        row = self.count
        if row >= self.matrix.shape[0]:
            grown = np.zeros((2 * self.matrix.shape[0], self.width), dtype=np.float32)
            grown[:row] = self.matrix
            self.matrix = grown
        self.matrix[row] = block
        self.count = row + 1
        self.index[key] = row
        return row


_STATIC_TABLE = _Table(_STATIC)
_ATTACK_TABLE = _Table(_encode.ATTACK_BLOCK_DIM)


def _board_slots(player):
    """(slot, pokemon) for the in-play Pokemon in encode.py's emission order: the Active
    (slot 0) then every occupied bench position (its list index + 1). Mirrors
    encode_details._slots, whose slot numbers the v3 columns encode."""
    rows = [(0, pokemon) for pokemon in (player["active"] or []) if pokemon is not None]
    rows += [(index + 1, pokemon) for index, pokemon in enumerate(player["bench"] or [])
             if pokemon is not None]
    return rows


def _card_slot(card_id):
    return _STATIC_TABLE.slot(("c", card_id),
                              lambda: _encode._card_static_features(card_id))


def _attack_static_slot(card_id, attack_id):
    return _STATIC_TABLE.slot(("a", card_id, attack_id),
                              lambda: _encode._attack_static_features(card_id, attack_id))


def _ability_static_slot(card_id, skill_index):
    return _STATIC_TABLE.slot(("s", card_id, skill_index),
                              lambda: _encode._ability_static_features(card_id, skill_index))


def _attack_block_slot(attack_id):
    return _ATTACK_TABLE.slot(attack_id, lambda: _encode._attack_block(attack_id))


class CachedV2Encoder:
    """One per (game, seat). `encode(observation, deck_counts=, knowledge=, history=)` returns
    exactly what `encode_observation_v2` returns for the same arguments -- or, with
    `v3=True`, exactly what `encode_observation_v3` returns."""

    def __init__(self, history_cap=None, v3=False):
        self.v3 = bool(v3)
        default_cap = _v2.V3_HISTORY_CAP if self.v3 else _v2.DEFAULT_HISTORY_CAP
        if history_cap is None:
            history_cap = default_cap
        if history_cap != default_cap:
            raise ValueError("CachedV2Encoder tracks the default history cap only")
        self.history_cap = history_cap
        self.width = _WIDTH_V3 if self.v3 else _WIDTH
        # history: static rows for EVERY event ever seen; only the 3 turn-relative scalars
        # move, and only when the turn advances. The v3 extra columns (target / attack
        # identity) are event-static, so they are filled once and never touched again.
        self._history_rows = np.zeros((0, _HISTORY_DIM), dtype=np.float32)
        self._history_v3_rows = np.zeros((0, _V3_EXTRA), dtype=np.float32)
        self._history_turns = np.zeros(0, dtype=np.int64)
        self._history_count = 0
        self._history_turn = None

    # ------------------------------------------------------------------ history
    def _history_grow(self, rows):
        capacity = max(rows, 2 * self._history_rows.shape[0] + 16)
        grown = np.zeros((capacity, _HISTORY_DIM), dtype=np.float32)
        grown[:self._history_rows.shape[0]] = self._history_rows
        self._history_rows = grown
        turns = np.zeros(capacity, dtype=np.int64)
        turns[:self._history_turns.shape[0]] = self._history_turns
        self._history_turns = turns
        if self.v3:
            grown_v3 = np.zeros((capacity, _V3_EXTRA), dtype=np.float32)
            grown_v3[:self._history_v3_rows.shape[0]] = self._history_v3_rows
            self._history_v3_rows = grown_v3

    def _history_window(self, events, current_turn):
        """The `history_cap` most recent event rows, byte-identical to
        `[history_block(event, current_turn) for event in events[-cap:]]`. With v3 on,
        returns (v2 window, v3 extra window); the two are row-aligned."""
        total = len(events)
        if total > self._history_rows.shape[0]:
            self._history_grow(total)
        if current_turn != self._history_turn and self._history_count:
            rows = self._history_rows[:self._history_count]
            turns_ago = np.maximum(0, current_turn - self._history_turns[:self._history_count])
            rows[:, _SCALAR_OFFSET + 4] = np.minimum(turns_ago, 20) / 20.0
            rows[:, _SCALAR_OFFSET + 5] = 1.0 / (1.0 + turns_ago)
            rows[:, _SCALAR_OFFSET + 7] = (turns_ago == 0)
        self._history_turn = current_turn
        while self._history_count < total:
            event = events[self._history_count]
            self._history_rows[self._history_count] = _v2.history_block(event, current_turn)
            if self.v3:
                self._history_v3_rows[self._history_count] = _v3.history_extra_row(event)
            self._history_turns[self._history_count] = event["turn"]
            self._history_count += 1
        start = max(0, total - self.history_cap)
        return (self._history_rows[start:total],
                self._history_v3_rows[start:total] if self.v3 else None)

    # ------------------------------------------------------------------ the encoder
    def encode(self, observation, deck_counts=None, knowledge=None, history=None,
               opponent_belief=None, belief_top_k=None,
               history_cap=None, rich_override=None):
        if history_cap is None:
            history_cap = self.history_cap
        if opponent_belief or belief_top_k is not None or rich_override is not None \
                or history_cap != self.history_cap:
            raise ValueError("CachedV2Encoder supports the training call shape only")
        current = observation["current"]
        me_index = current["yourIndex"]
        me = current["players"][me_index]
        opponent = current["players"][1 - me_index]

        my_unseen = _encode.unseen_my_counts(observation, deck_counts) if deck_counts else None
        if knowledge is not None:
            my_prize_belief = knowledge.prize_belief()
            my_prize_certainty = knowledge.prize_certainty()
            opponent_seen_hidden = knowledge.opponent_not_prized_counts(observation)
            my_known_top = knowledge.known_top_ids()
        else:
            my_prize_belief = opponent_seen_hidden = my_known_top = None
            my_prize_certainty = None

        # ONE dump_state decode per decision (the uncached path decodes twice: once in
        # encode_rich_observation, once in encode_full._entity_rich_rows).
        rich_cards, rich_players = _rich.rich_state(observation)
        rich_cards = rich_cards or {}
        rich_players = rich_players or {}
        summary = _v2.knowledge_summary(knowledge)
        my_deck_count = me.get("deckCount") or 0

        # --- in-play hosts, in encode_game's emission order (me active+bench, then theirs) ---
        # `host_slot_meta` is the same sequence with each host's BOARD slot (0 = Active,
        # bench index + 1), which is what encode_details.instance_row needs.
        me_active = [p for p in (me["active"] or []) if p is not None]
        me_bench = [p for p in (me["bench"] or []) if p is not None]
        opponent_active = [p for p in (opponent["active"] or []) if p is not None]
        opponent_bench = [p for p in (opponent["bench"] or []) if p is not None]
        hosts = me_active + me_bench + opponent_active + opponent_bench
        my_host_count = len(me_active) + len(me_bench)
        host_dynamic = np.zeros((len(hosts) + 1, _DYNAMIC), dtype=np.float32)
        host_rich = np.zeros((len(hosts) + 1, _rich.RICH_CARD_DIM), dtype=np.float32)
        for position, pokemon in enumerate(hosts):
            host_dynamic[position + 1] = _encode._dynamic_block(
                _encode._light_pokemon(pokemon))
            host_rich[position + 1] = _rich.card_block(rich_cards.get(pokemon["serial"]))
        host_extra = None
        if self.v3:
            host_extra = np.zeros((len(hosts) + 1, _V3_EXTRA), dtype=np.float32)
            position = 0
            for player in (me, opponent):
                for slot, pokemon in _board_slots(player):
                    host_extra[position + 1] = _v3.instance_row(
                        pokemon, slot, slot == 0, rich_cards.get(pokemon.get("serial")))
                    position += 1

        # --- walk the emission order, recording per-row gather indices --------------- #
        statics, host_slots, beliefs, attack_slots = [], [], [], []
        card_ids, owner_ids, zone_ids = [], [], []
        knowledge_rows = []                       # (row index, card id, zone) for the v2 block
        # v3 only, recorded the same way `knowledge_rows` is (row index + the fact), so the
        # v2 walk keeps its exact instruction count: the TRUE un-clamped copy count of a
        # count-weighted row, and the two flag-only row kinds.
        v3 = self.v3
        count_rows, facedown_rows, deck_view_rows = [], [], []
        add_static, add_host = statics.append, host_slots.append
        add_belief, add_attack = beliefs.append, attack_slots.append
        add_card, add_owner, add_zone = card_ids.append, owner_ids.append, zone_ids.append

        def emit(card_id, zone, owner, static_slot, host_slot, belief, attack_slot):
            add_static(static_slot)
            add_host(host_slot)
            add_belief(belief)
            add_attack(attack_slot)
            add_card(int(card_id or 0))
            add_owner(owner)
            add_zone(zone)

        # 1. board: me active, me bench, opponent active, opponent bench
        host_slot = 1
        for group, owner, zone in ((me_active, OWNER_ME, ZONE_ACTIVE),
                                   (me_bench, OWNER_ME, ZONE_BENCH),
                                   (opponent_active, OWNER_OPPONENT, ZONE_ACTIVE),
                                   (opponent_bench, OWNER_OPPONENT, ZONE_BENCH)):
            for pokemon in group:
                card_id = pokemon["id"]
                emit(card_id, zone, owner, _card_slot(card_id), host_slot, 0.0, 0)
                host_slot += 1

        # 2. my hand
        for card in (me["hand"] or []):
            card_id = card["id"]
            emit(card_id, ZONE_HAND, OWNER_ME, _card_slot(card_id), 0, 0.0, 0)

        # 3. both discard piles, count-weighted (one row per distinct id, first-seen order)
        for player, owner in ((me, OWNER_ME), (opponent, OWNER_OPPONENT)):
            for card_id, count in Counter(card["id"]
                                          for card in (player["discard"] or [])).items():
                if v3:
                    count_rows.append((len(statics), count))
                emit(card_id, ZONE_DISCARD, owner, _card_slot(card_id), 0,
                     min(count, _MAX_COPIES) / _MAX_COPIES, 0)

        # 4. my hidden library (deck + prizes)
        if my_unseen:
            for card_id, count in my_unseen.items():
                knowledge_rows.append((len(statics), card_id, ZONE_DECK))
                if v3:
                    count_rows.append((len(statics), count))
                emit(card_id, ZONE_DECK, OWNER_ME, _card_slot(card_id), 0,
                     min(count, _MAX_COPIES) / _MAX_COPIES, 0)

        # 5. opponent belief: never passed by training (guarded at the top)

        # 6. capability tokens: per in-play Pokemon, attacks then abilities, both sides
        for position, pokemon in enumerate(hosts):
            owner = OWNER_ME if position < my_host_count else OWNER_OPPONENT
            card_id = pokemon["id"]
            card = get_card(card_id)
            slot = position + 1
            for attack_id in card["attacks"]:
                emit(card_id, ZONE_ATTACK, owner,
                     _attack_static_slot(card_id, attack_id), slot, 0.0,
                     _attack_block_slot(attack_id))
            for skill_index in range(len(card["skills"])):
                emit(card_id, ZONE_ABILITY, owner,
                     _ability_static_slot(card_id, skill_index), slot, 0.0, 0)

        # 7. attached entities (tools, energy cards, pre-evolutions -- they carry their HOST's
        #    live state), then the Stadium, then the cards being looked at
        for position, pokemon in enumerate(hosts):
            owner = OWNER_ME if position < my_host_count else OWNER_OPPONENT
            slot = position + 1
            for entities, zone in ((pokemon.get("tools") or [], _full.ZONE_TOOL),
                                   (pokemon.get("energyCards") or [], _full.ZONE_ENERGY_CARD),
                                   (pokemon.get("preEvolution") or [],
                                    _full.ZONE_PRE_EVOLUTION)):
                for entity in entities:
                    entity_id = entity["id"]
                    emit(entity_id, zone, owner, _card_slot(entity_id), slot, 0.0, 0)
        for stadium in (current.get("stadium") or []):
            if stadium is None:
                continue
            if stadium.get("playerIndex") == me_index:
                owner = OWNER_ME
            elif stadium.get("playerIndex") == 1 - me_index:
                owner = OWNER_OPPONENT
            else:
                owner = OWNER_NEUTRAL
            card_id = stadium["id"]
            emit(card_id, _full.ZONE_STADIUM, owner, _card_slot(card_id), 0, 0.0, 0)
        for looked in (current.get("looking") or []):
            if looked is None:
                continue
            card_id = looked["id"]
            emit(card_id, _full.ZONE_LOOKING,
                 OWNER_ME if looked.get("playerIndex") == me_index else OWNER_OPPONENT,
                 _card_slot(card_id), 0, 0.0, 0)

        # 8. knowledge tokens: prize belief, opponent-seen, known top
        if my_prize_belief:
            for card_id, expected in sorted(my_prize_belief.items(), key=lambda item: -item[1]):
                knowledge_rows.append((len(statics), card_id, _full.ZONE_PRIZE))
                if v3:
                    count_rows.append((len(statics), expected))
                emit(card_id, _full.ZONE_PRIZE, OWNER_ME, _card_slot(card_id), 0,
                     min(expected, 4.0) / 4.0, 0)
        if opponent_seen_hidden:
            for card_id, copies in sorted(opponent_seen_hidden.items(),
                                          key=lambda item: -item[1]):
                if v3:
                    count_rows.append((len(statics), copies))
                emit(card_id, _full.ZONE_OPPONENT_SEEN, OWNER_OPPONENT, _card_slot(card_id),
                     0, min(copies, 4.0) / 4.0, 0)
        if my_known_top:
            for position, card_id in enumerate(my_known_top):
                knowledge_rows.append((len(statics), card_id, _full.ZONE_DECK_TOP))
                if v3:
                    count_rows.append((len(statics), 1))
                emit(card_id, _full.ZONE_DECK_TOP, OWNER_ME, _card_slot(card_id), 0,
                     1.0 / (1.0 + position), 0)

        # 9. select-context cards
        select = observation.get("select") or {}
        for card, zone in ((select.get("effect"), _full.ZONE_EFFECT_SOURCE),
                           (select.get("contextCard"), _full.ZONE_CONTEXT_CARD)):
            if card is None or not card.get("id"):
                continue
            card_id = card["id"]
            emit(card_id, zone, OWNER_ME if card.get("playerIndex") == me_index
                 else OWNER_OPPONENT, _card_slot(card_id), 0, 0.0, 0)

        # 10/11. v3 token kinds, appended LAST (encode_full.include_v3_zones): one row per
        # unidentifiable in-play slot, then one per card of the deck view being searched.
        if v3:
            for player, owner in ((me, OWNER_ME), (opponent, OWNER_OPPONENT)):
                for slots, active in (((player["active"] or []), True),
                                      ((player["bench"] or []), False)):
                    for index, pokemon in enumerate(slots):
                        if pokemon is not None:
                            continue
                        facedown_rows.append((len(statics), 0 if active else index + 1,
                                              active))
                        # No id to embed: the base 340 columns are all zeros by design.
                        emit(0, _full.ZONE_FACEDOWN, owner, 0, 0, 0.0, 0)
            for card in (select.get("deck") or []):
                if card is None or not card.get("id"):
                    continue
                deck_view_rows.append(len(statics))
                card_id = card["id"]
                emit(card_id, _full.ZONE_DECK_VIEW,
                     OWNER_ME if card.get("playerIndex") == me_index else OWNER_OPPONENT,
                     _card_slot(card_id), 0, 0.0, 0)

        # --- assemble: a handful of whole-matrix operations -------------------------- #
        card_row_count = len(statics)
        events = history.events if history is not None else []
        history_count = min(len(events), self.history_cap)
        tokens = np.zeros((card_row_count + history_count, self.width), dtype=np.float32)
        if card_row_count:
            block = tokens[:card_row_count]
            block[:, :_STATIC] = _STATIC_TABLE.matrix[statics]
            host_slots = np.asarray(host_slots, dtype=np.intp)
            block[:, _STATIC:_BELIEF_COLUMN] = host_dynamic[host_slots]
            block[:, _BELIEF_COLUMN] = beliefs
            block[:, _ATTACK_START:_TOKEN] = _ATTACK_TABLE.matrix[attack_slots]
            block[:, _RICH_START:_BASE] = host_rich[host_slots]
            for row, card_id, zone in knowledge_rows:
                block[row, _BASE:_EXTRA_END] = _v2.card_knowledge_block(
                    summary, int(card_id), zone, my_deck_count)
            if v3:
                # Same three shapes as above: gather the host instance rows, then patch the
                # rows that carry a count / a face-down flag / a deck-view flag.
                extra = block[:, _WIDTH:]
                extra[:] = host_extra[host_slots]
                if count_rows:
                    rows = np.fromiter((row for row, _ in count_rows), dtype=np.intp,
                                       count=len(count_rows))
                    counts = np.fromiter((value for _, value in count_rows),
                                         dtype=np.float64, count=len(count_rows))
                    extra[rows, _v3.X_COPIES_WIDE] = \
                        np.minimum(counts, _v3.WIDE_COPIES) / _v3.WIDE_COPIES
                    extra[rows, _v3.X_COPIES_DECK] = \
                        np.minimum(counts, _v3.DECK_SIZE) / _v3.DECK_SIZE
                for row, slot, is_active in facedown_rows:
                    extra[row] = _v3.facedown_row(slot, is_active)
                for row in deck_view_rows:
                    extra[row, _v3.X_DECK_VIEW] = 1.0
        if history_count:
            window, v3_window = self._history_window(events, current.get("turn", 0) or 0)
            tokens[card_row_count:, _EXTRA_END:_WIDTH] = window
            if v3:
                tokens[card_row_count:, _WIDTH:] = v3_window
            for event in events[len(events) - history_count:]:
                actor = event["actor"]
                add_owner(OWNER_ME if actor == 0
                          else (OWNER_OPPONENT if actor == 1 else OWNER_NEUTRAL))
                add_zone(_v2.ZONE_HISTORY)
                identity = event["card_id"]
                add_card(identity if 0 < identity < _v2.CARD_VOCAB else 0)

        # --- globals: base 22 | full extras 5 | select context 17 | rich 14 | v2 6 ----- #
        light_state = {
            "turn_count": current["turn"],
            "supporter_played": current["supporterPlayed"],
            "stadium_played": current["stadiumPlayed"],
            "energy_attached": current["energyAttached"],
            "retreated": current["retreated"],
            "stadium": current["stadium"],
            "me": {"prize_count": len(me["prize"]), "deck_count": me["deckCount"],
                   "hand_count": me["handCount"], "poisoned": me["poisoned"],
                   "burned": me["burned"], "asleep": me["asleep"],
                   "paralyzed": me["paralyzed"], "confused": me["confused"]},
            "opponent": {"prize_count": len(opponent["prize"]),
                         "deck_count": opponent["deckCount"],
                         "hand_count": opponent["handCount"],
                         "poisoned": opponent["poisoned"], "burned": opponent["burned"],
                         "asleep": opponent["asleep"], "paralyzed": opponent["paralyzed"],
                         "confused": opponent["confused"]},
        }
        global_stack = [
            _encode._encode_global(light_state),
            np.array([me.get("benchMax", 5) / 8.0, opponent.get("benchMax", 5) / 8.0,
                      float(current.get("firstPlayer", 0) == me_index),
                      min(current.get("turnActionCount", 0), 10) / 10.0,
                      float(my_prize_certainty or 0.0)], dtype=np.float32),
            _full._select_context_block(observation),
            _rich.player_block(rich_players.get(me_index)),
            _rich.player_block(rich_players.get(1 - me_index)),
            _v2._v2_global_block(knowledge, history, observation),
        ]
        if v3:
            global_stack.append(_v3.global_extra(
                observation, _v3.facedown_counts(observation), len(deck_view_rows),
                max(0, len(events) - self.history_cap)))
        global_features = np.concatenate(global_stack).astype(np.float32)

        return {"token_features": tokens,
                "owner_ids": np.array(owner_ids, dtype=np.int64),
                "zone_ids": np.array(zone_ids, dtype=np.int64),
                "global_features": global_features,
                "card_ids": np.clip(np.array(card_ids, dtype=np.int64), 0,
                                    _v2.CARD_VOCAB - 1)}


class CachedV3Encoder(CachedV2Encoder):
    """The same encoder in the v3 layout: byte-identical to `encode_observation_v3`."""

    def __init__(self, history_cap=None):
        super().__init__(history_cap=history_cap, v3=True)


def compare(fast, slow):
    """-> list of field names that differ bit-for-bit (dtype, shape or bytes)."""
    bad = []
    for field in ("token_features", "owner_ids", "zone_ids", "global_features", "card_ids"):
        left, right = fast.get(field), slow.get(field)
        if (left is None) != (right is None):
            bad.append(field)
            continue
        if left is None:
            continue
        if left.dtype != right.dtype or left.shape != right.shape \
                or left.tobytes() != right.tobytes():
            bad.append(field)
    return bad


def self_check(encoder, observation, deck_counts=None, knowledge=None, history=None):
    """Encode both ways and return (differing field names, fast, slow)."""
    fast = encoder.encode(observation, deck_counts=deck_counts, knowledge=knowledge,
                          history=history)
    reference = (_v2.encode_observation_v3 if encoder.v3 else _v2.encode_observation_v2)
    slow = reference(observation, deck_counts=deck_counts, knowledge=knowledge,
                     history=history)
    return compare(fast, slow), fast, slow
