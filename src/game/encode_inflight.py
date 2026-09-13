"""v5 input surface: the 2026-07-30 whole-pool-audit fix tier.

Closes every encoder gap the audit left open (experiments/encoder_audit_2026_07_30/
ENCODER_AUDIT_REPORT.md; COVERAGE_PLAN A1-A6 + finding N1), owner-approved 2026-07-30:

  N1  REVEALED PRIZES  -- a face-up card in EITHER prize pile becomes a token
      (ZONE_PRIZE_REVEALED, owner-tagged). Prize entries are None while face-down, so the
      rows exist only during reveals: zero steady-state cost.
  A1/A3/A4  IN-FLIGHT PICK MEMORY -- during a chained effect (SWITCH_ENERGY -> ATTACH_FROM,
      Energy Switch, attach-from-deck) the model finally sees WHICH board Pokemon the
      in-flight card came from / sits on: a per-token host marker, two per-option columns,
      and an in-flight-active global. Driven ONLY by engine facts: the select's own
      `contextCard` (bound to its host by serial scan) and picks the engine has ALREADY
      ACCEPTED via battle_select -- never by unconfirmed intent (fact-safe rule).
  A2  RESOLVED COUNT -- the number chosen at a COUNT step rides as a global through the
      rest of the chain (Munkidori: the placement step knows whether 10/20/30 lands).
  A5  HISTORY SERIALS -- ActionHistory records now carry the engine serial; history rows
      gain a "this exact copy is still in play" column.
  A6  MULLIGAN + FACE-DOWN MOVES -- ActionHistory(extended=True) emits HAS_BASIC_POKEMON
      reveals (KIND_MULLIGAN) and face-down card moves (KIND_MOVE_CARD, card_id 0, areas
      live); v5 encodes both (mulligan rows are built HERE because the frozen history
      block's kind one-hot has no 13th slot).

Layout discipline (the v3/v4 rule): every v5 block is APPENDED -- after the v3 token
columns, after the v4 option columns, after the v3 globals -- and every new token row is
appended after all existing rows, so a v3/v4 slice of a v5 array is exactly the v3/v4
array. v1-v4 modules are composed, not edited; the ONLY sibling edit is the additive
`extended` flag on ActionHistory.

The 32-event history clip (owner property) is enforced over the COMBINED event stream
(normal + extended), so total history rows never exceed 32.

Dims (next-run config): TOKEN_FEATURE_DIM_V5 / GLOBAL_FEATURE_DIM_V5 (+ the v4 loop's 3)
/ OPTION_FEATURE_DIM_V5; zones < NUM_ZONES_V5.
"""

from types import SimpleNamespace

import numpy as np

from src.game.action_history import KIND_MOVE_CARD, KIND_MULLIGAN, NUM_KINDS
from src.game.encode import _card_static_features
from src.game.encode_history import (_HISTORY_SCALAR_OFFSET, CARD_VOCAB, GLOBAL_FEATURE_DIM_V3,
                                HISTORY_FEATURE_DIM, HISTORY_SCALAR_DIM, NUM_ZONES_V3,
                                TOKEN_FEATURE_DIM_V3, V2_CARD_EXTRA_DIM, ZONE_HISTORY,
                                encode_observation_v3)
from src.game.encode_full import TOKEN_FEATURE_DIM_FULL_RICH
from src.game.encode_details import (V3_GLOBAL_EXTRA_DIM, _entity_at, _facedown_slots, _slots)
from src.game.encode_selection import (OPTION_FEATURE_DIM_V4, MultiSelect, base_option_matrix,
                                candidate_matrix, forced_answer)
from src.game.encode_selection import global_features as global_features_v4
from src.cards import get_card
from src.segments import OWNER_ME, OWNER_NEUTRAL, OWNER_OPPONENT

# --- per-token extra columns -------------------------------------------------------- #
T_IN_FLIGHT_HOST = 0        # this row's host Pokemon holds / was the source of the
                            # in-flight card (A1/A3/A4)
T_HISTORY_MULLIGAN = 1      # this history row is a mulligan reveal (A6)
T_HISTORY_FACEDOWN = 2      # this history row is a face-down card move (A6)
T_HISTORY_STILL_IN_PLAY = 3  # this history event's exact copy (serial) is on board (A5)
V5_TOKEN_EXTRA_DIM = 4

# --- per-option extra columns (after the v4 pair) ----------------------------------- #
O_TARGETS_IN_FLIGHT_HOST = 0  # option points at the in-flight host Pokemon
O_PICKS_IN_FLIGHT_CARD = 1    # option picks the in-flight card itself (by serial)
V5_OPTION_EXTRA_DIM = 2

# --- extra globals (after the v3 globals; the v4 loop's 3 are appended at forward time) #
G_IN_FLIGHT_ACTIVE = 0
G_IN_FLIGHT_COUNT = 1       # the engine-resolved COUNT pick / 10 (A2)
V5_GLOBAL_EXTRA_DIM = 2
COUNT_SCALE = 10.0

ZONE_PRIZE_REVEALED = NUM_ZONES_V3          # one new zone id, appended after the v3 set
NUM_ZONES_V5 = NUM_ZONES_V3 + 1

TOKEN_FEATURE_DIM_V5 = TOKEN_FEATURE_DIM_V3 + V5_TOKEN_EXTRA_DIM
GLOBAL_FEATURE_DIM_V5 = GLOBAL_FEATURE_DIM_V3 + V5_GLOBAL_EXTRA_DIM
OPTION_FEATURE_DIM_V5 = OPTION_FEATURE_DIM_V4 + V5_OPTION_EXTRA_DIM

HISTORY_CAP_V5 = 32                          # the owner's clip, COMBINED stream

_AREA_ACTIVE, _AREA_BENCH = 4, 5
_CONTEXT_MAIN = 0
_NON_CHAIN_CONTEXTS = frozenset({0, 1, 2, 41, 42})   # MAIN / setup / is-first / mulligan
_COUNT_CONTEXTS = frozenset({38, 39, 40})            # draw / damage-counter / remove counts


# ==================================================================================== #
# In-flight tracker (A1 / A2 / A3 / A4)
# ==================================================================================== #

def _host_of_serial(observation, serial):
    """The board Pokemon that IS or HOLDS the card with this serial, or None. Pure engine
    state scan: a Pokemon matches by its own serial or by an attached card's (tools /
    energyCards / preEvolution)."""
    if not serial:
        return None
    for player in observation["current"]["players"]:
        for pokemon, _slot, _is_active in _slots(player):
            if pokemon.get("serial") == serial:
                return pokemon
            for attached in ((pokemon.get("tools") or [])
                             + (pokemon.get("energyCards") or [])
                             + (pokemon.get("preEvolution") or [])):
                if attached and attached.get("serial") == serial:
                    return pokemon
    return None


class InFlightTracker:
    """Per-seat, per-game memory of the pick IN FLIGHT during a chained effect.

    Fact-safety (COVERAGE_PLAN rule): every field comes from the engine -- either the
    select's own `contextCard` (the engine names the in-flight card) or a pick the engine
    has ALREADY ACCEPTED via battle_select. Nothing is inferred about what the effect
    means; the tracker only remembers WHERE an accepted pick came from.

    Call order per decision: `observe(observation, select)` BEFORE encoding, and
    `record(observation, select, indices)` immediately AFTER the engine accepts
    `battle_select(indices)`. One tracker per seat; `reset()` between games."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.effect_serial = None
        self.host_serial = 0            # board Pokemon the in-flight card came from / sits on
        self.card_serial = 0            # the in-flight card itself, when the engine names it
        self.card_id = 0
        self.count = 0                  # the engine-resolved COUNT pick (A2)
        self.active = False
        # v6 addition (2026-08-03 chain-freeze audit): accepted picks per TARGET Pokemon
        # serial within the current effect chain. Batched effects (Phantom Dive) freeze
        # the observation across their placement selects, so this is the only record of
        # where earlier counters went. Written here, read ONLY by state_encoder -- v5
        # encodes never touch it, so v5 outputs stay byte-identical.
        self.chain_picks = {}

    def observe(self, observation, select):
        """Chain lifecycle + contextCard binding, from the CURRENT prompt."""
        if select is None:
            self.reset()
            return
        if select.get("context") in _NON_CHAIN_CONTEXTS:
            self.reset()
            return
        effect = select.get("effect") or {}
        effect_serial = effect.get("serial") or 0
        if effect_serial != self.effect_serial:
            self.reset()                              # a new effect chain begins
            self.effect_serial = effect_serial
        context_card = select.get("contextCard")
        if isinstance(context_card, dict) and context_card.get("serial"):
            self.card_serial = context_card["serial"]
            self.card_id = context_card.get("id") or 0
            host = _host_of_serial(observation, self.card_serial)
            if host is not None:
                self.host_serial = host.get("serial") or 0
            self.active = True

    def record(self, observation, select, indices):
        """Remember what the engine just ACCEPTED. Only chained-effect selects are
        recorded -- MAIN-context plays carry no in-flight state."""
        if select is None or select.get("context") in _NON_CHAIN_CONTEXTS:
            return
        if not select.get("effect"):
            return
        me_index = observation["current"]["yourIndex"]
        options = select["option"]
        for index in indices:
            if not 0 <= index < len(options):
                continue
            option = options[index]
            number = option.get("number")
            if number is not None and select.get("context") in _COUNT_CONTEXTS:
                self.count = int(number)
                self.active = True
            area = option.get("area")
            if area in (_AREA_ACTIVE, _AREA_BENCH):
                owner = option.get("playerIndex")
                owner = me_index if owner is None else owner
                _card, pokemon = _entity_at(observation, area, option.get("index"), owner)
                if pokemon is not None:
                    target_serial = pokemon.get("serial") or 0
                    if target_serial:
                        self.chain_picks[target_serial] = \
                            self.chain_picks.get(target_serial, 0) + 1
                    self.host_serial = target_serial
                    self.active = True
                    energy_index = option.get("energyIndex")
                    if energy_index is not None:
                        cards = pokemon.get("energyCards") or []
                        if 0 <= energy_index < len(cards):
                            self.card_serial = cards[energy_index].get("serial") or 0
                            self.card_id = cards[energy_index].get("id") or 0
                    tool_index = option.get("toolIndex")
                    if tool_index is not None:
                        cards = pokemon.get("tools") or []
                        if 0 <= tool_index < len(cards):
                            self.card_serial = cards[tool_index].get("serial") or 0
                            self.card_id = cards[tool_index].get("id") or 0


# ==================================================================================== #
# Row-walk mirror: per-CARD-row host serial (for the T_IN_FLIGHT_HOST column)
# ==================================================================================== #

def host_serial_rows(observation, my_unseen=None, opponent_belief=None, belief_top_k=None,
                     my_prize_belief=None, opponent_seen_hidden=None, my_known_top=None):
    """One int per CARD row of the v3 encoding, in emission order: the serial of the row's
    host Pokemon (board / capability / attached-entity rows), the row's own card serial
    (select-context rows), else 0. Mirrors encode_details.token_extra_rows' walk exactly and is
    asserted against the encoder's card-row count (same drift tripwire as v3)."""
    from collections import Counter

    current = observation["current"]
    me_index = current["yourIndex"]
    me, opponent = current["players"][me_index], current["players"][1 - me_index]
    rows = []

    for player in (me, opponent):                                     # 1. board
        rows.extend((pokemon.get("serial") or 0)
                    for pokemon, _slot, _is_active in _slots(player))
    rows.extend(0 for _ in (me["hand"] or []))                        # 2. my hand
    for player in (me, opponent):                                     # 3. discards
        rows.extend(0 for _ in Counter(card["id"]
                                       for card in (player["discard"] or [])))
    if my_unseen:                                                     # 4. my hidden library
        rows.extend(0 for _ in my_unseen)
    if opponent_belief:                                               # 5. opponent belief
        predicted = sorted(opponent_belief.items(), key=lambda item: -item[1])
        if belief_top_k is not None:
            predicted = predicted[:belief_top_k]
        rows.extend(0 for _ in predicted)
    for player in (me, opponent):                                     # 6. capability tokens
        for pokemon, _slot, _is_active in _slots(player):
            card = get_card(pokemon["id"])
            serial = pokemon.get("serial") or 0
            rows.extend(serial for _ in card["attacks"])
            rows.extend(serial for _ in card["skills"])
    for player in (me, opponent):                                     # 7. attached entities
        for pokemon, _slot, _is_active in _slots(player):
            serial = pokemon.get("serial") or 0
            attached = (len(pokemon.get("tools") or [])
                        + len(pokemon.get("energyCards") or [])
                        + len(pokemon.get("preEvolution") or []))
            rows.extend(serial for _ in range(attached))
    rows.extend(0 for stadium in (current.get("stadium") or [])       # stadium
                if stadium is not None)
    rows.extend(0 for looked in (current.get("looking") or [])        # looking
                if looked is not None)
    if my_prize_belief:                                               # 8. knowledge tokens
        rows.extend(0 for _ in my_prize_belief)
    if opponent_seen_hidden:
        rows.extend(0 for _ in opponent_seen_hidden)
    if my_known_top:
        rows.extend(0 for _ in my_known_top)
    select = observation.get("select") or {}                          # 9. context cards
    for card in (select.get("effect"), select.get("contextCard")):
        if card is not None and card.get("id"):
            rows.append(card.get("serial") or 0)
    for player in (me, opponent):                                     # 10. face-down slots
        rows.extend(0 for _ in _facedown_slots(player))
    rows.extend(0 for card in (select.get("deck") or [])              # 11. deck view
                if card is not None and card.get("id"))
    return rows


def _board_serials(observation):
    """Every serial currently visible in play (Pokemon + everything attached), both sides."""
    serials = set()
    for player in observation["current"]["players"]:
        for pokemon, _slot, _is_active in _slots(player):
            serials.add(pokemon.get("serial") or 0)
            for attached in ((pokemon.get("tools") or [])
                             + (pokemon.get("energyCards") or [])
                             + (pokemon.get("preEvolution") or [])):
                if attached:
                    serials.add(attached.get("serial") or 0)
    serials.discard(0)
    return serials


# ==================================================================================== #
# Appended rows: revealed prizes (N1) + mulligan history rows (A6)
# ==================================================================================== #

def _revealed_prizes(observation):
    """(card_id, owner) per FACE-UP prize entry. Entries are None while face-down, so this
    is empty outside reveal effects."""
    current = observation["current"]
    me_index = current["yourIndex"]
    revealed = []
    for player_index, player in enumerate(current["players"]):
        owner = OWNER_ME if player_index == me_index else OWNER_OPPONENT
        for entry in (player.get("prize") or []):
            if isinstance(entry, dict) and (entry.get("id") or 0) > 0:
                revealed.append((entry["id"], owner))
    return revealed


def _prize_row(card_id):
    """A full-width v3 token row for a revealed prize card: the static card block up front
    (same leading layout as every card row), zeros elsewhere -- the ZONE_PRIZE_REVEALED
    embedding carries the 'this is a revealed prize' meaning."""
    row = np.zeros(TOKEN_FEATURE_DIM_V3, dtype=np.float32)
    static = _card_static_features(card_id)
    row[:len(static)] = static
    return row


def _mulligan_row(event, current_turn):
    """A full-width v3 token row for a KIND_MULLIGAN event. Built here because the frozen
    history block's kind one-hot has no slot for kind 12: actor + scalars + fingerprint are
    laid out exactly like history_block, the kind one-hot stays zero, and the v5
    T_HISTORY_MULLIGAN column (added by the caller) says what it is."""
    features = np.zeros(HISTORY_FEATURE_DIM, dtype=np.float32)
    actor = min(max(event["actor"], 0), 2)
    features[actor] = 1.0
    offset = _HISTORY_SCALAR_OFFSET
    features[offset + 1] = float(event["flag"])       # hasBasicPokemon
    features[offset + 6] = min(event["index_in_turn"], 20) / 20.0
    features[offset + HISTORY_SCALAR_DIM:
             offset + HISTORY_SCALAR_DIM + len(event["fingerprint"])] = event["fingerprint"]
    turns_ago = max(0, current_turn - event["turn"])
    features[offset + 4] = min(turns_ago, 20) / 20.0
    features[offset + 5] = 1.0 / (1.0 + turns_ago)
    features[offset + 7] = float(turns_ago == 0)
    row = np.zeros(TOKEN_FEATURE_DIM_V3, dtype=np.float32)
    start = TOKEN_FEATURE_DIM_FULL_RICH + V2_CARD_EXTRA_DIM
    row[start:start + HISTORY_FEATURE_DIM] = features
    return row


_OWNER_BY_ACTOR = {0: OWNER_ME, 1: OWNER_OPPONENT, 2: OWNER_NEUTRAL}


# ==================================================================================== #
# The state encoder
# ==================================================================================== #

def encode_observation_v5(observation, deck_counts=None, knowledge=None, history=None,
                          in_flight=None, opponent_belief=None, belief_top_k=None,
                          history_cap=HISTORY_CAP_V5, rich_override=None):
    """v5 = v3 ++ the audit fix tier. Same output schema as encode_observation_v3;
    widths TOKEN_FEATURE_DIM_V5 / GLOBAL_FEATURE_DIM_V5, zones < NUM_ZONES_V5.

    history: an ActionHistory(extended=True) for this seat (a plain one also works --
    the extended rows are simply absent). in_flight: this seat's InFlightTracker (None ->
    the in-flight columns stay zero)."""
    # --- the owner's 32-event clip, over the COMBINED stream --------------------------- #
    all_events = history.events if history is not None else []
    dropped = 0
    window = all_events
    if history_cap is not None:
        dropped = max(0, len(all_events) - history_cap)
        window = all_events[-history_cap:]
    normal_events = [event for event in window if event["kind"] < NUM_KINDS]
    mulligan_events = [event for event in window if event["kind"] == KIND_MULLIGAN]
    view = None
    if history is not None:
        view = SimpleNamespace(events=normal_events, turns_seen=history.turns_seen)

    base = encode_observation_v3(observation, deck_counts=deck_counts, knowledge=knowledge,
                                 history=view, opponent_belief=opponent_belief,
                                 belief_top_k=belief_top_k, history_cap=None,
                                 rich_override=rich_override)
    global_features = base["global_features"].copy()
    # The view hides the pre-window stream, so two derived globals read low: the dropped-
    # events scalar (last v3 extra) and the total-events scalar (first of the v2 block --
    # which is all-zeros when knowledge is None, a contract the fixup must respect).
    global_features[-1] = min(dropped, 200) / 200.0
    if knowledge is not None:
        global_features[len(global_features) - V3_GLOBAL_EXTRA_DIM - 6] = \
            min(len(all_events), 200) / 200.0

    token_features = base["token_features"]
    owner_ids = base["owner_ids"]
    zone_ids = base["zone_ids"]
    card_ids = base["card_ids"]
    card_row_count = token_features.shape[0] - len(normal_events)
    current_turn = observation["current"].get("turn", 0) or 0

    # --- appended rows: mulligan history (A6), then revealed prizes (N1), LAST -------- #
    extra_rows, extra_owners, extra_zones, extra_card_ids = [], [], [], []
    extra_flags = []                                   # the row's v5 columns
    for event in mulligan_events:
        extra_rows.append(_mulligan_row(event, current_turn))
        extra_owners.append(_OWNER_BY_ACTOR.get(event["actor"], OWNER_NEUTRAL))
        extra_zones.append(ZONE_HISTORY)
        extra_card_ids.append(0)
        flags = np.zeros(V5_TOKEN_EXTRA_DIM, dtype=np.float32)
        flags[T_HISTORY_MULLIGAN] = 1.0
        extra_flags.append(flags)
    for card_id, owner in _revealed_prizes(observation):
        extra_rows.append(_prize_row(card_id))
        extra_owners.append(owner)
        extra_zones.append(ZONE_PRIZE_REVEALED)
        extra_card_ids.append(min(card_id, CARD_VOCAB - 1))
        extra_flags.append(np.zeros(V5_TOKEN_EXTRA_DIM, dtype=np.float32))

    # --- the v5 column block for every row -------------------------------------------- #
    total_rows = token_features.shape[0] + len(extra_rows)
    v5_columns = np.zeros((total_rows, V5_TOKEN_EXTRA_DIM), dtype=np.float32)

    if in_flight is not None and in_flight.active and in_flight.host_serial:
        knowledge_kwargs = {}
        if knowledge is not None:
            knowledge_kwargs = {
                "my_prize_belief": knowledge.prize_belief(),
                "opponent_seen_hidden": knowledge.opponent_not_prized_counts(observation),
                "my_known_top": knowledge.known_top_ids(),
            }
        from src.game.encode import unseen_my_counts
        my_unseen = unseen_my_counts(observation, deck_counts) if deck_counts else None
        serial_rows = host_serial_rows(
            observation, my_unseen=my_unseen, opponent_belief=opponent_belief,
            belief_top_k=belief_top_k, **knowledge_kwargs)
        assert len(serial_rows) == card_row_count, (
            f"v5 serial mirror drifted from the encoder: {len(serial_rows)} vs "
            f"{card_row_count} card rows -- host_serial_rows must track the emission order")
        for row, serial in enumerate(serial_rows):
            if serial and serial == in_flight.host_serial:
                v5_columns[row, T_IN_FLIGHT_HOST] = 1.0

    board = _board_serials(observation)
    for offset, event in enumerate(normal_events):
        row = card_row_count + offset
        if event["kind"] == KIND_MOVE_CARD and event["card_id"] == 0:
            v5_columns[row, T_HISTORY_FACEDOWN] = 1.0
        if event.get("serial") and event["serial"] in board:
            v5_columns[row, T_HISTORY_STILL_IN_PLAY] = 1.0
    base_extra = token_features.shape[0]
    for offset, flags in enumerate(extra_flags):
        v5_columns[base_extra + offset] = flags

    # --- assemble --------------------------------------------------------------------- #
    if extra_rows:
        token_features = np.concatenate([token_features, np.stack(extra_rows)])
        owner_ids = np.concatenate([owner_ids,
                                    np.array(extra_owners, dtype=owner_ids.dtype)])
        zone_ids = np.concatenate([zone_ids, np.array(extra_zones, dtype=zone_ids.dtype)])
        card_ids = np.concatenate([card_ids,
                                   np.array(extra_card_ids, dtype=card_ids.dtype)])
    token_features = np.concatenate([token_features, v5_columns], axis=1)

    in_flight_globals = np.zeros(V5_GLOBAL_EXTRA_DIM, dtype=np.float32)
    if in_flight is not None:
        in_flight_globals[G_IN_FLIGHT_ACTIVE] = float(in_flight.active)
        in_flight_globals[G_IN_FLIGHT_COUNT] = min(in_flight.count, COUNT_SCALE) / COUNT_SCALE
    global_features = np.concatenate([global_features, in_flight_globals])

    return {"token_features": token_features.astype(np.float32),
            "owner_ids": owner_ids, "zone_ids": zone_ids,
            "global_features": global_features.astype(np.float32),
            "card_ids": card_ids}


# ==================================================================================== #
# The option side + the v4 selection loop with v5 columns
# ==================================================================================== #

def _option_host_and_pick(observation, option, me_index):
    """(host serial the option points at, the attached/primary card serial it picks)."""
    owner = option.get("playerIndex")
    owner = me_index if owner is None else owner
    host_serial = pick_serial = 0
    for area, index in ((option.get("area"), option.get("index")),
                        (option.get("inPlayArea"), option.get("inPlayIndex"))):
        if area in (_AREA_ACTIVE, _AREA_BENCH) and index is not None:
            _card, pokemon = _entity_at(observation, area, index, owner)
            if pokemon is not None and not host_serial:
                host_serial = pokemon.get("serial") or 0
                energy_index = option.get("energyIndex")
                if energy_index is not None:
                    cards = pokemon.get("energyCards") or []
                    if 0 <= energy_index < len(cards):
                        pick_serial = cards[energy_index].get("serial") or 0
                tool_index = option.get("toolIndex")
                if tool_index is not None:
                    cards = pokemon.get("tools") or []
                    if 0 <= tool_index < len(cards):
                        pick_serial = cards[tool_index].get("serial") or 0
    if not pick_serial:
        card, _pokemon = _entity_at(observation, option.get("area"), option.get("index"),
                                    owner)
        if card is not None:
            pick_serial = card.get("serial") or 0
    return host_serial, pick_serial


def option_extra_v5(observation, option, in_flight):
    """The V5_OPTION_EXTRA_DIM columns for one option."""
    columns = np.zeros(V5_OPTION_EXTRA_DIM, dtype=np.float32)
    if in_flight is None or not in_flight.active:
        return columns
    me_index = observation["current"]["yourIndex"]
    host_serial, pick_serial = _option_host_and_pick(observation, option, me_index)
    if in_flight.host_serial and host_serial == in_flight.host_serial:
        columns[O_TARGETS_IN_FLIGHT_HOST] = 1.0
    if in_flight.card_serial and pick_serial == in_flight.card_serial:
        columns[O_PICKS_IN_FLIGHT_CARD] = 1.0
    return columns


def base_option_matrix_v5(observation, select, in_flight):
    """(v3 rows, v5 extra rows) for one select, both encoded once per select."""
    v3_matrix = base_option_matrix(observation, select)
    v5_extra = np.stack([option_extra_v5(observation, option, in_flight)
                         for option in select["option"]]).astype(np.float32)
    return v3_matrix, v5_extra


def candidate_matrix_v5(v3_matrix, v5_extra, multiselect, pending, stop_offered):
    """[K, OPTION_FEATURE_DIM_V5]: the v4 candidate rows ++ the v5 columns (zeros on the
    STOP row -- STOP points at nothing)."""
    v4_rows = candidate_matrix(v3_matrix, multiselect, pending, stop_offered)
    rows = np.zeros((v4_rows.shape[0], OPTION_FEATURE_DIM_V5), dtype=np.float32)
    rows[:, :OPTION_FEATURE_DIM_V4] = v4_rows
    if pending:
        rows[:len(pending), OPTION_FEATURE_DIM_V4:] = v5_extra[pending]
    return rows


def global_features_v5(base_globals, multiselect):
    """base_globals is the GLOBAL_FEATURE_DIM_V5 state vector; the v4 loop's three pick
    scalars are appended after it, same as v4."""
    return global_features_v4(base_globals, multiselect)


def resolve_with_v5(observation, select, choose, in_flight, stats=None):
    """encode_selection.resolve_with with v5 candidate rows. The two must stay in step."""
    forced, reason = forced_answer(select)
    if forced is not None:
        if stats is not None:
            stats["forced"] += 1
            stats["forced_" + reason] += 1
        return forced
    state = MultiSelect(observation, select)
    v3_matrix, v5_extra = base_option_matrix_v5(observation, select, in_flight)
    while not state.complete():
        forced_index = state.forced_index()
        if forced_index is not None:
            if stats is not None:
                stats["forced_pick"] += 1
            state.take(forced_index)
            continue
        pending = state.pending()
        stop_offered = state.stop_offered()
        rows = candidate_matrix_v5(v3_matrix, v5_extra, state, pending, stop_offered)
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
