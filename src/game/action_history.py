"""Ordered action history: one record per meaningful game event, as SEEN by one seat.

The board encoders are stateless -- each decision sees a snapshot, so play SEQUENCES are
invisible to the model ("they played Iono last turn", "they passed holding eight cards",
"they discarded a Rare Candy to an Ultra Ball"). This module keeps the ordered event stream
that the v2 encoder turns into history tokens (src/game/encode_history.py).

VISIBLE-ONLY (owner directive 2026-07-21): every record is something the engine reported to
THIS seat -- its own selections plus the incremental `logs` stream. Nothing is inferred or
synthesized. In particular opponent ABILITY use is NOT reconstructed from effect signatures:
the engine has no ability-use log at all (LogType 0..23), so the raw effect events (HP
changes, attached-card moves, status) go in uninterpreted and the model learns that
inference itself. A wrong guess written here would reach the model as fact.

Contract: call `update(observation)` EXACTLY ONCE per received observation -- logs are
incremental ("events since this seat's last selection"), so double-updating double-counts.
One tracker per SEAT per game (each seat sees its own log deliveries); `reset()` between
games. Enum ints are inlined (LogType / AreaType values) so this module stays DLL-free and
bundle-safe, like the rest of the trackers.

Fingerprint caveat: logs arrive BATCHED -- every event since our last selection lands in one
observation -- so the exact board state at each event is not recoverable from what we are
allowed to see. Each event carries the fingerprint of the observation where it was OBSERVED,
which is the honest visible approximation (and is what a human reading the log gets too).
"""

import numpy as np

# Inlined enum values (cg.api, stable per the official schema).
_LOG_HAS_BASIC_POKEMON = 1
_LOG_TURN_START, _LOG_TURN_END = 2, 3
_LOG_MOVE_CARD = 6
_LOG_SWITCH, _LOG_CHANGE = 8, 9
_LOG_PLAY, _LOG_ATTACH, _LOG_EVOLVE, _LOG_DEVOLVE = 10, 11, 12, 13
_LOG_MOVE_ATTACHED, _LOG_ATTACK, _LOG_HP_CHANGE = 14, 15, 16
_LOG_POISONED, _LOG_BURNED, _LOG_ASLEEP, _LOG_PARALYZED, _LOG_CONFUSED = 17, 18, 19, 20, 21
_LOG_COIN = 22

# AreaType 1..12 are the documented public values; the engine ALSO logs two internal ones it
# never documents -- Playing = 13 and DeckBottom = 14 (Types.h; CardMove.h logs toArea RAW
# while collapsing bottom->Deck for the state). Verified live: MOVE_CARD toArea=14 carries
# full identity, which is how "we put this exact card on the bottom" becomes an engine FACT.
AREA_DECK_BOTTOM = 14
NUM_AREAS = 15                     # 0 = absent, 1..14 real areas

# Event kinds (the history token vocabulary). PLAY/ATTACH/EVOLVE/ATTACK/retreat are the
# owner-specified action set; the rest are the RAW effect events that carry card identity.
KIND_PLAY = 0
KIND_ATTACH = 1
KIND_EVOLVE = 2
KIND_DEVOLVE = 3
KIND_MOVE_ATTACHED = 4
KIND_ATTACK = 5
KIND_SWITCH = 6                    # retreat / effect switch (Pokemon swapped)
KIND_CHANGE = 7                    # the in-play Pokemon itself was replaced
KIND_MOVE_CARD = 8                 # a face-up card moved zones (identity revealed)
KIND_HP_CHANGE = 9
KIND_STATUS = 10                   # poisoned / burned / asleep / paralyzed / confused
KIND_COIN = 11
NUM_KINDS = 12
# v5 EXTENDED kind (2026-07-30 audit, A6): deliberately OUTSIDE NUM_KINDS -- the v2/v3
# history block one-hots `kind` over NUM_KINDS slots, so a 12 would overflow into the
# status block. Emitted only by ActionHistory(extended=True) and consumed only by the v5
# encoder, which builds these rows itself (encode_inflight.py).
KIND_MULLIGAN = 12                 # HAS_BASIC_POKEMON log; flag = hasBasicPokemon

_STATUS_LOGS = {_LOG_POISONED: 0, _LOG_BURNED: 1, _LOG_ASLEEP: 2,
                _LOG_PARALYZED: 3, _LOG_CONFUSED: 4}
NUM_STATUS_TYPES = 5

FINGERPRINT_DIM = 16


def _energy_count(player_state):
    total = 0
    for pokemon in (player_state.get("active") or []) + (player_state.get("bench") or []):
        if pokemon is not None:
            total += len(pokemon.get("energies") or [])
    return total


def _damage_total(player_state):
    total = 0
    for pokemon in (player_state.get("active") or []) + (player_state.get("bench") or []):
        if pokemon is not None:
            total += max(0, (pokemon.get("maxHp") or 0) - (pokemon.get("hp") or 0))
    return total


def _active_hp_fraction(player_state):
    for pokemon in (player_state.get("active") or []):
        if pokemon is not None and (pokemon.get("maxHp") or 0) > 0:
            return (pokemon.get("hp") or 0) / pokemon["maxHp"]
    return 0.0


def _bench_count(player_state):
    return sum(1 for pokemon in (player_state.get("bench") or []) if pokemon is not None)


def fingerprint(observation, me_index):
    """FINGERPRINT_DIM me-centric scalars: 'what the situation looked like' when an event was
    observed. Public information only (hand SIZES, not contents)."""
    current = observation["current"]
    me = current["players"][me_index]
    opponent = current["players"][1 - me_index]
    turn = current.get("turn", 0) or 0
    first_player = current.get("firstPlayer", -1)
    if first_player < 0 or turn <= 0:
        my_turn = 0.0
    else:
        turn_player = first_player if turn % 2 == 1 else 1 - first_player
        my_turn = float(turn_player == me_index)
    return np.array([
        len(me.get("prize") or []) / 6.0,
        len(opponent.get("prize") or []) / 6.0,
        min(me.get("handCount") or 0, 15) / 15.0,
        min(opponent.get("handCount") or 0, 15) / 15.0,
        _bench_count(me) / 8.0,
        _bench_count(opponent) / 8.0,
        min(me.get("deckCount") or 0, 60) / 60.0,
        min(opponent.get("deckCount") or 0, 60) / 60.0,
        min(_damage_total(me), 500) / 500.0,
        min(_damage_total(opponent), 500) / 500.0,
        min(_energy_count(me), 15) / 15.0,
        min(_energy_count(opponent), 15) / 15.0,
        min(turn, 40) / 40.0,
        my_turn,
        _active_hp_fraction(me),
        _active_hp_fraction(opponent),
    ], dtype=np.float32)


class ActionHistory:
    """Per-seat, per-game ordered event stream. `events` is oldest-first; each record is a
    plain dict (see `_record`) carrying the engine's own fields plus the turn coordinates
    (`turn`, `index_in_turn`) that the encoder turns into ordering features.

    Records events ONLY. Any summarising of them ("they passed holding eight cards", "three
    turns since their last supporter") is the MODEL's job, not this tracker's -- hand-designed
    behavioural features are exactly what the owner ruled out for the learned agent.

    extended=False (the default) is byte-identical to the historical behavior -- every
    pre-v5 checkpoint keeps its exact input distribution. extended=True (v5 runs, 2026-07-30
    audit fixes A5/A6) additionally emits: mulligan reveals (KIND_MULLIGAN, flag =
    hasBasicPokemon) and FACE-DOWN card moves (KIND_MOVE_CARD with card_id 0 -- the
    from/to areas are the information: "an unknown card went deck-bottom"). Every record
    also carries the engine's `serial` (A5: which physical copy), which old encoders simply
    ignore."""

    def __init__(self, extended=False):
        self.extended = extended
        self.reset()

    def reset(self):
        self.events = []
        self.turns_seen = 0            # highest turn number observed
        self._turn_counts = {}         # turn -> events recorded so far in that turn

    # ------------------------------------------------------------------ #
    def update(self, observation):
        current = observation["current"]
        me_index = current["yourIndex"]
        turn = current.get("turn", 0) or 0
        self.turns_seen = max(self.turns_seen, turn)
        state = fingerprint(observation, me_index)

        for log in (observation.get("logs") or []):
            log_type = log.get("type")
            if log_type == _LOG_TURN_START or log_type == _LOG_TURN_END:
                continue                              # bookkeeping only; `turn` carries it
            record = self._translate(log, log_type, me_index)
            if record is None:
                continue
            record["turn"] = turn
            record["index_in_turn"] = self._turn_counts.get(turn, 0)
            self._turn_counts[turn] = record["index_in_turn"] + 1
            record["fingerprint"] = state
            self.events.append(record)
        return self

    def _translate(self, log, log_type, me_index):
        """One log entry -> one history record, or None for the events that carry no
        strategic content (face-down movement, draws, shuffles, setup, result)."""
        player_index = log.get("playerIndex")
        if player_index is None:
            actor = 2                                 # neutral (no owning player)
        elif player_index == me_index:
            actor = 0
        else:
            actor = 1

        if log_type == _LOG_PLAY:
            return self._record(KIND_PLAY, actor, card_id=log.get("cardId", 0),
                                serial=log.get("serial"))
        if log_type == _LOG_ATTACH:
            return self._record(KIND_ATTACH, actor, card_id=log.get("cardId", 0),
                                target_id=log.get("cardIdTarget", 0),
                                serial=log.get("serial"))
        if log_type == _LOG_EVOLVE:
            return self._record(KIND_EVOLVE, actor, card_id=log.get("cardId", 0),
                                target_id=log.get("cardIdTarget", 0),
                                serial=log.get("serial"))
        if log_type == _LOG_DEVOLVE:
            return self._record(KIND_DEVOLVE, actor, card_id=log.get("cardId", 0),
                                target_id=log.get("cardIdTarget", 0),
                                serial=log.get("serial"))
        if log_type == _LOG_MOVE_ATTACHED:
            return self._record(KIND_MOVE_ATTACHED, actor, card_id=log.get("cardId", 0),
                                target_id=log.get("cardIdAfter", 0),
                                serial=log.get("serial"))
        if log_type == _LOG_ATTACK:
            return self._record(KIND_ATTACK, actor, card_id=log.get("cardId", 0),
                                attack_id=log.get("attackId", 0),
                                serial=log.get("serial"))
        if log_type == _LOG_SWITCH:
            return self._record(KIND_SWITCH, actor, card_id=log.get("cardIdActive", 0),
                                target_id=log.get("cardIdBench", 0),
                                serial=log.get("serialActive"))
        if log_type == _LOG_CHANGE:
            return self._record(KIND_CHANGE, actor, card_id=log.get("cardIdAfter", 0),
                                target_id=log.get("cardIdBefore", 0),
                                serial=log.get("serialAfter"))
        if log_type == _LOG_MOVE_CARD:
            card_id = log.get("cardId", 0)
            if not card_id and not self.extended:     # face-down move: no identity to record
                return None
            return self._record(KIND_MOVE_CARD, actor, card_id=card_id,
                                area_from=log.get("fromArea") or 0,
                                area_to=log.get("toArea") or 0,
                                serial=log.get("serial"))
        if self.extended and log_type == _LOG_HAS_BASIC_POKEMON:
            return self._record(KIND_MULLIGAN, actor,
                                flag=bool(log.get("hasBasicPokemon")))
        if log_type == _LOG_HP_CHANGE:
            return self._record(KIND_HP_CHANGE, actor, card_id=log.get("cardId", 0),
                                value=log.get("value", 0) or 0,
                                flag=bool(log.get("putDamageCounter")),
                                serial=log.get("serial"))
        if log_type in _STATUS_LOGS:
            return self._record(KIND_STATUS, actor, card_id=log.get("cardId", 0),
                                status=_STATUS_LOGS[log_type],
                                flag=bool(log.get("isRecover")),
                                serial=log.get("serial"))
        if log_type == _LOG_COIN:
            return self._record(KIND_COIN, actor, flag=bool(log.get("head")))
        return None

    @staticmethod
    def _record(kind, actor, card_id=0, target_id=0, attack_id=0, value=0, flag=False,
                area_from=0, area_to=0, status=-1, serial=None):
        # `serial` (A5): the physical copy the engine named. Pre-v5 encoders ignore it.
        return {"kind": kind, "actor": actor, "card_id": int(card_id or 0),
                "target_id": int(target_id or 0), "attack_id": int(attack_id or 0),
                "value": int(value or 0), "flag": bool(flag),
                "area_from": int(area_from or 0), "area_to": int(area_to or 0),
                "status": int(status), "serial": int(serial or 0)}


# --------------------------------------------------------------------------- #
# Self-test: play a seeded game with a tracker on EACH seat and report the event
# inventory (the token-count budget the v2 encoder inherits).
#   ./.venv/Scripts/python.exe -m src.game.action_history
# --------------------------------------------------------------------------- #

def _self_test():
    import random
    from collections import Counter

    from cg import game

    KIND_NAMES = {KIND_PLAY: "PLAY", KIND_ATTACH: "ATTACH", KIND_EVOLVE: "EVOLVE",
                  KIND_DEVOLVE: "DEVOLVE", KIND_MOVE_ATTACHED: "MOVE_ATTACHED",
                  KIND_ATTACK: "ATTACK", KIND_SWITCH: "SWITCH", KIND_CHANGE: "CHANGE",
                  KIND_MOVE_CARD: "MOVE_CARD", KIND_HP_CHANGE: "HP_CHANGE",
                  KIND_STATUS: "STATUS", KIND_COIN: "COIN"}

    deck = [int(line) for line in
            open("submissions/submission_alakazam/deck.csv").read().split() if line.strip()]
    rng = random.Random(4)

    def random_legal(observation):
        select = observation["select"]
        count = len(select["option"])
        take = max(min(select["maxCount"], count), select["minCount"])
        return sorted(rng.sample(range(count), take)) if count else []

    totals = []
    for game_index in range(3):
        histories = [ActionHistory(), ActionHistory()]
        observation, _ = game.battle_start(list(deck), list(deck), seed=500 + game_index)
        moves = 0
        while observation["current"]["result"] == -1 and moves < 400:
            histories[observation["current"]["yourIndex"]].update(observation)
            select = observation.get("select")
            observation = game.battle_select(random_legal(observation) if select else [])
            moves += 1
        game.battle_finish()

        for seat, history in enumerate(histories):
            inventory = Counter(event["kind"] for event in history.events)
            totals.append(len(history.events))
            if seat == 0:
                print(f"game {game_index}: {moves} moves, turns {history.turns_seen}, "
                      f"seat0 events {len(history.events)}")
                print("   " + "  ".join(f"{KIND_NAMES[kind]}:{count}"
                                        for kind, count in sorted(inventory.items())))
        # Both seats must observe the same public events (deliveries differ, content doesn't).
        assert abs(len(histories[0].events) - len(histories[1].events)) < 40, \
            "seats disagree wildly on public event count"
        for event in histories[0].events[:0] + histories[0].events:
            assert 0 <= event["kind"] < NUM_KINDS
            assert event["fingerprint"].shape == (FINGERPRINT_DIM,)
            assert event["index_in_turn"] >= 0

    print(f"OK: mean {sum(totals) / len(totals):.0f} events/game/seat "
          f"(min {min(totals)}, max {max(totals)})")


if __name__ == "__main__":
    _self_test()
