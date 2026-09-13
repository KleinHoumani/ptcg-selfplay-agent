"""Grounded, NAMED game-state features for the factored value (learned reward-weight) experiment.

Each feature is a human-readable reading off the observation from PLAYER 0's perspective (me =
players[0], opponent = players[1]). The factored value learns a weight per feature = "how much
this thing being true predicts winning" -- so every weight is inspectable. These are candidate
signals; the model decides their value.

`game_features(observation)` -> np.array aligned to FEATURE_NAMES.
"""

import numpy as np

from src.cards import get_card, get_attack

_BEST_ATTACK = {}


def _best_attack(card_id):
    """(max attack damage, energy cost of that attack) for a card, cached."""
    if card_id not in _BEST_ATTACK:
        card = get_card(card_id) or {}
        best_damage, best_cost = 0, 0
        for attack_id in card.get("attacks", []):
            attack = get_attack(attack_id)
            if attack and attack.get("damage", 0) > best_damage:
                best_damage, best_cost = attack["damage"], len(attack.get("energies", []))
        _BEST_ATTACK[card_id] = (best_damage, best_cost)
    return _BEST_ATTACK[card_id]


def _stage(card_id):
    card = get_card(card_id) or {}
    return 2 if card.get("stage2") else (1 if card.get("stage1") else 0)


def _in_play(player):
    return [p for p in ((player.get("active") or []) + (player.get("bench") or [])) if p is not None]


def _active(player):
    return next((p for p in (player.get("active") or []) if p is not None), None)


def _hp_fraction(pokemon):
    return pokemon["hp"] / (pokemon["maxHp"] or 1) if pokemon else 0.0


FEATURE_NAMES = [
    "my_prizes_taken", "opp_prizes_taken", "prize_lead",
    "my_board_hp", "opp_board_hp", "my_active_hp", "opp_active_hp",
    "damage_on_opp_active", "damage_on_my_active",
    "my_bench_count", "opp_bench_count", "my_pokemon_in_play", "opp_pokemon_in_play",
    "my_total_energy", "opp_total_energy", "my_max_charged", "my_active_energy",
    "my_stage2_in_play", "my_ex_in_play", "opp_ex_in_play",
    "my_active_best_damage", "opp_active_best_damage", "my_active_ready_to_attack",
    "my_hand", "opp_hand", "my_deck", "opp_deck",
    "opp_poisoned", "opp_burned", "opp_asleep", "opp_paralyzed", "opp_confused",
    "my_afflicted", "stadium_in_play", "turn",
]


def game_features(observation):
    current = observation["current"]
    me, opp = current["players"][0], current["players"][1]
    my_play, opp_play = _in_play(me), _in_play(opp)
    my_active, opp_active = _active(me), _active(opp)

    my_active_energy = len(my_active.get("energies") or []) if my_active else 0
    active_damage, active_cost = _best_attack(my_active["id"]) if my_active else (0, 0)
    ready = 1.0 if (my_active and active_cost > 0 and my_active_energy >= active_cost) else 0.0

    values = [
        (6 - len(me["prize"])) / 6.0,                                       # my_prizes_taken
        (6 - len(opp["prize"])) / 6.0,                                      # opp_prizes_taken
        (len(opp["prize"]) - len(me["prize"])) / 6.0,                       # prize_lead
        sum(_hp_fraction(p) for p in my_play) / 6.0,                        # my_board_hp
        sum(_hp_fraction(p) for p in opp_play) / 6.0,                       # opp_board_hp
        _hp_fraction(my_active),                                            # my_active_hp
        _hp_fraction(opp_active),                                           # opp_active_hp
        1.0 - _hp_fraction(opp_active) if opp_active else 0.0,              # damage_on_opp_active
        1.0 - _hp_fraction(my_active) if my_active else 0.0,               # damage_on_my_active
        len([p for p in (me.get("bench") or []) if p]) / 5.0,               # my_bench_count
        len([p for p in (opp.get("bench") or []) if p]) / 5.0,              # opp_bench_count
        len(my_play) / 6.0,                                                 # my_pokemon_in_play
        len(opp_play) / 6.0,                                                # opp_pokemon_in_play
        sum(len(p.get("energies") or []) for p in my_play) / 12.0,          # my_total_energy
        sum(len(p.get("energies") or []) for p in opp_play) / 12.0,         # opp_total_energy
        max((len(p.get("energies") or []) for p in my_play), default=0) / 4.0,   # my_max_charged
        my_active_energy / 4.0,                                             # my_active_energy
        sum(_stage(p["id"]) == 2 for p in my_play) / 3.0,                   # my_stage2_in_play
        sum(bool((get_card(p["id"]) or {}).get("ex")) for p in my_play) / 3.0,   # my_ex_in_play
        sum(bool((get_card(p["id"]) or {}).get("ex")) for p in opp_play) / 3.0,  # opp_ex_in_play
        active_damage / 300.0,                                              # my_active_best_damage
        (_best_attack(opp_active["id"])[0] / 300.0) if opp_active else 0.0, # opp_active_best_damage
        ready,                                                              # my_active_ready_to_attack
        min(me["handCount"], 30) / 30.0,                                    # my_hand
        min(opp["handCount"], 30) / 30.0,                                   # opp_hand
        me["deckCount"] / 60.0,                                             # my_deck
        opp["deckCount"] / 60.0,                                            # opp_deck
        float(opp["poisoned"]), float(opp["burned"]), float(opp["asleep"]),
        float(opp["paralyzed"]), float(opp["confused"]),                    # opp status (me applying = good)
        float(any((me["poisoned"], me["burned"], me["asleep"], me["paralyzed"], me["confused"]))),  # my_afflicted
        1.0 if len(current["stadium"]) > 0 else 0.0,                        # stadium_in_play
        min(current["turn"], 50) / 50.0,                                    # turn
    ]
    return np.array(values, dtype=np.float32)


FEATURE_DIM = len(FEATURE_NAMES)
