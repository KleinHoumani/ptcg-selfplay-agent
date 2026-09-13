"""Prize-map solver: exact KO/prize arithmetic over cumulative damage, plus the ledger features
it produces for the factored value. The idea: multi-turn attack planning is not a tree -- damage
is additive and fungible, so "how many prizes can this side take in t turns" collapses to a small
allocation problem over each target's remaining HP. Computed for BOTH sides, from player 0's
perspective (same convention as src/game/features.py).

Rough-draft simplifications (deliberate; the learned weight per feature absorbs systematic bias):
- weakness/resistance, abilities, tools, protection effects and statuses are ignored
- attack profile per side = best damage / best spread among in-play Pokemon; turn 1 uses only the
  active's READY attacks (energy attached >= cost count), later turns assume the side can promote
  or charge its best in-play attacker
- spread damage ("put N damage counters ... in any way you like" attacks) is allocatable to bench
  targets; allocation is greedy by remaining HP ascending (near-optimal, microseconds)
- after the active is KOed, leftover cumulative active damage chains onto bench targets (they
  must promote); partial damage across two budgets on one target is not combined

`ledger_features(observation)` -> np.array aligned to LEDGER_FEATURE_NAMES.
"""

import re

import numpy as np

from src.cards import get_card, get_attack

_TURN_CAP = 15                    # turns_to_win scan horizon (saturates at "no path in sight")
_SPREAD_PATTERN = re.compile(
    r"[Pp]ut (\d+) damage counters on your opponent(?:'|’)s (Benched )?Pok\S?mon in any way you like")

_SPREAD_COUNTERS = {}
_PRIZE_VALUE = {}


def _spread_counters(attack_id):
    """Counters an attack places freely on the opponent's board (0 if not a free-spread attack)."""
    if attack_id not in _SPREAD_COUNTERS:
        attack = get_attack(attack_id) or {}
        match = _SPREAD_PATTERN.search(attack.get("text") or "")
        _SPREAD_COUNTERS[attack_id] = int(match.group(1)) if match else 0
    return _SPREAD_COUNTERS[attack_id]


def _prize_value(card_id):
    """Prizes taken for KOing this card (engine rule: ex = 2, mega ex = 3, else 1)."""
    if card_id not in _PRIZE_VALUE:
        card = get_card(card_id) or {}
        _PRIZE_VALUE[card_id] = 3 if card.get("megaEx") else (2 if card.get("ex") else 1)
    return _PRIZE_VALUE[card_id]


def _in_play(player):
    return [p for p in ((player.get("active") or []) + (player.get("bench") or [])) if p is not None]


def _active(player):
    return next((p for p in (player.get("active") or []) if p is not None), None)


def _attack_profile(player):
    """(active_damage_now, spread_now, damage_later, spread_later) for a side.

    "now" = the active's attacks whose energy cost is satisfied this turn; "later" = the best
    attack on ANY in-play Pokemon (the side can promote/charge it on future turns)."""
    active = _active(player)
    damage_now = spread_now = damage_later = spread_later = 0
    for pokemon in _in_play(player):
        is_active = active is not None and pokemon is active
        attached = len(pokemon.get("energies") or [])
        for attack_id in (get_card(pokemon["id"]) or {}).get("attacks", []):
            attack = get_attack(attack_id)
            if not attack:
                continue
            damage, spread = attack.get("damage", 0), _spread_counters(attack_id)
            damage_later = max(damage_later, damage)
            spread_later = max(spread_later, spread)
            if is_active and attached >= len(attack.get("energies") or []):
                damage_now = max(damage_now, damage)
                spread_now = max(spread_now, spread)
    return damage_now, spread_now, damage_later, spread_later


def _prizes_with_budgets(active_target, bench_targets, active_damage, spread_damage):
    """Greedy prize count for one attacker turn-budget: cumulative active damage must go through
    the active first (then chains onto promoted bench), spread damage picks off bench directly."""
    prizes = 0
    chain_open = False
    if active_target is not None:
        if active_damage >= active_target["hp"] > 0:
            prizes += _prize_value(active_target["id"])
            active_damage -= active_target["hp"]
            chain_open = True
    else:
        chain_open = True
    for target in sorted(bench_targets, key=lambda t: t["hp"]):
        if target["hp"] <= 0:
            continue
        if spread_damage >= target["hp"]:
            spread_damage -= target["hp"]
            prizes += _prize_value(target["id"])
        elif chain_open and active_damage >= target["hp"]:
            active_damage -= target["hp"]
            prizes += _prize_value(target["id"])
    return prizes


def prize_outlook(attacker, defender, horizon=3):
    """{"prizes_by_turn": [p1..p_horizon], "turns_to_win": t} for attacker vs defender's board.
    turns_to_win = first t (<= _TURN_CAP) whose greedy prize count reaches the attacker's
    remaining prize cards -- or, when the visible board holds fewer prizes than that, the turns
    to clear the whole board (future benchings aren't modeled). _TURN_CAP = no path in sight."""
    damage_now, spread_now, damage_later, spread_later = _attack_profile(attacker)
    active_target = _active(defender)
    bench_targets = [p for p in (defender.get("bench") or []) if p is not None]
    board_prizes = sum(_prize_value(p["id"]) for p in [active_target] + bench_targets
                       if p is not None and p["hp"] > 0)
    prizes_needed = min(len(attacker["prize"]), board_prizes) or _TURN_CAP * 6

    prizes_by_turn = []
    turns_to_win = _TURN_CAP
    for turn in range(1, _TURN_CAP + 1):
        active_budget = damage_now + (turn - 1) * damage_later
        spread_budget = 10 * (spread_now + (turn - 1) * spread_later)
        prizes = _prizes_with_budgets(active_target, bench_targets, active_budget, spread_budget)
        if turn <= horizon:
            prizes_by_turn.append(prizes)
        if prizes >= prizes_needed and turns_to_win == _TURN_CAP:
            turns_to_win = turn
            if turn >= horizon:
                break
    return {"prizes_by_turn": prizes_by_turn, "turns_to_win": turns_to_win}


LEDGER_FEATURE_NAMES = [
    "my_prizes_in_1", "my_prizes_in_2", "my_prizes_in_3",
    "opp_prizes_in_1", "opp_prizes_in_2", "opp_prizes_in_3",
    "my_turns_to_win", "opp_turns_to_win", "prize_race_margin",
    "my_can_ko_their_active", "opp_can_ko_my_active",
    "my_spread_ready", "their_cheapest_bench_prize", "my_cheapest_bench_liability",
]
LEDGER_DIM = len(LEDGER_FEATURE_NAMES)


def _cheapest_bench_hp(player):
    """Remaining HP of the cheapest bench KO on this player's board (their exposure)."""
    bench = [p["hp"] for p in (player.get("bench") or []) if p is not None and p["hp"] > 0]
    return min(bench) if bench else 0


def ledger_features(observation):
    current = observation["current"]
    me, opp = current["players"][0], current["players"][1]

    mine = prize_outlook(me, opp)
    theirs = prize_outlook(opp, me)
    my_damage_now, my_spread_now, _, _ = _attack_profile(me)
    opp_damage_now = _attack_profile(opp)[0]
    my_active, opp_active = _active(me), _active(opp)

    values = [
        mine["prizes_by_turn"][0] / 6.0,                                     # my_prizes_in_1
        mine["prizes_by_turn"][1] / 6.0,                                     # my_prizes_in_2
        mine["prizes_by_turn"][2] / 6.0,                                     # my_prizes_in_3
        theirs["prizes_by_turn"][0] / 6.0,                                   # opp_prizes_in_1
        theirs["prizes_by_turn"][1] / 6.0,                                   # opp_prizes_in_2
        theirs["prizes_by_turn"][2] / 6.0,                                   # opp_prizes_in_3
        mine["turns_to_win"] / _TURN_CAP,                                    # my_turns_to_win
        theirs["turns_to_win"] / _TURN_CAP,                                  # opp_turns_to_win
        (theirs["turns_to_win"] - mine["turns_to_win"]) / _TURN_CAP,         # prize_race_margin
        1.0 if (opp_active and my_damage_now >= opp_active["hp"] > 0) else 0.0,   # my_can_ko_their_active
        1.0 if (my_active and opp_damage_now >= my_active["hp"] > 0) else 0.0,    # opp_can_ko_my_active
        my_spread_now / 12.0,                                                # my_spread_ready
        (_cheapest_bench_hp(opp) or 300) / 300.0,                            # their_cheapest_bench_prize
        (_cheapest_bench_hp(me) or 300) / 300.0,                             # my_cheapest_bench_liability
    ]
    return np.array(values, dtype=np.float32)
