"""Unit matrix for pokepad_supporter_gate (owner rule 2026-08-16, sylveon deck):
Pokegear 3.0 (1122; target corrected from Poke Pad 08-16 night) is blocked once this turn's Supporter is spent, and on our going-first
turn 1; the block lifts vs known dragapult or a revealed Budew (item lock coming)."""
import sys

import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.game import state_encoder as ev  # noqa: E402

PAD, UB, BUDEW = 1122, 1121, 235   # PAD = Pokegear 3.0 (target corrected 08-16 night)
failures = []


def check(name, condition):
    print(("PASS " if condition else "FAIL ") + name)
    if not condition:
        failures.append(name)


def observation(turn, supporter_played=False, their_bench=()):
    return {"current": {"turn": turn, "yourIndex": 0,
                        "supporterPlayed": supporter_played,
                        "players": [{"hand": [{"id": PAD, "serial": 50},
                                              {"id": UB, "serial": 51}],
                                     "active": [], "bench": [], "discard": []},
                                    {"active": [],
                                     "bench": [{"id": i, "serial": 200 + n}
                                               for n, i in enumerate(their_bench)],
                                     "discard": []}]}}


MENU = {"context": 0, "minCount": 1, "maxCount": 1, "option": [
    {"type": 7, "index": 0}, {"type": 7, "index": 1}, {"type": 14}]}

ev._ACTION_RULES.clear()
ev.enable_action_rules(ev.RULE_POKEPAD_GATE)
ev.reset_opponent_context()

check("supporter spent: pad masked, ultra ball and end untouched",
      ev.pokepad_supporter_gate_mask(observation(5, supporter_played=True), MENU)
      == {1, 2})
check("going-first turn 1: pad masked",
      ev.pokepad_supporter_gate_mask(observation(1), MENU) == {1, 2})
check("turn 2 (second player's first turn): no restriction",
      ev.pokepad_supporter_gate_mask(observation(2), MENU) is None)
check("supporter still available mid-game: no restriction",
      ev.pokepad_supporter_gate_mask(observation(5), MENU) is None)
check("revealed budew lifts the block",
      ev.pokepad_supporter_gate_mask(
          observation(5, supporter_played=True, their_bench=(BUDEW,)), MENU) is None)
ev.note_dragapult_opponent()
check("known dragapult lifts the block",
      ev.pokepad_supporter_gate_mask(observation(1), MENU) is None)
ev.reset_opponent_context()
PAD_ONLY = {"context": 0, "minCount": 1, "maxCount": 1,
            "option": [{"type": 7, "index": 0}, {"type": 14}]}
check("pad + end menu: end stays available (never empty)",
      ev.pokepad_supporter_gate_mask(observation(1), PAD_ONLY) == {1})
ev._ACTION_RULES.clear()
check("rule disabled: inert",
      ev.pokepad_supporter_gate_mask(observation(1), MENU) is None)

print()
print("FAILURES:", failures if failures else "none -- all pass")
sys.exit(1 if failures else 0)
