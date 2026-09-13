"""Unit matrix for academy_at_night_gate (owner rule 2026-08-16, dragapult AND
sylveon): with Academy at Night in play, the stadium-USE option (ABILITY in the
STADIUM area) is masked while our deck has >= 1 card; at deck 0 the rule is inert
and the use is the model's call. Playing the card from hand and every other
ability stay untouched."""
import sys

import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.game import state_encoder as ev  # noqa: E402

ACADEMY = ev.ACADEMY_AT_NIGHT_ID
BATTLE_CAGE = ev.BATTLE_CAGE_ID
failures = []


def check(name, condition):
    print(("PASS " if condition else "FAIL ") + name)
    if not condition:
        failures.append(name)


def observation(deck=10, stadium_id=ACADEMY, your_index=0):
    players = [None, None]
    players[your_index] = {"deckCount": deck, "hand": [],
                           "active": [], "bench": [], "discard": []}
    players[1 - your_index] = {"deckCount": 20}
    return {"current": {"turn": 9, "yourIndex": your_index,
                        "stadium": [] if stadium_id is None
                        else [{"id": stadium_id, "serial": 300}],
                        "players": players}}


MENU = {"context": 0, "minCount": 1, "maxCount": 1, "option": [
    {"type": ev.OPTION_TYPE_ABILITY, "area": ev._AREA_STADIUM, "index": 0},
    {"type": 7, "index": 0},                       # play a card from hand
    {"type": ev.OPTION_TYPE_ABILITY, "area": 4, "index": 0},   # a Pokemon ability
    {"type": ev.OPTION_TYPE_END}]}

ev._ACTION_RULES.clear()
ev.enable_action_rules(ev.RULE_ACADEMY_GATE)

check("deck >= 1: stadium use masked, plays/abilities/end open",
      ev.academy_at_night_gate_mask(observation(), MENU) == {1, 2, 3})
check("deck exactly 1 still masks (only 0 lifts it)",
      ev.academy_at_night_gate_mask(observation(deck=1), MENU) == {1, 2, 3})
check("deck 0: inert (the save is the model's to take)",
      ev.academy_at_night_gate_mask(observation(deck=0), MENU) is None)
check("no stadium in play: inert",
      ev.academy_at_night_gate_mask(observation(stadium_id=None), MENU) is None)
check("a DIFFERENT stadium in play: inert",
      ev.academy_at_night_gate_mask(observation(stadium_id=BATTLE_CAGE), MENU)
      is None)
check("works from seat 1 (yourIndex=1 reads the right deck)",
      ev.academy_at_night_gate_mask(observation(your_index=1), MENU) == {1, 2, 3})
check("seat 1 at deck 0: inert",
      ev.academy_at_night_gate_mask(observation(deck=0, your_index=1), MENU) is None)

STADIUM_ONLY = {"context": 0, "minCount": 1, "maxCount": 1, "option": [
    {"type": ev.OPTION_TYPE_ABILITY, "area": ev._AREA_STADIUM, "index": 0}]}
check("stadium use is the only option: never mask to empty",
      ev.academy_at_night_gate_mask(observation(), STADIUM_ONLY) is None)

NO_STADIUM_OPTION = {"context": 0, "minCount": 1, "maxCount": 1, "option": [
    {"type": 7, "index": 0}, {"type": ev.OPTION_TYPE_END}]}
check("menu without the stadium option: no restriction",
      ev.academy_at_night_gate_mask(observation(), NO_STADIUM_OPTION) is None)

ev._ACTION_RULES.clear()
check("rule disabled: inert", ev.academy_at_night_gate_mask(observation(), MENU)
      is None)

print()
print("FAILURES:", failures if failures else "none -- all pass")
sys.exit(1 if failures else 0)
