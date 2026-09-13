"""Unit checks for the sunk-cost cleanup in FetchedCardLineRule / MeowthSupporterLineRule
observe_real (states set directly; observe_real called as main.py's PRE-pass does)."""
import sys

import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.game import state_encoder as ev  # noqa: E402

failures = []


def check(name, condition):
    print(("PASS " if condition else "FAIL ") + name)
    if not condition:
        failures.append(name)


def observation(turn=5, hand_serials=()):
    return {"current": {"turn": turn, "yourIndex": 0,
                        "players": [{"hand": [{"id": 1, "serial": s}
                                              for s in hand_serials],
                                     "active": [], "bench": [], "discard": []},
                                    {"active": [], "bench": [], "discard": []}]}}


def fetch_menu(area):
    return {"context": ev.SELECT_CONTEXT_TO_HAND, "minCount": 0, "maxCount": 1,
            "option": [{"type": ev.OPTION_TYPE_CARD, "area": area, "index": 0,
                        "playerIndex": 0}]}


MAIN_MENU = {"context": 0, "minCount": 1, "maxCount": 1,
             "option": [{"type": 7, "index": 0}]}

# ---- FetchedCardLineRule ------------------------------------------------------------
rule = ev.FetchedCardLineRule()
rule._real_turn = 5

# armed awaiting + the fetch menu on the table -> KEPT (obligation is live)
rule._real_state = (ev.ULTRA_BALL_CARD_ID, frozenset(), False)
rule.observe_real(observation(), fetch_menu(ev.AREA_DECK), None)
check("fetched: awaiting kept while its deck fetch menu is on the table",
      rule._real_state[0] == ev.ULTRA_BALL_CARD_ID)

# armed awaiting + a NON-fetch MAIN menu -> dropped (declined/whiffed for real)
rule.observe_real(observation(), MAIN_MENU, None)
check("fetched: awaiting dropped once back at a main menu",
      rule._real_state[0] is None)

# armed awaiting + a mid-chain COST menu (UB's discard-2, context DISCARD) -> KEPT
# (2026-08-15 fix: the cost menu arrives between the play and the fetch menu; it
# belongs to the item's own resolution chain, not to the passed-window case).
COST_MENU = {"context": 8, "minCount": 2, "maxCount": 2,
             "option": [{"type": ev.OPTION_TYPE_CARD, "area": 2, "index": i,
                         "playerIndex": 0} for i in range(4)]}
rule._real_state = (ev.ULTRA_BALL_CARD_ID, frozenset(), False)
rule.observe_real(observation(), COST_MENU, None)
check("fetched: awaiting KEPT at the item's own cost menu (UB discard-2)",
      rule._real_state[0] == ev.ULTRA_BALL_CARD_ID)

# stretcher awaiting stays live only for a DISCARD menu
rule._real_state = (ev.NIGHT_STRETCHER_CARD_ID, frozenset(), False)
rule.observe_real(observation(), fetch_menu(ev.AREA_DISCARD), None)
check("fetched: stretcher awaiting kept at its discard menu",
      rule._real_state[0] == ev.NIGHT_STRETCHER_CARD_ID)
rule._real_state = (ev.NIGHT_STRETCHER_CARD_ID, frozenset(), False)
rule.observe_real(observation(), fetch_menu(ev.AREA_DECK), None)
check("fetched: stretcher awaiting KEPT at a mid-chain non-main menu (2026-08-15)",
      rule._real_state[0] == ev.NIGHT_STRETCHER_CARD_ID)
rule.observe_real(observation(), MAIN_MENU, None)
check("fetched: ...and dropped at the next main menu",
      rule._real_state[0] is None)

# real whiff flag is sunk
rule._real_state = (None, frozenset(), True)
rule.observe_real(observation(), MAIN_MENU, None)
check("fetched: real whiffed flag cleared", rule._real_state[2] is False)

# due serial still in hand -> kept; gone from hand -> pruned
rule._real_state = (None, frozenset({77}), False)
rule.observe_real(observation(hand_serials=(77, 78)), MAIN_MENU, None)
check("fetched: due kept while the card is in hand",
      rule._real_state[1] == frozenset({77}))
rule.observe_real(observation(hand_serials=(78,)), MAIN_MENU, None)
check("fetched: due pruned once the card left our hand for real",
      rule._real_state[1] == frozenset())

# violated() on the pruned state is clean
check("fetched: pruned state no longer violates",
      not rule.violated(rule.root_state()))

# turn change still resets everything
rule._real_state = (ev.ULTRA_BALL_CARD_ID, frozenset({5}), True)
rule.observe_real(observation(turn=6), MAIN_MENU, None)
check("fetched: turn change resets", rule._real_state == rule.initial_state())

# ---- MeowthSupporterLineRule --------------------------------------------------------
meowth = ev.MeowthSupporterLineRule()
meowth._real_turn = 5

meowth._real_state = (True, True, None, False)          # benched, fetch pending
meowth.observe_real(observation(), fetch_menu(ev.AREA_DECK), None)
check("meowth: awaiting kept at the supporter fetch menu",
      meowth._real_state == (True, True, None, False))
meowth.observe_real(observation(), MAIN_MENU, None)
check("meowth: obligation dropped once the fetch window passed",
      meowth._real_state == meowth.initial_state())

meowth._real_state = (True, False, 42, False)           # fetched supporter serial 42
meowth.observe_real(observation(hand_serials=(42,)), MAIN_MENU, None)
check("meowth: obligation kept while the fetched supporter is in hand",
      meowth._real_state == (True, False, 42, False))
meowth.observe_real(observation(hand_serials=()), MAIN_MENU, None)
check("meowth: obligation dropped once the supporter left our hand",
      meowth._real_state == meowth.initial_state())

meowth._real_state = (True, False, 42, True)            # satisfied: untouched
meowth.observe_real(observation(hand_serials=()), MAIN_MENU, None)
check("meowth: satisfied state untouched by the cleanup",
      meowth._real_state == (True, False, 42, True))

print()
print("FAILURES:", failures if failures else "none -- all pass")
sys.exit(1 if failures else 0)
