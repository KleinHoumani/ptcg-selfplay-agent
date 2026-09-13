"""Unit matrix for meowth_supporter_gate_mask's two surfaces, incl. the going-first
turn-1 fetch mask (owner rule 2026-08-15). Engine fact the mask encodes: supporters
are blocked only on turn 1 (GameProc.h `state.turn <= 1`), the going-first player's
turn; the second player's first turn is turn 2 and unrestricted."""
import sys

import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.game import state_encoder as ev  # noqa: E402

MEOWTH, DREEPY, BOSS = 1071, 119, 1182
failures = []


def check(name, condition):
    print(("PASS " if condition else "FAIL ") + name)
    if not condition:
        failures.append(name)


def observation(turn, deck_view=(), hand_ids=(), supporter_played=False):
    return {"current": {"turn": turn, "yourIndex": 0,
                        "supporterPlayed": supporter_played,
                        "players": [{"hand": [{"id": i, "serial": 50 + n}
                                              for n, i in enumerate(hand_ids)],
                                     "active": [], "bench": [], "discard": []},
                                    {}]},
            "select": {"deck": [{"id": i, "serial": 100 + n}
                                for n, i in enumerate(deck_view)]}}


def fetch_menu(n):
    return {"context": ev.SELECT_CONTEXT_TO_HAND, "minCount": 0, "maxCount": 1,
            "option": [{"type": ev.OPTION_TYPE_CARD, "area": ev.AREA_DECK,
                        "index": i, "playerIndex": 0} for i in range(n)]}


def play_menu(n):
    return {"context": 0, "minCount": 1, "maxCount": 1,
            "option": [{"type": 7, "index": i} for i in range(n)] + [{"type": 14}]}


ev._ACTION_RULES.clear()
ev.enable_action_rules(ev.RULE_MEOWTH_SUPPORTER)

# (b) the turn-1 fetch surface
o = observation(1, deck_view=(DREEPY, MEOWTH, DREEPY))
check("t1 fetch: meowth pick masked, others allowed",
      ev.meowth_supporter_gate_mask(o, fetch_menu(3)) == {0, 2})
o2 = observation(2, deck_view=(DREEPY, MEOWTH, DREEPY))
check("t2 fetch (second player's first turn): no restriction",
      ev.meowth_supporter_gate_mask(o2, fetch_menu(3)) is None)
o3 = observation(1, deck_view=(MEOWTH,))
check("t1 fetch: meowth-only menu -> None (never mask to empty)",
      ev.meowth_supporter_gate_mask(o3, fetch_menu(1)) is None)
o4 = observation(1, hand_ids=(MEOWTH, BOSS))
check("t1 non-fetch menu: playing from hand NOT blocked by the fetch surface",
      ev.meowth_supporter_gate_mask(o4, play_menu(2)) is None)

# (a) the supporter-spent play surface, regression
o5 = observation(5, hand_ids=(MEOWTH, BOSS), supporter_played=True)
check("supporter spent: meowth play masked",
      ev.meowth_supporter_gate_mask(o5, play_menu(2)) == {1, 2})
o6 = observation(5, hand_ids=(MEOWTH, BOSS), supporter_played=False)
check("supporter available: no restriction",
      ev.meowth_supporter_gate_mask(o6, play_menu(2)) is None)

# (c) the dragapult mirror exception (owner rule 2026-08-15 evening): vs a KNOWN
# dragapult opponent the going-first turn-1 Meowth fetch is ALLOWED, and holding the
# fetched Meowth is waived (bench it turn 2, where the Supporter can be played).
ev.note_dragapult_opponent()
check("mirror t1 fetch: meowth pick allowed",
      ev.meowth_supporter_gate_mask(o, fetch_menu(3)) is None)
armed = (ev.ULTRA_BALL_CARD_ID, frozenset(), False)
labels = ev.FetchedCardLineRule.option_labels(o, fetch_menu(3))
after = ev.FetchedCardLineRule.update(armed, labels[1])
check("mirror t1 take: waived -- awaiting cleared, no due, no violation",
      after == (None, frozenset(), False)
      and not ev.FetchedCardLineRule.violated(after))
o_t5 = observation(5, deck_view=(DREEPY, MEOWTH, DREEPY))
labels5 = ev.FetchedCardLineRule.option_labels(o_t5, fetch_menu(3))
after5 = ev.FetchedCardLineRule.update(armed, labels5[1])
check("mirror t5 take: NOT waived (the meowth hold exception is turn-1 only)",
      after5[1] == frozenset({101}) and ev.FetchedCardLineRule.violated(after5))
ev.reset_opponent_context()
check("flag off t1 fetch: meowth pick still masked",
      ev.meowth_supporter_gate_mask(o, fetch_menu(3)) == {0, 2})
after_off = ev.FetchedCardLineRule.update(
    armed, ev.FetchedCardLineRule.option_labels(o, fetch_menu(3))[1])
check("flag off t1 take: due created as before", after_off[1] == frozenset({101}))

# dead_fetch_item_mask synergy: turn-1 UB whose only hidden Pokemon is Meowth with a
# FULL bench (basic unplayable) is dead without the flag, live with it.
ev.enable_action_rules(ev.RULE_FETCHED_CARD)
ev.note_our_decklist({MEOWTH: 1, ev.ULTRA_BALL_CARD_ID: 1})
o_dead = observation(1, hand_ids=(ev.ULTRA_BALL_CARD_ID,))
me = o_dead["current"]["players"][0]
me["bench"] = [{"id": DREEPY, "serial": 200 + i} for i in range(5)]
me["benchMax"] = 5
me["deckCount"] = 1
me["prize"] = []
check("dead mask, flag off: turn-1 meowth-only UB is dead",
      ev.dead_fetch_item_mask(o_dead, play_menu(1)) == {1})
ev.note_dragapult_opponent()
check("dead mask, mirror: the waived meowth keeps UB live",
      ev.dead_fetch_item_mask(o_dead, play_menu(1)) is None)
ev.reset_opponent_context()
ev.note_our_decklist(None)

# (d) the obligation exit mask (owner go 2026-08-15 night, Crispin-over-Lillie's):
# while the fetched supporter is in hand and the slot unspent, rival supporters,
# ATTACK and END are masked; items/attach/the obligated play stay open.
LILLIES, CRISPIN_ID = 1227, 1218
rule = ev.MeowthSupporterLineRule()
rule._real_turn = 5


def pending_observation(supporter_played=False, hand_ids=(LILLIES, CRISPIN_ID, 1017)):
    return {"current": {"turn": 5, "yourIndex": 0,
                        "supporterPlayed": supporter_played,
                        "players": [{"hand": [{"id": i, "serial": 50 + n}
                                              for n, i in enumerate(hand_ids)],
                                     "active": [], "bench": [], "discard": []},
                                    {}]}}


MAIN = {"context": 0, "minCount": 1, "maxCount": 1, "option": [
    {"type": 7, "index": 0},                       # play the fetched Lillie's (50)
    {"type": 7, "index": 1},                       # play rival supporter Crispin
    {"type": 7, "index": 2},                       # play an item
    {"type": 8, "area": 2, "index": 0},            # attach
    {"type": ev.OPTION_TYPE_ATTACK, "attackId": 154},
    {"type": ev.OPTION_TYPE_END}]}

rule._real_state = (True, False, 50, False)        # obligation live: serial 50
rule.observe_real(pending_observation(), MAIN, None)
check("exit: obligation live -> rival supporter, attack and end masked",
      ev.meowth_obligation_exit_mask(pending_observation(), MAIN) == {0, 2, 3})
rule._real_state = (True, False, 50, True)         # satisfied
rule.observe_real(pending_observation(), MAIN, None)
check("exit: satisfied -> no restriction",
      ev.meowth_obligation_exit_mask(pending_observation(), MAIN) is None)
rule._real_state = (True, False, 50, False)
rule.observe_real(pending_observation(supporter_played=True), MAIN, None)
check("prune: slot spent on another card -> obligation dropped (poisoning fix)",
      rule._real_state == rule.initial_state()
      and ev.meowth_obligation_exit_mask(pending_observation(True), MAIN) is None)
rule._real_state = (True, False, 99, False)        # fetched card no longer in hand
rule.observe_real(pending_observation(), MAIN, None)
check("prune: fetched supporter gone from hand -> obligation dropped, mask down",
      rule._real_state == rule.initial_state()
      and ev.meowth_obligation_exit_mask(pending_observation(), MAIN) is None)
rule._real_state = (True, False, 50, False)
rule.observe_real(pending_observation(), MAIN, None)
ONLY_CLOSING = {"context": 0, "minCount": 1, "maxCount": 1, "option": [
    {"type": 7, "index": 0}, {"type": ev.OPTION_TYPE_END}]}
check("exit: obligated play + end -> forces the play (never empty)",
      ev.meowth_obligation_exit_mask(pending_observation(), ONLY_CLOSING) == {0})
ev._set_meowth_pending(None)

# rule off
ev._ACTION_RULES.clear()
check("rule disabled: inert",
      ev.meowth_supporter_gate_mask(o, fetch_menu(3)) is None)

print()
print("FAILURES:", failures if failures else "none -- all pass")
sys.exit(1 if failures else 0)
