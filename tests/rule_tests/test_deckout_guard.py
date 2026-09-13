"""Unit matrix for prevent_deck_out (owner rule 2026-08-16): with our deck at 0 and
theirs at 1+, a playable Lillie's whose redraw leaves >= 1 card closes END until it
is played; the line rule condemns non-winning leaves that skipped it. Inert when
their deck is 0 (incl. both-at-0), when the hand is too small, when the Supporter
slot is spent, or without Lillie's in hand."""
import sys

import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.game import state_encoder as ev  # noqa: E402

LILLIES = ev.LILLIES_DETERMINATION_CARD_ID
failures = []


def check(name, condition):
    print(("PASS " if condition else "FAIL ") + name)
    if not condition:
        failures.append(name)


def observation(our_deck=0, their_deck=10, hand_size=10, prizes=5,
                supporter_played=False, lillies=True):
    hand = [{"id": LILLIES if (lillies and n == 0) else 1121, "serial": 50 + n}
            for n in range(hand_size)]
    return {"current": {"turn": 9, "yourIndex": 0,
                        "supporterPlayed": supporter_played,
                        "players": [{"hand": hand, "deckCount": our_deck,
                                     "prize": [{}] * prizes,
                                     "active": [], "bench": [], "discard": []},
                                    {"deckCount": their_deck}]}}


MENU = {"context": 0, "minCount": 1, "maxCount": 1, "option": [
    {"type": 7, "index": 0},                      # play Lillie's
    {"type": 7, "index": 1},                      # play an item
    {"type": ev.OPTION_TYPE_ATTACK, "attackId": 154},
    {"type": ev.OPTION_TYPE_END}]}

ev._ACTION_RULES.clear()
ev.enable_action_rules(ev.RULE_DECKOUT_GUARD)

# hand 10 -> 9 excl one Lillie's; prizes 5 -> draw 6; 9 > 6 -> armed
check("armed: END closed, everything else (attacks, boss-class plays) open",
      ev.deckout_guard_mask(observation(), MENU) == {0, 1, 2})
check("their deck 0 -> inert (they deck out first)",
      ev.deckout_guard_mask(observation(their_deck=0), MENU) is None)
check("BOTH decks 0 -> inert (owner: keep the turn free to heal/survive)",
      ev.deckout_guard_mask(observation(our_deck=0, their_deck=0), MENU) is None)
check("our deck 1 -> inert (not decked yet)",
      ev.deckout_guard_mask(observation(our_deck=1), MENU) is None)
check("hand exactly draw-back size -> inert (redraw would empty the deck again)",
      ev.deckout_guard_mask(observation(hand_size=7), MENU) is None)
check("6 prizes remaining -> draw 8: hand 9-excl forces",
      ev.deckout_guard_mask(observation(hand_size=10, prizes=6), MENU) == {0, 1, 2})
check("6 prizes remaining -> draw 8: hand 8-excl is inert",
      ev.deckout_guard_mask(observation(hand_size=9, prizes=6), MENU) is None)
check("supporter already spent -> inert",
      ev.deckout_guard_mask(observation(supporter_played=True), MENU) is None)
check("no lillies in hand -> inert",
      ev.deckout_guard_mask(observation(lillies=False), MENU) is None)

# line rule
rule = ev.DeckoutGuardLineRule()
rule.observe_real(observation(), MENU, None)
check("line: armed at the root", rule.root_state() == (True, False))
labels = rule.option_labels(observation(), MENU)
after = rule.update(rule.root_state(), labels[0])
check("line: playing lillies saves the line",
      after == (True, True) and not rule.violated(after))
check("line: ending without lillies violates", rule.violated((True, False)))
rule.observe_real(observation(their_deck=0), MENU, None)
check("line: disarmed when their deck is empty",
      rule.root_state() == (False, False) and not rule.violated(rule.root_state()))

ev._ACTION_RULES.clear()
check("rule disabled: inert", ev.deckout_guard_mask(observation(), MENU) is None)

# ---- deckout_judge_stamp extension (owner 2026-08-16 evening, dragapult only) ----
JUDGE, STAMP = ev.JUDGE_CARD_ID, ev.UNFAIR_STAMP_CARD_ID


def observation_ids(hand_ids, our_deck=0, their_deck=10, prizes=5,
                    supporter_played=False):
    hand = [{"id": i, "serial": 50 + n} for n, i in enumerate(hand_ids)]
    return {"current": {"turn": 9, "yourIndex": 0,
                        "supporterPlayed": supporter_played,
                        "players": [{"hand": hand, "deckCount": our_deck,
                                     "prize": [{}] * prizes,
                                     "active": [], "bench": [], "discard": []},
                                    {"deckCount": their_deck}]}}


FILLER = 1121
ev._ACTION_RULES.clear()
ev.enable_action_rules(ev.RULE_DECKOUT_GUARD)
check("base rule only (sylveon parity): judge in hand does NOT arm",
      ev.deckout_guard_mask(observation_ids([JUDGE] + [FILLER] * 8), MENU) is None)

ev.enable_action_rules(ev.RULE_DECKOUT_EXTENDED)
check("extension: judge alone arms (hand 7 -> 6 kept > 4 drawn)",
      ev.deckout_guard_mask(observation_ids([JUDGE] + [FILLER] * 6), MENU)
      == {0, 1, 2})
check("extension: judge hand too small (5 -> 4 kept, draws 4) stays inert",
      ev.deckout_guard_mask(observation_ids([JUDGE] + [FILLER] * 4), MENU) is None)
check("extension: judge blocked once the supporter slot is spent",
      ev.deckout_guard_mask(observation_ids([JUDGE] + [FILLER] * 6,
                                            supporter_played=True), MENU) is None)

STAMP_MENU = {"context": 0, "minCount": 1, "maxCount": 1, "option": [
    {"type": 7, "index": 0},                      # play the stamp (engine-offered)
    {"type": ev.OPTION_TYPE_ATTACK, "attackId": 154},
    {"type": ev.OPTION_TYPE_END}]}
check("extension: OFFERED stamp arms even with the supporter spent (item)",
      ev.deckout_guard_mask(observation_ids([STAMP] + [FILLER] * 6,
                                            supporter_played=True), STAMP_MENU)
      == {0, 1})
NO_PLAY_MENU = {"context": 0, "minCount": 1, "maxCount": 1, "option": [
    {"type": ev.OPTION_TYPE_ATTACK, "attackId": 154},
    {"type": ev.OPTION_TYPE_END}]}
check("extension: stamp in hand but NOT offered -> inert (ace-spec timing)",
      ev.deckout_guard_mask(observation_ids([STAMP] + [FILLER] * 6,
                                            supporter_played=True), NO_PLAY_MENU)
      is None)
check("extension: stamp hand too small (6 -> 5 kept, draws 5) stays inert",
      ev.deckout_guard_mask(observation_ids([STAMP] + [FILLER] * 5,
                                            supporter_played=True), STAMP_MENU)
      is None)

PREF_MENU = {"context": 0, "minCount": 1, "maxCount": 1, "option": [
    {"type": 7, "index": 0},                      # lillies
    {"type": 7, "index": 1},                      # judge
    {"type": 7, "index": 2},                      # stamp
    {"type": ev.OPTION_TYPE_END}]}
o = observation_ids([LILLIES, JUDGE, STAMP] + [FILLER] * 7)
check("preference: lillies armed -> judge AND stamp plays masked with END",
      ev.deckout_guard_mask(o, PREF_MENU) == {0})
o = observation_ids([JUDGE, STAMP] + [FILLER] * 7)
JS_MENU = {"context": 0, "minCount": 1, "maxCount": 1, "option": [
    {"type": 7, "index": 0}, {"type": 7, "index": 1},
    {"type": ev.OPTION_TYPE_END}]}
check("preference: no lillies -> judge preferred, stamp play masked",
      ev.deckout_guard_mask(o, JS_MENU) == {0})

rule = ev.DeckoutGuardLineRule()
o = observation_ids([JUDGE] + [FILLER] * 6)
rule.observe_real(o, MENU, None)
check("line: judge-armed root arms the obligation", rule.root_state() == (True, False))
labels = rule.option_labels(o, MENU)
check("line: playing judge saves the line",
      labels[0] is not None
      and not rule.violated(rule.update(rule.root_state(), labels[0])))

ev._ACTION_RULES.clear()
check("extension disabled again: everything inert",
      ev.deckout_guard_mask(observation_ids([JUDGE] + [FILLER] * 6), MENU) is None)

print()
print("FAILURES:", failures if failures else "none -- all pass")
sys.exit(1 if failures else 0)
