"""Unit matrix for stadium_discipline (owner rule 2026-08-15): mask surfaces,
alakazam line rule, ownership tracking, exceptions, resets."""
import sys

import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.game import state_encoder as ev  # noqa: E402

WT, JT, MEOWTH, NS, BOSS = 1256, 1246, 1071, 1097, 1182
failures = []


def check(name, condition):
    print(("PASS " if condition else "FAIL ") + name)
    if not condition:
        failures.append(name)


def fresh():
    ev._ACTION_RULES.clear()
    ev.enable_action_rules(ev.RULE_STADIUM_DISCIPLINE)
    ev._MATCHUP_FLAGS.clear()
    ev._OUR_STADIUM_SERIALS.clear()
    ev._MEOWTH_PRIZE_PROVEN = False
    ev.note_our_decklist({MEOWTH: 1, NS: 3, WT: 1, JT: 1})


def obs(hand_ids=(), stadium=None, bench_ids=(), discard_ids=(), prizes=6):
    hand = [{"id": i, "serial": 500 + n} for n, i in enumerate(hand_ids)]
    return {"current": {
        "turn": 5, "yourIndex": 0,
        "stadium": [stadium] if stadium else [],
        "players": [
            {"hand": hand, "active": [], "prize": [{}] * prizes,
             "bench": [{"id": i, "serial": 700 + n, "hp": 50, "maxHp": 50}
                       for n, i in enumerate(bench_ids)],
             "discard": [{"id": i, "serial": 800 + n}
                         for n, i in enumerate(discard_ids)]},
            {"hand": None, "active": [], "bench": [], "discard": [], "prize": [{}] * 6},
        ]}}


def play_menu(hand_ids):
    return {"context": 0, "minCount": 1, "maxCount": 1,
            "option": [{"type": 7, "index": n} for n in range(len(hand_ids))]
            + [{"type": 14}]}


def mask(o, hand_ids):
    return ev.stadium_discipline_mask(o, play_menu(hand_ids))


# ---- mask: gating ---------------------------------------------------------------
fresh()
hand = (WT, BOSS)
check("mask: no matchup flag -> None", mask(obs(hand), hand) is None)
ev.note_matchup("grimmsnarl")
check("mask: grimmsnarl, no opp stadium -> WT blocked",
      mask(obs(hand), hand) == {1, 2})
opp_stadium = {"id": 1234, "serial": 999}
check("mask: opponent stadium in play -> WT allowed",
      mask(obs(hand, stadium=opp_stadium), hand) is None)
ev._note_our_stadium(999)
check("mask: OUR stadium in play -> WT still blocked",
      mask(obs(hand, stadium=opp_stadium), hand) == {1, 2})

# ---- mask: meowth exceptions ----------------------------------------------------
fresh(); ev.note_matchup("grimmsnarl")
check("mask: meowth on bench -> WT allowed",
      mask(obs(hand, bench_ids=(MEOWTH,)), hand) is None)
check("mask: meowth+all stretchers discarded -> WT allowed",
      mask(obs(hand, discard_ids=(MEOWTH, NS, NS, NS)), hand) is None)
check("mask: meowth discarded but a stretcher remains -> blocked",
      mask(obs(hand, discard_ids=(MEOWTH, NS, NS)), hand) == {1, 2})
ev.note_meowth_prize_proven()
check("mask: meowth provably prized -> WT allowed", mask(obs(hand), hand) is None)
check("mask: prized but meowth DRAWN to hand -> blocked again",
      mask(obs(hand + (MEOWTH,)), hand + (MEOWTH,)) == {1, 2, 3})
check("mask: prized but meowth in discard w/ stretchers live -> blocked",
      mask(obs(hand, discard_ids=(MEOWTH,)), hand) == {1, 2})
check("mask: prized->drawn->discarded w/ ALL stretchers gone -> allowed",
      mask(obs(hand, discard_ids=(MEOWTH, NS, NS, NS)), hand) is None)

# ---- mask: lucario jamming clause -----------------------------------------------
fresh(); ev.note_matchup("lucario")
jam_up = {"id": JT, "serial": 998}
check("mask: lucario + jamming in play -> blocked even with meowth benched",
      mask(obs(hand, stadium=jam_up, bench_ids=(MEOWTH,)), hand) == {1, 2})
check("mask: lucario + THEIR jamming in play -> blocked even though it bumps",
      mask(obs(hand, stadium={"id": JT, "serial": 12345}), hand) == {1, 2})
fresh(); ev.note_matchup("grimmsnarl")
check("mask: non-lucario + jamming(theirs) in play -> WT allowed (bumps theirs)",
      mask(obs(hand, stadium=jam_up), hand) is None)

# ---- mask: festival_lead --------------------------------------------------------
fresh(); ev.note_matchup("festival_lead")
hand2 = (WT, JT, BOSS)
check("mask: festival -> ALL our stadiums blocked, others fine",
      mask(obs(hand2), hand2) == {2, 3})
check("mask: festival + opponent stadium -> allowed",
      mask(obs(hand2, stadium=opp_stadium), hand2) is None)
check("mask: festival, only-stadium menu all blocked -> None (never empty)",
      mask(obs((WT,)), (WT,)) == {1})   # END option index 1 stays allowed

# ---- line rule: labels + sequence -----------------------------------------------
fresh()
rule = ev.StadiumDisciplineLineRule()
wt_up = {"id": WT, "serial": 997}
o = obs((JT, MEOWTH, BOSS), stadium=wt_up)
menu = play_menu((JT, MEOWTH, BOSS))
check("line: labels None without alakazam flag",
      rule.option_labels(o, menu) is None)
ev.note_matchup("alakazam")
labels = rule.option_labels(o, menu)
check("line: jam labeled only with watchtower up",
      labels[0] == ("jam_over_watchtower", None)
      and labels[1] == ("meowth_benched", None)
      and labels[2] == ("supporter_played", 502))
o_nowt = obs((JT, MEOWTH, BOSS))
labels2 = rule.option_labels(o_nowt, menu)
check("line: no watchtower -> jamming unlabeled",
      labels2 is None or labels2[0] is None)

state = rule.initial_state()
state = rule.update(state, ("meowth_benched", None))
check("line: pre-jam meowth ignored", state == rule.initial_state())
state = rule.update(state, ("jam_over_watchtower", None))
check("line: jam arms the obligation", rule.violated(state))
state = rule.update(state, ("meowth_benched", None))
state = rule.update(state, ("supporter_fetched", 42))
check("line: fetched but unplayed still violates", rule.violated(state))
state = rule.update(state, ("supporter_played", 41))
check("line: WRONG serial still violates", rule.violated(state))
state = rule.update(state, ("supporter_played", 42))
check("line: full jam->meowth->fetch->play sequence satisfies",
      not rule.violated(state))

# ---- observe_real: ownership + sunk pruning -------------------------------------
fresh(); ev.note_matchup("alakazam")
rule = ev.StadiumDisciplineLineRule()
rule._real_turn = 5
o3 = obs((WT, BOSS))
rule.observe_real(o3, play_menu((WT, BOSS)), [0])       # we really play Watchtower
check("real: our stadium serial recorded", 500 in ev._OUR_STADIUM_SERIALS)
check("real: recorded stadium reads as OURS",
      not ev._opponent_stadium_in_play(obs((), stadium={"id": WT, "serial": 500})))

rule._real_state = (True, True, None, False)             # jammed, fetch pending
fetch_menu = {"context": ev.SELECT_CONTEXT_TO_HAND, "minCount": 0, "maxCount": 1,
              "option": [{"type": ev.OPTION_TYPE_CARD, "area": ev.AREA_DECK,
                          "index": 0, "playerIndex": 0}]}
rule.observe_real(obs(()), fetch_menu, None)
check("real: jam obligation kept at its fetch menu",
      rule._real_state == (True, True, None, False))
rule.observe_real(obs(()), play_menu(()), None)
check("real: jam obligation dropped once fetch window passed (sunk)",
      rule._real_state == rule.initial_state())
rule._real_state = (True, False, 42, False)
rule.observe_real(obs(()), play_menu(()), None)          # supporter 42 not in hand
check("real: fetched supporter gone -> sunk, dropped",
      rule._real_state == rule.initial_state())

rule.reset_episode()
check("reset: flags/serials/prized cleared",
      not ev._MATCHUP_FLAGS and not ev._OUR_STADIUM_SERIALS
      and not ev._MEOWTH_PRIZE_PROVEN)

# ---- non-sticky flags (owner fix 2026-08-15) ------------------------------------
fresh()
ev.set_matchup_flags(["grimmsnarl"])
ev.set_matchup_flags(["lucario"])
check("flags: set_matchup_flags REPLACES (dethroned lock lifts)",
      ev.matchup_known("lucario") and not ev.matchup_known("grimmsnarl"))
ev.set_matchup_flags([])
check("flags: empty replacement clears all restrictions",
      mask(obs(hand), hand) is None)

# ---- Battle Cage clause (owner rule 2026-08-15) ---------------------------------
import copy

fresh()
cage = {"id": ev.BATTLE_CAGE_ID, "serial": 900}
card = __import__("src.cards", fromlist=["get_card"]).get_card(ev.BATTLE_CAGE_ID)
check("cage: card id 1264 resolves to Battle Cage",
      card is not None and card.get("name") == "Battle Cage")

o_cage = obs((WT, BOSS), stadium=cage, bench_ids=(112,))
o_cage["current"]["players"][0]["bench"][0]["energyCards"] = [{"id": 7}]
menu_cage = {"context": 0, "minCount": 1, "maxCount": 1, "option": [
    {"type": 7, "index": 0},                       # play WT (stadium)
    {"type": 7, "index": 1},                       # play Boss
    {"type": 10, "area": 5, "index": 0},           # Adrena-Brain (dark munkidori)
    {"type": 13, "attackId": ev.PHANTOM_DIVE_ATTACK_ID},
    {"type": 13, "attackId": 153},                 # Jet Headbutt: counters uninvolved
    {"type": 14}]}
blocked = ev.battle_cage_blocked_action_indices(o_cage, menu_cage)
check("cage: dark Adrena + Phantom Dive flagged, Jet Headbutt not",
      blocked == {2, 3})
check("cage: stadium bump candidates found",
      ev.stadium_bump_candidates(o_cage, menu_cage) == [0])

o_nodark = copy.deepcopy(o_cage)
o_nodark["current"]["players"][0]["bench"][0]["energyCards"] = [{"id": 5}]
check("cage: psychic-only munkidori NOT flagged (ability unusable-for-move anyway)",
      ev.battle_cage_blocked_action_indices(o_nodark, menu_cage) == {3})

o_nocage = obs((WT, BOSS), stadium={"id": 1234, "serial": 901}, bench_ids=(112,))
o_nocage["current"]["players"][0]["bench"][0]["energyCards"] = [{"id": 7}]
check("cage: other stadium in play -> nothing flagged",
      ev.battle_cage_blocked_action_indices(o_nocage, menu_cage) == set())

ev._ACTION_RULES.clear()
check("cage: rule disabled -> nothing flagged, no candidates",
      ev.battle_cage_blocked_action_indices(o_cage, menu_cage) == set()
      and ev.stadium_bump_candidates(o_cage, menu_cage) == [])
ev.enable_action_rules(ev.RULE_STADIUM_DISCIPLINE)

# ---- mirror Watchtower hold (owner rule 2026-08-16) -----------------------------
fresh()
ev.reset_opponent_context()
wt_up_mirror = {"id": WT, "serial": 996}
hand_m = (JT, BOSS)


def mirror_obs(**kwargs):
    return obs(hand_m, stadium=wt_up_mirror, **kwargs)


check("mirror: no dragapult flag -> None (no matchup flags either)",
      mask(mirror_obs(bench_ids=(MEOWTH,)), hand_m) is None)
ev.note_dragapult_opponent()
check("mirror: WT up + our meowth benched + theirs absent -> stadium plays blocked"
      " (binds with EMPTY matchup flags)",
      mask(mirror_obs(bench_ids=(MEOWTH,)), hand_m) == {1, 2})
check("mirror: our meowth NOT yet spent -> no hold",
      mask(mirror_obs(), hand_m) is None)
check("mirror: our meowth unrecoverable (all stretchers gone) -> hold",
      mask(mirror_obs(discard_ids=(MEOWTH, NS, NS, NS)), hand_m) == {1, 2})
o_their_meowth = mirror_obs(bench_ids=(MEOWTH,))
o_their_meowth["current"]["players"][1]["bench"] = [
    {"id": MEOWTH, "serial": 950, "hp": 50, "maxHp": 50}]
check("mirror: THEIR meowth in play -> no hold (Watchtower dead for both)",
      mask(o_their_meowth, hand_m) is None)
check("mirror: no Watchtower up (other stadium) -> no hold",
      mask(obs(hand_m, stadium={"id": 1234, "serial": 995},
               bench_ids=(MEOWTH,)), hand_m) is None)
check("mirror: non-stadium plays and END never touched",
      1 in mask(mirror_obs(bench_ids=(MEOWTH,)), hand_m)
      and 2 in mask(mirror_obs(bench_ids=(MEOWTH,)), hand_m))
ev.reset_opponent_context()
check("mirror: reset_opponent_context lifts the hold",
      mask(mirror_obs(bench_ids=(MEOWTH,)), hand_m) is None)

# ---- registration ---------------------------------------------------------------
check("registered: mask in OPTION_MASK_RULES",
      ev.stadium_discipline_mask in ev.OPTION_MASK_RULES)
line = ev.line_rule_for([ev.RULE_STADIUM_DISCIPLINE])
check("registered: line_rule_for returns the rule",
      line is not None and line.name == ev.RULE_STADIUM_DISCIPLINE)

print()
print("FAILURES:", failures if failures else "none -- all pass")
sys.exit(1 if failures else 0)
