"""Unit matrix for damage_solver (owner REWORK 2026-08-16): win tier (MegaEx = 3
prizes, engine State.h) > KO sweep over the archetype PRIORITY LIST (most-setup
tiebreak with the extra-KO exception) > the UNIVERSAL staging ladder (prize-gated
sections, global "n" Munkidori budget, their-healer thresholds, evolution-aware
passes) > model. Plus the unchanged surfaces: ctx-14 Mist shielding (never ctx 13),
doomed-active exclusion, forced max count, the Adrena exit mask + interception."""
import sys

import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.game import state_encoder as ev  # noqa: E402

DRAGAPULT, DRAKLOAK, DREEPY, MUNKIDORI = 121, 120, 119, 112
FEZ, MEOWTH, BUDEW = 140, 1071, 235
MEGA_KANGA, SLOWKING, MEGA_LUCARIO = 756, 163, 678
HARIYAMA, MAKUHITA, LUNATONE, DUNSPARCE = 674, 673, 675, 65
GRIMM_EX, MORGREM, IMPIDIMP, HYDRAPPLE_EX = 648, 647, 646, 150
SPIRITOMB, GIBLE, GARCHOMP_EX = 387, 379, 381
DARK, MIST, CAPE = 7, 11, 1159
failures = []


def check(name, condition):
    print(("PASS " if condition else "FAIL ") + name)
    if not condition:
        failures.append(name)


def pokemon(card_id, hp, energy=(), tools=()):
    from src.cards import get_card
    return {"id": card_id, "hp": hp, "serial": 900 + card_id,
            "maxHp": (get_card(card_id) or {}).get("hp") or hp,
            "energyCards": [{"id": e} for e in energy],
            "tools": [{"id": t} for t in tools]}


def observation(their_bench, their_active=None, our_active=None, our_bench=(),
                our_prizes=6):
    return {"current": {"turn": 8, "yourIndex": 0, "players": [
        {"active": [our_active] if our_active else [],
         "bench": list(our_bench), "prize": [{}] * our_prizes, "hand": []},
        {"active": [their_active] if their_active else [],
         "bench": list(their_bench)}]}}


def counter_menu(context, count, remain=None):
    select = {"context": context, "minCount": 1, "maxCount": 1,
              "option": [{"type": ev.OPTION_TYPE_CARD, "area": 5, "index": i,
                          "playerIndex": 1} for i in range(count)]}
    if remain is not None:
        select["remainDamageCounter"] = remain
    return select


def with_active_row(select):
    select["option"].append({"type": ev.OPTION_TYPE_CARD, "area": 4, "index": 0,
                             "playerIndex": 1})
    return select


OUR_MUNKI = pokemon(MUNKIDORI, 110, energy=(DARK,))
ev._ACTION_RULES.clear()
ev.enable_action_rules(ev.RULE_DAMAGE_SOLVER)
ev.note_dragapult_opponent()      # mirror list unless set_solver_matchup overrides
CTX14, CTX13, CTX40 = 14, 13, 40

# -- prize yields (engine State.h getPrizeCount) --------------------------------------
check("prizes: MegaEx=3, ex=2, plain=1",
      ev._prize_yield(pokemon(MEGA_KANGA, 300)) == 3
      and ev._prize_yield(pokemon(FEZ, 210)) == 2
      and ev._prize_yield(pokemon(DREEPY, 70)) == 1)

# -- ctx 14: KO tier (mirror priority list) -------------------------------------------
o = observation([pokemon(DRAKLOAK, 40), pokemon(DREEPY, 20)])
check("ko: priority order picks drakloak over dreepy",
      ev.damage_solver_mask(o, counter_menu(CTX14, 2, remain=6)) == {0})
o = observation([pokemon(DRAKLOAK, 40), pokemon(DREEPY, 20), pokemon(DRAGAPULT, 60)])
check("ko: strict priority takes the dragapult over two smaller KOs",
      ev.damage_solver_mask(o, counter_menu(CTX14, 3, remain=6)) == {2})
o = observation([pokemon(DRAKLOAK, 30), pokemon(DRAKLOAK, 30, energy=(7,))])
check("ko tiebreak: more energy wins",
      ev.damage_solver_mask(o, counter_menu(CTX14, 2, remain=6)) == {1})
o = observation([pokemon(DRAKLOAK, 30, tools=(999,)), pokemon(DRAKLOAK, 30)])
check("ko tiebreak: tool beats no tool (no energy anywhere)",
      ev.damage_solver_mask(o, counter_menu(CTX14, 2, remain=6)) == {0})
o = observation([pokemon(DRAKLOAK, 30, tools=(999,)),
                 pokemon(DRAKLOAK, 30, tools=(CAPE,))])
check("ko tiebreak: both have tools -> Hero's Cape wins",
      ev.damage_solver_mask(o, counter_menu(CTX14, 2, remain=6)) == {1})
o = observation([pokemon(MUNKIDORI, 60, energy=(DARK,)), pokemon(MUNKIDORI, 30),
                 pokemon(BUDEW, 30)])
check("ko exception: cheaper same-name target preserves an ADDITIONAL ko",
      ev.damage_solver_mask(o, counter_menu(CTX14, 3, remain=6)) == {1})
o = observation([pokemon(MUNKIDORI, 60, energy=(DARK,)), pokemon(MUNKIDORI, 30)])
check("ko exception inert without an extra ko -> most setup wins",
      ev.damage_solver_mask(o, counter_menu(CTX14, 2, remain=6)) == {0})

# -- KO tier: conditional entries -----------------------------------------------------
ev.set_solver_matchup("lucario")
o = observation([pokemon(HARIYAMA, 50, energy=(1, 2)), pokemon(MAKUHITA, 40)])
check("lucario: 2-energy hariyama outranks makuhita",
      ev.damage_solver_mask(o, counter_menu(CTX14, 2, remain=6)) == {0})
o = observation([pokemon(HARIYAMA, 50), pokemon(MAKUHITA, 40)])
check("lucario: energyless hariyama drops below makuhita",
      ev.damage_solver_mask(o, counter_menu(CTX14, 2, remain=6)) == {1})
o = observation([pokemon(LUNATONE, 50), pokemon(DUNSPARCE, 60)])
check("lucario: LONE lunatone outranks dunsparce",
      ev.damage_solver_mask(o, counter_menu(CTX14, 2, remain=6)) == {0})
o = observation([pokemon(LUNATONE, 50), pokemon(LUNATONE, 50), pokemon(DUNSPARCE, 60)])
check("lucario: two lunatone -> the late entry, equal pair ties to the model",
      ev.damage_solver_mask(o, counter_menu(CTX14, 3, remain=6)) == {0, 1})
ev.set_solver_matchup("cynthia_garchomp")
o = observation([pokemon(SPIRITOMB, 30), pokemon(GIBLE, 30),
                 pokemon(GARCHOMP_EX, 130)])
check("cynthia: 200+ bench damage promotes spiritomb",
      ev.damage_solver_mask(o, counter_menu(CTX14, 3, remain=6)) == {0})
o = observation([pokemon(SPIRITOMB, 30), pokemon(GIBLE, 30),
                 pokemon(GARCHOMP_EX, 330)])
check("cynthia: below 200 bench damage the gible outranks spiritomb",
      ev.damage_solver_mask(o, counter_menu(CTX14, 3, remain=6)) == {1})

# -- win tier -------------------------------------------------------------------------
ev.set_solver_matchup("slowking")
o = observation([pokemon(SLOWKING, 40), pokemon(MEGA_KANGA, 60)], our_prizes=3)
check("win: MEGA = 3 prizes closes the game over the ko order",
      ev.damage_solver_mask(o, counter_menu(CTX14, 2, remain=6)) == {1})
ev.set_solver_matchup(None)
o = observation([pokemon(DREEPY, 10), pokemon(DREEPY, 10)], our_prizes=2)
check("win: two-target combination found (no single target wins)",
      ev.damage_solver_mask(o, counter_menu(CTX14, 2, remain=6)) == {0})
check("win: pending prizes from the already-dead active count toward the win",
      ev.damage_solver_mask(
          observation([pokemon(DREEPY, 10), pokemon(FEZ, 30)], our_prizes=4,
                      their_active=pokemon(DRAGAPULT, 0)),
          counter_menu(CTX14, 2, remain=6)) == {1})

# -- staging: universal ladder (mirror list) ------------------------------------------
o = observation([pokemon(DRAGAPULT, 250)])
check("staging: dragapult chipped to 200 (no healers)",
      ev.damage_solver_mask(o, counter_menu(CTX14, 1, remain=6)) == {0})
o = observation([pokemon(DRAGAPULT, 195), pokemon(MUNKIDORI, 110, energy=(DARK,))])
check("staging: their dark munki deepens the tier to 170",
      ev.damage_solver_mask(o, counter_menu(CTX14, 2, remain=6)) == {0})
o = observation([pokemon(DRAGAPULT, 195), pokemon(MUNKIDORI, 110)])
check("staging: energyless munki heals nothing -> 200 met, munki to 60 instead",
      ev.damage_solver_mask(o, counter_menu(CTX14, 2, remain=6)) == {1})

# n budget: our dark munkidori opens the put-to-30 step (targets chosen so nothing
# is KO-able inside the budget: dragapult 105 -> to-60 only, drakloak 85 -> to-30)
o = observation([pokemon(DRAGAPULT, 105), pokemon(DRAKLOAK, 85)],
                our_bench=[OUR_MUNKI])
check("n: our dark munki -> drakloak straight to 30",
      ev.damage_solver_mask(o, counter_menu(CTX14, 2, remain=6)) == {1})
o = observation([pokemon(DRAGAPULT, 105), pokemon(DRAKLOAK, 85)])
check("n=0: the 30 step is skipped, dragapult to 60 instead",
      ev.damage_solver_mask(o, counter_menu(CTX14, 2, remain=6)) == {0})
o = observation([pokemon(DRAGAPULT, 105), pokemon(DRAKLOAK, 85),
                 pokemon(DREEPY, 25, energy=(MIST,))], our_bench=[OUR_MUNKI])
check("n consumed: a (shielded) target already at <=30 uses the slot",
      ev.damage_solver_mask(o, counter_menu(CTX14, 3, remain=6)) == {0})

# evolution-aware passes (budgets below the KO cost so the staging tier decides)
o = observation([pokemon(DREEPY, 30)])
check("evolution: dreepy 30 -> 10 so a future drakloak sits at 30",
      ev.damage_solver_mask(o, counter_menu(CTX14, 1, remain=2)) == {0})
o = observation([pokemon(DREEPY, 45), pokemon(DRAGAPULT, 205)])
check("evolution: the evo pass of an early step runs before the next step",
      ev.damage_solver_mask(o, counter_menu(CTX14, 2, remain=1)) == {0})

# prize-gated sections
o = observation([pokemon(MEGA_KANGA, 250), pokemon(MUNKIDORI, 110)], our_prizes=3)
check("sections: at <=3 prizes the 3-prize mega is staged first",
      ev.damage_solver_mask(o, counter_menu(CTX14, 2, remain=6)) == {0})
o = observation([pokemon(MEGA_KANGA, 250), pokemon(MUNKIDORI, 110)], our_prizes=6)
check("sections: at 6 prizes the same board stages the munki (priority walk)",
      ev.damage_solver_mask(o, counter_menu(CTX14, 2, remain=6)) == {1})
o = observation([pokemon(FEZ, 100), pokemon(MUNKIDORI, 110)], our_prizes=2)
check("sections: at <=2 prizes the 2-prize fez is staged first",
      ev.damage_solver_mask(o, counter_menu(CTX14, 2, remain=6)) == {0})
o = observation([pokemon(FEZ, 100), pokemon(MUNKIDORI, 110)], our_prizes=6)
check("sections: at 6 prizes the munki outranks the fez",
      ev.damage_solver_mask(o, counter_menu(CTX14, 2, remain=6)) == {1})

# deep steps
ev.set_solver_matchup("lucario")
o = observation([pokemon(MEGA_LUCARIO, 335)])
check("deep steps: 340-hp mega reached through the 290 tier",
      ev.damage_solver_mask(o, counter_menu(CTX14, 1, remain=6)) == {0})
caped = pokemon(MEGA_LUCARIO, 435, tools=(CAPE,))
caped["maxHp"] = 440
o = observation([caped])
check("deep steps: the 400 tier reaches a caped mega (owner 2026-08-16)",
      ev.damage_solver_mask(o, counter_menu(CTX14, 1, remain=6)) == {0})
ev.set_solver_matchup(None)

# festival_lead: the Rellor pre-load (owner 2026-08-16)
ev.set_solver_matchup("festival_lead")
RELLOR, DIPPLIN = 73, 93
o = observation([pokemon(RELLOR, 50), pokemon(DIPPLIN, 90)])
check("festival: rellor to 30 comes before every ladder step",
      ev.damage_solver_mask(o, counter_menu(CTX14, 2, remain=4)) == {0})
o = observation([pokemon(RELLOR, 30), pokemon(DIPPLIN, 90)])
check("festival: then rellor to 10 (the pre-loaded rabsca kill)",
      ev.damage_solver_mask(o, counter_menu(CTX14, 2, remain=4)) == {0})
o = observation([pokemon(RELLOR, 50)])
check("festival: a ko-able rellor is simply knocked out (list head)",
      ev.damage_solver_mask(o, counter_menu(CTX14, 1, remain=6)) == {0})
ev.set_solver_matchup(None)

# archaludon: the 230-damage cap was REMOVED (owner 2026-08-16 evening: "the rule
# was wrong") -- Archaludon ex now stages like anything else in its list
ev.set_solver_matchup("archaludon")
ARCH_EX = 190
o = observation([pokemon(ARCH_EX, 100)])
check("archaludon: no damage cap -- arch ex stages to 60 like any target",
      ev.damage_solver_mask(o, counter_menu(CTX14, 1, remain=6)) == {0})
o = observation([pokemon(ARCH_EX, 60)])
check("archaludon: arch ex KO unrestricted",
      ev.damage_solver_mask(o, counter_menu(CTX14, 1, remain=6)) == {0})
ev.set_solver_matchup(None)

# the Freezing Shroud discount (owner 2026-08-16; widened same day: triggers on
# card 104 IN PLAY, any matchup)
ev.set_solver_matchup("grimmsnarl")
FROSLASS = 104
o = observation([pokemon(MUNKIDORI, 50), pokemon(FROSLASS, 90)])
check("froslass: chip discounts the ability target's ko (50hp needs 4, not 5)",
      ev.damage_solver_mask(o, counter_menu(CTX14, 2, remain=4)) == {0})
o = observation([pokemon(MUNKIDORI, 70), pokemon(FROSLASS, 90), pokemon(FROSLASS, 90)])
check("froslass: two copies stack (70hp munki dies to 20 chip -> 5-counter ko)",
      ev.damage_solver_mask(o, counter_menu(CTX14, 3, remain=5)) == {0})
o = observation([pokemon(MUNKIDORI, 70), pokemon(FROSLASS, 90)])
check("froslass: one copy is not enough there -> ladder stages froslass instead",
      ev.damage_solver_mask(o, counter_menu(CTX14, 2, remain=5)) == {1})
o = observation([pokemon(MUNKIDORI, 10), pokemon(FROSLASS, 90)])
check("froslass: a target already dying to the chip gets NO counters",
      ev.damage_solver_mask(o, counter_menu(CTX14, 2, remain=6)) == {1})
ev.set_solver_matchup("starmie")
o = observation([pokemon(MUNKIDORI, 70), pokemon(FROSLASS, 90), pokemon(FROSLASS, 90)])
check("froslass: the discount follows the CARD, not the matchup (starmie board)",
      ev.damage_solver_mask(o, counter_menu(CTX14, 3, remain=5)) == {0})
o = observation([pokemon(MUNKIDORI, 70), pokemon(FROSLASS, 90)])
check("froslass: starmie single copy insufficient -> ladder stages froslass",
      ev.damage_solver_mask(o, counter_menu(CTX14, 2, remain=5)) == {1})
ev.set_solver_matchup(None)

# -- dive-combo setups at ctx 13 (owner rule 2026-08-16 evening) ----------------------
DIVE_ACTIVE = pokemon(DRAGAPULT, 320, energy=(7, 7))
SMOOCHUM = 183
ev.set_solver_matchup("slowking")
o = observation([pokemon(SMOOCHUM, 30), pokemon(MEGA_KANGA, 90)],
                our_active=DIVE_ACTIVE, our_prizes=3)
check("combo win: <=3 prizes -> load the 90hp mega for the dive, not the 30hp kill",
      ev.damage_solver_mask(o, counter_menu(CTX13, 2, remain=3)) == {1})
o = observation([pokemon(SMOOCHUM, 30), pokemon(MEGA_KANGA, 90)],
                our_active=DIVE_ACTIVE, our_prizes=6)
check("combo sweep: at 6 prizes the mega setup still outranks the low direct kill",
      ev.damage_solver_mask(o, counter_menu(CTX13, 2, remain=3)) == {1})
o = observation([pokemon(SLOWKING, 30), pokemon(MEGA_KANGA, 90)],
                our_active=DIVE_ACTIVE, our_prizes=6)
check("combo sweep: a direct kill HIGHER in the list still wins",
      ev.damage_solver_mask(o, counter_menu(CTX13, 2, remain=3)) == {0})
o = observation([pokemon(SLOWKING, 30), pokemon(SLOWKING, 80)],
                our_active=DIVE_ACTIVE, our_prizes=6)
check("combo sweep: same entry -> the certain direct KO beats the setup",
      ev.damage_solver_mask(o, counter_menu(CTX13, 2, remain=3)) == {0})
o = observation([pokemon(MEGA_KANGA, 90, energy=(MIST,)), pokemon(SMOOCHUM, 30)],
                our_active=DIVE_ACTIVE, our_prizes=6)
check("combo: a shielded bench target is no setup (dive counters would blank)",
      ev.damage_solver_mask(o, counter_menu(CTX13, 2, remain=3)) == {1})
o = observation([pokemon(SMOOCHUM, 30), pokemon(MEGA_KANGA, 90)],
                our_active=pokemon(MUNKIDORI, 110), our_prizes=3)
check("combo: dive not armed -> behavior identical to before (direct kill)",
      ev.damage_solver_mask(o, counter_menu(CTX13, 2, remain=3)) == {0})
o = observation([pokemon(SMOOCHUM, 30), pokemon(MEGA_KANGA, 90)],
                our_active=DIVE_ACTIVE, our_prizes=6)
check("combo: ctx 14 untouched (no setups at the attack's own menus)",
      ev.damage_solver_mask(o, counter_menu(CTX14, 2, remain=6)) == {0})
ev.set_solver_matchup("dragapult")
o = observation([pokemon(MUNKIDORI, 50)], their_active=pokemon(DRAGAPULT, 230),
                our_active=DIVE_ACTIVE, our_prizes=6)
check("combo active window: 230 active loaded to 200 for the dive's damage",
      ev.damage_solver_mask(o, with_active_row(counter_menu(CTX13, 1, remain=3)))
      == {1})
o = observation([pokemon(MUNKIDORI, 50)], their_active=pokemon(DRAGAPULT, 231),
                our_active=DIVE_ACTIVE, our_prizes=6)
check("combo active window: 231 is out of reach -> ladder stages the munki",
      ev.damage_solver_mask(o, with_active_row(counter_menu(CTX13, 1, remain=3)))
      == {0})
o = observation([pokemon(MUNKIDORI, 50)], their_active=pokemon(DRAGAPULT, 200),
                our_active=DIVE_ACTIVE, our_prizes=6)
check("combo active window: a 200 active already dies to the dive (doomed, excluded)",
      ev.damage_solver_mask(o, with_active_row(counter_menu(CTX13, 1, remain=3)))
      == {0})
ev.set_solver_matchup(None)

# healer detection: the grimmsnarl assumption + hydrapple ex. NB a Morgrem on the
# board itself triggers the impidimp/morgrem assumption, so the unhealed fixtures
# must not contain the Grimmsnarl line's pre-evolutions.
ev.set_solver_matchup("grimmsnarl")
o = observation([pokemon(GRIMM_EX, 195), pokemon(MUNKIDORI, 110)])
check("healers: energyless munki alone heals nothing -> munki to 60",
      ev.damage_solver_mask(o, counter_menu(CTX14, 2, remain=6)) == {1})
o = observation([pokemon(GRIMM_EX, 195), pokemon(MUNKIDORI, 110),
                 pokemon(IMPIDIMP, 70, energy=(MIST,))])   # shielded: board-only
check("healers: an impidimp in play flips every munki to dark (assumption)",
      ev.damage_solver_mask(o, counter_menu(CTX14, 3, remain=6)) == {0})
o = observation([pokemon(GRIMM_EX, 195)])
check("healers: no healer, 195 sits between every tier -> model's pick",
      ev.damage_solver_mask(o, counter_menu(CTX14, 1, remain=6)) is None)
o = observation([pokemon(GRIMM_EX, 195), pokemon(HYDRAPPLE_EX, 330)])
check("healers: an in-play hydrapple ex counts like a dark munki (170 tier opens)",
      ev.damage_solver_mask(o, counter_menu(CTX14, 2, remain=6)) == {0})
ev.set_solver_matchup(None)

# -- unknown deck (owner answer 6: same sections + munki factor, prizes-first order) --
ev.reset_opponent_context()
o = observation([pokemon(DREEPY, 20), pokemon(FEZ, 30)])
check("unknown ko: max prizes beats the mirror name order (fez over dreepy)",
      ev.damage_solver_mask(o, counter_menu(CTX14, 2, remain=6)) == {1})
ev.note_dragapult_opponent()
check("matchup switch: same board under the mirror list picks the dreepy",
      ev.damage_solver_mask(o, counter_menu(CTX14, 2, remain=6)) == {0})
ev.reset_opponent_context()
o = observation([pokemon(FEZ, 100), pokemon(MUNKIDORI, 110)])
check("unknown staging: biggest-prize-first walk (fez to 60 before munki)",
      ev.damage_solver_mask(o, counter_menu(CTX14, 2, remain=6)) == {0})
o = observation([pokemon(MEGA_KANGA, 250), pokemon(FEZ, 100)], our_prizes=3)
check("unknown sections: <=3 prizes stages the mega to 200 first",
      ev.damage_solver_mask(o, counter_menu(CTX14, 2, remain=6)) == {0})
o = observation([pokemon(MEGA_KANGA, 250), pokemon(FEZ, 100)], our_prizes=6)
check("unknown sections: at 6 prizes the fez-to-60 step comes first",
      ev.damage_solver_mask(o, counter_menu(CTX14, 2, remain=6)) == {1})
ev.note_dragapult_opponent()

# crustle: benched-Crustle staging priorities 240 (caped) / 120 / 70 (owner
# 2026-08-16 evening; "giant cape" = Hero's Cape 1159, the only +100 tool in pool)
ev.set_solver_matchup("crustle")
CRUSTLE, DWEBBLE_ID = 345, 344


def caped_crustle(hp):
    row = pokemon(CRUSTLE, hp, tools=(CAPE,))
    row["maxHp"] = 250
    return row


o = observation([caped_crustle(250), pokemon(DWEBBLE_ID, 70)])
check("crustle: caped 250 bench crustle chipped to 240 first",
      ev.damage_solver_mask(o, counter_menu(CTX14, 2, remain=6)) == {0})
o = observation([pokemon(CRUSTLE, 150), pokemon(DWEBBLE_ID, 70)])
check("crustle: uncaped bench crustle to 120 next",
      ev.damage_solver_mask(o, counter_menu(CTX14, 2, remain=6)) == {0})
o = observation([pokemon(CRUSTLE, 120), pokemon(DWEBBLE_ID, 70)])
check("crustle: then to 70",
      ev.damage_solver_mask(o, counter_menu(CTX14, 2, remain=6)) == {0})
o = observation([pokemon(CRUSTLE, 70)])
check("crustle: at 70 the priorities are met, the normal ladder continues (to 60)",
      ev.damage_solver_mask(o, counter_menu(CTX14, 1, remain=6)) == {0})
o = observation([pokemon(DWEBBLE_ID, 70)], their_active=caped_crustle(250))
check("crustle: the ACTIVE caped crustle is not a priority (bench only)",
      ev.damage_solver_mask(o, with_active_row(counter_menu(CTX14, 1, remain=6)))
      == {0})
ev.set_solver_matchup(None)

# -- eligibility ----------------------------------------------------------------------
o = observation([pokemon(DRAKLOAK, 30, energy=(MIST,)), pokemon(DREEPY, 20)])
check("mist: shielded target skipped at the attack menu",
      ev.damage_solver_mask(o, counter_menu(CTX14, 2, remain=6)) == {1})
o = observation([pokemon(DRAKLOAK, 0), pokemon(DREEPY, 20)])
check("dead target never picked",
      ev.damage_solver_mask(o, counter_menu(CTX14, 2, remain=6)) == {1})
o = observation([pokemon(DRAKLOAK, 0)])
check("all dead -> stand down", ev.damage_solver_mask(
    o, counter_menu(CTX14, 1, remain=6)) is None)
o = observation([pokemon(BUDEW, 20, energy=(MIST,)), pokemon(DREEPY, 25)])
check("exhausted waterfall STILL enforces exclusions (shielded row masked out)",
      ev.damage_solver_mask(o, counter_menu(CTX14, 2, remain=1)) == {1})

# -- ctx 13 (Adrena-Brain target) -----------------------------------------------------
o = observation([pokemon(FEZ, 30, energy=(MIST,))], our_prizes=2)
check("ctx13: mist does NOT block the ability target",
      ev.damage_solver_mask(o, counter_menu(CTX13, 1, remain=3)) == {0})
o = observation([pokemon(DRAGAPULT, 220)],
                their_active=pokemon(DRAGAPULT, 150),
                our_active=pokemon(DRAGAPULT, 320, energy=(7, 7)))
check("ctx13: the active our ready dive kills is excluded",
      ev.damage_solver_mask(o, with_active_row(counter_menu(CTX13, 1, remain=3)))
      == {0})
o = observation([], their_active=pokemon(DREEPY, 20),
                our_active=pokemon(MUNKIDORI, 110))
check("ctx13: active pickable when our dive is NOT ready",
      ev.damage_solver_mask(o, with_active_row(counter_menu(CTX13, 0, remain=3)))
      == {0})
o = observation([pokemon(DREEPY, 10), pokemon(DREEPY, 10)], our_prizes=2)
check("ctx13: win tier single-target only; identical kos tie -> model chooses",
      ev.damage_solver_mask(o, counter_menu(CTX13, 2, remain=3)) == {0, 1})

# -- ctx 40 (count menu) --------------------------------------------------------------
count_menu = {"context": CTX40, "minCount": 1, "maxCount": 1,
              "option": [{"type": ev.OPTION_TYPE_NUMBER, "number": n}
                         for n in (1, 2, 3)]}
check("ctx40: max count forced",
      ev.damage_solver_mask(observation([]), count_menu) == {2})

# -- adrena timing --------------------------------------------------------------------
def main_menu(our_active, our_bench):
    obs = observation([pokemon(DRAGAPULT, 320)], our_active=our_active,
                      our_bench=our_bench)
    sel = {"context": 0, "minCount": 1, "maxCount": 1, "option": [
        {"type": 7, "index": 0},
        {"type": ev.OPTION_TYPE_ABILITY, "area": 5, "index": 0, "playerIndex": 0},
        {"type": ev.OPTION_TYPE_ATTACK, "attackId": 154},
        {"type": 14}]}
    return obs, sel


obs, sel = main_menu(pokemon(DRAGAPULT, 290, energy=(7, 7)),
                     [pokemon(MUNKIDORI, 110, energy=(DARK,))])
check("no hold mask: adrena is not in the mask registry (owner: model's call)",
      not hasattr(ev, "adrena_hold_mask")
      and all(m.__name__ != "adrena_hold_mask" for m in ev.OPTION_MASK_RULES))
check("exit: attack and end masked while adrena is available (ability stays open)",
      ev.adrena_exit_mask(obs, sel) == {0, 1})
sel_used = {"context": 0, "minCount": 1, "maxCount": 1, "option": [
    {"type": 7, "index": 0}, {"type": ev.OPTION_TYPE_ATTACK, "attackId": 154},
    {"type": ev.OPTION_TYPE_END}]}
check("exit: no restriction once the ability option is gone",
      ev.adrena_exit_mask(obs, sel_used) is None)
obs_m, sel_m = main_menu(pokemon(MUNKIDORI, 80, energy=(DARK,)), [])
sel_m["option"][1] = {"type": ev.OPTION_TYPE_ABILITY, "area": 4, "index": 0,
                      "playerIndex": 0}
sel_m["option"].append({"type": ev.OPTION_TYPE_RETREAT})
check("exit: dark-munkidori-active retreat is also closed",
      ev.adrena_exit_mask(obs_m, sel_m) == {0, 1})
obs_d, sel_d = main_menu(pokemon(DRAGAPULT, 290), [pokemon(MUNKIDORI, 100,
                                                           energy=(DARK,))])
sel_d["option"].append({"type": ev.OPTION_TYPE_RETREAT})
check("exit: retreat of a non-munkidori active stays open",
      ev.adrena_exit_mask(obs_d, sel_d) == {0, 1, 4})
sel_forced = {"context": 0, "minCount": 1, "maxCount": 1, "option": [
    {"type": ev.OPTION_TYPE_ABILITY, "area": 5, "index": 0, "playerIndex": 0},
    {"type": ev.OPTION_TYPE_END}]}
check("exit: ability+end menu forces the ability (never masks to empty)",
      ev.adrena_exit_mask(obs, sel_forced) == {0})

# -- interception trigger -------------------------------------------------------------
obs, sel = main_menu(pokemon(DRAGAPULT, 290, energy=(7, 7)),
                     [pokemon(MUNKIDORI, 110, energy=(DARK,))])
check("force: chosen attack becomes adrena first",
      ev.adrena_first_index(obs, sel, [2]) == 1)
check("force: chosen END fires adrena first (turn-5 regression, 08-15)",
      ev.adrena_first_index(obs, sel, [3]) == 1)
check("force: non-end, non-attack play is untouched",
      ev.adrena_first_index(obs, sel, [0]) is None)
obs4, sel4 = main_menu(pokemon(MUNKIDORI, 80, energy=(DARK,)), [])
sel4["option"][1] = {"type": ev.OPTION_TYPE_ABILITY, "area": 4, "index": 0,
                     "playerIndex": 0}
sel4["option"].append({"type": ev.OPTION_TYPE_RETREAT})
check("force: retreat of the dark active munkidori fires adrena first",
      ev.adrena_first_index(obs4, sel4, [4]) == 1)
obs5, sel5 = main_menu(pokemon(DRAGAPULT, 290), [pokemon(MUNKIDORI, 110,
                                                         energy=(DARK,))])
sel5["option"].append({"type": ev.OPTION_TYPE_RETREAT})
check("force: retreat of a non-munkidori active is untouched",
      ev.adrena_first_index(obs5, sel5, [4]) is None)

# -- registration ---------------------------------------------------------------------
check("all 15 archetype lists registered, every list ends in anything-else",
      len(ev.SOLVER_PRIORITY) == 15
      and set(ev.SOLVER_TABLE_MATCHUPS) == set(ev.SOLVER_PRIORITY)
      and all(entries[-1] == (None, None)
              and all(name for name, _c in entries[:-1])
              for entries in ev.SOLVER_PRIORITY.values()))
ev.set_solver_matchup("starmie")
check("set_solver_matchup: accepts every archetype key",
      ev._SOLVER_MATCHUP == "starmie")
ev.set_solver_matchup("bogus")
check("set_solver_matchup: unknown name -> None (default behavior)",
      ev._SOLVER_MATCHUP is None)

ev._ACTION_RULES.clear()
check("rule disabled: inert",
      ev.damage_solver_mask(observation([pokemon(DREEPY, 20)]),
                            counter_menu(CTX14, 1, remain=6)) is None)

print()
print("FAILURES:", failures if failures else "none -- all pass")
sys.exit(1 if failures else 0)
