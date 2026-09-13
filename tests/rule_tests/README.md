# rule_tests -- durable unit matrices for the opt-in action rules

Run each against the repo src (they sys.path the repo root; test_root_veto loads the
canonical template turn_search.py). Older rules' suites lived in session scratchpads
and are gone; new/changed rules add their matrix here so rebuilds have a regression
gate.

- test_damage_solver.py (2026-08-15, REWORKED 2026-08-16 + evening addenda:
  the 400 tier, festival Rellor 30/10 pre-load, the REMOVED Archaludon cap (now stages freely),
  the grimmsnarl Freezing-Shroud KO discount, the ctx-13 dive-combo Adrena tier
  incl. window boundaries and the 30-nonex-vs-90-mega scenario, the crustle
  benched-240/120/70 priorities, the card-in-play Froslass trigger): damage_solver,
  90 checks -- win tier (MegaEx = 3 prizes per engine State.h getPrizeCount, incl.
  pending-dead prizes), the KO sweep over all-15 archetype PRIORITY LISTS
  (most-setup tiebreak energy > tool > Hero's Cape, the extra-KO exception,
  conditional entries: 2-energy Hariyama, lone-vs-spare Lunatone, Cynthia's
  Spiritomb 200+ bench gate), the UNIVERSAL staging ladder (prize-gated 3/2-prize
  sections, the global "n" dark-Munkidori budget, their-healer thresholds
  30/170/230 incl. the impidimp/morgrem assumption and Hydrapple ex, deep
  230/260/290 steps, evolution-aware passes: dreepy 30 -> 10 for a drakloak at
  30), the unknown-deck prizes-first default, Mist shielding at ctx 14 but never
  ctx 13, the doomed-active exclusion, forced max at ctx 40, and the Adrena exit
  mask + interception triggers.
- test_stadium_rule.py (2026-08-15, +mirror hold 2026-08-16): stadium_discipline,
  49 checks -- mask gating per matchup flag, Watchtower/Meowth exceptions (in play /
  discard-exhausted / provably-prized-while-invisible), lucario Jamming clause,
  festival freeze, alakazam jam-over-watchtower line rule + sunk pruning +
  ownership tracking, non-sticky flag replacement, Battle Cage clause (trigger
  detection + bump candidates), and the dragapult-MIRROR Watchtower hold (no
  stadium play over an in-play Watchtower once OUR Meowth fetch is spent and
  THEIRS is unfired; runs on _DRAGAPULT_OPPONENT, not the matchup flags).
- test_bank_curve.py (2026-08-15, re-anchored 2026-08-16 to the restored 256 top):
  the continuous bank curve in search_budget, 18 checks -- anchors (600 s -> 256
  sims @ 9.0 s, 400 -> ~208, 100 -> ~80), 40 s raw floor + sub-16-sim tail,
  monotone/quantized/no-cliff shape, slow-early-steep-late owner shape, linear
  deadline scaling, live-bank clamp + wall-clock fallback.
- test_root_veto.py (2026-08-15): the search root veto, 6 checks -- a >=8-visit
  child at Q<=-0.9 cannot win on visit count (episode-93157396 override); visit
  floor, non-veto at Q=-0.5, all-vetoed fallback, unvisited safety, allowed-set
  composition.
- test_meowth_gate.py (2026-08-15): meowth_supporter_gate_mask, 14 checks -- the
  going-first turn-1 Meowth FETCH mask (engine fact: supporters blocked only on
  turn 1, GameProc.h `state.turn <= 1`), the supporter-spent play surface, never-
  empty and disabled-rule contracts, and the DRAGAPULT MIRROR exception (owner
  2026-08-15 evening): vs known dragapult the turn-1 fetch is allowed, the hold is
  waived in FetchedCardLineRule (_fetch_hold_waived meowth clause, turn-1 only),
  and the waiver keeps a meowth-only Ultra Ball live in dead_fetch_item_mask.
- test_sunk_pruning.py (2026-08-14/15): FetchedCard + Meowth observe_real sunk-cost
  cleanup, 16 checks -- window keep-alive incl. the mid-chain COST-menu fix
  (obligations prune only at MAIN menus), due pruning lag, whiff clearing, turn
  reset.
- test_pokepad_gate.py (2026-08-16): pokepad_supporter_gate (sylveon), 8 checks --
  block on spent Supporter / going-first turn 1, dragapult + revealed-Budew
  exceptions, never-empty and disabled-rule contracts.
- test_deckout_guard.py (2026-08-16): prevent_deck_out, 14 checks -- the arming
  boundary in both prize regimes (draw-back 8 at exactly 6 prizes, else 6), inert
  cases incl. BOTH-decks-0, END-only masking, and the line rule's
  arm/save/violate/disarm cycle.
- test_academy_gate.py (2026-08-16): academy_at_night_gate (dragapult AND sylveon),
  10 checks -- stadium-USE (ABILITY in the STADIUM area) masked while our deck has
  >= 1 card, inert at deck 0 / other stadium / no stadium, seat-1 deck read,
  never-empty and disabled-rule contracts.
