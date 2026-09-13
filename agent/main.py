"""Kaggle cabt submission: v5-encoding PPO policy (Dragapult). Blocker C1's fix.

WHAT THIS UNBLOCKS. Every bundle before the v5 family serves a v1-v4 checkpoint. The current
training line (experiments/selfplay_ppo, --encoding v5) produces checkpoints that NO shipped
bundle could load: v5 widens the state/option/global vectors, adds a zone id, and -- the part
that is not just a width -- carries a THIRD per-seat tracker whose contract is a call order,
not a function call. Without this file the whole line is unshippable.

v5 = v3/v4 inputs ++ the 2026-07-30 pool-audit fix tier (src/game/encode_inflight.py): revealed-prize
tokens on their own zone id, in-flight pick memory, history serials, mulligan / face-down
events. Widths 593 / 122(+3) / 2046, 23 zones. The v4 ACTION space is unchanged -- v5 changes
what the model SEES, not how it answers.

THE STATEFUL ADDITION: `encode_inflight.InFlightTracker`. During a chained effect the engine asks
several questions in a row about ONE card (play Crispin -> which energy types -> onto which
Pokemon). The prompts alone never say which card the chain is about, so the tracker remembers
it -- strictly from engine-supplied fact: the select's own `contextCard`, or a pick the engine
has ALREADY ACCEPTED. It never infers what an effect means. Its contract is a call ORDER:

    reset()   on a new game, and on every no-menu observation (no chain in progress)
    observe() BEFORE encoding this prompt
    record()  AFTER the engine has accepted the answer

Training satisfies that order inside its own game loop (train_ppo.py `play_probe_game` /
the generator). A Kaggle agent CANNOT: `agent()` returns a move and is never told the engine
accepted it. So `record()` is DEFERRED here -- the answered observation, its select and the
move are stashed, and the record runs at the top of the NEXT call, before the trackers advance.
By then the engine has consumed the move, which is exactly the point in training's loop where
record() fires. The ordering is therefore identical, not merely similar; `_flush_pending` is
the only caller and runs exactly once per stashed move. EVERY returned move is stashed --
forced answers and fallbacks included -- because training records the move it actually sent,
whatever produced it. `_validation/parity.py` proves the whole loop move-for-move against a
transcription of train_ppo's driver.

NO ENGINE PROMPT IS ANSWERED BY HAND-WRITTEN POLICY (owner directive 2026-07-29). Every
submission before the v4 family carried a trivial-move helper that decided, in code, "take all
of them" whenever `maxCount >= len(option)`, "the first k in engine order" for every
multi-pick, and "[0]" for every single-option prompt even when `minCount == 0` made declining
legal. Those are real decisions -- which cards to discard, which two to search out, whether to
use an ability at all -- and this model TRAINS making them.

So there is no such helper here, and the build script asserts its name is absent from this
file. The only prompts answered without a model forward are the ones `encode_inflight.forced_answer`
(the v4 predicate, re-exported) PROVES have exactly one legal answer: with `low = minCount`,
`high = min(maxCount, n)`, that is `low == high and (low == 0 or low == n)` -- and NOT when the
answer is an ordering (`ORDER_SENSITIVE_CONTEXTS`, e.g. SKILL_ORDER: "take all n" has n! legal
answers, so the model picks the order). It is a predicate, not a case list, so a select shape
this deck has never produced is still handled correctly.

Everything else goes through `encode_inflight.resolve_with_v5`, which IS the selection loop. At every
step the model scores the options not yet picked plus a synthetic STOP row -- offered only once
`minCount` picks are in hand -- and the answer is submitted when STOP wins or the budget is
spent. Legality is by construction: indices are unique, in range, and the length lands inside
[minCount, maxCount] no matter what the net says. Consequence for latency: a multi-pick prompt
costs SEVERAL forwards (one per sub-pick); the board encode is shared across them (the board
cannot change until the answer is sent).

INFERENCE MIRRORS TRAINING EXACTLY (`train_ppo._encode_decision` / `_v4_probe_move` /
`_local_forward` under ENCODING == "v5"):
  * encode_observation_v5(observation, deck_counts=OUR 60-card counts, knowledge=, history=,
    in_flight=) -- token/global widths 593/122, 23 zones.
  * encode_option_v3 rows via encode_inflight.base_option_matrix_v5, encoded ONCE per select.
  * per sub-pick: candidate_matrix_v5 appends the v4 columns (is_stop_action,
    same_id_already_chosen) and the two v5 columns (targets_in_flight_host,
    picks_in_flight_card) -> width 2046; global_features_v5 appends the three v4 pick scalars
    (picks_so_far, minCount, maxCount) -> width 125.
  * SOFTMAX THEN argmax_tiebreak, not argmax over raw logits. train_ppo's probe scores through
    `_local_forward`, which returns `softmax(logits)`, and `argmax_tiebreak` splits EXACT ties
    uniformly -- softmax is monotone so the winner is the same, but it can MERGE two distinct
    logits into one float probability, creating a tie that the raw-logit path would not see.
    Matching the transform is what makes the tie sets identical.
  * ONE CardKnowledge + ONE ActionHistory(extended=True) + ONE InFlightTracker per game, for
    THIS seat only. The history MUST be extended: v5 reads mulligan and face-down rows that a
    plain ActionHistory never emits. Knowledge/history are updated EXACTLY ONCE per received
    observation -- including the no-menu ones, exactly as training's loops do (they update
    before the `select is None` check). Their logs are incremental, so a second update on the
    same observation double-counts events: `_update_trackers` is the only writer and `agent`
    is the only caller, once, before any decision work.
  * trackers are recreated when a new game is detected (a `current`-less observation, or a turn
    counter that went backwards -- Kaggle reuses one process for several episodes).

Runtime rules this bundle obeys (CLAUDE.md "Submission packaging"):
  * The loader `exec`s this source, so the module-path dunder is undefined -- bundled files are
    located via os.getcwd() / /kaggle_simulations/agent / the sys.path entry the loader appends
    for the agent's own directory (get_last_callable appends it before the exec).
  * `agent` is the LAST callable bound in this module. get_last_callable returns
    `[v for v in env.values() if callable(v)][-1]`, so anything callable defined after `agent`
    -- a class, a helper, a lambda -- would be handed to the engine instead. The build script
    asserts this.
  * The engine (bundle-local cg/) is imported ONCE at module load, inside its own
    try/except, purely to cache it in sys.modules while the loader's sys.path window is
    open -- the Kaggle runtime restores sys.path after load, so deferred bare imports
    fail (observed on-ladder 2026-08-03). Consumers: turn_search (search sessions) and
    src/game/encode_rich.py, which decodes
    observation["search_begin_input"] through the source-built engine's DumpState export to
    fill the encoder's RICH block (16 token + 14 global dims), which was LIVE during this run's
    training (train_ppo defaults CG_DLL to engine_src/build/cg.dll). If the engine cannot load
    -- or is absent entirely -- that block is zeros and play continues. `dump_state(blob)` is a
    pure decode with no battle pointer, so it cannot disturb the host engine running the match.
  * No PyYAML anywhere in the bundled src/ subset.
  * CPU-only, 2 torch threads (eval hardware = 2 vCPU / ~6.5 GB / no GPU). Weights are STORED
    fp16 and cast to fp32 at load: compute is fp32 (bf16/fp16 compute is a known instability
    here, and fp16 CPU matmul is slow anyway).
  * Every decision is wrapped: any failure returns a random legal move (a crash = a loss).

SEARCH EXTENSION POINT: `search_client.SearchClient` is a documented stub with the
`evaluate(states) -> (priors, values)` interface the separate search tree (search_gen.py) will
plug into. `USE_SEARCH` is False and the stub is never imported at load. Raw-policy serving
below is complete and self-sufficient without it; see search_client.py for the contract.

`_STATS` is the no-auto-answer census (forwards vs forced-by-proof, per reason). A validation
harness that loads this module with runpy reads it straight off the module globals.
"""

import copy
import os
import random
import sys
import threading
import time
import traceback
from collections import Counter

# The loader execs this source, so the module-path dunder is meaningless. Kaggle puts it at
# /kaggle_simulations/agent; a local harness usually chdirs into it; and
# kaggle_environments.agent.get_last_callable appends the agent file's own directory to
# sys.path before the exec, so that entry is a third, loader-supplied candidate.
_CANDIDATES = [os.getcwd(), "/kaggle_simulations/agent"] + [p for p in sys.path if p]
AGENT_DIR = next((d for d in _CANDIDATES
                  if os.path.exists(os.path.join(d, "ppo.pt"))), os.getcwd())
if AGENT_DIR not in sys.path:
    sys.path.insert(0, AGENT_DIR)


def _ensure_agent_path():
    """The Kaggle loader keeps AGENT_DIR on sys.path only WHILE main.py loads, then
    restores the path (observed on-ladder 2026-08-03, two episodes: every load-time
    import works, every deferred import -- turn_search, cg -- raises
    ModuleNotFoundError). Called at the top of every agent() call and before any lazy
    package import: idempotent, two comparisons when the path is already present."""
    if AGENT_DIR not in sys.path:
        sys.path.insert(0, AGENT_DIR)

with open(os.path.join(AGENT_DIR, "deck.csv")) as _deck_file:
    DECK = [int(line) for line in _deck_file if line.strip()]

# Search hook: 256x1 OUR-TURN determinized PUCT (turn_search.py, owner spec 2026-08-02).
# Single-pick (min==max==1) selects with >= 2 options are searched when the episode time
# bank allows; every other decision -- and every search failure of any kind -- plays the
# raw policy exactly as before. turn_search imports lazily inside _get_search so a broken
# search module can never take down raw-policy serving. V5_DISABLE_SEARCH=1 turns the hook
# off for harnesses that need raw-policy determinism (the parity drill compares decisions
# against reference drivers that do not search).
USE_SEARCH = os.environ.get("V5_DISABLE_SEARCH") != "1"

_MODEL = None
_DECK_COUNTS = None
_CardKnowledge = _ActionHistory = _InFlightTracker = None
_encode_observation = _argmax_tiebreak = None
_forced_answer = _global_features = _resolve_with = None
_ACTION_RULES = []                       # from the checkpoint envelope (see model load)
_combined_option_mask = None
_LINE_RULE = None                        # search line rule object (state_encoder.line_rule_for)
_REVEAL_SAMPLING = []                    # card ids whose in-search plays sample redraws
_FETCH_BRANCH = False                    # envelope "fetch_branch": search fetch menus
_FORCED_EVOLUTION_OPTIONS = None         # force_evolve_before_shuffle_draw helpers
_SHUFFLE_PLAY_INDICES = None
_BATTLE_CAGE_BLOCKED = None              # stadium_discipline Battle Cage clause helpers
_STADIUM_BUMP_CANDIDATES = None
_ADRENA_FIRST = None                     # damage_solver pre-attack Adrena-Brain forcing
_np = _torch = None
# Tie-break stream. Fixed seed: the choice among EXACT ties is still uniform, but a replayed
# game reproduces, which is what made ties measurable in the first place.
_TIE_RNG = random.Random(20260729)
# The census. `forwards` = model decisions; `forced*` = answers proved unique by
# encode_inflight.forced_answer; `forced_pick` = a sub-pick with one option left and STOP not yet
# legal. Nothing else may ever answer the engine.
_STATS = Counter()
try:
    import numpy as _np
    import torch as _torch

    _torch.set_num_threads(2)                      # eval hardware = 2 vCPU, no GPU

    from src.decks.card_knowledge import CardKnowledge as _CardKnowledge
    from src.game.action_history import ActionHistory as _ActionHistory
    from src.models.transformer import (GameStateTransformer,
                                        GameStateTransformerConfig)
    from src.tiebreak import argmax_tiebreak as _argmax_tiebreak

    _state = _torch.load(os.path.join(AGENT_DIR, "ppo.pt"), map_location="cpu",
                         weights_only=False)
    # THE ENCODER IS CHOSEN BY THE CHECKPOINT, not hardcoded: one hand-written main.py
    # serves every bundle in the family, so a fix here reaches all of them. v6 = v5 ++ the
    # two chain-progress option columns (batched-effect fix, src/game/state_encoder.py); its
    # state encoder and pick-scalar globals ARE v5's (re-exported), only the option rows
    # widen 2046 -> 2048. The builder ships exactly the encoder module this line needs.
    _ENCODING = _state.get("encoding")
    assert _ENCODING in ("v5", "v6"), f"unservable checkpoint encoding: {_ENCODING}"
    if _ENCODING == "v6":
        from src.game.state_encoder import (InFlightTracker as _InFlightTracker,
                                        encode_observation_v6 as _encode_observation,
                                        forced_answer as _forced_answer,
                                        global_features_v6 as _global_features,
                                        resolve_with_v6 as _resolve_with)
        # Opt-in action rules TRAVEL WITH THE CHECKPOINT (2026-08-10): a model trained
        # under a rule (d128_mask: counter_cap, live from iteration 1) must play under
        # it -- unmasked menus would show it options outside its training distribution.
        # Enabling here makes resolve_with_v6 mask every raw pick; the search's branch
        # points get the same mask via the option_mask_hook set in _turn_search_module.
        _ACTION_RULES = [str(rule) for rule in (_state.get("action_rules") or [])]
        if _ACTION_RULES:
            from src.game.state_encoder import (
                combined_option_mask as _combined_option_mask,
                enable_action_rules as _enable_action_rules,
                line_rule_for as _line_rule_for)
            _enable_action_rules(*_ACTION_RULES)
            _LINE_RULE = _line_rule_for(_ACTION_RULES)
            print(f"[agent] action rules enabled from envelope: {_ACTION_RULES}"
                  + (f" (line rule: {_LINE_RULE.name})" if _LINE_RULE else ""),
                  file=sys.stderr, flush=True)
            if "force_evolve_before_shuffle_draw" in _ACTION_RULES:
                try:
                    from src.game.state_encoder import (
                        forced_evolution_options as _FORCED_EVOLUTION_OPTIONS,
                        shuffle_play_indices as _SHUFFLE_PLAY_INDICES)
                except ImportError:
                    pass           # older bundled encode: the rule stays inert
            if "stadium_discipline" in _ACTION_RULES:
                try:
                    from src.game.state_encoder import (
                        battle_cage_blocked_action_indices as _BATTLE_CAGE_BLOCKED,
                        stadium_bump_candidates as _STADIUM_BUMP_CANDIDATES)
                except ImportError:
                    pass           # older bundled encode: the clause stays inert
            if "damage_solver" in _ACTION_RULES:
                try:
                    from src.game.state_encoder import (
                        adrena_first_index as _ADRENA_FIRST)
                except ImportError:
                    pass           # older bundled encode: the rule stays inert
        # OPT-IN sampled chance nodes (owner 2026-08-14): envelope "reveal_sampling"
        # lists card ids whose IN-SEARCH plays are valued as the MEAN over up to
        # REVEAL_SAMPLE_CAP independent engine redraws (turn_search reveal nodes;
        # measured 2026-08-13: re-stepping a shuffle-draw play re-randomizes) instead
        # of planning into one sampled redraw. Absent (every existing bundle) =
        # frozenset() in turn_search = search behavior bit-identical to before.
        _REVEAL_SAMPLING = [int(card_id)
                            for card_id in (_state.get("reveal_sampling") or [])]
        if _REVEAL_SAMPLING:
            print(f"[agent] reveal sampling enabled from envelope: {_REVEAL_SAMPLING}",
                  file=sys.stderr, flush=True)
        # OPT-IN fetch branching v2 (owner 2026-08-14): envelope "fetch_branch" makes
        # optional deck/discard CARD menus real branch points -- in-search AND at the
        # real decision (_search_move widens its gate; a searched DECLINE answers []).
        # Duplicate candidate copies are merged and the FETCH_TOPK strongest distinct
        # cards kept (turn_search._fetch_groups). Absent = search behavior identical
        # to before.
        _FETCH_BRANCH = bool(_state.get("fetch_branch"))
        if _FETCH_BRANCH:
            print("[agent] fetch branching enabled from envelope",
                  file=sys.stderr, flush=True)
    else:
        from src.game.encode_inflight import (InFlightTracker as _InFlightTracker,
                                        encode_observation_v5 as _encode_observation,
                                        forced_answer as _forced_answer,
                                        global_features_v5 as _global_features,
                                        resolve_with_v5 as _resolve_with)
    _MODEL = GameStateTransformer(GameStateTransformerConfig(
        token_feature_dim=_state["token_feature_dim"],
        global_feature_dim=_state["global_feature_dim"],
        option_feature_dim=_state["option_feature_dim"],
        num_zones=_state["num_zones"],
        d_model=_state["d_model"],
        num_layers=_state["num_layers"],
        num_heads=_state["num_heads"],
        feedforward_dim=_state["feedforward_dim"],
        card_vocab=_state.get("card_vocab", 0),
        card_embedding_dim=_state.get("card_embedding_dim", 64)))
    # Stored fp16, computed fp32: the cast happens ONCE here, so no forward pays for it and
    # no activation is ever half precision.
    _MODEL.load_state_dict({key: value.float()
                            for key, value in _state["state_dict"].items()})
    _MODEL.eval()
    _DECK_COUNTS = dict(Counter(DECK))
    if _ENCODING == "v6":
        try:
            # Arms require_play_fetched_card's dead-fetch MASK surface (it needs our
            # decklist to bound what a deck search could offer); inert without the
            # rule in the envelope, and an older bundled encode just stays mask-less.
            from src.game.state_encoder import note_our_decklist as _note_our_decklist
            _note_our_decklist(_DECK_COUNTS)
        except ImportError:
            pass
    print(f"[agent] v5 policy ready: iteration {_state.get('iteration')}, "
          f"d_model {_state['d_model']}, dims {_state['token_feature_dim']}/"
          f"{_state['global_feature_dim']}/{_state['option_feature_dim']}, "
          f"zones {_state['num_zones']}, weights {_state.get('weight_dtype', 'unknown')}",
          file=sys.stderr, flush=True)
except Exception:
    print("[agent] model construction failed:\n" + traceback.format_exc(),
          file=sys.stderr, flush=True)
    _MODEL = None

# ----------------------------------------------------------------- archetype router ---- #
# Model bank + router (2026-08-08, owner spec; template refreshed 2026-08-12 onto the
# counter_cap + knowledge_hook family main.py). When the bundle ships router_data.json,
# every decision re-picks the serving model by the router's posterior over the opponent's
# revealed cards (tau_enter/tau_exit in the data file; stateless rule, may switch more than
# once -- evidence only accumulates, so oscillation is impossible). EVERY switch is printed
# to stderr as a "[router]" line so Kaggle replay logs show exactly when and why the model
# changed. Router failure of any kind permanently falls back to the base model.
#
# LAZY SPECIALIST LOADING (owner spec 2026-08-14): only the BASE model loads at setup --
# module load was paying ~9 torch.loads on the first act's clock while a typical game
# routes to at most one specialist. Setup now only VALIDATES the manifest (a missing
# checkpoint still fails the bundle loudly, where drills catch it); each specialist
# checkpoint loads ON FIRST ROUTE to its archetype, a one-time sub-second block inside a
# decision. Blocking there is safe by design: cabt has no per-move limit, only the
# episode bank, and the lazy total is strictly cheaper than the old load-everything
# setup. A specialist whose lazy load fails is pinned out permanently (base keeps
# serving) without retiring the router.
_MODEL_BANK = {}
_ACTIVE_MODEL = "base"
_ROUTER = None
_BASE_STATE_META = None
_SPECIALIST_PATHS = {}
_FAILED_SPECIALISTS = set()


def _load_policy(path):
    """Construct a policy from a bundle-format checkpoint (same recipe as the base load
    above; dims and encoding MUST match the base -- one encoder serves the whole bank)."""
    state = _torch.load(path, map_location="cpu", weights_only=False)
    assert state.get("encoding") == _ENCODING, \
        f"bank checkpoint encoding {state.get('encoding')} != base {_ENCODING}"
    for key in ("token_feature_dim", "global_feature_dim", "option_feature_dim",
                "num_zones", "d_model", "num_layers", "num_heads", "feedforward_dim"):
        assert state[key] == _BASE_STATE_META[key], \
            f"bank checkpoint {key}={state[key]} != base {_BASE_STATE_META[key]}"
    model = GameStateTransformer(GameStateTransformerConfig(
        token_feature_dim=state["token_feature_dim"],
        global_feature_dim=state["global_feature_dim"],
        option_feature_dim=state["option_feature_dim"],
        num_zones=state["num_zones"],
        d_model=state["d_model"],
        num_layers=state["num_layers"],
        num_heads=state["num_heads"],
        feedforward_dim=state["feedforward_dim"],
        card_vocab=state.get("card_vocab", 0),
        card_embedding_dim=state.get("card_embedding_dim", 64)))
    model.load_state_dict({key: value.float()
                           for key, value in state["state_dict"].items()})
    model.eval()
    return model, state.get("iteration")


try:
    if _MODEL is not None:
        _MODEL_BANK["base"] = _MODEL
        _BASE_STATE_META = _state
        _router_data_path = os.path.join(AGENT_DIR, "router_data.json")
        if os.path.exists(_router_data_path):
            import importlib.util as _importlib_util
            import json as _json
            _router_spec = _importlib_util.spec_from_file_location(
                "bundle_router", os.path.join(AGENT_DIR, "router.py"))
            _router_module = _importlib_util.module_from_spec(_router_spec)
            _router_spec.loader.exec_module(_router_module)
            with open(_router_data_path, encoding="utf-8") as _router_file:
                _router_config = _json.load(_router_file)
            _router_candidate = _router_module.ArchetypeRouter(_router_config)
            for _archetype, _relative in sorted(_router_candidate.specialists.items()):
                _specialist_path = os.path.join(AGENT_DIR, _relative)
                assert os.path.exists(_specialist_path), \
                    f"specialist checkpoint missing: {_relative}"
                _SPECIALIST_PATHS[_archetype] = _specialist_path
            _ROUTER = _router_candidate
            print(f"[router] ready: {len(_router_candidate.lists)} lists / "
                  f"{len(_router_candidate.archetypes)} archetypes, "
                  f"specialists={sorted(_router_candidate.specialists)} (lazy), "
                  f"tau_enter={_router_candidate.tau_enter} "
                  f"tau_exit={_router_candidate.tau_exit}",
                  file=sys.stderr, flush=True)
except Exception:
    print("[router] construction failed (base model serves alone):\n"
          + traceback.format_exc(), file=sys.stderr, flush=True)
    _ROUTER = None


def _route_model(obs_dict):
    """Re-pick the serving model from the opponent's revealed cards. Never raises; any
    failure retires the router for the rest of the process and the base model serves."""
    global _MODEL, _ACTIVE_MODEL, _ROUTER
    if _ROUTER is None or _MODEL is None:
        return
    try:
        _ROUTER.update(obs_dict)
        # require_play_fetched_card's dragapult exception (owner 2026-08-13): the
        # router posterior IS the "recognizer says dragapult" half of the condition
        # (the Budew half comes from the line rule's own real-observation scan). Same
        # threshold that would engage a dragapult specialist. Sticky inside the encode
        # module; guarded so an older bundled encode without the hook is a no-op and
        # an encode hiccup can never retire the router.
        try:
            _posterior = getattr(_ROUTER, "_posterior", None) or {}
            if _posterior.get("dragapult", 0.0) >= _ROUTER.tau_enter:
                from src.game.state_encoder import note_dragapult_opponent
                note_dragapult_opponent()
            # force_evolve_before_shuffle_draw's Safeguard gate: crustle IS a pool
            # archetype (sylveon is not -- reveals only, scanned by the tracker).
            if _posterior.get("crustle", 0.0) >= _ROUTER.tau_enter:
                from src.game.state_encoder import note_safeguard_opponent
                note_safeguard_opponent()
            # stadium_discipline's matchup flags (owner rule 2026-08-15): REPLACED
            # from the live posterior each decision -- a dethroned early lock lifts
            # its restrictions (owner fix 2026-08-15); reset_episode clears.
            from src.game.state_encoder import (STADIUM_RULE_MATCHUPS,
                                            set_matchup_flags)
            set_matchup_flags(_archetype for _archetype in STADIUM_RULE_MATCHUPS
                              if _posterior.get(_archetype, 0.0)
                              >= _ROUTER.tau_enter)
            # damage_solver's matchup table (owner tables 2026-08-15 night): the
            # strongest posterior among the authored tables, REPLACED each decision
            # (None = the DEFAULT table). Guarded like the rest: an older bundled
            # encode without the hook keeps the solver on its fallback tables.
            try:
                from src.game.state_encoder import (SOLVER_TABLE_MATCHUPS,
                                                set_solver_matchup)
                _best_table = None
                for _archetype in SOLVER_TABLE_MATCHUPS:
                    _p = _posterior.get(_archetype, 0.0)
                    if _p >= _ROUTER.tau_enter \
                            and (_best_table is None or _p > _best_table[1]):
                        _best_table = (_archetype, _p)
                set_solver_matchup(_best_table[0] if _best_table else None)
            except ImportError:
                pass
            # Its provable-prized-Meowth exception: prizes are fixed at game start,
            # so once the never-seen pool has shrunk to EXACTLY the remaining prize
            # count (any full deck view does this), every never-seen card is provably
            # prized -- a never-seen Meowth ex included.
            _knowledge = _GAME.get("knowledge")
            if _knowledge is not None:
                _never_seen = _knowledge.never_seen_counts()
                _me = obs_dict["current"]["players"][obs_dict["current"]["yourIndex"]]
                if _never_seen.get(1071, 0) > 0 \
                        and sum(_never_seen.values()) == len(_me.get("prize") or []):
                    from src.game.state_encoder import note_meowth_prize_proven
                    note_meowth_prize_proven()
        except Exception:
            pass
        desired, top, probability = _ROUTER.choose(_ACTIVE_MODEL)
        if desired != _ACTIVE_MODEL and desired not in _MODEL_BANK \
                and desired in _SPECIALIST_PATHS \
                and desired not in _FAILED_SPECIALISTS:
            # First route to this archetype: load its checkpoint NOW and only announce
            # the switch below once the load has succeeded, so the census and replay
            # logs never claim a switch that then fell back. A failed load pins the
            # archetype out for the rest of the process; the router itself survives.
            try:
                _load_started = time.time()
                _specialist, _iteration = _load_policy(_SPECIALIST_PATHS[desired])
                _MODEL_BANK[desired] = _specialist
                print(f"[router] specialist loaded lazily: {desired} "
                      f"(iteration {_iteration}, {time.time() - _load_started:.2f}s)",
                      file=sys.stderr, flush=True)
            except Exception:
                _FAILED_SPECIALISTS.add(desired)
                print(f"[router] specialist load FAILED: {desired} "
                      f"({_ACTIVE_MODEL} keeps serving):\n" + traceback.format_exc(),
                      file=sys.stderr, flush=True)
        if desired != _ACTIVE_MODEL and desired in _MODEL_BANK:
            print(f"[router] MODEL SWITCH {_ACTIVE_MODEL} -> {desired} "
                  f"(P[{top}]={probability:.3f}, seen={_ROUTER.seen_count} cards, "
                  f"decision {_STATS['selects']})", file=sys.stderr, flush=True)
            _STATS[f"router_switch_to_{desired}"] += 1
            _ACTIVE_MODEL = desired
            _MODEL = _MODEL_BANK[desired]
        elif (top is not None and top not in _ROUTER.specialists
                and probability >= _ROUTER.tau_enter
                and not _STATS[f"router_seen_{top}"]):
            _STATS[f"router_seen_{top}"] += 1     # once per archetype per process
            print(f"[router] recognized {top} (P={probability:.3f}) -- no specialist, "
                  f"base continues", file=sys.stderr, flush=True)
    except Exception:
        print("[router] runtime error (retired; base model serves):\n"
              + traceback.format_exc(), file=sys.stderr, flush=True)
        _ROUTER = None
        if "base" in _MODEL_BANK:
            _ACTIVE_MODEL = "base"
            _MODEL = _MODEL_BANK["base"]


# cg preloaded at MODULE LOAD, while the loader's sys.path window is provably open (the
# src.* imports above just used it): decision-time `from cg import api` then resolves
# from the sys.modules cache no matter what happens to sys.path afterwards. Guarded on
# its own -- an engine that cannot load must never take the model down (search and the
# rich block both degrade gracefully without it).
try:
    _ensure_agent_path()
    import cg.api as _cg_api_preloaded          # noqa: F401
    print("[agent] cg preloaded at module load", file=sys.stderr, flush=True)
except Exception:
    print("[agent] cg preload failed (search + rich features degrade):\n"
          + traceback.format_exc(), file=sys.stderr, flush=True)

# ---------------------------------------------------------------- traced inference ---- #
# TorchScript serving (owner go 2026-08-14). Each bank model's policy_value/forward is
# re-served through a torch.jit.trace + freeze + optimize_for_inference graph pair, built
# lazily AT RUNTIME on the first forward that model takes (~0.4s each; no serialized
# artifacts, so nothing depends on the eval box's torch version). The traced graphs drop
# the padding mask: serving is always batch-1, so the mask is all-False by construction
# (both drivers below build it that way) and eliding it is most of the win. Measured
# 2026-08-14 (idle box, same-seed searched game): 1.10x end-to-end, 1.30x under CPU
# contention; per-forward 2.39 -> 2.05 ms.
#
# NOT bit-identical: trace parity is ~1e-5 on logits, which can flip EXACT softmax ties.
# Harnesses that compare decisions against an eager reference driver must therefore set
# V5_DISABLE_JIT=1 (same convention as V5_DISABLE_SEARCH). Failure of any kind -- trace
# build, parity self-check, or a traced call at decision time -- permanently retires the
# trace for that model and the eager module serves exactly as before.
USE_JIT = os.environ.get("V5_DISABLE_JIT") != "1"
_TRACE_CACHE = {}                    # id(model) -> (policy_graph, value_graph) | None


def _build_traced_graphs(model):
    """Trace one bank model into a (policy, value) graph pair and self-check parity
    against the eager module at two off-trace shapes. Raises on any mismatch; the caller
    turns that into a permanent eager fallback for this model."""

    class PolicyGraph(_torch.nn.Module):
        """model.policy_value with the always-all-False padding mask elided."""

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, tokens, owners, zones, globals_t, options, card_ids):
            module = self.module
            batch_size = tokens.shape[0]
            cards = module.card_projection(
                _torch.cat([tokens, module.card_embedding(card_ids)], dim=-1))
            cards = cards + module.owner_embedding(owners) + module.zone_embedding(zones)
            global_token = module.global_projection(globals_t).unsqueeze(1)
            owner = _torch.full((batch_size, 1), 2)
            zone = _torch.full((batch_size, 1), 0)
            global_token = (global_token + module.owner_embedding(owner)
                            + module.zone_embedding(zone))
            sequence = _torch.cat([global_token, cards], dim=1)
            encoded = module.encoder(sequence, src_key_padding_mask=None)
            context = encoded[:, 0]
            value = _torch.tanh(module.value_head(context)).squeeze(-1)
            broadcast = context.unsqueeze(1).expand(-1, options.shape[1], -1)
            logits = module.policy_score(
                _torch.cat([broadcast, options], dim=-1)).squeeze(-1)
            return logits, value

    class ValueGraph(_torch.nn.Module):
        """model.forward (value only), padding mask elided."""

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, tokens, owners, zones, globals_t, card_ids):
            module = self.module
            batch_size = tokens.shape[0]
            cards = module.card_projection(
                _torch.cat([tokens, module.card_embedding(card_ids)], dim=-1))
            cards = cards + module.owner_embedding(owners) + module.zone_embedding(zones)
            global_token = module.global_projection(globals_t).unsqueeze(1)
            owner = _torch.full((batch_size, 1), 2)
            zone = _torch.full((batch_size, 1), 0)
            global_token = (global_token + module.owner_embedding(owner)
                            + module.zone_embedding(zone))
            sequence = _torch.cat([global_token, cards], dim=1)
            encoded = module.encoder(sequence, src_key_padding_mask=None)
            return _torch.tanh(module.value_head(encoded[:, 0])).squeeze(-1)

    def example(token_count, option_count):
        return (_torch.randn(1, token_count, _state["token_feature_dim"]),
                _torch.randint(0, 3, (1, token_count)),
                _torch.randint(0, _state["num_zones"], (1, token_count)),
                _torch.randn(1, _state["global_feature_dim"]),
                _torch.randn(1, option_count, _state["option_feature_dim"]),
                _torch.randint(0, max(_state.get("card_vocab", 0), 1),
                               (1, token_count)))

    with _torch.inference_mode():
        tokens, owners, zones, globals_t, options, card_ids = example(90, 8)
        policy_graph = _torch.jit.trace(
            PolicyGraph(model), (tokens, owners, zones, globals_t, options, card_ids))
        policy_graph = _torch.jit.optimize_for_inference(
            _torch.jit.freeze(policy_graph.eval()))
        value_graph = _torch.jit.trace(
            ValueGraph(model), (tokens, owners, zones, globals_t, card_ids))
        value_graph = _torch.jit.optimize_for_inference(
            _torch.jit.freeze(value_graph.eval()))
        # Parity self-check at two OFF-TRACE shapes: the trace must shape-generalize AND
        # agree with the eager module, or it never serves. policy_value's value output and
        # forward's are the same head over the same context, so one eager reference
        # checks both graphs.
        for token_count, option_count in ((60, 3), (110, 14)):
            tokens, owners, zones, globals_t, options, card_ids = example(token_count,
                                                                          option_count)
            padding = _torch.zeros(1, token_count, dtype=_torch.bool)
            option_mask = _torch.ones(1, option_count, dtype=_torch.bool)
            eager_logits, eager_value = model.policy_value(
                tokens, owners, zones, padding, globals_t, options, option_mask,
                card_ids=card_ids)
            traced_logits, traced_value = policy_graph(tokens, owners, zones, globals_t,
                                                       options, card_ids)
            delta = max(float((traced_logits - eager_logits).abs().max()),
                        float((traced_value - eager_value).abs().max()))
            assert delta < 1e-4, f"policy trace parity broke: {delta}"
            leaf_value = value_graph(tokens, owners, zones, globals_t, card_ids)
            delta = float((leaf_value - eager_value).abs().max())
            assert delta < 1e-4, f"value trace parity broke: {delta}"
    return policy_graph, value_graph


def _traced(model):
    """The traced graph pair for `model`, built on first use. None = eager serves (flag
    off, a model without the card-identity embedding, or a previous failure)."""
    if not USE_JIT or _torch is None \
            or getattr(model, "card_embedding", None) is None:
        return None
    key = id(model)
    if key not in _TRACE_CACHE:
        try:
            started = time.time()
            _TRACE_CACHE[key] = _build_traced_graphs(model)
            _STATS["jit_traced_models"] += 1
            print(f"[agent] jit trace ready ({time.time() - started:.1f}s)",
                  file=sys.stderr, flush=True)
        except Exception:
            _TRACE_CACHE[key] = None
            print("[agent] jit trace failed (eager path serves):\n"
                  + traceback.format_exc(), file=sys.stderr, flush=True)
    return _TRACE_CACHE[key]


def _policy_forward(tokens, owners, zones, padding, globals_tensor, option_tensor,
                    option_mask, identity):
    """One policy+value forward through the ACTIVE model: the traced graph when
    available, the eager module otherwise. A traced call that raises retires that
    model's trace and re-runs eager -- serving can only ever degrade to the old path.
    Batched-search worker threads (thread-local set by _ForwardBatcher.enter_worker)
    are routed into the batcher instead: same contract, coalesced execution."""
    batcher = getattr(_SEARCH_WORKER, "batcher", None)
    if batcher is not None and identity is not None:
        return batcher.submit("policy", (tokens, owners, zones, globals_tensor,
                                         option_tensor, identity))
    graphs = _traced(_MODEL) if identity is not None else None
    if graphs is not None:
        try:
            return graphs[0](tokens, owners, zones, globals_tensor, option_tensor,
                             identity)
        except Exception:
            _TRACE_CACHE[id(_MODEL)] = None
            _STATS["jit_runtime_fallbacks"] += 1
    return _MODEL.policy_value(tokens, owners, zones, padding, globals_tensor,
                               option_tensor, option_mask, card_ids=identity)


def _value_forward(tokens, owners, zones, padding, globals_tensor, identity):
    """Value-only forward (search leaf evaluation), same trace-first contract."""
    batcher = getattr(_SEARCH_WORKER, "batcher", None)
    if batcher is not None and identity is not None:
        return batcher.submit("value", (tokens, owners, zones, globals_tensor, None,
                                        identity))
    graphs = _traced(_MODEL) if identity is not None else None
    if graphs is not None:
        try:
            return graphs[1](tokens, owners, zones, globals_tensor, identity)
        except Exception:
            _TRACE_CACHE[id(_MODEL)] = None
            _STATS["jit_runtime_fallbacks"] += 1
    return _MODEL.forward(tokens, owners, zones, padding, globals_tensor,
                          card_ids=identity)


# ------------------------------------------------------------ batched search forwards - #
# BATCHED LEAF EVALUATION support (owner go 2026-08-14; the OPT-IN flag
# V5_BATCHED_SEARCH=1 is read by turn_search -- this side only offers the capability).
# turn_search's descent threads land in _policy_forward/_value_forward like every other
# caller; the thread-local set by enter_worker routes their tensors here, where serve()
# -- run by the search's own calling thread -- coalesces everything queued into ONE
# padded batch-N EAGER forward. The traced graphs are batch-1-shaped and elide the
# padding mask, so the batch path uses the eager module with a REAL mask instead; the
# batch amortization, not the trace, is the win here. Torch therefore still runs on
# exactly one thread at a time (the serving thread), keeping set_num_threads(2) honest.
_SEARCH_WORKER = threading.local()


class _ForwardBatcher:
    """One instance per searched decision, created by turn_search via the injected
    factory. submit() blocks the calling worker until its slice of the batched forward
    arrives; close() releases every still-blocked submitter with an error so the search
    can always tear down."""

    def __init__(self, width):
        self.width = width
        self.condition = threading.Condition()
        self.requests = []
        self.closed = False

    def enter_worker(self):
        _SEARCH_WORKER.batcher = self

    def exit_worker(self):
        _SEARCH_WORKER.batcher = None

    def submit(self, kind, tensors):
        request = {"kind": kind, "tensors": tensors, "result": None, "error": None,
                   "event": threading.Event()}
        with self.condition:
            if self.closed:
                raise RuntimeError("forward batcher closed")
            self.requests.append(request)
            self.condition.notify_all()
        request["event"].wait()
        if request["error"] is not None:
            raise request["error"]
        return request["result"]

    def close(self):
        with self.condition:
            self.closed = True
            leftovers, self.requests = self.requests, []
        for request in leftovers:
            request["error"] = RuntimeError("forward batcher closed")
            request["event"].set()

    def serve(self, workers_alive):
        """Coalesce and run queued forwards until every worker thread has exited. No
        artificial batching delay: the first request wakes the loop, and every request
        that lands while a forward is running joins the NEXT batch -- the batch width
        self-regulates to however many workers are parked."""
        while True:
            with self.condition:
                while not self.requests:
                    if not workers_alive():
                        return
                    self.condition.wait(0.005)
                batch, self.requests = self.requests, []
            try:
                self._run_batch(batch)
            except Exception as error:
                for request in batch:
                    if not request["event"].is_set():
                        request["error"] = error
                        request["event"].set()
                raise

    def _run_batch(self, batch):
        for kind in ("policy", "value"):
            group = [request for request in batch if request["kind"] == kind]
            if not group:
                continue
            _STATS["batch_forward_calls"] += 1
            _STATS["batch_forward_rows"] += len(group)
            try:
                with _torch.inference_mode():
                    if kind == "policy":
                        self._run_policy(group)
                    else:
                        self._run_value(group)
            finally:
                for request in group:
                    if request["result"] is None and request["error"] is None:
                        request["error"] = RuntimeError("batched forward failed")
                    request["event"].set()

    @staticmethod
    def _pad_boards(group):
        """Common board-side padding: returns (tokens, owners, zones, globals, padding,
        card_ids, token_counts) batched over the group."""
        token_counts = [request["tensors"][0].shape[1] for request in group]
        count, token_max = len(group), max(token_counts)
        sample_tokens = group[0]["tensors"][0]
        sample_globals = group[0]["tensors"][3]
        tokens_b = _torch.zeros(count, token_max, sample_tokens.shape[2])
        owners_b = _torch.zeros(count, token_max, dtype=_torch.int64)
        zones_b = _torch.zeros(count, token_max, dtype=_torch.int64)
        globals_b = _torch.zeros(count, sample_globals.shape[1])
        padding_b = _torch.ones(count, token_max, dtype=_torch.bool)
        identity_b = _torch.zeros(count, token_max, dtype=_torch.int64)
        for row, request in enumerate(group):
            tokens, owners, zones, globals_t, _options, identity = request["tensors"]
            token_count = token_counts[row]
            tokens_b[row, :token_count] = tokens[0]
            owners_b[row, :token_count] = owners[0]
            zones_b[row, :token_count] = zones[0]
            globals_b[row] = globals_t[0]
            padding_b[row, :token_count] = False
            identity_b[row, :token_count] = identity[0]
        return tokens_b, owners_b, zones_b, globals_b, padding_b, identity_b

    def _run_policy(self, group):
        (tokens_b, owners_b, zones_b, globals_b, padding_b,
         identity_b) = self._pad_boards(group)
        option_counts = [request["tensors"][4].shape[1] for request in group]
        count, option_max = len(group), max(option_counts)
        sample_options = group[0]["tensors"][4]
        options_b = _torch.zeros(count, option_max, sample_options.shape[2])
        option_mask_b = _torch.zeros(count, option_max, dtype=_torch.bool)
        for row, request in enumerate(group):
            options = request["tensors"][4]
            option_count = option_counts[row]
            options_b[row, :option_count] = options[0]
            option_mask_b[row, :option_count] = True
        logits, values = _MODEL.policy_value(tokens_b, owners_b, zones_b, padding_b,
                                             globals_b, options_b, option_mask_b,
                                             card_ids=identity_b)
        for row, request in enumerate(group):
            request["result"] = (logits[row, :option_counts[row]].unsqueeze(0),
                                 values[row].unsqueeze(0))

    def _run_value(self, group):
        (tokens_b, owners_b, zones_b, globals_b, padding_b,
         identity_b) = self._pad_boards(group)
        values = _MODEL.forward(tokens_b, owners_b, zones_b, padding_b, globals_b,
                                card_ids=identity_b)
        for row, request in enumerate(group):
            request["result"] = values[row].unsqueeze(0)


# Per-game state. `turn` is the last turn number seen, for new-game detection. `pending` holds
# (observation, select, move) for the deferred InFlightTracker.record -- see the module
# docstring: the engine has consumed that move by the time the next observation arrives.
_GAME = {"knowledge": None, "history": None, "in_flight": None, "turn": -1, "pending": None}


def _reset_game():
    """Fresh trackers for a new game (never mid-game: history logs are incremental, so a
    reset permanently loses the events already consumed)."""
    global _MODEL, _ACTIVE_MODEL
    _GAME["turn"] = -1
    _GAME["pending"] = None
    if _LINE_RULE is not None:
        _LINE_RULE.reset_episode()
    # New game: forget the previous opponent's reveals and serve the base model again.
    if _ROUTER is not None:
        try:
            _ROUTER.reset()
        except Exception:
            pass
        if _ACTIVE_MODEL != "base" and "base" in _MODEL_BANK:
            print(f"[router] MODEL SWITCH {_ACTIVE_MODEL} -> base (new episode)",
                  file=sys.stderr, flush=True)
            _ACTIVE_MODEL = "base"
            _MODEL = _MODEL_BANK["base"]
    if _CardKnowledge is None:
        _GAME["knowledge"] = _GAME["history"] = _GAME["in_flight"] = None
        return
    try:
        _GAME["knowledge"] = _CardKnowledge(dict(_DECK_COUNTS or {}))
        # extended=True is REQUIRED by v5: it reads mulligan / face-down rows.
        _GAME["history"] = _ActionHistory(extended=True)
        _GAME["in_flight"] = _InFlightTracker()
    except Exception:
        _GAME["knowledge"] = _GAME["history"] = _GAME["in_flight"] = None


def _flush_pending():
    """Deferred InFlightTracker.record for the PREVIOUS decision. Runs at the top of the next
    call -- the engine has accepted that move by now -- which is the same point in the sequence
    where training's loop records it. Clears the stash first so a raising record cannot replay.
    """
    pending = _GAME["pending"]
    _GAME["pending"] = None
    if pending is None or _GAME["in_flight"] is None:
        return
    observation, select, move = pending
    try:
        _GAME["in_flight"].record(observation, select, move)
    except Exception:
        pass                                       # a stale marker beats a crash


def _update_trackers(obs_dict):
    """The ONLY writer of the v5 trackers -- called exactly once per received observation."""
    turn = obs_dict["current"].get("turn") or 0
    if _GAME["knowledge"] is None or turn < _GAME["turn"]:
        _reset_game()                              # new game in this process
    _GAME["turn"] = turn
    if _GAME["knowledge"] is None:
        return
    try:
        _GAME["knowledge"].update(obs_dict)
        _GAME["history"].update(obs_dict)
    except Exception:
        pass                                       # stale trackers beat a crash


def _random_legal(select):
    count = len(select["option"])
    if count == 0:
        return []
    take = max(min(select["maxCount"], count), select["minCount"])
    return sorted(random.sample(range(count), take))


def _safe_fallback(obs_dict):
    _STATS["fallbacks"] += 1
    try:
        return _random_legal(obs_dict["select"])
    except Exception:
        return []


def _v5_move(obs_dict, select):
    """The v5 selection loop: encode the board + the option rows ONCE, then one forward per
    sub-pick over `candidate_matrix_v5` (pending options ++ STOP when legal), greedy over
    SOFTMAX probabilities with exact ties split uniformly. Mirrors train_ppo's
    `_v4_probe_move` + `_local_forward` under ENCODING == "v5", which is the driver the
    reported win rates were measured with."""
    in_flight = _GAME["in_flight"]
    encoded = _encode_observation(obs_dict, deck_counts=_DECK_COUNTS,
                                     knowledge=_GAME["knowledge"],
                                     history=_GAME["history"],
                                     in_flight=in_flight)
    base_globals = encoded["global_features"]
    # Board tensors are constant across the sub-picks of one select (no battle_select happens
    # until the answer is complete), so they are built once; only globals + rows change.
    tokens = _torch.from_numpy(encoded["token_features"]).float().unsqueeze(0)
    owners = _torch.from_numpy(encoded["owner_ids"].astype(_np.int64)).unsqueeze(0)
    zones = _torch.from_numpy(encoded["zone_ids"].astype(_np.int64)).unsqueeze(0)
    padding = _torch.zeros(1, tokens.shape[1], dtype=_torch.bool)
    card_ids = encoded.get("card_ids")
    identity = (_torch.from_numpy(card_ids.astype(_np.int64)).unsqueeze(0)
                if card_ids is not None else None)

    def choose(rows, multiselect):
        globals_vector = _global_features(base_globals, multiselect)
        globals_tensor = _torch.from_numpy(globals_vector).float().unsqueeze(0)
        option_tensor = _torch.from_numpy(rows).float().unsqueeze(0)
        option_mask = _torch.ones(1, option_tensor.shape[1], dtype=_torch.bool)
        with _torch.inference_mode():
            logits, _value = _policy_forward(tokens, owners, zones, padding,
                                             globals_tensor, option_tensor, option_mask,
                                             identity)
            # softmax, then tie-break: train_ppo._local_forward returns probabilities and
            # argmax_tiebreak's tie test is EXACT equality, so the transform is part of the
            # policy, not cosmetic.
            probabilities = _torch.softmax(logits[0], dim=-1).numpy()
        return _argmax_tiebreak(probabilities, _TIE_RNG)

    return _resolve_with(obs_dict, select, choose, in_flight, _STATS)


# ------------------------------------------------------------------ turn search ------- #
# Evaluator + greedy resolver handed to turn_search.TurnSearch. ROOT-FROZEN CONTEXT: every
# in-search encode uses the trackers as they stand at the REAL decision (_GAME), the same
# convention the training targets were generated with (train_ppo.SearchLeafEncoder). The
# in_flight passed to in-search resolves is a deepcopy so simulation never mutates real
# play state.

_SEARCH = None
_TURN_SEARCH = None


def _turn_search_module():
    """turn_search loaded BY FILE PATH (importlib), never by bare import: the Kaggle
    runtime execs main.py in a context where `import turn_search` does NOT resolve
    against the agent directory even though ppo.pt loads from it fine (observed
    on-ladder 2026-08-03, ModuleNotFoundError with the model serving normally; local
    drills never reproduce it because the local loader appends the agent dir to
    sys.path). Loading from AGENT_DIR directly removes the sys.path dependency.
    Cached; False = tried and failed, never retried."""
    global _TURN_SEARCH
    if _TURN_SEARCH is None:
        try:
            import importlib.util
            path = os.path.join(AGENT_DIR, "turn_search.py")
            spec = importlib.util.spec_from_file_location("turn_search", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules["turn_search"] = module      # one shared instance (clock state)
            if _ACTION_RULES and _combined_option_mask is not None:
                # Same mask at every surface: raw picks (resolve_with_v6), in-search
                # greedy resolves (same function), and now the search's branch points.
                module.option_mask_hook = _combined_option_mask
            if _LINE_RULE is not None and hasattr(module, "line_rule"):
                # Turn-level line rule (capability injection like greedy_resolve_pinned:
                # older turn_search copies without the attribute are never wired).
                module.line_rule = _LINE_RULE
            if hasattr(module, "knowledge_hook"):
                # Serial-level prize deduction feeds determinization: sighted cards are
                # never sampled into our prizes, exact prizes after a full deck view.
                module.knowledge_hook = lambda: _GAME["knowledge"]
            if hasattr(module, "forward_batcher"):
                # Batched leaf evaluation (capability injection; turn_search only uses
                # it when ITS opt-in flag V5_BATCHED_SEARCH=1 is set, and older
                # turn_search copies without the attribute are never wired).
                module.forward_batcher = _ForwardBatcher
            if _REVEAL_SAMPLING and hasattr(module, "REVEAL_HORIZON_CARDS"):
                # Sampled chance nodes at these cards' in-search plays (capability
                # injection: older turn_search copies without the machinery stay raw).
                module.REVEAL_HORIZON_CARDS = frozenset(_REVEAL_SAMPLING)
            if _FETCH_BRANCH and hasattr(module, "FETCH_TOPK"):
                # Fetch menus become branch points. Gate on FETCH_TOPK -- the v2
                # (dedup + cap) marker -- NOT on the FETCH_BRANCH attribute, which
                # older copies also carry: enabling the un-deduped v1 by accident
                # would branch every physical copy separately.
                module.FETCH_BRANCH = True
            _TURN_SEARCH = module
        except Exception:
            try:
                listing = sorted(os.listdir(AGENT_DIR))[:40]
            except Exception:
                listing = ["<listdir failed>"]
            print("[agent] turn_search load failed (raw policy only). dir=" + repr(listing)
                  + "\n" + traceback.format_exc(), file=sys.stderr, flush=True)
            _TURN_SEARCH = False
    return _TURN_SEARCH or None


def _search_forward(obs_dict, select, capture, pinned_first=None):
    """The _v5_move forward, but capturing (probabilities, value) for the search instead of
    feeding the census. select=None -> value-only forward (leaf evaluation).
    pinned_first: force the FIRST sub-pick to this pending-row index (fetch-branch child
    identity -- on the first sub-pick, pending rows are the options in ascending order, so
    the row index equals the option index); the policy chooses every later sub-pick."""
    encoded = _encode_observation(obs_dict, deck_counts=_DECK_COUNTS,
                                     knowledge=_GAME["knowledge"],
                                     history=_GAME["history"],
                                     in_flight=_GAME["in_flight"])
    tokens = _torch.from_numpy(encoded["token_features"]).float().unsqueeze(0)
    owners = _torch.from_numpy(encoded["owner_ids"].astype(_np.int64)).unsqueeze(0)
    zones = _torch.from_numpy(encoded["zone_ids"].astype(_np.int64)).unsqueeze(0)
    padding = _torch.zeros(1, tokens.shape[1], dtype=_torch.bool)
    card_ids = encoded.get("card_ids")
    identity = (_torch.from_numpy(card_ids.astype(_np.int64)).unsqueeze(0)
                if card_ids is not None else None)
    if select is None:
        # Value-only forward (no pending select): the pick-scalar extras still need a
        # multiselect-shaped object -- a bool here crashed every turn-end leaf eval
        # until the path-restoration drill caught it (2026-08-03).
        null_pick = type("NullPick", (), {"chosen": (), "min_take": 0, "max_take": 0})()
        globals_tensor = _torch.from_numpy(
            _global_features(encoded["global_features"],
                                null_pick)).float().unsqueeze(0)
        with _torch.inference_mode():
            value = _value_forward(tokens, owners, zones, padding, globals_tensor,
                                   identity)
        capture["value"] = float(value[0])
        return None

    pin_state = {"pending": pinned_first}

    def choose(rows, multiselect):
        globals_tensor = _torch.from_numpy(
            _global_features(encoded["global_features"], multiselect)).float().unsqueeze(0)
        option_tensor = _torch.from_numpy(rows).float().unsqueeze(0)
        option_mask = _torch.ones(1, option_tensor.shape[1], dtype=_torch.bool)
        with _torch.inference_mode():
            logits, value = _policy_forward(tokens, owners, zones, padding,
                                            globals_tensor, option_tensor, option_mask,
                                            identity)
            probabilities = _torch.softmax(logits[0], dim=-1).numpy()
        capture["probabilities"] = probabilities
        capture["value"] = float(value[0])
        if pin_state["pending"] is not None:
            forced = pin_state["pending"]
            pin_state["pending"] = None
            if forced < len(rows):
                return forced
        return _argmax_tiebreak(probabilities, _TIE_RNG)

    return _resolve_with(obs_dict, select, choose,
                            copy.deepcopy(_GAME["in_flight"]), Counter())


def _forced_evolution_substitute(obs_dict, select, move):
    """force_evolve_before_shuffle_draw (owner rule 2026-08-14, owner-specified
    mechanism): menus are never masked -- when the move the model just chose PLAYS a
    shuffle-draw card while a forcible dragapult-line evolution is on the same menu
    (conditions + exceptions live in state_encoder.forced_evolution_options), send the
    evolution (or the evolving Drakloak's unused Recon Directive) instead; the model
    re-decides from the post-evolve state on the next call. With several forcible
    options the MODEL's own scores pick which one (code picks the class, the model
    picks the instance). Never raises; any failure returns the original move."""
    try:
        if len(move) != 1 or move[0] not in _SHUFFLE_PLAY_INDICES(obs_dict, select):
            return move
        candidates = _FORCED_EVOLUTION_OPTIONS(obs_dict, select)
        if not candidates:
            return move
        chosen = candidates[0]
        if len(candidates) > 1:
            try:
                capture = {}
                _search_forward(obs_dict, select, capture)
                probabilities = capture.get("probabilities")
                count = len(select.get("option") or [])
                if probabilities is not None and len(probabilities) < count \
                        and _combined_option_mask is not None:
                    # Masked-select scatter: probabilities arrive in PENDING order
                    # (one entry per allowed option, ascending), same as
                    # turn_search._expand_priors.
                    allowed = _combined_option_mask(obs_dict, select)
                    if allowed is not None:
                        scattered = [0.0] * count
                        for position, original in enumerate(sorted(allowed)):
                            if position < len(probabilities):
                                scattered[original] = float(probabilities[position])
                        probabilities = scattered
                if probabilities is not None:
                    chosen = max(candidates,
                                 key=lambda index: probabilities[index]
                                 if index < len(probabilities) else 0.0)
            except Exception:
                chosen = candidates[0]
        _STATS["evolve_forced"] += 1
        return [chosen]
    except Exception:
        return move


def _adrena_first_substitute(obs_dict, select, move):
    """damage_solver (owner rule 2026-08-15): a chosen attack or END -- or a
    retreat that would strip the dark off an active Munkidori -- while an unused
    Adrena-Brain is available becomes the ability use first; the model re-decides from the
    post-move state on the next call (the ability option is gone after the real
    use, so this cannot loop). Runs BEFORE the Battle Cage substitute so a forced
    use under Battle Cage still becomes the stadium bump. Never raises."""
    try:
        index = _ADRENA_FIRST(obs_dict, select, move)
        if index is not None:
            _STATS["adrena_forced"] += 1
            return [int(index)]
    except Exception:
        pass
    return move


def _battle_cage_substitute(obs_dict, select, move):
    """stadium_discipline's Battle Cage clause (owner rule 2026-08-15): when the
    chosen move uses Adrena-Brain or attacks with Phantom Dive while the opponent's
    Battle Cage would blank the placed counters, and one of our stadiums is playable
    on the same menu, send the stadium play instead (bump first); the model
    re-decides from the post-bump state on the next call. With several stadiums the
    MODEL's scores pick which. Never raises; any failure returns the original move."""
    try:
        if len(move) != 1 or move[0] not in _BATTLE_CAGE_BLOCKED(obs_dict, select):
            return move
        candidates = _STADIUM_BUMP_CANDIDATES(obs_dict, select)
        if not candidates:
            return move
        chosen = candidates[0]
        if len(candidates) > 1:
            try:
                capture = {}
                _search_forward(obs_dict, select, capture)
                probabilities = capture.get("probabilities")
                count = len(select.get("option") or [])
                if probabilities is not None and len(probabilities) < count \
                        and _combined_option_mask is not None:
                    allowed = _combined_option_mask(obs_dict, select)
                    if allowed is not None:
                        scattered = [0.0] * count
                        for position, original in enumerate(sorted(allowed)):
                            if position < len(probabilities):
                                scattered[original] = float(probabilities[position])
                        probabilities = scattered
                if probabilities is not None:
                    chosen = max(candidates,
                                 key=lambda index: probabilities[index]
                                 if index < len(probabilities) else 0.0)
            except Exception:
                chosen = candidates[0]
        _STATS["stadium_bump_forced"] += 1
        return [chosen]
    except Exception:
        return move


def _search_evaluate(obs_dict, select):
    """turn_search contract: (probabilities_or_None, value), value seat-relative."""
    capture = {}
    _search_forward(obs_dict, select, capture)
    if "value" not in capture:
        # The select resolved entirely by forced-answer proof -- zero model forwards
        # ran, so no value was captured. Take it from the value-only forward instead.
        _search_forward(obs_dict, None, capture)
    return capture.get("probabilities"), capture["value"]


def _search_greedy(obs_dict, select):
    """turn_search contract: resolve a non-branch select with the policy (model decides)."""
    capture = {}
    return _search_forward(obs_dict, select, capture)


def _search_greedy_pinned(obs_dict, select, first_option_index):
    """Fetch-branch (multi-target searches): resolve the select with the FIRST pick forced
    to `first_option_index`; the policy chooses the remaining picks conditioned on it."""
    capture = {}
    return _search_forward(obs_dict, select, capture, pinned_first=first_option_index)


def _get_search():
    global _SEARCH
    if _SEARCH is None and _MODEL is not None:
        try:
            turn_search = _turn_search_module()
            if turn_search is None:
                _SEARCH = False
                return None
            _SEARCH = turn_search.TurnSearch(_search_evaluate, _search_greedy,
                                             _DECK_COUNTS, _STATS)
            # Optional capability injection: multi-target fetch branching (FETCH_BRANCH)
            # pins the first pick of a deck-search multi-pick; older turn_search copies
            # simply never read the attribute.
            _SEARCH.greedy_resolve_pinned = _search_greedy_pinned
        except Exception:
            print("[agent] turn_search unavailable:\n" + traceback.format_exc(),
                  file=sys.stderr, flush=True)
            _SEARCH = False                        # do not retry every decision
    return _SEARCH or None


def _search_move(obs_dict, select):
    """The searched answer for a searchable select, or None -> raw policy. Never
    raises. Searchable: mandatory single picks (min==max==1, >= 2 options) and --
    when the envelope enables fetch branching -- optional deck/discard fetch menus,
    where the search may answer DECLINE (the empty move)."""
    if not USE_SEARCH or _MODEL is None:
        return None
    try:
        turn_search = _turn_search_module()
        if turn_search is None:
            return None
        options = select.get("option") or []
        single_pick = (select.get("minCount") == 1 and select.get("maxCount") == 1
                       and len(options) >= 2)
        fetch_root = (not single_pick and _FETCH_BRANCH
                      and getattr(turn_search, "FETCH_BRANCH", False)
                      and hasattr(turn_search, "_fetch_menu_shape")
                      and turn_search._fetch_menu_shape(select, options))
        if not ((single_pick or fetch_root)
                and obs_dict.get("search_begin_input")):
            return None
        budget = turn_search.search_budget(obs_dict)   # bank curve: sims taper as it drains
        if budget is None:
            return None
        simulations_budget, per_move_deadline = budget
        search = _get_search()
        if search is None:
            return None
        started = time.time()
        chosen = search.run(obs_dict, select, started + per_move_deadline,
                            simulations_budget=simulations_budget)
        turn_search._clock["search_seconds"] += time.time() - started
        if chosen is None:
            return None
        _STATS["searched"] += 1
        _STATS[f"search_tier_{simulations_budget}"] += 1
        if fetch_root:
            _STATS["search_root_fetch"] += 1
            if int(chosen) == len(options):
                return []                      # the searched DECLINE
        return [int(chosen)]
    except Exception:
        _STATS["search_errors"] += 1
        if _STATS["search_errors"] == 1:
            print("[agent] search error (raw policy takes over):\n"
                  + traceback.format_exc(), file=sys.stderr, flush=True)
        return None


def agent(obs_dict: dict) -> list:
    try:
        _ensure_agent_path()                   # the loader restores sys.path post-load
        current = obs_dict.get("current")
        if current is None:
            _reset_game()                          # deck selection (or a finished episode)
            try:
                turn_search = _turn_search_module()
                if turn_search is not None:
                    turn_search.note_episode_start()   # the search's episode bank clock
            except Exception:
                pass
            return list(DECK)
        # The engine has consumed the previous move -- record it BEFORE the trackers advance,
        # which is where training's loop records it.
        _flush_pending()
        _update_trackers(obs_dict)                 # exactly once per observation
        _route_model(obs_dict)                     # archetype router: re-pick the model
        select = obs_dict.get("select")
        if select is None:
            if _GAME["in_flight"] is not None:
                _GAME["in_flight"].reset()         # no menu = no chain in progress
            return []                              # advance the engine
        if _GAME["in_flight"] is not None:
            try:
                _GAME["in_flight"].observe(obs_dict, select)   # BEFORE encoding
            except Exception:
                pass
        if _LINE_RULE is not None:
            # Context-only pass BEFORE this decision's search: real-observation signals
            # the line rule reads at its root (turn change, the opponent-Budew scan for
            # require_play_fetched_card's dragapult exception) must land ahead of the
            # root seeding. chosen=None folds no labels; the after-move call below is
            # the one that records the decision.
            _LINE_RULE.observe_real(obs_dict, select, None)
        move = None
        if _forced_answer is not None:
            # The ONE sanctioned non-model answer: exactly one legal answer exists. Checked
            # before encoding (as training does) so the cheap case stays cheap; resolve_with_v5
            # re-checks it and will not double-count.
            forced, reason = _forced_answer(select)
            if forced is not None:
                _STATS["forced"] += 1
                _STATS["forced_" + reason] += 1
                move = forced
        if move is None:
            if _MODEL is None:
                move = _safe_fallback(obs_dict)
            else:
                try:
                    move = _search_move(obs_dict, select)   # 48x1 our-turn search or None
                    if move is None:
                        move = _v5_move(obs_dict, select)
                except Exception:
                    if _STATS["fallbacks"] == 0:
                        print("[agent] policy error:\n" + traceback.format_exc(),
                              file=sys.stderr, flush=True)
                    move = _safe_fallback(obs_dict)
                if _STATS["selects"] == 1 or _STATS["selects"] % 50 == 0:
                    print(f"[agent] model selects={_STATS['selects']} "
                          f"forwards={_STATS['forwards']} picks={_STATS['picks']} "
                          f"forced={_STATS['forced']} fallbacks={_STATS['fallbacks']}",
                          file=sys.stderr, flush=True)
        if move and select is not None and _FORCED_EVOLUTION_OPTIONS is not None:
            # force_evolve_before_shuffle_draw intercepts at SUBMISSION time: a chosen
            # shuffle-draw play becomes the forced evolution (or Recon) and the model
            # re-decides from the post-evolve state on the next call.
            move = _forced_evolution_substitute(obs_dict, select, move)
        if move and select is not None and _ADRENA_FIRST is not None:
            # damage_solver: attack (or dark-stripping retreat) with an unused
            # Adrena-Brain available -> use the ability first.
            move = _adrena_first_substitute(obs_dict, select, move)
        if move and select is not None and _BATTLE_CAGE_BLOCKED is not None:
            # stadium_discipline's Battle Cage clause: a chosen Adrena-Brain or
            # Phantom Dive under Battle Cage becomes the stadium bump instead.
            move = _battle_cage_substitute(obs_dict, select, move)
        # Stash for the deferred record: training records the move it actually sent, whatever
        # produced it (model, forced-by-proof or fallback).
        _GAME["pending"] = (obs_dict, select, move)
        if _LINE_RULE is not None:
            # Real decisions feed the line rule's turn state, so the NEXT search's root
            # inherits any outstanding obligation (observe_real never raises).
            _LINE_RULE.observe_real(obs_dict, select, move)
        return move
    except Exception:
        return _safe_fallback(obs_dict)
