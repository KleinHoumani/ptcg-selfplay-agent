"""Rich engine-state encoder -- the SECOND, separate encoder next to src/game/encode.py.

encode.py encodes only the official observation API and stays frozen so existing checkpoints
keep working. This module encodes the engine's hidden effect state (the fields the observation
JSON omits: ramp-attack damage memory, shields, cost modifiers, once-per-turn ability usage,
item/supporter locks, exact poison counters), decoded from observation["search_begin_input"]
via the source-built engine's DumpState export.

Output contract (row-aligned with encode.py):
    encode_rich_observation(observation, **same zone kwargs as encode_observation) ->
        {"token_rich": [T, RICH_CARD_DIM], "global_rich": [RICH_GLOBAL_DIM]}
    where row i annotates token i of encode_observation(observation, **same kwargs) -- pass the
    SAME opponent_belief/belief_top_k/my_unseen/include_discard/capability_tokens or the rows
    will not line up. In-play card rows carry that card's effect state, capability rows carry
    their host's, and rows for zones without per-instance state (hand, discard, unseen library,
    belief) are zeros. Rich-aware models concatenate along the feature axis (or process
    separately); old models simply never call this.

Availability: needs the source-built engine (CG_DLL=engine_src/build/cg.dll or a bundled build
with the DumpState export). With the official binary, or no cg module at all (the Kaggle policy
runtime), every value is zeros and a single warning is printed -- shapes stay correct, nothing
crashes. The traversal below mirrors encode.encode_game's token-emission order; if that order
ever changes, update this walk in lockstep."""

import os
import sys
from collections import Counter

import numpy as np

from src.cards import get_card

RICH_CARD_DIM = 16
RICH_PLAYER_DIM = 7
RICH_GLOBAL_DIM = 2 * RICH_PLAYER_DIM

_DAMAGE_SCALE = 100.0
_RETREAT_SCALE = 4.0
_ATTACK_COST_SCALE = 3.0

_ZERO_CARD = np.zeros(RICH_CARD_DIM, dtype=np.float32)
_ZERO_CARD.flags.writeable = False

_PROTECTION_KEYS = (
    "noDamageAndEffectEnemyExAttackNextEnemyTurn", "noDamageLessEqualAttackNextEnemyTurn",
    "noDamageAndEffectAttackNextEnemyTurn", "noDamageAndEffectEnemyAttackNextEnemyTurn",
    "noDamageAttackNextEnemyTurn", "noDamageBasicAttackNextEnemyTurn",
    "noDamageBasicColorAttackNextEnemyTurn", "noDamageAbilityAttackNextEnemyTurn",
    "noDamageEnemyAbilityPokemonAttack", "noDamageEnemyExAttack", "noDamageEnemyBasicExAttack",
    "noDamageAndEffectEnemyTerastalAttack", "noDamageAndEffectEnemySpecialEnergyAttack",
    "noDamageEnemyAttack", "noDamageGreaterEqual")

_dump_state = 0                       # 0 = unprobed, None = unavailable, else cg.api.dump_state


def rich_state(observation, extras_out=None):
    """(serial -> card effect dict, playerIndex -> restriction dict) for this observation, or
    (None, None) when the engine's DumpState export is unavailable.

    `extras_out`: an optional dict that receives the TOP-LEVEL dump fields the pair drops --
    today `knockouts` (the build_v25 cumulative KO record; absent-when-empty in the dump,
    always a tuple key here). Keyword-only-in-spirit and additive: every existing caller and
    the feature output are byte-unchanged."""
    global _dump_state
    blob = observation.get("search_begin_input")
    if blob is None or _dump_state is None:
        return None, None
    if os.environ.get("RICH_DISABLE") == "1":
        # Measurement hook (2026-08-11): force the Kaggle deploy condition (zeroed rich
        # block) locally, keeping the seeded v25 engine so probes stay protocol-paired.
        # Probe-only -- never set in training (it would also blank knockout labels).
        return None, None
    try:
        if _dump_state == 0:
            from cg import api as cg_api                     # lazy: cg is absent on Kaggle
            _dump_state = cg_api.dump_state
        dump = _dump_state(blob)
    except Exception as error:
        _dump_state = None
        print(f"[encode_rich] DumpState unavailable ({error!r}) -> rich features are zeros",
              file=sys.stderr, flush=True)
        return None, None
    if extras_out is not None:
        # None when the key is ABSENT -- which means either a pre-v25 engine (no export)
        # or a v25 engine with zero KOs so far (the dump omits empty lists). The consumer
        # must only treat key-PRESENT as an authoritative scan: that is sound in both
        # worlds, because before the first KO record there are no prize events for the
        # fallback inference to misread either.
        extras_out["knockouts"] = dump.get("knockouts")
    return ({card["serial"]: card for card in dump["cards"]},
            {player["playerIndex"]: player for player in dump["players"]})


def card_block(rich):
    """16 features for one card's engine effect state (zeros when it carries none). DumpState
    emits only non-zero fields, so every read defaults to 0."""
    if not rich:
        return _ZERO_CARD
    this_turn = rich.get("thisTurn") or {}
    next_turn = rich.get("nextTurn") or {}
    damage_out = (this_turn.get("damageChange", 0) + this_turn.get("damageChangeActive", 0)
                  + this_turn.get("damageChangeMyAttack", 0) + rich.get("damageChangeContinual", 0)
                  + rich.get("damageChangeActiveContinual", 0) + rich.get("damageChangeThisTurn", 0)
                  + rich.get("damageChangeExThisTurn", 0))
    damage_in = (rich.get("takeDamageChange", 0) + rich.get("takeEnemyAttackDamageChange", 0)
                 + rich.get("takeDamageChangeThisTurnEnemy", 0)
                 + rich.get("takeDamageChangeNextEnemyTurn", 0))
    retreat_mod = this_turn.get("retreatCostChange", 0) + rich.get("retreatCostChange", 0)
    attack_cost_mod = (this_turn.get("attackCostChange", 0) + rich.get("attackCostChangeColorless", 0)
                       - rich.get("attackCostDown", 0) - rich.get("attackCostDownColorlessOwnAttack", 0))
    return np.array([
        rich.get("takeAttackDamagePreTurn", 0) / _DAMAGE_SCALE,         # ramp-attack memory
        rich.get("takeAttackDamageThisTurn", 0) / _DAMAGE_SCALE,
        rich.get("hpChange", 0) / _DAMAGE_SCALE,                        # e.g. Cape max-hp buff
        damage_out / _DAMAGE_SCALE,                                     # outgoing damage modifiers
        damage_in / _DAMAGE_SCALE,                                      # incoming (negative = shield)
        retreat_mod / _RETREAT_SCALE,
        attack_cost_mod / _ATTACK_COST_SCALE,
        float(bool(this_turn.get("cannotAttack") or rich.get("cannotAttack")
                   or this_turn.get("cannotAttackLessEqualEnergy2"))),
        float(bool(next_turn.get("cannotAttack") or next_turn.get("cannotUseAttackId")
                   or next_turn.get("cannotUseAttackId2"))),
        float(bool(this_turn.get("cannotRetreat") or rich.get("cannotRetreat"))),
        float(bool(this_turn.get("cannotUseAttackId") or this_turn.get("cannotUseAttackId2")
                   or rich.get("cannotUseAttackIdNonActive"))),
        float(any(rich.get(key) for key in _PROTECTION_KEYS)),          # shielded next enemy turn
        float(bool(rich.get("noAbility"))),
        float(bool(rich.get("abilityUsed"))),                           # once-per-turn spent
        float(bool(rich.get("canUsePreEvolutionAttack"))),
        float(bool(rich.get("evolved"))),
    ], dtype=np.float32)


def player_block(locks):
    """7 restriction features for one player (zeros when unavailable)."""
    if not locks:
        return np.zeros(RICH_PLAYER_DIM, dtype=np.float32)
    return np.array([
        float(bool(locks.get("cannotPlayItemThisTurn") or locks.get("cannotPlayItem"))),
        float(bool(locks.get("cannotPlaySupporterThisTurn"))),
        float(bool(locks.get("cannotPlayStadiumThisTurn") or locks.get("cannotPlayStadium"))),
        float(bool(locks.get("cannotPlaySpecialEnergyThisTurn"))),
        float(bool(locks.get("cannotEvolveThisTurn"))),
        float(bool(locks.get("cannotPlayItemNextTurn") or locks.get("cannotPlaySupporterNextTurn")
                   or locks.get("cannotPlayStadiumNextTurn")
                   or locks.get("cannotPlaySpecialEnergyNextTurn")
                   or locks.get("cannotEvolveNextTurn"))),
        min(locks.get("poisonDamageCounter", 0), 4) / 4.0,              # exact poison stack
    ], dtype=np.float32)


def encode_rich_observation(observation, opponent_belief=None, belief_top_k=None,
                            my_unseen=None, include_discard=False, capability_tokens=False,
                            rich_override=None):
    """Row-aligned rich annotations for encode_observation(observation, <same kwargs>) -- see
    the module docstring for the contract. Returns {"token_rich": [T, 16], "global_rich": [14]}.

    rich_override: a (rich_cards, rich_players) pair to use INSTEAD of decoding this
      observation's own blob -- the search-side ROOT FREEZE. Forward-search states carry no
      state blob at all (probed: search_begin_input is None at the root and every depth), so a
      search-side encoder would otherwise see an all-zero rich block where training saw live
      effect state. Passing the real decision's rich_state() here re-binds that effect state
      by SERIAL to whatever sits on the search node's board (board serials persist battle ->
      search root -> stepped states, probe-verified); cards first played inside the simulation
      simply get zeros, which is approximately right for fresh cards. None = decode this
      observation, i.e. exactly the original behaviour."""
    current = observation["current"]
    me_index = current["yourIndex"]
    me, opponent = current["players"][me_index], current["players"][1 - me_index]
    rich_cards, rich_players = (rich_state(observation) if rich_override is None
                                else rich_override)
    rich_cards = rich_cards or {}
    rich_players = rich_players or {}

    def in_play(player):                                     # mirrors encode.py's None filtering
        return ([p for p in (player["active"] or []) if p is not None]
                + [p for p in (player["bench"] or []) if p is not None])

    rows = []
    # 1. board tokens: me active+bench, then opponent active+bench (encode_game's order)
    for player in (me, opponent):
        for pokemon in in_play(player):
            rows.append(card_block(rich_cards.get(pokemon["serial"])))
    # 2. my hand: no per-instance effect state
    rows.extend(_ZERO_CARD for _ in (me["hand"] or []))
    # 3. optional zones without per-instance state -> zero rows, counts must match encode.py
    if include_discard:
        for player in (me, opponent):
            rows.extend(_ZERO_CARD for _ in Counter(card["id"] for card in player["discard"]))
    if my_unseen:
        rows.extend(_ZERO_CARD for _ in my_unseen)
    if opponent_belief:
        count = len(opponent_belief)
        if belief_top_k is not None:
            count = min(count, belief_top_k)
        rows.extend(_ZERO_CARD for _ in range(count))
    # 4. capability tokens inherit their host's effect state (attacks first, then abilities)
    if capability_tokens:
        for player in (me, opponent):
            for pokemon in in_play(player):
                host = card_block(rich_cards.get(pokemon["serial"]))
                card = get_card(pokemon["id"])
                rows.extend(host for _ in card["attacks"])
                rows.extend(host for _ in card["skills"])

    token_rich = (np.stack(rows).astype(np.float32) if rows
                  else np.zeros((0, RICH_CARD_DIM), dtype=np.float32))
    global_rich = np.concatenate([player_block(rich_players.get(me_index)),
                                  player_block(rich_players.get(1 - me_index))])
    return {"token_rich": token_rich, "global_rich": global_rich}
