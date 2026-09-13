"""ENGINE-computed result of a damage-counter placement, for the v6 chain column.

Why this exists (2026-08-05 audit). The v6 column answered "what HP will this target have
if I place here?" with hand arithmetic -- `hp - 10` -- which embeds two assumptions:
one counter is worth exactly 10, and the counter LANDS. The second is false often: Crustle's
Mysterious Rock Inn ({ex} immunity), Team Rocket's Articuno's Repelling Veil (placement is
an attack EFFECT), benched Tera and others absorb it entirely. Measured against the engine
on Phantom Dive placements, the hand formula is wrong on ~10% of options (12 of 123): the
counter is fully absorbed and HP does not move, while `hp - 10` claims progress.

(The 10-per-counter part of the assumption is CORRECT for Phantom Dive -- engine deltas are
exactly {0, 10}, n=123. An earlier draft of this module claimed 20-damage placements and a
39% error rate; both were artefacts of lumping Munkidori -- which MOVES several counters at
once -- in with Phantom Dive. Always attribute measurements per EFFECT CARD.)

So stop computing it. `search_begin` forks the live state and `search_step` applies one
candidate pick; the resulting observation carries the engine's own post-placement HP, with
every immunity, prevention and damage modifier already resolved. No card knowledge, no
edge-case table to maintain -- whatever the engine does is what we report.

Measured cost (200 games, build_v23): search_begin 0.373 ms per placement select,
search_step + read 0.164 ms per option. Placement selects are 0.39% of decisions, so at
512 games/iter this is ~0.3 s against a ~100 s iteration.

The caller NEVER falls back to arithmetic. Where this returns a value the column carries it;
where it does not, the column carries the engine's CURRENT HP for the target. An unknown
result is reported as the present state, never as an invented one.

SCOPE / when this returns None:
  * the select is not single-pick. `search_step` answers a WHOLE select, so probing one
    option at a time is only meaningful when minCount == maxCount == 1. Phantom Dive's
    chain is six single-pick selects, which is the case that matters.
  * no determinization could be built. Measured 2026-08-05, this is ~never: 100% in
    self-play (the worker SAMPLED the opponent deck and passes the truth) AND 100% at
    inference with DeckRecognizer.consistent_deck (121/121). An earlier measurement here
    claimed 54% at inference; that was a harness bug -- OpponentTracker was fed BOTH seats'
    observations, so "revealed" held 87-95 cards including our own, which no legal deck can
    contain. One tracker per seat, fed only that seat's observations.
  * any engine error. Never let a probe kill a game.

State-pool safety: every stepped state is released, and the root is released before
returning. The old ~256-512-live-state crash was fixed in the June-30 engine release and
re-verified on build_v23 (889,856 live states, zero crashes), so this is hygiene, not a
workaround.

IF YOU EVER ADD A "JUST FILL IT WITH ANYTHING LEGAL" DECK FALLBACK, SCOPE IT TO HERE.
This probe is 1-ply and reads only the visible board, so ANY legal deck consistent with the
reveals answers it exactly -- padding the unknowns with basic energy (exempt from the
4-copy limit) would be perfectly sound for THIS use.

It is NOT sound for MCTS determinization. That path simulates the opponent PLAYING the
deck, so an energy-padded world produces nonsense rollouts and a mis-scored tree; it needs
a PLAUSIBLE deck, which is what DeckRecognizer.consistent_deck exists to give it. The two
callers have genuinely different requirements from the same-looking input, so never
"improve" consistent_deck by relaxing it -- add a separate filler here instead.

(No such fallback is needed today: consistent_deck measured 100% at inference and the
self-play worker passes the real deck. A `probe_deck` energy filler was written on
2026-08-05 and REMOVED once the 46% failure that motivated it turned out to be a harness
bug -- OpponentTracker fed both seats' observations. Recorded so the idea is not
re-invented without the scoping caveat above.)
"""

import numpy as np

from src.game.encode_details import _entity_at

_AREA_ACTIVE, _AREA_BENCH = 4, 5
_DAMAGE_CONTEXTS = frozenset((13, 14, 15))

__all__ = ["engine_projected_hp"]


def _hp_by_serial(observation):
    """serial -> hp for every in-play Pokemon, from EITHER observation shape: the dict the
    agent receives, or the cg.api.Observation dataclass search_step returns."""
    is_dict = isinstance(observation, dict)
    current = observation["current"] if is_dict else observation.current
    players = current["players"] if is_dict else current.players
    out = {}
    for player in players or ():
        if not player:
            continue
        if isinstance(player, dict):
            slots = (player.get("active") or []) + (player.get("bench") or [])
        else:
            slots = list(getattr(player, "active", None) or []) + \
                    list(getattr(player, "bench", None) or [])
        for slot in slots:
            if slot is None:
                continue
            if isinstance(slot, dict):
                out[slot.get("serial")] = slot.get("hp")
            else:
                out[getattr(slot, "serial", None)] = getattr(slot, "hp", None)
    return out


def engine_projected_hp(observation, select, my_deck, opponent_deck, rng=None):
    """-> {option index: HP that option's target ends up with}, or None if unavailable.

    Only options that point at an in-play Pokemon appear in the mapping; the caller keeps
    its own value for the rest. Pass the opponent's REAL deck where you have it (self-play
    does); DeckRecognizer.consistent_deck is the right source where you don't, and measured
    100% at both.
    """
    if select is None or not select.get("option"):
        return None
    if select.get("minCount") != 1 or select.get("maxCount") != 1:
        return None                      # search_step answers a whole select; see SCOPE

    from cg import api                                     # local: keep cg off the import
    from src.search.determinize import build_determinization   # path for bundles

    try:
        args = build_determinization(observation, list(my_deck), list(opponent_deck),
                                     rng=rng)
        root = api.search_begin(api.to_observation_class(observation), *args)
    except Exception:
        return None
    root_id = getattr(root, "searchId", None)
    if not root_id:
        return None

    me_index = observation["current"]["yourIndex"]
    projected = {}
    try:
        for position, option in enumerate(select["option"]):
            if option.get("area") not in (_AREA_ACTIVE, _AREA_BENCH):
                continue
            owner = option.get("playerIndex")
            owner = me_index if owner is None else owner
            _card, pokemon = _entity_at(observation, option.get("area"),
                                        option.get("index"), owner)
            if pokemon is None:
                continue
            serial = pokemon.get("serial")
            try:
                stepped = api.search_step(root_id, [position])
            except Exception:
                continue                 # one bad option must not lose the whole select
            try:
                # The engine's own HP for this target after taking this option. Whatever
                # the engine resolved -- one counter, an absorbed counter, or the rest of
                # the effect if this pick ends the chain -- this is the board that results
                # from choosing it. Nothing is computed or assumed here.
                #
                # ABSENT means the placement KILLED it: a KO'd Pokemon leaves the board, so
                # its serial is gone from the stepped observation. That is the single most
                # decision-relevant outcome, and reporting "unknown" would let the caller
                # fall back to the target's PRE-placement HP -- reading a lethal placement
                # as "still alive". Record it as 0, i.e. dead.
                hp = _hp_by_serial(stepped.observation).get(serial)
                projected[position] = 0 if hp is None else hp
            finally:
                try:
                    api.search_release(stepped.searchId)
                except Exception:
                    pass
    finally:
        try:
            api.search_release(root_id)
        except Exception:
            pass
    return projected or None
