"""Full-information, entity-granular encoding (the 2026-07-10 directive: the model must see
every knowable detail of the game state, with one token per independently-addressable entity).

Composes the FROZEN base encoder -- encode.py is not touched, so every existing checkpoint
keeps loading and producing byte-identical encodings. This module:

  1. turns ON every detail the base encoder already supports but the trained models never
     used: both discard piles, my unseen library (deck+prizes multiset), the opponent-deck
     belief, and per-attack / per-ability capability tokens for all in-play Pokemon;
  2. adds the entities the base path DROPS entirely, as new token types (same 340-wide card
     token, new zone ids): the Stadium in play (identity, not just a flag), each attached
     Pokemon Tool, each attached Energy CARD (identity -- special vs basic), and each card in
     an evolution stack (pre-evolutions, for devolution effects). Attached-entity tokens carry
     their HOST's live state in the dynamic block (same binding trick capability tokens use).

Only models sized for NUM_ZONES_FULL understand this output -- build them with
`"num_zones": <NUM_ZONES_FULL>` in the model config. Old models keep NUM_ZONES and the old
encoders; nothing changes for them.

What remains invisible, and why:
  - opponent hand / exact deck order / prize identities: genuinely hidden information; the
    belief tokens + counts in the global vector are the honest ceiling.
  - the engine's hidden effect state: separate rich encoder (src/game/encode_rich.py), by
    the compatibility directive.
"""

import numpy as np

from src.game.encode import (TOKEN_FEATURE_DIM, encode_card_token, encode_observation,
                             unseen_my_counts, _light_pokemon)
from src.game.encode_rich import (RICH_CARD_DIM, RICH_GLOBAL_DIM, _ZERO_CARD as _ZERO_RICH,
                                  card_block, encode_rich_observation, rich_state)
from src.game.features import FEATURE_DIM, game_features
from src.solvers.prize_map import LEDGER_DIM, ledger_features
from src.segments import NUM_ZONES, OWNER_ME, OWNER_OPPONENT, OWNER_NEUTRAL

# New entity zones, appended AFTER the frozen base ids (segments.py stays untouched so the
# old models' embedding tables keep their shapes).
ZONE_STADIUM = NUM_ZONES + 0        # the Stadium card in play (owner = who played it)
ZONE_TOOL = NUM_ZONES + 1           # a Pokemon Tool attached to an in-play Pokemon
ZONE_ENERGY_CARD = NUM_ZONES + 2    # an Energy CARD attached to an in-play Pokemon
ZONE_PRE_EVOLUTION = NUM_ZONES + 3  # a card underneath an evolved in-play Pokemon
ZONE_LOOKING = NUM_ZONES + 4        # a card currently being looked at (mid-effect reveal)
ZONE_PRIZE = NUM_ZONES + 5          # MY prize belief: expected prized copies per card id
                                    # (exact after a full-deck view; see CardKnowledge)
ZONE_OPPONENT_SEEN = NUM_ZONES + 6  # a card of THEIRS we have seen that is hidden again:
                                    # certainly in their hand/deck, certainly NOT prized
ZONE_DECK_TOP = NUM_ZONES + 7       # a card we KNOW is coming up in MY deck (known order;
                                    # belief scalar encodes draw proximity)
ZONE_EFFECT_SOURCE = NUM_ZONES + 8  # the card whose pending effect is asking (select.effect)
ZONE_CONTEXT_CARD = NUM_ZONES + 9   # the card being acted on (select.contextCard)
NUM_ZONES_FULL = NUM_ZONES + 10

# v3 zones (include_v3_zones=True only -- see encode_details.py). NUM_ZONES_FULL + 0 is already
# taken by encode_history's ZONE_HISTORY, so these start one past it; NUM_ZONES_V3 = +3.
ZONE_FACEDOWN = NUM_ZONES_FULL + 1   # a Pokemon in play we cannot identify (setup, face-down)
ZONE_DECK_VIEW = NUM_ZONES_FULL + 2  # a card in the select.deck view currently being searched

# Decision-context extension (include_select_context=True; for POLICY models and any value
# model retrained to evaluate mid-effect states): 6 scalars + SelectType one-hot (11).
# The queue-state fields (remainDamageCounter counting down per placement pick,
# remainEnergyCost) are what tell a policy "this is my last counter".
SELECT_CONTEXT_DIM = 6 + 11
GLOBAL_FEATURE_DIM_FULL_SELECT = None   # set below, after GLOBAL_FEATURE_DIM_FULL

# Extra global scalars the frozen 22-dim global vector lacks (appended, so full models use
# GLOBAL_FEATURE_DIM_FULL as their global_feature_dim): bench capacity is a live rule variable
# (Area Zero -> 8), and going first changes the whole game's tempo parity.
GLOBAL_FEATURE_DIM_FULL = 22 + 5    # + my benchMax/8, opp benchMax/8, I went first,
                                    #   turnActionCount/10 (actions taken this turn),
                                    #   my prize certainty (1.0 = ZONE_PRIZE tokens are
                                    #   exact deductions, lower = hypergeometric guess)

# With include_rich=True, every token gains the 16 engine-effect features and the global
# vector gains both players' 7 restriction features (item/supporter/stadium/special-energy/
# evolve locks, next-turn locks, exact poison) -- decoded from the source-built engine's
# DumpState. On observations without the state blob (forward-SEARCH states, or the official
# binary) the rich block is zeros: shapes stay correct, information degrades gracefully.
TOKEN_FEATURE_DIM_FULL_RICH = TOKEN_FEATURE_DIM + RICH_CARD_DIM         # 340 + 16
GLOBAL_FEATURE_DIM_FULL_RICH = GLOBAL_FEATURE_DIM_FULL + RICH_GLOBAL_DIM
GLOBAL_FEATURE_DIM_FULL_SELECT = GLOBAL_FEATURE_DIM_FULL + SELECT_CONTEXT_DIM

# With include_solver_features=True the global vector gains the NAMED arithmetic features
# the net otherwise has to rediscover from win/loss: the 35 grounded board readings
# (src/game/features.py) + the 14 prize-race ledger scalars (src/solvers/prize_map.py --
# exact greedy KO/prize arithmetic both ways). Appended LAST, after any select-context and
# rich blocks, so every existing dim composition keeps its offsets.
SOLVER_GLOBAL_DIM = FEATURE_DIM + LEDGER_DIM                             # 35 + 14 = 49


def _select_context_block(observation):
    """SELECT_CONTEXT_DIM features of the pending menu (zeros when no menu): queue state,
    menu shape, and the SelectType one-hot."""
    features = np.zeros(SELECT_CONTEXT_DIM, dtype=np.float32)
    select = observation.get("select")
    if select is None:
        return features
    features[0] = 1.0                                                  # a menu is pending
    features[1] = min(select.get("remainDamageCounter", 0) or 0, 10) / 10.0
    features[2] = min(select.get("remainEnergyCost", 0) or 0, 5) / 5.0
    features[3] = min(select.get("minCount", 0), 8) / 8.0
    features[4] = min(select.get("maxCount", 0), 8) / 8.0
    features[5] = min(len(select.get("option") or []), 40) / 40.0
    select_type = select.get("type", 0)
    if 0 <= select_type < 11:
        features[6 + select_type] = 1.0
    return features


def _attached_entity_tokens(observation):
    """Tokens for every attached/underneath card and the Stadium -- the entities the base
    encoder reduces to counts or a boolean. Returns (features, owner_ids, zone_ids) lists."""
    features, owners, zones = [], [], []
    current = observation["current"]
    me_index = current["yourIndex"]

    for owner, player in ((OWNER_ME, current["players"][me_index]),
                          (OWNER_OPPONENT, current["players"][1 - me_index])):
        in_play = [p for p in ((player["active"] or []) + (player["bench"] or []))
                   if p is not None]
        for pokemon in in_play:
            host = _light_pokemon(pokemon)             # live hp/energy binds token to host
            for tool in pokemon.get("tools") or []:
                features.append(encode_card_token(tool["id"], host))
                owners.append(owner)
                zones.append(ZONE_TOOL)
            for energy_card in pokemon.get("energyCards") or []:
                features.append(encode_card_token(energy_card["id"], host))
                owners.append(owner)
                zones.append(ZONE_ENERGY_CARD)
            for pre_evolution in pokemon.get("preEvolution") or []:
                features.append(encode_card_token(pre_evolution["id"], host))
                owners.append(owner)
                zones.append(ZONE_PRE_EVOLUTION)

    for stadium in current.get("stadium") or []:
        if stadium is None:
            continue
        if stadium.get("playerIndex") == me_index:
            owner = OWNER_ME
        elif stadium.get("playerIndex") == 1 - me_index:
            owner = OWNER_OPPONENT
        else:
            owner = OWNER_NEUTRAL
        features.append(encode_card_token(stadium["id"]))
        owners.append(owner)
        zones.append(ZONE_STADIUM)

    # Cards currently revealed mid-effect (a search in progress, a look at deck tops).
    for looked in current.get("looking") or []:
        if looked is None:
            continue
        features.append(encode_card_token(looked["id"]))
        owners.append(OWNER_ME if looked.get("playerIndex") == me_index else OWNER_OPPONENT)
        zones.append(ZONE_LOOKING)

    return features, owners, zones


def _entity_rich_rows(observation, rich_override=None):
    """Rich-effect rows aligned with _attached_entity_tokens' emission order: attached
    entities (tools / energy cards / pre-evolutions) inherit their HOST's effect block;
    Stadium and looking cards carry no per-instance engine state -> zeros.
    rich_override: the (rich_cards, rich_players) pair to use instead of decoding this
    observation -- see encode_rich.encode_rich_observation."""
    rich_cards, _rich_players = (rich_state(observation) if rich_override is None
                                 else rich_override)
    rich_cards = rich_cards or {}
    rows = []
    current = observation["current"]
    me_index = current["yourIndex"]
    for player in (current["players"][me_index], current["players"][1 - me_index]):
        in_play = [p for p in ((player["active"] or []) + (player["bench"] or []))
                   if p is not None]
        for pokemon in in_play:
            host = card_block(rich_cards.get(pokemon["serial"]))
            attached = (len(pokemon.get("tools") or []) + len(pokemon.get("energyCards") or [])
                        + len(pokemon.get("preEvolution") or []))
            rows.extend(host for _ in range(attached))
    rows.extend(_ZERO_RICH for stadium in (current.get("stadium") or []) if stadium is not None)
    rows.extend(_ZERO_RICH for looked in (current.get("looking") or []) if looked is not None)
    return rows


def solver_global_features(observation):
    """The SOLVER_GLOBAL_DIM named arithmetic features, me-centric. game_features and
    ledger_features read players[0] as 'me'; during search the deciding player can be
    index 1, so swap the players list to match the me-centric token encoding."""
    current = observation["current"]
    if current["yourIndex"] == 1:
        current = dict(current)
        current["players"] = [current["players"][1], current["players"][0]]
        observation = {"current": current}
    return np.concatenate([game_features(observation), ledger_features(observation)])


def encode_observation_full(observation, deck_counts=None, opponent_belief=None,
                            belief_top_k=None, include_rich=False, my_prize_belief=None,
                            my_prize_certainty=None, opponent_seen_hidden=None,
                            my_known_top=None, include_select_context=False,
                            include_solver_features=False, rich_override=None,
                            include_v3_zones=False):
    """observation -> the full-detail token arrays. Same output schema as encode_observation
    (token_features [T, F] / owner_ids / zone_ids / global_features), with all base detail
    flags on plus the attached-entity and Stadium tokens.

    deck_counts: {card_id: copies} of MY 60-card decklist -> my unseen library tokens.
    opponent_belief: {card_id: expected_hidden_copies} (e.g. DeckRecognizer.expected_unseen).
    include_rich: append the engine effect state (see TOKEN/GLOBAL_FEATURE_DIM_FULL_RICH) --
      per-card shields/locks/modifiers on every token, player-level item/supporter/stadium/
      energy/evolve locks + exact poison in the globals. Zeros where the state blob is absent.
    my_prize_belief: {card_id: expected prized copies} (CardKnowledge.prize_belief()) ->
      ZONE_PRIZE tokens, my deduced/inferred prizes. (These cards also appear in the merged
      ZONE_DECK unseen tokens; the zone embedding distinguishes the two readings.)
    my_prize_certainty: CardKnowledge.prize_certainty() -> global scalar (1.0 = the prize
      tokens are exact deductions; 0 when unknown/not provided).
    opponent_seen_hidden: CardKnowledge.opponent_not_prized_counts(observation) ->
      ZONE_OPPONENT_SEEN tokens: their cards we saw that are hidden again (hand/deck,
      certainly not prized).
    my_known_top: CardKnowledge.known_top_ids() -> ZONE_DECK_TOP tokens, ordered known
      upcoming draws; belief scalar = draw proximity (1.0 = next draw).
    include_solver_features: append the SOLVER_GLOBAL_DIM named arithmetic features
      (me-centric game_features + the prize-race ledger) to the global vector, LAST.
      Zeros if the solver computation fails on an exotic state.
    rich_override: with include_rich, the (rich_cards, rich_players) pair to apply instead of
      decoding THIS observation's state blob -- the search-side root freeze (see
      encode_rich.encode_rich_observation). None = decode this observation, unchanged.
    include_v3_zones: emit the two v3 token kinds (OFF by default so v1/v2 stay
      byte-identical) -- one ZONE_FACEDOWN token per unidentifiable in-play slot (the base
      path drops those slots entirely, so the model could not even tell the slot was
      occupied) and one ZONE_DECK_VIEW token per card of a `select.deck` search view (which
      otherwise reaches the model only through the individual option vectors). Both are
      appended LAST, after the select-context cards, so every existing row index is
      unchanged. See STATE_ENCODING_AUDIT M7 / M8.
    """
    my_unseen = unseen_my_counts(observation, deck_counts) if deck_counts else None
    encoded = encode_observation(observation, opponent_belief=opponent_belief,
                                 belief_top_k=belief_top_k, my_unseen=my_unseen,
                                 include_discard=True, capability_tokens=True)

    features, owners, zones = _attached_entity_tokens(observation)
    extra_token_count = 0
    if my_prize_belief:
        for card_id, expected_copies in sorted(my_prize_belief.items(),
                                               key=lambda item: -item[1]):
            features.append(encode_card_token(card_id,
                                              belief=min(expected_copies, 4.0) / 4.0))
            owners.append(OWNER_ME)
            zones.append(ZONE_PRIZE)
            extra_token_count += 1
    if opponent_seen_hidden:
        for card_id, copies in sorted(opponent_seen_hidden.items(),
                                      key=lambda item: -item[1]):
            features.append(encode_card_token(card_id, belief=min(copies, 4.0) / 4.0))
            owners.append(OWNER_OPPONENT)
            zones.append(ZONE_OPPONENT_SEEN)
            extra_token_count += 1
    if my_known_top:
        for position, card_id in enumerate(my_known_top):
            features.append(encode_card_token(card_id, belief=1.0 / (1.0 + position)))
            owners.append(OWNER_ME)
            zones.append(ZONE_DECK_TOP)
            extra_token_count += 1
    if include_select_context:
        select = observation.get("select") or {}
        me_index = observation["current"]["yourIndex"]
        for card, zone in ((select.get("effect"), ZONE_EFFECT_SOURCE),
                           (select.get("contextCard"), ZONE_CONTEXT_CARD)):
            if card is None or not card.get("id"):
                continue
            features.append(encode_card_token(card["id"]))
            owners.append(OWNER_ME if card.get("playerIndex") == me_index
                          else OWNER_OPPONENT)
            zones.append(zone)
            extra_token_count += 1
    if include_v3_zones:
        current_v3 = observation["current"]
        me_index_v3 = current_v3["yourIndex"]
        for owner, player in ((OWNER_ME, current_v3["players"][me_index_v3]),
                              (OWNER_OPPONENT, current_v3["players"][1 - me_index_v3])):
            for slots in ((player["active"] or []), (player["bench"] or [])):
                for pokemon in slots:
                    if pokemon is not None:
                        continue
                    # No id to embed: the token IS the "something is here and I cannot see
                    # what" fact. Its owner/zone segments and the v3 slot columns carry the
                    # rest (encode_details.facedown_row).
                    features.append(np.zeros(TOKEN_FEATURE_DIM, dtype=np.float32))
                    owners.append(owner)
                    zones.append(ZONE_FACEDOWN)
                    extra_token_count += 1
        for card in ((observation.get("select") or {}).get("deck") or []):
            if card is None or not card.get("id"):
                continue
            features.append(encode_card_token(card["id"]))
            owners.append(OWNER_ME if card.get("playerIndex") == me_index_v3
                          else OWNER_OPPONENT)
            zones.append(ZONE_DECK_VIEW)
            extra_token_count += 1
    if features:
        encoded["token_features"] = np.concatenate(
            [encoded["token_features"], np.stack(features).astype(np.float32)])
        encoded["owner_ids"] = np.concatenate(
            [encoded["owner_ids"], np.array(owners, dtype=np.int64)])
        encoded["zone_ids"] = np.concatenate(
            [encoded["zone_ids"], np.array(zones, dtype=np.int64)])

    # Extra globals (see GLOBAL_FEATURE_DIM_FULL): live bench capacity + turn-order parity.
    current = observation["current"]
    me_index = current["yourIndex"]
    me = current["players"][me_index]
    opponent = current["players"][1 - me_index]
    encoded["global_features"] = np.concatenate([
        encoded["global_features"],
        np.array([me.get("benchMax", 5) / 8.0, opponent.get("benchMax", 5) / 8.0,
                  float(current.get("firstPlayer", 0) == me_index),
                  min(current.get("turnActionCount", 0), 10) / 10.0,
                  float(my_prize_certainty or 0.0)], dtype=np.float32),
    ])
    if include_select_context:
        encoded["global_features"] = np.concatenate(
            [encoded["global_features"], _select_context_block(observation)])

    if include_rich:
        rich = encode_rich_observation(observation, opponent_belief=opponent_belief,
                                       belief_top_k=belief_top_k, my_unseen=my_unseen,
                                       include_discard=True, capability_tokens=True,
                                       rich_override=rich_override)
        entity_rows = _entity_rich_rows(observation, rich_override=rich_override)
        if extra_token_count:                     # knowledge tokens carry no effect state
            entity_rows = entity_rows + [_ZERO_RICH] * extra_token_count
        token_rich = rich["token_rich"]
        if entity_rows:
            token_rich = np.concatenate([token_rich, np.stack(entity_rows)])
        assert token_rich.shape[0] == encoded["token_features"].shape[0], \
            "rich rows misaligned with full token layout"
        encoded["token_features"] = np.concatenate(
            [encoded["token_features"], token_rich.astype(np.float32)], axis=1)
        encoded["global_features"] = np.concatenate(
            [encoded["global_features"], rich["global_rich"].astype(np.float32)])

    if include_solver_features:
        try:
            solver_block = solver_global_features(observation).astype(np.float32)
        except Exception:                     # degrade like the rich block: zeros, right shape
            solver_block = np.zeros(SOLVER_GLOBAL_DIM, dtype=np.float32)
        encoded["global_features"] = np.concatenate(
            [encoded["global_features"], solver_block])
    return encoded


# --------------------------------------------------------------------------- #
# Self-test: play a short random game, encode mid-game states, report the token
# inventory per zone, and verify the frozen base path is untouched.
#   ./.venv/Scripts/python.exe -m src.game.encode_full
# --------------------------------------------------------------------------- #

def _self_test():
    import random
    from collections import Counter

    from cg import game

    ZONE_NAMES = {0: "GLOBAL", 1: "ACTIVE", 2: "BENCH", 3: "HAND", 4: "DECK", 5: "DISCARD",
                  6: "PREDICTED", 7: "ATTACK", 8: "ABILITY", ZONE_STADIUM: "STADIUM",
                  ZONE_TOOL: "TOOL", ZONE_ENERGY_CARD: "ENERGY_CARD",
                  ZONE_PRE_EVOLUTION: "PRE_EVOLUTION", ZONE_LOOKING: "LOOKING",
                  ZONE_PRIZE: "PRIZE", ZONE_OPPONENT_SEEN: "OPP_SEEN",
                  ZONE_DECK_TOP: "DECK_TOP"}

    from src.decks.card_knowledge import CardKnowledge

    deck_path = "submissions/submission_alakazam/deck.csv"
    deck = [int(line) for line in open(deck_path).read().split() if line.strip()]
    deck_counts = dict(Counter(deck))
    knowledge = CardKnowledge(Counter(deck))

    rng = random.Random(7)

    def random_legal(observation):
        select = observation["select"]
        count = len(select["option"])
        take = max(min(select["maxCount"], count), select["minCount"])
        return sorted(rng.sample(range(count), take)) if count else []

    observation, _ = game.battle_start(list(deck), list(deck), seed=42)
    moves = 0
    checked = 0
    while observation["current"]["result"] == -1 and moves < 400:
        if observation["current"]["yourIndex"] == 0:
            knowledge.update(observation)
        select = observation.get("select")
        observation = game.battle_select(random_legal(observation) if select else [])
        moves += 1
        if moves % 60 == 0:
            base = encode_observation(observation)
            full = encode_observation_full(observation, deck_counts=deck_counts)
            assert full["token_features"].shape[1] == TOKEN_FEATURE_DIM
            assert full["token_features"].shape[0] >= base["token_features"].shape[0]
            assert int(full["zone_ids"].max()) < NUM_ZONES_FULL
            assert full["global_features"].shape[0] == GLOBAL_FEATURE_DIM_FULL
            assert base["global_features"].shape[0] == 22       # frozen base untouched
            prize_belief = knowledge.prize_belief()
            rich = encode_observation_full(
                observation, deck_counts=deck_counts, include_rich=True,
                my_prize_belief=prize_belief,
                my_prize_certainty=knowledge.prize_certainty(),
                opponent_seen_hidden=knowledge.opponent_not_prized_counts(observation),
                my_known_top=knowledge.known_top_ids())
            everything = encode_observation_full(
                observation, deck_counts=deck_counts, include_select_context=True)
            assert everything["global_features"].shape[0] == GLOBAL_FEATURE_DIM_FULL_SELECT
            assert int(everything["zone_ids"].max()) < NUM_ZONES_FULL
            assert rich["token_features"].shape[1] == TOKEN_FEATURE_DIM_FULL_RICH
            assert rich["global_features"].shape[0] == GLOBAL_FEATURE_DIM_FULL_RICH
            rich_block = rich["token_features"][:, TOKEN_FEATURE_DIM:]
            live = float((np.abs(rich_block) > 0).any(axis=1).mean())
            exactness = "EXACT" if knowledge.prizes_exact() is not None \
                else f"{knowledge.prize_certainty():.2f}"
            print(f"          rich: width {rich['token_features'].shape[1]}, "
                  f"{live:.0%} live effect state | prize tokens "
                  f"{len(prize_belief)} (certainty {exactness})")
            inventory = Counter(int(z) for z in full["zone_ids"])
            print(f"move {moves:3d}: base {base['token_features'].shape[0]:3d} tokens -> "
                  f"full {full['token_features'].shape[0]:3d} | "
                  + "  ".join(f"{ZONE_NAMES.get(z, z)}:{n}"
                              for z, n in sorted(inventory.items())))
            checked += 1
    game.battle_finish()
    print(f"OK: {checked} states encoded, token width {TOKEN_FEATURE_DIM}, "
          f"zones < {NUM_ZONES_FULL}, base path untouched")


if __name__ == "__main__":
    _self_test()
