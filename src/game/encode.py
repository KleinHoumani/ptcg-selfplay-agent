
from collections import Counter
from functools import lru_cache

import numpy as np

from src.cards import get_card, get_attack
from src.card_effects import (effect_features, attack_effect_features, skill_effect_features,
                              EFFECT_FEATURE_DIM)
from src.embeddings import ability_embedding, attack_embedding, card_embedding
from src.segments import (OWNER_ME, OWNER_OPPONENT, ZONE_ACTIVE, ZONE_BENCH, ZONE_DECK,
                          ZONE_DISCARD, ZONE_HAND, ZONE_PREDICTED, ZONE_ATTACK, ZONE_ABILITY)


@lru_cache(maxsize=None)
def _card_details_built(card_id):
    """The part of a built card that depends ONLY on the id -- everything except the
    per-instance serial: name, types, printed stats, and the (heavy) skill/attack text
    embeddings. Cached because it never changes; the embedding lookups here are the bulk of
    build_game's cost. build_card() copies this and attaches the live serial, so the cached
    structure is never mutated and every instance keeps its own serial (no data lost)."""
    card_details = get_card(card_id)

    card_built = {
        "id": card_id,
        "name": card_details["name"],
        "card_type": card_details["cardType"],
        "pokemon_type": card_details["pokemonType"],
        "evolution_type": card_details["evolutionType"],
        "retreat_cost": card_details["retreatCost"],
        "hp": card_details["hp"],
        "weakness": card_details["weakness"],
        "resistance": card_details["resistance"],
        "energy_type": card_details["energyType"],
        "basic": card_details["basic"],
        "stage1": card_details["stage1"],
        "stage2": card_details["stage2"],
        "ex": card_details["ex"],
        "mega_ex": card_details["megaEx"],
        "tera": card_details["tera"],
        "ace_spec": card_details["aceSpec"],
        "evolves_from": card_details["evolvesFrom"],
        "skills": [],
        "attacks": [],
        "tags": []
    }

    for index, skill in enumerate(card_details["skills"]):
        skill_built = {
            "name": skill["name"],
            "text": skill["text"],
            "text_embedding": ability_embedding(card_id, index),
            "effects": []  # manual labels, added later
        }
        card_built["skills"].append(skill_built)

    for attack_id in card_details["attacks"]:
        attack = get_attack(attack_id)
        attack_built = {
            "id": attack_id,
            "name": attack["name"],
            "text": attack["text"],
            "text_embedding": attack_embedding(attack_id),
            "damage": attack["damage"],
            "energies": attack["energies"]
        }
        card_built["attacks"].append(attack_built)

    return card_built


def build_card(card):
    """A built card for `card` (id + per-instance serial). Id-derived fields come from the
    `_card_details_built` cache; serial is attached fresh per instance."""
    card_built = dict(_card_details_built(card["id"]))   # shallow copy -> never mutate the cache
    card_built["serial"] = card["serial"]
    return card_built


def build_pokemon(pokemon):
    pokemon_built = {
        "id": pokemon["id"],
        "serial": pokemon["serial"],
        "hp": pokemon["hp"],
        "max_hp": pokemon["maxHp"],
        "appear_this_turn": pokemon["appearThisTurn"],
        "energies": list(pokemon["energies"]),  # resolved EnergyType list (what attacks read)
        "energy_cards": [],
        "tools": [],
        "pre_evolution": []
    }

    for energy in pokemon["energyCards"]:
        pokemon_built["energy_cards"].append({"id": energy["id"], "serial": energy["serial"]})

    for tool in pokemon["tools"]:
        pokemon_built["tools"].append(build_card(tool))

    for pre_evolution in pokemon["preEvolution"]:
        pokemon_built["pre_evolution"].append(build_card(pre_evolution))

    return pokemon_built


def build_player(player):
    return {
        "hand_count": player["handCount"],
        "deck_count": player["deckCount"],
        "bench_max": player["benchMax"],
        "poisoned": player["poisoned"],
        "burned": player["burned"],
        "asleep": player["asleep"],
        "paralyzed": player["paralyzed"],
        "confused": player["confused"],
        "prize_count": len(player["prize"]),
        "active": [build_pokemon(pokemon) for pokemon in player["active"] if pokemon is not None],
        "bench": [build_pokemon(pokemon) for pokemon in player["bench"] if pokemon is not None],
        # hand is visible only for you -- the opponent's is None (only handCount is known)
        "hand": [build_card(card) for card in player["hand"]] if player["hand"] is not None else [],
        "discard": [build_card(card) for card in player["discard"]],
    }


def build_game(observation):
    cur = observation["current"]
    me = cur["players"][cur["yourIndex"]]
    opp = cur["players"][1 - cur["yourIndex"]]

    # Player details
    me_built = build_player(me)
    opp_built = build_player(opp)

    # Turn details
    turn_count = cur["turn"]
    turn_action_count = cur["turnActionCount"]
    first_player_index = cur["firstPlayer"]
    supporter_played = cur["supporterPlayed"]
    stadium_played = cur["stadiumPlayed"]
    energy_attached = cur["energyAttached"]
    retreated = cur["retreated"]
    stadium = [build_card(card) for card in cur["stadium"]]
    looking = [build_card(card) for card in (cur["looking"] or [])]

    game_state = {
        "me": me_built,
        "opponent": opp_built,
        "turn_count": turn_count,
        "turn_action_count": turn_action_count,
        "first_player_index": first_player_index,
        "supporter_played": supporter_played,
        "stadium_played": stadium_played,
        "energy_attached": energy_attached,
        "retreated": retreated,
        "stadium": stadium,
        "looking": looking
    }

    return game_state
    

# --- Vectorize: game_state -> transformer input arrays -------------------------------
# Owner / zone segment ids come from src/segments.py (shared with the model).
# Normalisers (rough scales so features land near [0, 1] -- tune freely).
MAX_HP = 400.0
MAX_RETREAT = 4.0
MAX_TURN = 50.0          # saturation scale (clamped) -- turns are unbounded
MAX_DECK = 60.0
MAX_HAND = 30.0          # saturation scale (clamped) -- draw decks reach 30+; true cap ~58
NUM_PRIZES = 6.0
NUM_ENERGY_TYPES = 12    # runtime EnergyType reaches 11 (special energies); guarded below for safety
NUM_CARD_TYPES = 7
MAX_COPIES = 4.0         # belief-token scale: a 4-of saturates (basic energy clamps here too)


def _one_hot(index, size):
    vector = np.zeros(size, dtype=np.float32)
    if index is not None and 0 <= index < size:
        vector[index] = 1.0
    return vector


# Attack capability-token extras (zero on every non-attack token): normalised damage + per-type
# energy cost. Tokens share one width, so this rides on the end of every token's vector.
ATTACK_DAMAGE_SCALE = 100.0
ATTACK_COST_SCALE = 3.0
ATTACK_BLOCK_DIM = 1 + NUM_ENERGY_TYPES
_ZERO_ATTACK_BLOCK = np.zeros(ATTACK_BLOCK_DIM, dtype=np.float32)
_ZERO_ATTACK_BLOCK.flags.writeable = False


@lru_cache(maxsize=None)
def _card_type_stats(card_id):
    """Type/energy/weakness one-hots + printed stats -- the host-context block shared by a card's
    own token and its attack/ability (capability) tokens."""
    card = get_card(card_id)
    block = np.concatenate([
        _one_hot(card["cardType"], NUM_CARD_TYPES),
        _one_hot(card["energyType"], NUM_ENERGY_TYPES),
        _one_hot(card["weakness"], NUM_ENERGY_TYPES),
        np.array([
            card["hp"] / MAX_HP,                              # printed (max) hp
            card["retreatCost"] / MAX_RETREAT,
            float(card["basic"]), float(card["stage1"]), float(card["stage2"]),
            float(card["ex"]), float(card["megaEx"]), float(card["tera"]), float(card["aceSpec"]),
        ], dtype=np.float32),
    ]).astype(np.float32)
    block.flags.writeable = False
    return block


def _read_only(array):
    array = array.astype(np.float32)
    array.flags.writeable = False
    return array


@lru_cache(maxsize=None)
def _card_static_features(card_id):
    """Static (card-id-only) block of a CARD token: identity embedding + type/stats + the
    whole-card effect union. Cached + read-only (downstream only concatenates/stacks it)."""
    return _read_only(np.concatenate([
        card_embedding(card_id), _card_type_stats(card_id), effect_features(card_id)]))


@lru_cache(maxsize=None)
def _attack_static_features(card_id, attack_id):
    """Static block of an ATTACK capability token: the attack's embedding + the host's type/stats
    + THIS attack's own effect vector (so two attacks of a card stay distinct)."""
    return _read_only(np.concatenate([
        attack_embedding(attack_id), _card_type_stats(card_id), attack_effect_features(attack_id)]))


@lru_cache(maxsize=None)
def _ability_static_features(card_id, skill_index):
    """Static block of an ABILITY capability token: the ability's embedding + host type/stats +
    that ability's own effect vector."""
    return _read_only(np.concatenate([
        ability_embedding(card_id, skill_index), _card_type_stats(card_id),
        skill_effect_features(card_id, skill_index)]))


@lru_cache(maxsize=None)
def _attack_block(attack_id):
    """An attack token's trailing extras: normalised damage + per-energy-type cost counts."""
    attack = get_attack(attack_id)
    cost = np.zeros(NUM_ENERGY_TYPES, dtype=np.float32)
    for energy_type in attack["energies"]:
        if 0 <= energy_type < NUM_ENERGY_TYPES:
            cost[energy_type] += 1.0 / ATTACK_COST_SCALE
    return _read_only(np.concatenate([[attack["damage"] / ATTACK_DAMAGE_SCALE], cost]))


def _dynamic_block(pokemon):
    """An in-play card's per-instance state: current hp ratio, tool/evolution counts,
    appeared-this-turn, and attached-energy counts (zeros when the card is not in play)."""
    if pokemon is None:
        return np.zeros(4 + NUM_ENERGY_TYPES, dtype=np.float32)
    energy_counts = np.zeros(NUM_ENERGY_TYPES, dtype=np.float32)
    for energy_type in pokemon["energies"]:
        if 0 <= energy_type < NUM_ENERGY_TYPES:               # guard unknown runtime energy enums
            energy_counts[energy_type] += 1
    return np.concatenate([
        np.array([
            pokemon["hp"] / pokemon["max_hp"],                # current hp ratio
            len(pokemon["tools"]) / 2.0,
            len(pokemon["pre_evolution"]) / 2.0,              # evolution depth
            float(pokemon["appear_this_turn"]),
        ], dtype=np.float32),
        energy_counts,
    ])


def encode_card_token(card_id, pokemon=None, belief=0.0):
    """One card token = static (card id, cached) + dynamic (filled from `pokemon` in play, else
    zero -- the zone embedding says which) + a trailing belief scalar + a zero attack block (only
    attack capability tokens use that). `belief` is the predicted presence for a belief/count
    token in [0, 1]; 0 for every real, observed card."""
    return np.concatenate([_card_static_features(card_id), _dynamic_block(pokemon),
                           [np.float32(belief)], _ZERO_ATTACK_BLOCK]).astype(np.float32)


def encode_attack_token(card_id, attack_id, pokemon):
    """A capability token for ONE attack of an in-play Pokemon: the attack's embedding + its own
    effect vector + the host's type/stats and live hp/energy + the attack's damage & cost. Lets
    the model reason about each specific attack (not the card-level union) in board context."""
    return np.concatenate([_attack_static_features(card_id, attack_id), _dynamic_block(pokemon),
                           [np.float32(0.0)], _attack_block(attack_id)]).astype(np.float32)


def encode_ability_token(card_id, skill_index, pokemon):
    """A capability token for ONE ability of an in-play Pokemon (its embedding + own effect vector
    + the host's type/stats and live hp/energy). No damage/cost -- abilities aren't attacks."""
    return np.concatenate([_ability_static_features(card_id, skill_index), _dynamic_block(pokemon),
                           [np.float32(0.0)], _ZERO_ATTACK_BLOCK]).astype(np.float32)


# Width of one token's feature vector (probed once so it tracks the encoders).
TOKEN_FEATURE_DIM = len(encode_card_token(1))


def _encode_global(game_state):
    """The CLS token's scalar features: turn flags + per-side counts + conditions."""
    me, opponent = game_state["me"], game_state["opponent"]
    return np.array([
        min(game_state["turn_count"], MAX_TURN) / MAX_TURN,
        float(game_state["supporter_played"]),
        float(game_state["stadium_played"]),
        float(game_state["energy_attached"]),
        float(game_state["retreated"]),
        float(len(game_state["stadium"]) > 0),                # a stadium is in play
        me["prize_count"] / NUM_PRIZES, opponent["prize_count"] / NUM_PRIZES,
        me["deck_count"] / MAX_DECK, opponent["deck_count"] / MAX_DECK,
        min(me["hand_count"], MAX_HAND) / MAX_HAND, min(opponent["hand_count"], MAX_HAND) / MAX_HAND,
        float(me["poisoned"]), float(me["burned"]), float(me["asleep"]),
        float(me["paralyzed"]), float(me["confused"]),
        float(opponent["poisoned"]), float(opponent["burned"]), float(opponent["asleep"]),
        float(opponent["paralyzed"]), float(opponent["confused"]),
    ], dtype=np.float32)


def encode_game(game_state, opponent_belief=None, belief_top_k=None,
                my_unseen=None, include_discard=False, capability_tokens=False):
    """game_state -> per-token arrays for ONE state (collate pads + batches + ->torch).

    Cards with per-instance state (board, my hand) get one token each. The optional arguments
    add the bulk unordered zones as count-weighted tokens (one per distinct card id), so the
    model can see full composition, not just the board -- each defaults off, leaving the base
    encoding untouched:
      opponent_belief: {card_id: expected_hidden_copies} (e.g. DeckRecognizer.expected_unseen)
        -> OPPONENT/ZONE_PREDICTED tokens, the model's view of the opponent's hidden cards.
        belief_top_k caps how many (None = the full predicted deck).
      my_unseen: {card_id: copies} of my decklist I can't see (see unseen_my_counts) -> my
        ME/ZONE_DECK hidden library (deck + prizes; the split sizes are in the global features).
      include_discard: emit both players' (public) discard piles as ZONE_DISCARD tokens.
      capability_tokens: emit one ZONE_ATTACK token per attack and one ZONE_ABILITY token per
        ability of every in-play Pokemon (both sides), each carrying that specific action's own
        effects + host state -- lossless per-action detail vs the card-level union (transformer
        only; the mean-pool baseline would just blur them)."""
    token_features, owner_ids, zone_ids = [], [], []

    for owner, player in ((OWNER_ME, game_state["me"]), (OWNER_OPPONENT, game_state["opponent"])):
        for zone_id, zone_key in ((ZONE_ACTIVE, "active"), (ZONE_BENCH, "bench")):
            for pokemon in player[zone_key]:
                token_features.append(encode_card_token(pokemon["id"], pokemon))
                owner_ids.append(owner)
                zone_ids.append(zone_id)

    # Hand is visible only for you -> tokens; the opponent's is just a count (in global).
    for card in game_state["me"]["hand"]:
        token_features.append(encode_card_token(card["id"]))
        owner_ids.append(OWNER_ME)
        zone_ids.append(ZONE_HAND)

    # Discard piles are public for both players -> count-weighted tokens.
    if include_discard:
        for owner, player in ((OWNER_ME, game_state["me"]), (OWNER_OPPONENT, game_state["opponent"])):
            for card_id, count in Counter(card["id"] for card in player["discard"]).items():
                token_features.append(
                    encode_card_token(card_id, belief=min(count, MAX_COPIES) / MAX_COPIES))
                owner_ids.append(owner)
                zone_ids.append(ZONE_DISCARD)

    # My hidden library (the copies of my known decklist I can't see -- deck + prizes).
    if my_unseen:
        for card_id, count in my_unseen.items():
            token_features.append(
                encode_card_token(card_id, belief=min(count, MAX_COPIES) / MAX_COPIES))
            owner_ids.append(OWNER_ME)
            zone_ids.append(ZONE_DECK)

    if opponent_belief:
        predicted = sorted(opponent_belief.items(), key=lambda item: -item[1])
        if belief_top_k is not None:
            predicted = predicted[:belief_top_k]
        for card_id, expected_copies in predicted:
            token_features.append(
                encode_card_token(card_id, belief=min(expected_copies, MAX_COPIES) / MAX_COPIES))
            owner_ids.append(OWNER_OPPONENT)
            zone_ids.append(ZONE_PREDICTED)

    # Per-action capability tokens for in-play Pokemon (both sides): one per attack + per ability.
    if capability_tokens:
        for owner, player in ((OWNER_ME, game_state["me"]), (OWNER_OPPONENT, game_state["opponent"])):
            for pokemon in player["active"] + player["bench"]:
                card = get_card(pokemon["id"])
                for attack_id in card["attacks"]:
                    token_features.append(encode_attack_token(pokemon["id"], attack_id, pokemon))
                    owner_ids.append(owner)
                    zone_ids.append(ZONE_ATTACK)
                for skill_index in range(len(card["skills"])):
                    token_features.append(encode_ability_token(pokemon["id"], skill_index, pokemon))
                    owner_ids.append(owner)
                    zone_ids.append(ZONE_ABILITY)

    tokens = (np.stack(token_features).astype(np.float32) if token_features
              else np.zeros((0, TOKEN_FEATURE_DIM), dtype=np.float32))   # empty board -> [0, F]
    return {
        "token_features": tokens,                                        # [T, F]
        "owner_ids": np.array(owner_ids, dtype=np.int64),                # [T]
        "zone_ids": np.array(zone_ids, dtype=np.int64),                  # [T]
        "global_features": _encode_global(game_state),                   # [G]
    }


# --- Fast path: observation -> arrays, skipping the rich build_game intermediate ---------
# encode_game/_encode_global only read a SUBSET of build_game's output (active/bench pokemon
# stats, our hand card ids, per-side counts/conditions). The rest -- discard piles, stadium,
# looking cards, and every card's text-embedding/skills/attacks -- is built and thrown away.
# encode_observation builds only the keys the encoders read, then reuses encode_game, so the
# output is byte-identical while skipping that wasted work.

def _light_pokemon(pokemon):
    return {"id": pokemon["id"], "hp": pokemon["hp"], "max_hp": pokemon["maxHp"],
            "energies": pokemon["energies"], "tools": pokemon["tools"],          # only len() read
            "pre_evolution": pokemon["preEvolution"], "appear_this_turn": pokemon["appearThisTurn"]}


def _light_player(player):
    return {
        "active": [_light_pokemon(p) for p in (player["active"] or []) if p is not None],
        "bench": [_light_pokemon(p) for p in (player["bench"] or []) if p is not None],
        "hand": [{"id": card["id"]} for card in (player["hand"] or [])],         # opponent's is None -> []
        "discard": [{"id": card["id"]} for card in (player["discard"] or [])],   # public both sides
        "prize_count": len(player["prize"]), "deck_count": player["deckCount"],
        "hand_count": player["handCount"],
        "poisoned": player["poisoned"], "burned": player["burned"], "asleep": player["asleep"],
        "paralyzed": player["paralyzed"], "confused": player["confused"],
    }


def encode_observation(observation, opponent_belief=None, belief_top_k=None,
                       my_unseen=None, include_discard=False, capability_tokens=False):
    """Same output as encode_game(build_game(observation)) but without building the rich
    intermediate -- the hot encoding path used by the agents and training. The trailing
    arguments pass straight through to encode_game (belief / my-deck / discard / capability)."""
    current = observation["current"]
    me_index = current["yourIndex"]
    me, opponent = current["players"][me_index], current["players"][1 - me_index]
    light = {
        "me": _light_player(me),
        "opponent": _light_player(opponent),
        "turn_count": current["turn"],
        "supporter_played": current["supporterPlayed"],
        "stadium_played": current["stadiumPlayed"],
        "energy_attached": current["energyAttached"],
        "retreated": current["retreated"],
        "stadium": current["stadium"],                                           # only len() read
    }
    return encode_game(light, opponent_belief, belief_top_k, my_unseen, include_discard,
                       capability_tokens)


def unseen_my_counts(observation, deck_counts):
    """{card_id: copies of mine I can't see} = my decklist minus every card visible on my side
    (hand, board incl. attached energy/tools/evolution stack, discard, a Stadium I played). The
    remainder is split between my deck and prizes (both face-down) -- we know the multiset, not
    the split. Pass the result to encode as `my_unseen`. `deck_counts` is a {card_id: copies}
    of the 60-card deck. No cross-observation history is needed: every one of my cards is, at
    any instant, either visible here or in a hidden zone (deck/prize).

    Exact at stable decision points. It can over-count by one at two transient states, both
    harmless (a single low-weight phantom deck token, and the card in question is carried by the
    option/context features anyway): a face-down own Pokémon during setup (shown as None -- no id
    to read), and a card mid-effect-resolution that has left my hand but not yet reached its
    destination zone."""
    current = observation["current"]
    my_index = current["yourIndex"]
    me = current["players"][my_index]
    seen = Counter()
    for card in me["hand"] or []:
        seen[card["id"]] += 1
    for card in me["discard"]:
        seen[card["id"]] += 1
    for zone in ("active", "bench"):
        for pokemon in me[zone] or []:
            if pokemon is None:
                continue
            seen[pokemon["id"]] += 1
            for energy in pokemon["energyCards"]:
                seen[energy["id"]] += 1
            for tool in pokemon["tools"]:
                seen[tool["id"]] += 1
            for pre in pokemon["preEvolution"]:
                seen[pre["id"]] += 1
    # A Stadium I played lives in the shared in-play area, not my player dict -- count it (only
    # mine; cards carry an owner) or it'd look like it's still in my deck. (No lost zone exists in
    # cabt; `looking` cards stay counted in deckCount, so neither needs subtracting here.)
    for card in current["stadium"]:
        if card.get("playerIndex") == my_index:
            seen[card["id"]] += 1
    return {card_id: count - seen[card_id] for card_id, count in deck_counts.items()
            if count - seen[card_id] > 0}


def board_summary(encoded):
    """Mean-pooled board token vector (zeros if the board has no tokens) -- the policy's
    cheap board context."""
    tokens = encoded["token_features"]
    if tokens.shape[0] == 0:
        return np.zeros(tokens.shape[1], dtype=np.float32)
    return tokens.mean(axis=0).astype(np.float32)


def collate(encoded):
    """Pad a list of encode_game() outputs into one batch of torch tensors for the model:
    (token_features, owner_ids, zone_ids, padding_mask, global_features)."""
    import torch

    batch = len(encoded)
    max_tokens = max(item["token_features"].shape[0] for item in encoded)
    feature_dim = encoded[0]["token_features"].shape[1]
    global_dim = encoded[0]["global_features"].shape[0]

    token_features = torch.zeros(batch, max_tokens, feature_dim)
    owner_ids = torch.zeros(batch, max_tokens, dtype=torch.long)
    zone_ids = torch.zeros(batch, max_tokens, dtype=torch.long)
    padding_mask = torch.ones(batch, max_tokens, dtype=torch.bool)   # True = padded
    global_features = torch.zeros(batch, global_dim)

    for i, item in enumerate(encoded):
        count = item["token_features"].shape[0]
        token_features[i, :count] = torch.from_numpy(item["token_features"])
        owner_ids[i, :count] = torch.from_numpy(item["owner_ids"])
        zone_ids[i, :count] = torch.from_numpy(item["zone_ids"])
        padding_mask[i, :count] = False
        global_features[i] = torch.from_numpy(item["global_features"])

    return token_features, owner_ids, zone_ids, padding_mask, global_features


# --- Option features (for the policy: score each legal select.option) ----------------
NUM_OPTION_TYPES = 16
STATIC_FEATURE_DIM = len(_card_static_features(1))   # the structured card block, reused for option cards
_AREA_ZONE = {2: "hand", 3: "discard", 4: "active", 5: "bench", 6: "prize"}


def _option_card(card_id):
    """Full STATIC card features (text embedding + cardType/energyType/weakness one-hots +
    printed stats) for a card an option references -- the same block board tokens get -- or
    zeros when the option references no card. This is what lets the policy tell e.g. a Fire from
    a Psychic energy when choosing which to attach, instead of two near-identical text embeddings."""
    if card_id is None:
        return np.zeros(STATIC_FEATURE_DIM, dtype=np.float32)
    return _card_static_features(card_id)


def _resolve_card_id(observation, area, index, player_index):
    """The card id at (area, index) for player_index, or None (face-down / out of range)."""
    current = observation["current"]
    if area == 1:                                   # DECK -> select.deck
        deck = observation["select"].get("deck")
        card = deck[index] if deck is not None and index < len(deck) else None
    elif area == 7:                                 # STADIUM
        stadium = current["stadium"]
        card = stadium[index] if index < len(stadium) else None
    else:
        zone = _AREA_ZONE.get(area)
        sequence = current["players"][player_index].get(zone) if zone else None
        card = sequence[index] if sequence is not None and index < len(sequence) else None
    return card["id"] if card else None


def encode_option(observation, option):
    """Feature vector for one legal option: its type + the cards/attack it references."""
    your_index = observation["current"]["yourIndex"]
    type_one_hot = _one_hot(option["type"], NUM_OPTION_TYPES)

    area, index = option.get("area"), option.get("index")
    if option["type"] == 7 and index is not None:   # PLAY -> hand[index]
        primary_id = _resolve_card_id(observation, 2, index, your_index)
    elif area is not None and index is not None:
        owner = option.get("playerIndex")
        primary_id = _resolve_card_id(observation, area, index, owner if owner is not None else your_index)
    else:
        primary_id = None

    in_play_area, in_play_index = option.get("inPlayArea"), option.get("inPlayIndex")
    target_id = (_resolve_card_id(observation, in_play_area, in_play_index, your_index)
                 if in_play_area is not None and in_play_index is not None else None)

    attack_id = option.get("attackId")
    attack = get_attack(attack_id) if attack_id is not None else None
    attack_damage = (attack["damage"] / 100.0) if attack else 0.0
    number = option.get("number")
    number = (number / 10.0) if number is not None else 0.0

    return np.concatenate([
        type_one_hot,
        _option_card(primary_id),                   # full static card block (was: text embedding only)
        _option_card(target_id),
        attack_embedding(attack_id),                # zeros if None
        attack_effect_features(attack_id),          # THIS attack's structured effects (not the card union)
        np.array([attack_damage, number], dtype=np.float32),
    ]).astype(np.float32)


# Width of one option's feature vector (computed so it tracks encode_option -- import this
# instead of hardcoding it).
OPTION_FEATURE_DIM = (NUM_OPTION_TYPES + 2 * STATIC_FEATURE_DIM + len(attack_embedding(0))
                      + EFFECT_FEATURE_DIM + 2)
    

