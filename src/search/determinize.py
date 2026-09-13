"""Build a determinization (a concrete guess of all hidden information) for the cabt search
API. MCTS needs perfect-info worlds to search; `cg.api.search_begin` takes one as its
arguments. We know our own full deck list; the opponent's full deck is sampled from the
belief model (DeckRecognizer). Given a full deck for each side, the split into the hidden
zones (deck order / prizes / opponent hand / face-down active) is sampled here.

Verified against the engine: the produced tuple is accepted by search_begin, and counts
match the observation's deckCount / prize / handCount.
"""

import random
from collections import Counter

from src.cards import get_card


def _visible_ids(player):
    """Card ids the owner of `player` can see in their own non-hidden zones (board + attached
    + discard). Hand is handled separately (visible only for us); prizes/deck are hidden."""
    ids = []
    for pokemon in (player.get("active") or []) + player.get("bench", []):
        if pokemon is None:
            continue
        ids.append(pokemon["id"])
        for card in (pokemon.get("energyCards", []) + pokemon.get("tools", [])
                     + pokemon.get("preEvolution", [])):
            ids.append(card["id"])
    for card in player.get("discard", []):
        ids.append(card["id"])
    return ids


def _pool(full_deck, taken_visible, hand_ids=()):
    """full_deck (a 60-id list) minus the cards already accounted for -> the hidden pool.
    Raises ValueError if a visible card isn't in the assumed deck (inconsistent belief)."""
    pool = Counter(full_deck)
    for card_id in list(taken_visible) + list(hand_ids):
        pool[card_id] -= 1
        if pool[card_id] < 0:
            raise ValueError("assumed deck does not contain a revealed card")
    return list(pool.elements())


def _fit(pool, target, rng):
    """Trim a hidden pool to `target` cards (dropping random surplus) so a small accounting
    SURPLUS -- an in-play card we didn't subtract (a Stadium, a mid-effect in-transit card) --
    degrades to a slightly-imperfect search world instead of a failed determinization (which
    would fall back to the no-search policy). A deficit can't be invented, so the caller still
    raises and falls back on too-few."""
    if len(pool) > target:
        rng.shuffle(pool)
        pool = pool[:target]
    return pool


def _apply_known_positions(deck, top_ids, bottom_ids):
    """Reorder a determinized deck so known-position cards sit where we know they are
    (top_ids first in order, bottom_ids last). Best-effort: ids not present are skipped."""
    if not top_ids and not bottom_ids:
        return deck
    remaining = list(deck)
    top, bottom = [], []
    for card_id in top_ids:
        if card_id in remaining:
            remaining.remove(card_id)
            top.append(card_id)
    for card_id in bottom_ids:
        if card_id in remaining:
            remaining.remove(card_id)
            bottom.append(card_id)
    return top + remaining + bottom


def _split_by_never_seen(pool, never_seen_counts):
    """Partition a hidden-card id list into (never-seen part, seen-somewhere part) against a
    {card_id: copies never sighted} multiset. Prizes can only be never-seen cards."""
    budget = Counter(never_seen_counts)
    never_part, seen_part = [], []
    for card_id in pool:
        if budget[card_id] > 0:
            budget[card_id] -= 1
            never_part.append(card_id)
        else:
            seen_part.append(card_id)
    return never_part, seen_part


def build_determinization(observation, my_deck, opponent_deck, rng=None, knowledge=None):
    """observation: the dict the agent received. my_deck / opponent_deck: full 60-card id
    lists (ours is known; the opponent's is sampled from the belief). Returns the 6 positional
    args for cg.api.search_begin: (your_deck, your_prize, opp_deck, opp_prize, opp_hand,
    opp_active).

    knowledge: optional src.decks.card_knowledge.CardKnowledge. When given, sampled worlds
    respect what is actually known: MY prizes are drawn only from never-sighted cards (exact
    when the tracker has deduced them -- e.g. after a full-deck search), and the OPPONENT's
    prizes avoid cards we have ever seen of theirs. None -> the original behavior, unchanged."""
    rng = rng or random.Random()
    current = observation["current"]
    me_index = current["yourIndex"]
    me, opponent = current["players"][me_index], current["players"][1 - me_index]

    # cards being "looked at" are pulled from a player's deck temporarily -> account for them
    looking = current.get("looking") or []
    my_looking = [c["id"] for c in looking if c is not None and c.get("playerIndex") == me_index]
    opp_looking = [c["id"] for c in looking if c is not None and c.get("playerIndex") == 1 - me_index]

    # a Stadium in play lives in current.stadium, NOT a player's zones -> subtract it too, or it
    # stays counted as hidden and the pool overshoots by one (then the whole search falls back).
    stadium = current.get("stadium") or []
    my_stadium = [c["id"] for c in stadium if c.get("playerIndex") == me_index]

    # our side: deck + prize are hidden even to us; carve them from what we can't see
    my_hand = [card["id"] for card in (me["hand"] or [])]
    my_hidden = _pool(my_deck, _visible_ids(me) + my_looking + my_stadium, my_hand)
    deck_count, prize_count = me["deckCount"], len(me["prize"])
    my_hidden = _fit(my_hidden, deck_count + prize_count, rng)        # absorb any residual surplus
    if len(my_hidden) != deck_count + prize_count:
        raise ValueError(f"our hidden pool {len(my_hidden)} < deck {deck_count} + prize {prize_count}")
    rng.shuffle(my_hidden)
    if knowledge is not None:
        # Prizes are a uniform subset of the cards never sighted anywhere -- exact when the
        # tracker has pinned them (the shuffle above already randomized order, so taking the
        # first prize_count of the never-seen part IS a uniform draw).
        never_part, seen_part = _split_by_never_seen(my_hidden, knowledge.never_seen_counts())
        if len(never_part) >= prize_count:
            your_prize = never_part[:prize_count]
            your_deck = never_part[prize_count:] + seen_part
            rng.shuffle(your_deck)
            your_deck = _apply_known_positions(your_deck, knowledge.known_top_ids(),
                                               knowledge.known_bottom_ids())
        else:                                             # transient inconsistency: fall back
            your_deck = my_hidden[:deck_count]
            your_prize = my_hidden[deck_count:deck_count + prize_count]
    else:
        your_deck = my_hidden[:deck_count]
        your_prize = my_hidden[deck_count:deck_count + prize_count]

    # opponent side: deck + prize + hand (+ a face-down active) are all hidden to us. We don't
    # precisely subtract their Stadium (it may be outside the sampled belief deck) -- _fit trims
    # the surplus instead.
    opp_hidden = _pool(opponent_deck, _visible_ids(opponent) + opp_looking)
    opp_deck_count, opp_prize_count, opp_hand_count = (
        opponent["deckCount"], len(opponent["prize"]), opponent["handCount"])
    active_list = opponent.get("active") or []
    need_active = len(active_list) > 0 and active_list[0] is None
    expected = opp_deck_count + opp_prize_count + opp_hand_count + (1 if need_active else 0)
    opp_hidden = _fit(opp_hidden, expected, rng)
    if len(opp_hidden) != expected:
        raise ValueError(f"opponent hidden pool {len(opp_hidden)} < expected {expected}")
    rng.shuffle(opp_hidden)

    opp_active = []
    if need_active:                                  # a face-down active must be a Basic Pokemon
        for position, card_id in enumerate(opp_hidden):
            card = get_card(card_id)
            if card and card.get("basic"):
                opp_active = [opp_hidden.pop(position)]
                break
        if not opp_active:
            raise ValueError("no Basic Pokemon available for the opponent's face-down active")

    if knowledge is not None:
        # Their prizes avoid every card of theirs we have EVER seen (transient reveals
        # included) -- those copies are hidden in hand/deck, never prized.
        not_prized = knowledge.opponent_not_prized_counts(observation)
        never_part, seen_part = _split_by_never_seen(opp_hidden, Counter(opp_hidden)
                                                     - Counter(not_prized))
        # never_part here = the portion NOT known-seen (eligible for prizes).
        if len(never_part) >= opp_prize_count:
            opp_prize = never_part[:opp_prize_count]
            rest = never_part[opp_prize_count:] + seen_part
            rng.shuffle(rest)
            opp_hand = rest[:opp_hand_count]
            opp_deck = rest[opp_hand_count:]
            return your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active
    opp_hand = opp_hidden[:opp_hand_count]
    opp_prize = opp_hidden[opp_hand_count:opp_hand_count + opp_prize_count]
    opp_deck = opp_hidden[opp_hand_count + opp_prize_count:]
    return your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active
