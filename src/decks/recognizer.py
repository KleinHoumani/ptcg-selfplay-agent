"""Recognize an opponent's deck by matching revealed cards against a corpus of real
ladder decklists (built by scripts/parse_top_episode_decks.py -> data/decks/corpus.json).

Method (Bayesian-ish): the opponent's deck must be one that CONTAINS every card we've
seen, so we keep the corpus decks consistent with the reveals and rank them by popularity
(games played). With one signature card the belief is broad; as more cards reveal, the
consistent set collapses to a confident identification. `most_likely` returns the best-
guess full 60-card list -- the prior MCTS determinization samples the opponent's hidden
cards from.

Matching the actual decklists is the strong baseline (we have ground truth). A later
refinement is clustering the corpus into archetypes, for generalization to unseen-but-
similar lists and meta summarization.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

from src.cards import get_card

_BASIC_ENERGY = 5   # CardType.BASIC_ENERGY (no 4-copy limit)

_CORPUS_PATH = Path(__file__).resolve().parents[2] / "data" / "decks" / "corpus.json"


def _contains(deck, revealed):
    return all(deck.get(card, 0) >= count for card, count in revealed.items())


def _overlap(deck, revealed):
    return sum(min(deck.get(card, 0), count) for card, count in revealed.items())


def _as_counter(revealed):
    """Accept either a {card_id: count} mapping (e.g. OpponentTracker.revealed_counts()) or a
    flat iterable of card ids -> Counter. (Iterating a Counter yields keys only, so a mapping
    must be handled explicitly or every count collapses to 1.)"""
    if isinstance(revealed, dict):
        return Counter({int(card): int(count) for card, count in revealed.items()})
    return Counter(int(card) for card in revealed)


class DeckRecognizer:
    def __init__(self, corpus_path=_CORPUS_PATH):
        records = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
        # Merge the per-(username, deck) records into unique deck compositions, summing
        # popularity -- popularity is the prior over what an unknown opponent is running.
        merged = {}
        for record in records:
            cards = {int(card): count for card, count in record["cards"].items()}
            key = tuple(sorted(cards.items()))
            entry = merged.setdefault(key, {"cards": cards, "games": 0, "players": set()})
            entry["games"] += record["games"]
            entry["players"].add(record["username"])
        self.decks = [{"cards": e["cards"], "games": e["games"], "players": len(e["players"])}
                      for e in merged.values()]
        self.decks.sort(key=lambda deck: -deck["games"])

    def _distribution(self, seen):
        """Corpus decks consistent with `seen` (a Counter) + their normalised popularity
        probabilities; falls back to the closest decks by overlap if none are consistent."""
        consistent = [deck for deck in self.decks if _contains(deck["cards"], seen)]
        if not consistent:
            consistent = sorted(self.decks,
                                key=lambda deck: (-_overlap(deck["cards"], seen), -deck["games"]))[:5]
        total = sum(deck["games"] for deck in consistent) or 1
        return [(deck, deck["games"] / total) for deck in consistent]

    def beliefs(self, revealed, top=5):
        """revealed: iterable of opponent card ids seen so far (a multiset). Returns ranked
        [(deck cards, probability)] over corpus decks consistent with the reveals."""
        distribution = sorted(self._distribution(_as_counter(revealed)),
                              key=lambda item: -item[1])
        return [(deck["cards"], prob) for deck, prob in distribution[:top]]

    def most_likely(self, revealed):
        """Best-guess full deck (card id -> count), or None if the corpus is empty."""
        belief = self.beliefs(revealed, top=1)
        return belief[0][0] if belief else None

    def consistent_deck(self, revealed, deck_size=60):
        """A full deck (card_id -> count) GUARANTEED to contain `revealed`, for MCTS
        determinization (search_begin rejects a deck that lacks a revealed card). Prefers the
        most-likely real corpus deck when it's consistent; otherwise keeps the revealed cards
        and fills the rest with the most expected cards (respecting the 4-copy limit, energy
        excepted). Returns None only if it can't reach deck_size (caller falls back).

        Callers (checked 2026-08-05): src/agents/mcts_agent.py and experiments' search
        agents -- i.e. the OLDER MCTS line, plus train_ppo's --search-gen when enabled.
        NOT the current line: training is pure PPO (--search-gen off) and the shipped v6
        bundle does not even ship this module -- its turn_search.py builds its own
        deliberately "engine-legal, not accurate" opponent side, because on an
        our-turn-only search their pieces never move.

        Given that, DON'T relax this to guarantee a deck (e.g. padding with basic energy).
        Its remaining callers SIMULATE the opponent playing the deck, where plausibility is
        the whole point; a padded world gives nonsense rollouts. Anything that only needs
        LEGALITY -- a 1-ply probe reading the visible board, or turn_search's
        determinize() -- should build its own filler, as turn_search already does.

        Measured 2026-08-05: returns a deck 100% of the time (121/121) when OpponentTracker
        is fed only its OWN seat's observations; earlier reports of ~46% None came from a
        tracker fed both seats, producing impossible 87-95 card reveal sets."""
        seen = _as_counter(revealed)
        best = self.most_likely(revealed)
        if best is not None and _contains(best, seen):
            return best
        deck = Counter(seen)
        total = sum(deck.values())
        for card_id, _ in sorted(self.card_probabilities(revealed).items(), key=lambda item: -item[1]):
            if total >= deck_size:
                break
            limit = deck_size if (get_card(card_id) or {}).get("cardType") == _BASIC_ENERGY else 4
            add = min(limit - deck.get(card_id, 0), deck_size - total)
            if add > 0:
                deck[card_id] += add
                total += add
        return dict(deck) if total == deck_size else None

    def card_probabilities(self, revealed):
        """Per-card marginal belief: expected number of copies in the opponent's deck,
        averaged over the deck distribution -> {card_id: expected_count}. (A 4-of in a
        confidently-identified deck approaches 4.0; effectively a soft probability x copies.)"""
        expected = defaultdict(float)
        for deck, prob in self._distribution(_as_counter(revealed)):
            for card_id, count in deck["cards"].items():
                expected[card_id] += prob * count
        return dict(expected)

    def presence_probabilities(self, revealed):
        """Per-card percent chance: probability the opponent's deck contains AT LEAST ONE
        copy, marginalised over the deck distribution -> {card_id: probability in [0, 1]}.
        (`card_probabilities` is the richer expected-COUNT view; this is the literal
        'what are the odds they run card X'.)"""
        probability = defaultdict(float)
        for deck, prob in self._distribution(_as_counter(revealed)):
            for card_id in deck["cards"]:
                probability[card_id] += prob
        return dict(probability)

    def expected_unseen(self, revealed):
        """Per-card expected count still HIDDEN = deck belief minus what's revealed ->
        {card_id: expected_hidden_count} for cards with >0 expected hidden copies. This is
        the 'what's left in their deck/hand/prizes' prediction."""
        seen = _as_counter(revealed)
        unseen = {card: count - seen.get(card, 0)
                  for card, count in self.card_probabilities(revealed).items()}
        return {card: round(count, 3) for card, count in unseen.items() if count > 0.01}


def revealed_opponent_cards(player_state):
    """Card ids currently VISIBLE in a player's state: active/bench Pokemon plus their
    attached energy/tools/pre-evolutions, and the discard pile (hand and prizes are hidden).

    This is one snapshot. To track everything the opponent has revealed across a game, the
    caller should accumulate -- e.g. keep a {serial: card_id} of every opponent card ever
    seen (deduped by serial) and feed Counter(those ids) to `beliefs`."""
    revealed = []
    for pokemon in (player_state.get("active") or []) + player_state.get("bench", []):
        if pokemon is None:
            continue
        revealed.append(pokemon["id"])
        for card in (pokemon.get("energyCards", []) + pokemon.get("tools", [])
                     + pokemon.get("preEvolution", [])):
            revealed.append(card["id"])
    for card in player_state.get("discard", []):
        revealed.append(card["id"])
    return revealed
