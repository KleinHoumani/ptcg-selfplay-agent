"""Serial-level card knowledge: prize deduction + inference for BOTH sides (2026-07-11).

Engine facts this is built on (probe-verified, 15 games): every physical card has a stable
serial that never changes identity, deck views / looking expose serials, and shuffles do NOT
re-randomize them. Prizes are six specific physical cards fixed at game start. Therefore:

  OWN side (decklist known exactly):
    - every serial ever sighted is NOT prized, permanently;
    - the never-seen remainder's id-multiset is known (decklist minus seen ids);
    - by exchangeability the prize posterior is "a uniform 6-subset of the never-seen
      multiset" (plain hypergeometric) -- the pro's draw-frequency reasoning collapses into
      this counting: seeing more copies anywhere shrinks what can be prized;
    - after enough sightings (e.g. any full-deck search view) never-seen == prizes left and
      the prized cards are EXACT.

  OPPONENT side (decklist is a belief):
    - the same seen-serial set retains TRANSIENT reveals (cards logged leaving their hand,
      reveal effects) that the public-zone OpponentTracker forgets;
    - any of their cards ever sighted is not prized -- a hard constraint for determinization
      (today's sampler can wrongly prize a card we watched in their hand).

Contract: call `update(observation)` EXACTLY ONCE per received observation (logs are
incremental). Sightings never expire; zone knowledge is recomputed per observation.
"""

from collections import Counter

# Inlined log types (cg.api values; module stays DLL-free like the rest of src/decks).
_LOG_SHUFFLE = 0
_LOG_DRAW = 4
_LOG_MOVE_CARD = 6
_LOG_PLAY, _LOG_ATTACH, _LOG_EVOLVE, _LOG_DEVOLVE = 10, 11, 12, 13

# Undocumented-but-logged AreaType: the engine collapses a bottom placement to Deck for the
# STATE (CardMove.h) but logs the raw destination, so a face-up "to the bottom" move arrives
# as MOVE_CARD toArea=14 WITH identity (verified live: Drakloak's look-then-bottom).
_AREA_DECK_BOTTOM = 14


def _board_cards(player_state):
    """(serial, card_id) for everything visible on one side's board + discard."""
    for pokemon in (player_state.get("active") or []) + (player_state.get("bench") or []):
        if pokemon is None:
            continue
        yield pokemon["serial"], pokemon["id"]
        for card in ((pokemon.get("energyCards") or []) + (pokemon.get("tools") or [])
                     + (pokemon.get("preEvolution") or [])):
            yield card["serial"], card["id"]
    for card in (player_state.get("discard") or []):
        yield card["serial"], card["id"]


class CardKnowledge:
    """Per-game seen-serial knowledge for both sides. `my_deck_counts` = {card_id: copies}
    of OUR 60-card decklist (exact); the opponent's side works without a decklist."""

    def __init__(self, my_deck_counts):
        self.my_deck_counts = Counter(my_deck_counts)
        self.my_seen = {}          # serial -> card_id, every one of OUR cards ever sighted
        self.opponent_seen = {}    # serial -> card_id, every one of THEIRS ever sighted
        self._my_prizes_left = 6
        self._observed_any = False
        # Prizes are dealt AFTER the setup draw / mulligan reveals, so a serial sighted in
        # that window is not evidence of anything -- it can still end up prized. Recording
        # it anyway is what made `never_seen_total < prizes_left` (arithmetically impossible)
        # true on 16% of decisions and pushed prize_certainty to 2.0 (STATE audit M1).
        # Latched, because prizes_left also returns to 0 when the game is WON.
        self._prizes_dealt = False
        # Known deck POSITIONS (identity knowledge never expires; position knowledge does):
        # serials we know sit on top (first = next draw) / at the bottom of MY deck. Filled
        # by the note_* hooks (the agent knows its own to-top/to-bottom selections), wiped by
        # our SHUFFLE logs, self-correcting against DRAW logs.
        self.known_top = []
        self.known_bottom = []

    # ------------------------------------------------------------------ #
    def update(self, observation):
        current = observation["current"]
        me_index = current["yourIndex"]
        me = current["players"][me_index]
        opponent = current["players"][1 - me_index]
        self._my_prizes_left = len(me.get("prize") or [])
        if self._my_prizes_left > 0:
            self._prizes_dealt = True
        elif not self._prizes_dealt:
            # Setup / mulligan window: the prize pile does not exist yet, so nothing seen
            # here constrains it. Skip every sighting (both sides -- both piles are dealt
            # at the same moment) and every position hook; the very next observation after
            # the deal re-reads the whole visible board and hand anyway.
            return self
        self._observed_any = True

        # Own zones: hand + board + discard.
        for card in (me.get("hand") or []):
            self.my_seen[card["serial"]] = card["id"]
        for serial, card_id in _board_cards(me):
            self.my_seen[serial] = card_id
        # Opponent public zones (board + discard).
        for serial, card_id in _board_cards(opponent):
            self.opponent_seen[serial] = card_id

        # Cards being looked at / offered from a deck view (both carry owner playerIndex).
        select = observation.get("select") or {}
        for card in ((current.get("looking") or []) + (select.get("deck") or [])):
            if card is None:
                continue
            self._sight(card.get("playerIndex"), me_index, card.get("serial"), card.get("id"))

        # A card played on/attached to a side we might not re-derive; the Stadium.
        for stadium in (current.get("stadium") or []):
            if stadium is not None:
                self._sight(stadium.get("playerIndex"), me_index,
                            stadium.get("serial"), stadium.get("id"))

        # Logs: face-up card movements carry (serial, cardId) for BOTH sides -- this is what
        # retains transient reveals (their Ultra Ball discards, revealed hands, our draws).
        for log in (observation.get("logs") or []):
            log_type = log.get("type")
            if log_type in (_LOG_DRAW, _LOG_MOVE_CARD, _LOG_PLAY, _LOG_ATTACH,
                            _LOG_EVOLVE, _LOG_DEVOLVE):
                serial = log.get("serial")
                card_id = log.get("cardId", 0)
                if serial is not None and card_id and card_id > 0:
                    self._sight(log.get("playerIndex"), me_index, serial, card_id)
            # Position-knowledge maintenance (mine only).
            if log.get("playerIndex") == me_index:
                if log_type == _LOG_SHUFFLE:
                    self.known_top.clear()
                    self.known_bottom.clear()
                elif log_type == _LOG_DRAW and self.known_top:
                    if log.get("serial") == self.known_top[0]:
                        self.known_top.pop(0)            # predicted draw confirmed
                    else:
                        self.known_top.clear()           # order was wrong -> stop trusting it
                elif log_type == _LOG_MOVE_CARD \
                        and log.get("toArea") == _AREA_DECK_BOTTOM:
                    serial = log.get("serial")
                    if serial is not None:
                        self.note_sent_to_bottom([serial])

        # A bottomed card that has resurfaced is no longer at the bottom. Only zones that
        # are definitely OUT of the deck count (`looking` does not -- that is where cards
        # sit on their way TO the bottom).
        if self.known_bottom:
            out_of_deck = {card["serial"] for card in (me.get("hand") or [])}
            out_of_deck.update(serial for serial, _card_id in _board_cards(me))
            if out_of_deck:
                self.known_bottom = [serial for serial in self.known_bottom
                                     if serial not in out_of_deck]
        return self

    # ---- deck-position hooks (the agent knows its own ordering choices) --- #
    def note_sent_to_bottom(self, serials):
        """Record cards WE placed on the bottom of OUR deck (e.g. a Drakloak-style
        look-then-bottom). Order given = order placed (last call ends up lowest)."""
        for serial in serials:
            if serial in self.known_top:
                self.known_top.remove(serial)
            if serial not in self.known_bottom:
                self.known_bottom.append(serial)

    def note_sent_to_top(self, serials):
        """Record cards WE placed on top of OUR deck (first = the very top / next draw)."""
        for serial in reversed(list(serials)):
            if serial in self.known_bottom:
                self.known_bottom.remove(serial)
            if serial in self.known_top:
                self.known_top.remove(serial)
            self.known_top.insert(0, serial)

    def known_top_ids(self):
        """Ordered card ids we know are about to be drawn (identity via seen serials)."""
        return [self.my_seen[serial] for serial in self.known_top if serial in self.my_seen]

    def known_bottom_ids(self):
        return [self.my_seen[serial] for serial in self.known_bottom
                if serial in self.my_seen]

    def _sight(self, player_index, me_index, serial, card_id):
        if serial is None or card_id is None or card_id <= 0:
            return
        if player_index == me_index:
            self.my_seen[serial] = card_id
        elif player_index == 1 - me_index:
            self.opponent_seen[serial] = card_id

    # ---- own side ------------------------------------------------------ #
    def never_seen_counts(self):
        """{card_id: copies of MY decklist never sighted anywhere}. Prizes are a uniform
        subset of exactly this multiset."""
        remaining = self.my_deck_counts - Counter(self.my_seen.values())
        return +remaining                                   # drop non-positive entries

    def prizes_left(self):
        return self._my_prizes_left

    def prizes_exact(self):
        """The exact multiset of my remaining prized cards, or None while uncertainty
        remains. Exact when every non-prized card has been sighted at least once (a full
        deck view usually gets there)."""
        never_seen = self.never_seen_counts()
        total = sum(never_seen.values())
        if self._observed_any and total == self._my_prizes_left:
            return never_seen
        return None

    def prize_belief(self):
        """{card_id: expected copies among my remaining prizes} under the uniform-subset
        (hypergeometric) posterior over the never-seen multiset."""
        never_seen = self.never_seen_counts()
        total = sum(never_seen.values())
        if total == 0 or self._my_prizes_left == 0:
            return {}
        scale = min(1.0, self._my_prizes_left / total)
        return {card_id: copies * scale for card_id, copies in never_seen.items()}

    def prize_certainty(self):
        """0..1: how localized the prize knowledge is (1.0 = exact). Clamped: the ratio can
        only exceed 1 when the deduction is corrupt, and the global vector documents this
        column as a [0,1] confidence (STATE audit M1 measured 2.0 on 16% of decisions)."""
        never_seen_total = sum(self.never_seen_counts().values())
        if never_seen_total == 0:
            return 1.0
        return min(1.0, self._my_prizes_left / never_seen_total)

    # ---- opponent side --------------------------------------------------- #
    def opponent_seen_counts(self):
        """{card_id: distinct physical copies of THEIRS ever sighted} -- a superset of
        OpponentTracker's public-zone counts (retains transient reveals)."""
        return Counter(self.opponent_seen.values())

    def opponent_not_prized_counts(self, observation):
        """{card_id: copies of theirs known NOT prized but currently HIDDEN (hand/deck)} --
        seen-ever minus currently-visible. Determinization should keep these ids out of
        their sampled prizes."""
        current = observation["current"]
        opponent_index = 1 - current["yourIndex"]
        opponent = current["players"][opponent_index]
        visible = Counter(card_id for _serial, card_id in _board_cards(opponent))
        # Their Stadium in play and their cards shown in `looking` are ON THE TABLE, not
        # hidden -- _board_cards covers neither zone, so without this each counted as a
        # phantom hidden copy (false "they hold another one" fact).
        for card in ((current.get("stadium") or []) + (current.get("looking") or [])):
            if card is not None and card.get("playerIndex") == opponent_index \
                    and (card.get("id") or 0) > 0:
                visible[card["id"]] += 1
        hidden_but_seen = Counter(self.opponent_seen.values()) - visible
        return +hidden_but_seen


# --------------------------------------------------------------------------- #
# Self-test: consistency against the engine. Prizes picked up during play are ground
# truth -- every one must be inside the tracker's posterior support, and NEVER contradict
# an exact deduction. Also reports when exactness is reached.
#   ./.venv/Scripts/python.exe -m src.decks.card_knowledge
# --------------------------------------------------------------------------- #

def _self_test():
    import random
    import sys
    from collections import Counter as C

    from cg import game

    deck = [int(line) for line in
            open("submissions/submission_alakazam/deck.csv").read().split() if line.strip()]
    deck_counts = C(deck)
    rng = random.Random(1)

    def random_legal(observation):
        select = observation["select"]
        count = len(select["option"])
        take = max(min(select["maxCount"], count), select["minCount"])
        return sorted(rng.sample(range(count), take)) if count else []

    AREA_PRIZE, AREA_HAND = 6, 2
    games = 20
    pickups = support_hits = exact_hits = exact_contradictions = 0
    # Serial-level checks (the id-level ones above pass by luck whenever the deck runs
    # duplicates -- exactly why the setup-sighting bug survived, STATE audit M1):
    #   serial_seen_before_pickup: a prized SERIAL that the tracker had already recorded as
    #     sighted, i.e. it claimed "this physical card is not prized" and was wrong;
    #   impossible_states: decisions where never_seen_total < prizes_left, which cannot
    #     happen if every recorded sighting really is a non-prized card;
    #   certainty_out_of_range: prize_certainty() outside [0, 1].
    serial_seen_before_pickup = impossible_states = certainty_out_of_range = 0
    decisions = 0
    exact_reached_turns = []

    for game_index in range(games):
        knowledge = CardKnowledge(deck_counts)
        exact_at = None
        observation, _ = game.battle_start(list(deck), list(deck), seed=900 + game_index)
        moves = 0
        while observation["current"]["result"] == -1 and moves < 400:
            if observation["current"]["yourIndex"] == 0:
                # Check prize pickups BEFORE updating (the pickup log tests the PRIOR belief).
                for log in (observation.get("logs") or []):
                    if log.get("type") == _LOG_MOVE_CARD and log.get("playerIndex") == 0 \
                            and log.get("fromArea") == AREA_PRIZE \
                            and log.get("toArea") == AREA_HAND and log.get("cardId", 0) > 0:
                        pickups += 1
                        exact = knowledge.prizes_exact()
                        belief = knowledge.prize_belief()
                        if belief.get(log["cardId"], 0) > 0:
                            support_hits += 1
                        if exact is not None:
                            if exact.get(log["cardId"], 0) > 0:
                                exact_hits += 1
                            else:
                                exact_contradictions += 1
                        if log.get("serial") is not None \
                                and log["serial"] in knowledge.my_seen:
                            serial_seen_before_pickup += 1
                knowledge.update(observation)
                decisions += 1
                never_seen_total = sum(knowledge.never_seen_counts().values())
                if never_seen_total < knowledge.prizes_left():
                    impossible_states += 1
                if not 0.0 <= knowledge.prize_certainty() <= 1.0:
                    certainty_out_of_range += 1
                if exact_at is None and knowledge.prizes_exact() is not None:
                    exact_at = observation["current"]["turn"]
            select = observation.get("select")
            observation = game.battle_select(random_legal(observation) if select else [])
            moves += 1
        game.battle_finish()
        if exact_at is not None:
            exact_reached_turns.append(exact_at)

    print(f"games {games} | prize pickups checked {pickups} | "
          f"in posterior support {support_hits}/{pickups}")
    print(f"exact-mode pickups {exact_hits} confirmed, {exact_contradictions} CONTRADICTED")
    print(f"SERIAL check: {serial_seen_before_pickup}/{pickups} prized serials were already "
          f"recorded as sighted (must be 0)")
    print(f"decisions {decisions} | impossible never_seen<prizes_left: {impossible_states} "
          f"| prize_certainty out of [0,1]: {certainty_out_of_range}")
    print(f"exact prize knowledge reached in {len(exact_reached_turns)}/{games} games "
          f"(median turn {sorted(exact_reached_turns)[len(exact_reached_turns) // 2] if exact_reached_turns else '-'})")
    if exact_contradictions or support_hits < pickups or serial_seen_before_pickup \
            or impossible_states or certainty_out_of_range:
        print("FAIL: tracker contradicted ground truth", file=sys.stderr)
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    _self_test()
