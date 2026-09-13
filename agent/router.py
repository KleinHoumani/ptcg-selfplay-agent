"""Archetype router (bundle-local module, shipped by scripts/build_router_bundle.py).

Matches every card the OPPONENT has visibly revealed against the archetype pools'
decklists and produces a posterior over archetypes (Bayes: ladder-share prior x
containment likelihood with a soft penalty per unexplained copy -- the standard
naive-Bayes-over-lists construction, measured 103/103 correct locks at tau 0.9 on
2026-08-08).

Model choice is STATELESS with a hysteresis band (owner spec 2026-08-08: switching more
than once is fine -- evidence only accumulates, so a dethroned archetype can never
recover and oscillation is structurally impossible):

    desired = specialist[top]  if P(top) >= tau_enter
              current          if current == specialist[top] and P(top) >= tau_exit
              base             otherwise

The router NEVER answers the engine and never touches game state: it only says which
model should score the options. All logging is the caller's job (main.py prints
"[router]" lines to stderr so switches are visible in Kaggle replay logs).
"""
from collections import Counter


class ArchetypeRouter:
    def __init__(self, data):
        """data: the bundle's router_data.json --
        {"tau_enter": 0.8, "tau_exit": 0.7, "penalty": 0.05,
         "archetypes": {name: {"prior": float, "lists": [[card_id, ...], ...]}},
         "specialists": {name: "models/<name>.pt"}}"""
        self.tau_enter = float(data.get("tau_enter", 0.8))
        self.tau_exit = float(data.get("tau_exit", 0.7))
        self.penalty = float(data.get("penalty", 1e-6))
        self.specialists = dict(data.get("specialists", {}))
        # Print-blindness: every card id folds to its name's canonical id, both in the
        # lists (done at build) and in live reveals (done in update()). A different PRINT
        # of Abra must still read as Abra.
        self.canonical = {int(card_id): canon
                          for card_id, canon in (data.get("canonical") or {}).items()}
        self.lists = []                    # (archetype, Counter(card_id -> copies), weight)
        for name, entry in data["archetypes"].items():
            lists = entry.get("lists") or []
            if not lists:
                continue
            prior = float(entry.get("prior", 1.0))
            # Per-list ladder mass when the build supplies it (games-weighted, owner spec
            # 2026-08-08); uniform split otherwise. Either way the archetype's lists sum
            # to its prior.
            shares = entry.get("weights")
            if not shares or len(shares) != len(lists):
                shares = [1.0 / len(lists)] * len(lists)
            for ids, share in zip(lists, shares):
                self.lists.append((name, Counter(ids), prior * float(share)))
        self.archetypes = sorted({name for name, _deck, _w in self.lists})
        self._seen = Counter()             # opponent card_id -> copies revealed
        self._posterior = None             # recomputed only when _seen changes

    # -- observation -> revealed opponent cards ----------------------------------------- #

    @staticmethod
    def _visible_ids(player):
        """Every opponent card id that is legitimately face-up: active + bench stacks
        (with attached energy, tools and pre-evolutions) and the discard pile. The same
        zones the search's determinize() reads."""
        ids = []
        for pokemon in (player.get("active") or []) + (player.get("bench") or []):
            if pokemon is None:
                continue
            ids.append(pokemon["id"])
            for card in ((pokemon.get("energyCards") or []) + (pokemon.get("tools") or [])
                         + (pokemon.get("preEvolution") or [])):
                ids.append(card["id"])
        for card in (player.get("discard") or []):
            ids.append(card["id"])
        return ids

    def update(self, observation_dict):
        """Refresh the seen-set from an observation. Cheap no-op when nothing new."""
        current = observation_dict.get("current")
        if not current:
            return
        me = current.get("yourIndex")
        players = current.get("players") or []
        if me is None or len(players) != 2:
            return
        seen = Counter(self.canonical.get(card_id, card_id)
                       for card_id in self._visible_ids(players[1 - me]))
        # Monotone merge: zones can shrink (cards shuffled back in), but a reveal is
        # knowledge we keep -- once seen, always seen.
        for card_id, count in seen.items():
            if count > self._seen[card_id]:
                self._seen[card_id] = count
        if sum(self._seen.values()) and (self._posterior is None or seen):
            self._posterior = self._score()

    def _score(self):
        mass = {name: 0.0 for name in self.archetypes}
        seen = self._seen
        for name, deck, weight in self.lists:
            unexplained = 0
            for card_id, count in seen.items():
                over = count - deck[card_id]
                if over > 0:
                    unexplained += over
            mass[name] += weight * (self.penalty ** unexplained)
        total = sum(mass.values())
        if total <= 0:
            return None
        return {name: value / total for name, value in mass.items()}

    # -- the routing rule ---------------------------------------------------------------- #

    def choose(self, active_name):
        """(desired_model_name, top_archetype, top_probability). `active_name` is the
        name currently serving ("base" or an archetype)."""
        if not self._posterior:
            return "base", None, 0.0
        top = max(self._posterior, key=self._posterior.get)
        probability = self._posterior[top]
        if top in self.specialists:
            if probability >= self.tau_enter:
                return top, top, probability
            if active_name == top and probability >= self.tau_exit:
                return top, top, probability
        return "base", top, probability

    @property
    def seen_count(self):
        return sum(self._seen.values())

    def reset(self):
        self._seen = Counter()
        self._posterior = None
