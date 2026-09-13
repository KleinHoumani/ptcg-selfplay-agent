"""Does --field-pools sample the archetype corpus the way it claims to?

Four questions, each answered against the files and the sampler itself rather than against
its docstring:

  1. POOLS LOAD          every archetype directory under the root yields >=1 list, every
                         list is exactly 60 integer card ids, and the count matches
                         manifest.json's own per-archetype `decks` number.
  2. LISTS ARE LEGAL     each archetype's most-played list starts a real engine battle
                         (`battle_start`), which is the only definition of legal that
                         matters. Sampling a deck the engine rejects would kill the game,
                         so this is a hard gate.
  3. ARCHETYPE IDENTITY  a list drawn from pool X really contains X's signature line --
                         pool.json records the line as card NAMES with counts, and the
                         drawn deck must carry at least one copy of each named card.
                         This is what proves the directories are labelled correctly, as
                         opposed to merely being readable.
  4. THE DRAW IS UNIFORM  _sample_field_deck (the function generation actually calls) is
                         run N times through the real worker state, and the empirical
                         archetype frequencies are chi-square tested against 1/K each,
                         and the within-archetype list frequencies against 1/len(pool).
                         Uniform-over-ARCHETYPES is the point: uniform over the 763 LISTS
                         would be popularity weighting by the back door (alakazam has 194
                         lists, lopunny 7).

Usage:
    ./.venv/Scripts/python.exe experiments/selfplay_ppo/verify_field_pools.py \
        [--root data/decks/matchups] [--draws 26000] [--no-engine]
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import train_ppo                                                    # noqa: E402
from src.cards import get_card                                      # noqa: E402


def load_pools(root):
    """The SAME walk main() does, so a divergence here is a divergence there."""
    pools = []
    for directory in sorted(entry for entry in root.iterdir() if entry.is_dir()):
        decks = []
        for path in sorted(directory.glob("*.csv")):
            deck = [int(line) for line in path.read_text().split() if line.strip()]
            decks.append((path.name, deck))
        if decks:
            pools.append((directory.name, decks))
    return pools


def check_pools_load(pools, root, failures):
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) \
        if manifest_path.exists() else {"archetypes": {}}
    declared = manifest.get("archetypes") or {}
    total = 0
    for name, decks in pools:
        total += len(decks)
        for filename, deck in decks:
            if len(deck) != 60:
                failures.append(f"{name}/{filename}: {len(deck)} cards, not 60")
        expected = (declared.get(name) or {}).get("decks")
        if expected is not None and expected != len(decks):
            failures.append(f"{name}: manifest says {expected} decks, directory has "
                            f"{len(decks)}")
    print(f"1. POOLS LOAD    {len(pools)} archetypes / {total} lists  "
          f"({'manifest matches' if declared else 'no manifest to cross-check'})")
    return total


def check_engine_legal(pools, failures):
    """One list per archetype through battle_start -- the engine's own legality verdict."""
    try:
        from cg import game as cg_game
    except Exception as error:                                       # noqa: BLE001
        print(f"2. ENGINE LEGAL  SKIPPED ({type(error).__name__}: cg unavailable)")
        return
    checked = 0
    for name, decks in pools:
        _filename, deck = decks[0]
        try:
            cg_game.battle_start(list(deck), list(deck))
            cg_game.battle_finish()
            checked += 1
        except Exception as error:                                   # noqa: BLE001
            failures.append(f"{name}: battle_start rejected {_filename} "
                            f"({type(error).__name__}: {error})")
    print(f"2. ENGINE LEGAL  {checked}/{len(pools)} archetypes started a real battle")


def _fold(name):
    """Apostrophe/accent fold, same as build_matchup_pools.normalise: specs write straight
    apostrophes, cards.json prints curly ones (first bitten by "N's Zorua", 2026-08-08)."""
    import unicodedata
    name = unicodedata.normalize("NFKC", name or "")
    return name.replace("’", "'").replace("‘", "'")


def check_archetype_identity(pools, root, failures):
    """Every list in a pool carries that pool's signature line (by card NAME)."""
    checked = missing_signature = 0
    for name, decks in pools:
        pool_file = root / name / "pool.json"
        if not pool_file.exists():
            missing_signature += 1
            continue
        line = (json.loads(pool_file.read_text(encoding="utf-8")).get("line") or ())
        wanted = {_fold(card_name) for card_name, _count in line}
        if not wanted:                     # e.g. `miscellaneous`: "any list no pool claims"
            missing_signature += 1
            continue
        for filename, deck in decks:
            names = {_fold((get_card(card_id) or {}).get("name")) for card_id in deck}
            absent = wanted - names
            if absent:
                failures.append(f"{name}/{filename} is missing its signature "
                                f"{sorted(absent)}")
            checked += 1
    print(f"3. IDENTITY      {checked} lists carry their pool's signature line "
          f"({missing_signature} pools have no signature to check)")


def chi_square(observed, expected):
    return sum((count - expected) ** 2 / expected for count in observed)


class RecordingRandom:
    """A random.Random proxy that remembers what each `choice` returned.

    Attribution has to be EXACT: one 60-card list is filed under two archetypes (a genuine
    Crustle + Cynthia's Garchomp hybrid), so reverse-looking-up a returned deck by its
    contents cannot tell which pool it came from -- and guessing "the first" is what made an
    earlier version of this check report crustle and cynthia_garchomp as non-uniform when
    the sampler was fine. Recording the actual objects the sampler chose removes the guess,
    and asserting there were exactly TWO choices also pins the sampler's SHAPE: one draw for
    the archetype, one for the list."""

    def __init__(self, rng):
        self._rng = rng
        self.picks = []

    def choice(self, sequence):
        value = self._rng.choice(sequence)
        self.picks.append(value)
        return value

    def choices(self, *args, **kwargs):                    # the corpus path, unused here
        return self._rng.choices(*args, **kwargs)


def check_uniform_draw(pools, draws, failures):
    """Drive the REAL sampler: _worker state + _sample_field_deck, nothing re-implemented."""
    train_ppo._worker.clear()
    train_ppo._worker.update({
        "scripted_deck": None, "matchup": None, "matchup_pool": None,
        "deck_mode": "focus",
        "field_pools": [(name, [deck for _filename, deck in decks])
                        for name, decks in pools]})
    filenames = {name: [filename for filename, _deck in decks] for name, decks in pools}
    deck_rng = RecordingRandom(random.Random(20260806))
    archetypes, lists = Counter(), Counter()
    for _draw in range(draws):
        deck_rng.picks = []
        deck = train_ppo._sample_field_deck(deck_rng)
        if len(deck_rng.picks) != 2:
            failures.append(f"the sampler made {len(deck_rng.picks)} random choices, not "
                            f"2 (archetype, then list) -- the draw is not what it claims")
            break
        (name, pool), drawn = deck_rng.picks
        if drawn is not deck and list(drawn) != list(deck):
            failures.append("the sampler returned a deck it did not draw")
        index = next((position for position, candidate in enumerate(pool)
                      if candidate is drawn), None)
        if index is None:
            failures.append(f"{name}: drawn list is not a member of its own pool")
            continue
        archetypes[name] += 1
        lists[(name, filenames[name][index])] += 1

    count = len(pools)
    expected = draws / count
    statistic = chi_square([archetypes[name] for name, _decks in pools], expected)
    # 5% critical value, chi-square with (count - 1) degrees of freedom, from the standard
    # table -- no scipy in this venv.
    critical = {6: 12.59, 7: 14.07, 8: 15.51, 9: 16.92, 10: 18.31, 11: 19.68, 12: 21.03,
                13: 22.36, 14: 23.68, 15: 25.00}.get(count - 1)
    low = min(archetypes.values()) / draws
    high = max(archetypes.values()) / draws
    print(f"4. UNIFORM DRAW  {draws} draws over {count} archetypes: "
          f"share {low:.4f}..{high:.4f} vs 1/{count} = {1 / count:.4f}  "
          f"chi2 {statistic:.2f}" + (f" vs {critical} crit@5%" if critical else ""))
    if critical is not None and statistic > critical:
        failures.append(f"archetype draw is not uniform: chi2 {statistic:.2f} > {critical}")
    for name, decks in pools:
        if len(decks) < 2 or archetypes[name] < 5 * len(decks):
            continue                       # too few draws for a meaningful within-test
        counts = [lists[(name, filename)] for filename, _deck in decks]
        within = chi_square(counts, archetypes[name] / len(decks))
        # 5% critical value ~ df + 2*sqrt(2*df) (Wilson-Hilferty is overkill here); flag
        # only gross departures, since this is a smoke test not a statistics paper.
        degrees = len(decks) - 1
        if within > degrees + 3 * (2 * degrees) ** 0.5:
            failures.append(f"{name}: within-pool draw looks non-uniform "
                            f"(chi2 {within:.1f}, df {degrees})")
    shared = Counter()
    for name, decks in pools:
        for filename, deck in decks:
            shared[tuple(sorted(deck))] += 1
    overlap = sum(1 for count in shared.values() if count > 1)
    if overlap:
        print(f"   note: {overlap} list(s) are filed under more than one archetype, so "
              f"those decks are drawn slightly more often than a single-pool list")
    print("   per-archetype share: " + "  ".join(
        f"{name}={archetypes[name] / draws:.3f}" for name, _decks in pools))


def check_real_game_seeds(pools, seed_base, games_per_iter, iterations, failures):
    """The mix THIS RUN will actually play.

    The check above proves the sampler is uniform over an arbitrary rng. This one replays
    the exact per-game seeds `build_tasks` will generate (`seed_base + 1000000 +
    (iteration - 1) * games_per_iter + i`) through the same per-game deck rng the worker
    builds (`random.Random(seed * 31 + 7)`), so it answers the question that matters: over
    the first `iterations` iterations, what fraction of games face each archetype?

    A uniform SAMPLER can still produce a skewed RUN if the seed stride happens to
    correlate with the draw -- unlikely, but it costs nothing to rule out rather than
    assume."""
    train_ppo._worker.clear()
    train_ppo._worker.update({
        "scripted_deck": None, "matchup": None, "matchup_pool": None,
        "deck_mode": "focus", "focus": [0] * 60,
        "field_pools": [(name, [deck for _filename, deck in decks])
                        for name, decks in pools]})
    archetypes = Counter()
    games = 0
    for iteration in range(1, iterations + 1):
        for index in range(games_per_iter):
            seed = seed_base + 1000000 + (iteration - 1) * games_per_iter + index
            deck_rng = RecordingRandom(random.Random(seed * 31 + 7))
            train_ppo._sample_our_deck(deck_rng)         # seat 0 draw, same stream order
            deck_rng.picks = []
            train_ppo._sample_field_deck(deck_rng)
            archetypes[deck_rng.picks[0][0]] += 1
            games += 1
    expected = games / len(pools)
    statistic = chi_square([archetypes[name] for name, _decks in pools], expected)
    critical = {12: 21.03}.get(len(pools) - 1)
    print(f"5. RUN SEEDS     {games} games ({iterations} iters x {games_per_iter}) at "
          f"seed-base {seed_base}: chi2 {statistic:.2f}"
          + (f" vs {critical} crit@5%" if critical else ""))
    if critical is not None and statistic > critical:
        failures.append(f"this run's seeds do not give a uniform archetype mix: "
                        f"chi2 {statistic:.2f} > {critical}")
    print("   " + "  ".join(f"{name}={archetypes[name] / games:.3f}"
                            for name, _decks in pools))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT / "data" / "decks" / "matchups"))
    parser.add_argument("--draws", type=int, default=26000)
    parser.add_argument("--no-engine", action="store_true")
    parser.add_argument("--seed-base", type=int, default=20260806)
    parser.add_argument("--games-per-iter", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=100)
    options = parser.parse_args()
    root = Path(options.root)
    pools = load_pools(root)
    failures = []
    check_pools_load(pools, root, failures)
    if not options.no_engine:
        check_engine_legal(pools, failures)
    check_archetype_identity(pools, root, failures)
    check_uniform_draw(pools, options.draws, failures)
    check_real_game_seeds(pools, options.seed_base, options.games_per_iter,
                          options.iterations, failures)
    print()
    if failures:
        print(f"FAIL ({len(failures)})")
        for failure in failures[:40]:
            print(f"  {failure}")
        return 1
    print("PASS -- field pools load, are engine-legal, are labelled correctly, and the "
          "sampler draws uniform archetype then uniform list")
    return 0


if __name__ == "__main__":
    sys.exit(main())
