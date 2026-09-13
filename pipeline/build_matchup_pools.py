"""Split the ladder deck corpus into per-archetype decklist POOLS for matchup fine-tuning.

  ./.venv/Scripts/python.exe scripts/build_matchup_pools.py
  ./.venv/Scripts/python.exe scripts/build_matchup_pools.py --archetype alakazam
  ./.venv/Scripts/python.exe scripts/build_matchup_pools.py --min-games-raw 5
  ./.venv/Scripts/python.exe scripts/build_matchup_pools.py --no-validate

Each archetype is defined by the owner's EVOLUTION-LINE requirement (a "3-3-2 Alakazam
line" = at least 3 Abra, 3 Kadabra, 2 Alakazam) and/or a set of required cards. Every
corpus list that clears the bar joins that archetype's pool, so a matchup fine-tune trains
against the real SPREAD of lists people ladder with rather than one fixed 60.

Counting is by card NAME, aggregated over printings: the engine has several prints of the
same Pokemon (two Abra, three Riolu, four Applin...) and `evolvesFrom` is a name, so every
print of "Riolu" really is a legal basic for Mega Lucario ex. Requirements that name a
specific print (the basic box) are given as card ids instead.

Output, under data/decks/matchups/:
    manifest.json                     every archetype: spec, pool size, ladder share
    <archetype>/pool.json             per-deck popularity + line shape, ranked
    <archetype>/<archetype>_000.csv   60 card ids, one per line -- train_ppo deck format

`games` is the corpus's recency-decayed game count (7-day half-life) and `games_raw` the
true count; both are summed over the duplicate corpus entries that share a list. Use them
to weight sampling, or ignore them for a uniform pool -- popularity inside a pool is very
skewed (one Cynthia's Garchomp list holds 97% of that archetype's games), so replicating it
would defeat the point of having a pool.
"""

import argparse
import hashlib
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # the engine wrapper (cg/) lives at the repo root

CARDS_PATH = ROOT / "data" / "cards" / "cards.json"
CORPUS_PATH = ROOT / "data" / "decks" / "corpus.json"
OUT_DIR = ROOT / "data" / "decks" / "matchups"
# OWNER RULE (2026-08-09, made rebuild-proof 08-12): the eval-panel decks must always
# be present in the miscellaneous pool so training sees them. Source of truth lives
# here; build() injects each as panel_<name>.csv UNLESS the corpus already contains an
# identical list naturally (then the natural copy is the presence and no duplicate is
# written -- a duplicate would double that list's sampling weight).
PANEL_DIR = ROOT / "data" / "decks" / "panel_decks"

# archetype -> (line requirements, required card ids, human-readable line notation).
# `line` entries are (card name, minimum copies) in evolution order; a 0 minimum is kept
# in the table so the notation and the requirement stay side by side.
ARCHETYPES = {
    "grimmsnarl": {
        "line": [("Marnie's Impidimp", 3), ("Marnie's Morgrem", 0),
                 ("Marnie's Grimmsnarl ex", 2)],
        "cards": {},
        "notation": "3-0-2 Marnie's Grimmsnarl ex",
    },
    "alakazam": {
        "line": [("Abra", 3), ("Kadabra", 3), ("Alakazam", 2)],
        "cards": {},
        "notation": "3-3-2 Alakazam",
    },
    "dragapult": {
        "line": [("Dreepy", 3), ("Drakloak", 2), ("Dragapult ex", 2)],
        "cards": {},
        "notation": "3-2-2 Dragapult ex",
    },
    "crustle": {
        "line": [("Dwebble", 2), ("Crustle", 2)],
        "cards": {},
        "notation": "2-2 Crustle",
    },
    "basic_box": {
        # A named-print requirement, not a line: one copy each of the five box pieces.
        "line": [],
        "cards": {756: "Mega Kangaskhan ex", 272: "Lillie's Clefairy ex",
                  108: "Wellspring Mask Ogerpon ex", 96: "Teal Mask Ogerpon ex",
                  184: "Latias ex"},
        "notation": "1+ each of Mega Kangaskhan ex / Lillie's Clefairy ex / "
                    "Wellspring Ogerpon ex / Teal Mask Ogerpon ex / Latias ex",
    },
    "lucario": {
        "line": [("Riolu", 2), ("Mega Lucario ex", 2)],
        "cards": {},
        "notation": "2-2 Mega Lucario ex",
    },
    "team_rocket": {
        "line": [("Team Rocket's Tarountula", 3), ("Team Rocket's Spidops", 3)],
        "cards": {},
        "notation": "3-3 Team Rocket's Spidops",
    },
    "festival_lead": {
        "line": [("Grookey", 2), ("Thwackey", 2), ("Applin", 3), ("Dipplin", 3)],
        "cards": {},
        "notation": "2-2 Thwackey + 3-3 Dipplin",
    },
    "lopunny": {
        # "Dudunsparce" is the plain Stage 1 (id 66); Dudunsparce ex and Larry's
        # Dudunsparce ex are different cards on different lines and are NOT counted.
        # No Mega Lopunny list in the corpus runs either of them, so this costs nothing.
        "line": [("Buneary", 2), ("Mega Lopunny ex", 2),
                 ("Dunsparce", 2), ("Dudunsparce", 2)],
        "cards": {},
        "notation": "2-2 Mega Lopunny ex + 2-2 Dudunsparce",
    },
    "cynthia_garchomp": {
        "line": [("Cynthia's Gible", 3), ("Cynthia's Gabite", 2),
                 ("Cynthia's Garchomp ex", 2)],
        "cards": {},
        "notation": "3-2-2 Cynthia's Garchomp ex",
    },
    "starmie": {
        "line": [("Staryu", 2), ("Mega Starmie ex", 2)],
        "cards": {},
        "notation": "2-2 Mega Starmie ex",
    },
    "archaludon": {
        # "Archaludon ex" (190), NOT the plain "Archaludon" (170/840): they are separate
        # names on the same Duraludon basic, the owner's specialist runs 4-4 of the ex, and
        # the plain card clears the bar in exactly 1 corpus list versus 86 for the ex.
        "line": [("Duraludon", 3), ("Archaludon ex", 2)],
        "cards": {},
        "notation": "3-2 Archaludon ex",
    },
    # --- owner additions 2026-08-08 (specs given verbatim) ------------------------------ #
    "slowking": {
        # Plain "Kyurem" (the non-ex partner), NOT "Kyurem ex" / "Black Kyurem ex".
        "line": [("Slowpoke", 3), ("Slowking", 2), ("Kyurem", 1)],
        "cards": {},
        "notation": "3-2 Slowking + 1 Kyurem",
    },
    "ns_zoroark": {
        # cards.json spells these with a curly apostrophe; normalise() folds it.
        "line": [("N's Zorua", 3), ("N's Zoroark ex", 3), ("N's Zekrom", 1)],
        "cards": {},
        "notation": "3-3 N's Zoroark ex + 1 N's Zekrom",
    },
    "hydrapple": {
        # Plain "Meganium", NOT "Mega Meganium ex".
        #
        # OWNER CHANGE 2026-08-14: widened from
        #   [("Hydrapple ex", 1), ("Meganium", 1), ("Teal Mask Ogerpon ex", 2)]
        # to the Meganium line alone. The old spec was not restrictive -- it already caught
        # every Hydrapple ex list in the corpus (all 24 run 2/2/4, far above its 1/1/2 bar) --
        # but it MISSED the Meganium lists that skip Hydrapple ex entirely and play the same
        # Chikorita/Bayleef/Meganium engine behind 4 Teal Mask Ogerpon ex. Those are the same
        # deck to play against, so they belong in the same pool.
        # Measured: 11 -> 21 distinct lists, 1.55% -> 2.24% ladder share, the old 11 remain a
        # strict subset, and the widened set stays disjoint from every other archetype.
        # Name kept as "hydrapple" (owner): it is what every pool dir, seed offset, run name
        # and bundle already calls this archetype.
        "line": [("Meganium", 2)],
        "cards": {},
        "notation": "2+ Meganium",
    },
    # Catch-all, kept LAST: every list no other archetype claims (see `matches`). Its pool is
    # heterogeneous by construction -- it is the tail, not a deck -- which is exactly what
    # makes it useful as a sampling bucket: with it present the archetypes partition the
    # corpus and the shares sum to 1.0 instead of 95.5%.
    "miscellaneous": {
        "line": [],
        "cards": {},
        "catch_all": True,
        "notation": "any list no other archetype claims",
    },
}


def normalise(name):
    """Fold the curly apostrophe / accent variants so a spec name matches cards.json."""
    name = unicodedata.normalize("NFKC", name)
    return name.replace("’", "'").replace("‘", "'").replace("�", "'")


def load_cards():
    cards = json.loads(CARDS_PATH.read_text(encoding="utf-8"))
    ids_by_name = defaultdict(list)
    name_by_id = {}
    for card in cards:
        ids_by_name[normalise(card["name"])].append(card["cardId"])
        name_by_id[card["cardId"]] = card["name"]
    return ids_by_name, name_by_id


def load_unique_lists():
    """Corpus entries collapsed to unique 60-card multisets, popularity summed.

    2048 ladder entries carry only 762 distinct lists -- a popular list is re-uploaded by
    everyone who copied it -- so the pool must be over lists, not over entries."""
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    unique = {}
    for entry in corpus:
        counts = {int(card_id): n for card_id, n in entry["cards"].items()}
        key = tuple(sorted(counts.items()))
        deck = unique.setdefault(key, {"counts": counts, "games": 0.0, "games_raw": 0,
                                       "usernames": []})
        deck["games"] += entry["games"]
        deck["games_raw"] += entry["games_raw"]
        deck["usernames"].append(entry["username"])
    for deck in unique.values():
        deck["usernames"].sort()
    return list(unique.values())


def name_count(counts, ids_by_name, name):
    """Copies of `name` in a deck, summed over every printing of that name."""
    return sum(counts.get(card_id, 0) for card_id in ids_by_name[normalise(name)])


def matches(deck, spec, ids_by_name):
    if spec.get("catch_all"):
        # The COMPLEMENT of every real archetype, read off ARCHETYPES rather than off the
        # `--archetype` selection, so `--archetype miscellaneous` alone still means "the
        # ones nothing else claims" and not "everything".
        return not any(matches(deck, other, ids_by_name)
                       for other in ARCHETYPES.values() if not other.get("catch_all"))
    counts = deck["counts"]
    if any(name_count(counts, ids_by_name, name) < least
           for name, least in spec["line"]):
        return False
    return all(counts.get(card_id, 0) >= 1 for card_id in spec["cards"])


def deck_card_ids(deck):
    """The 60 card ids, ascending -- the order the hand-built deck.csv files use."""
    ids = []
    for card_id in sorted(deck["counts"]):
        ids.extend([card_id] * deck["counts"][card_id])
    return ids


def list_signature(card_ids):
    """Content identity for a 60-card list: file names get renumbered every rebuild,
    the sorted-ids hash does not."""
    return hashlib.sha1(",".join(str(i) for i in sorted(card_ids)).encode()).hexdigest()[:16]


# OWNER-EXCLUDED lists (2026-08-11): keyed by list_signature so rebuilds keep them out
# no matter how files are renumbered. The first entry is a Cynthia's Garchomp + Crustle
# hybrid that satisfied BOTH archetype lines and landed in two pools (was crustle_089 /
# cynthia_garchomp_027); the owner wants it in NEITHER pool.
EXCLUDED_LISTS = {
    "fa95ad4fb6e8b2a3": "Cynthia's Garchomp + Crustle hybrid "
                        "(was crustle_089 / cynthia_garchomp_027)",
}


def overlap_exclusions(decks, ids_by_name):
    """STANDING RULE (owner 2026-08-12): a list that satisfies two or more real
    archetype specs belongs to NEITHER pool -- hybrids would double-count in the
    partition and blur what a matchup fine-tune trains against. Returns
    {signature: reason}, printed by the caller."""
    real = {name: spec for name, spec in ARCHETYPES.items()
            if not spec.get("catch_all")}
    banned = {}
    for deck in decks:
        hits = [name for name, spec in real.items()
                if matches(deck, spec, ids_by_name)]
        if len(hits) >= 2:
            banned[list_signature(deck_card_ids(deck))] = (
                f"matches multiple archetypes: {' + '.join(hits)} "
                f"({deck['games_raw']} raw games)")
    return banned


def validate(card_ids):
    """-> None if the engine accepts the deck, else the failure reason.

    battle_start is the authority on deck legality: it returns a None observation and an
    errorType for anything the engine refuses to deal."""
    from cg.game import battle_start, battle_finish
    observation, start_data = battle_start(list(card_ids), list(card_ids))
    if observation is None:
        return f"errorType={getattr(start_data, 'errorType', '?')}"
    battle_finish()
    return None


def build(name, spec, decks, ids_by_name, name_by_id, total_games, args, banned):
    pool = [d for d in decks if matches(d, spec, ids_by_name)]
    pool = [d for d in pool if d["games_raw"] >= args.min_games_raw]
    for deck in pool:
        signature = list_signature(deck_card_ids(deck))
        if signature in banned:
            print(f"  {name}: excluding {signature} ({banned[signature]})")
    pool = [d for d in pool
            if list_signature(deck_card_ids(d)) not in banned]
    pool.sort(key=lambda d: (-d["games"], -d["games_raw"], deck_card_ids(d)))

    archetype_dir = OUT_DIR / name
    archetype_dir.mkdir(parents=True, exist_ok=True)
    for stale in archetype_dir.glob(f"{name}_*.csv"):
        stale.unlink()

    records, rejected, written_signatures = [], [], set()
    for deck in pool:
        card_ids = deck_card_ids(deck)
        if args.validate:
            reason = validate(card_ids)
            if reason is not None:
                rejected.append((deck, reason))
                continue
        written_signatures.add(list_signature(card_ids))
        filename = f"{name}_{len(records):03d}.csv"
        (archetype_dir / filename).write_text(
            "".join(f"{card_id}\n" for card_id in card_ids), encoding="ascii")
        line_shape = {card: name_count(deck["counts"], ids_by_name, card)
                      for card, _ in spec["line"]}
        line_shape.update({name_by_id[card_id]: deck["counts"][card_id]
                           for card_id in spec["cards"]})
        records.append({
            "file": filename,
            "games": round(deck["games"], 3),
            "games_raw": deck["games_raw"],
            "corpus_entries": len(deck["usernames"]),
            "line": line_shape,
            "usernames": deck["usernames"],
        })

    panel_files = []
    if spec.get("catch_all") and PANEL_DIR.exists():
        for panel_source in sorted(PANEL_DIR.glob("*.csv")):
            panel_ids = [int(x) for x in panel_source.read_text().split()]
            panel_target = archetype_dir / f"panel_{panel_source.stem}.csv"
            if list_signature(panel_ids) in written_signatures:
                panel_target.unlink(missing_ok=True)
                print(f"  {name}: panel {panel_source.stem} already in the pool "
                      f"naturally -- no injected copy")
            else:
                panel_target.write_text(
                    "".join(f"{card_id}\n" for card_id in panel_ids), encoding="ascii")
                panel_files.append(panel_target.name)
                print(f"  {name}: injected panel {panel_target.name}")

    pool_games = sum(r["games"] for r in records)
    (archetype_dir / "pool.json").write_text(json.dumps({
        "archetype": name,
        "notation": spec["notation"],
        "line": spec["line"],
        "cards": {str(k): v for k, v in spec["cards"].items()},
        "source": str(CORPUS_PATH.relative_to(ROOT)).replace("\\", "/"),
        "min_games_raw": args.min_games_raw,
        "engine_validated": args.validate,
        "decks": len(records) + len(panel_files),
        "panel_files": panel_files,
        "games": round(pool_games, 3),
        "ladder_share": round(pool_games / total_games, 5),
        "deck_list": records,
    }, indent=2), encoding="utf-8")

    for deck, reason in rejected:
        print(f"  !! {name}: engine rejected a list ({reason}), "
              f"{deck['games_raw']} raw games -- dropped", file=sys.stderr)
    return {"archetype": name, "notation": spec["notation"],
            "decks": len(records) + len(panel_files),
            "games": round(pool_games, 3),
            "ladder_share": round(pool_games / total_games, 5),
            "rejected": len(rejected)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archetype", action="append", choices=sorted(ARCHETYPES),
                        help="build only these (repeatable); default = all")
    parser.add_argument("--min-games-raw", type=int, default=1,
                        help="drop lists with fewer true ladder games than this "
                             "(default 1 = keep everything the corpus recorded)")
    parser.add_argument("--no-validate", dest="validate", action="store_false",
                        help="skip the engine battle_start legality probe (needs cg/)")
    args = parser.parse_args()

    ids_by_name, name_by_id = load_cards()
    for name, spec in ARCHETYPES.items():
        for card, _ in spec["line"]:
            assert ids_by_name[normalise(card)], f"{name}: no card named {card!r}"
        for card_id, expected in spec["cards"].items():
            assert normalise(name_by_id[card_id]) == normalise(expected), \
                f"{name}: card {card_id} is {name_by_id[card_id]!r}, not {expected!r}"

    decks = load_unique_lists()
    total_games = sum(d["games"] for d in decks)
    wanted = args.archetype or list(ARCHETYPES)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    banned = dict(EXCLUDED_LISTS)
    banned.update(overlap_exclusions(decks, ids_by_name))
    print(f"corpus: {len(decks)} unique lists, {total_games:.0f} decayed games"
          f"{'' if args.validate else '  (engine validation OFF)'} | "
          f"{len(banned)} banned list(s)\n")
    summaries = [build(name, ARCHETYPES[name], decks, ids_by_name, name_by_id,
                       total_games, args, banned) for name in wanted]

    manifest_path = OUT_DIR / "manifest.json"
    manifest = {"source": str(CORPUS_PATH.relative_to(ROOT)).replace("\\", "/"),
                "corpus_lists": len(decks), "corpus_games": round(total_games, 3),
                "min_games_raw": args.min_games_raw,
                "engine_validated": args.validate, "archetypes": {}}
    if manifest_path.exists():          # a partial --archetype run keeps the others
        manifest["archetypes"] = json.loads(
            manifest_path.read_text(encoding="utf-8")).get("archetypes", {})
    for summary in summaries:
        manifest["archetypes"][summary["archetype"]] = summary
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"{'archetype':18s} {'lists':>6s} {'ladder share':>13s}   line")
    covered = 0.0
    for summary in sorted(summaries, key=lambda s: -s["ladder_share"]):
        covered += summary["ladder_share"]
        print(f"{summary['archetype']:18s} {summary['decks']:6d} "
              f"{summary['ladder_share']:12.2%}   {summary['notation']}")
    print(f"\n{sum(s['decks'] for s in summaries)} lists, "
          f"{covered:.1%} of decayed ladder games -> {OUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
