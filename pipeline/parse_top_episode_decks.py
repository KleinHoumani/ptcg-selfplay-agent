"""Parse Kaggle cabt episode replays into a compact deck corpus.

Each replay has the two players' usernames at info.TeamNames and their 60-card decks at
steps[1][player]["action"] (the deck submission). We dedup to one record per unique
(username, deck-composition) and count how many games it appeared in -- so a player who
ran the same deck many times is stored once (with games=N), and a player who switched
decks gets multiple records.

  ./.venv/Scripts/python.exe scripts/parse_top_episode_decks.py [--since M_DD] \
      [--half-life-days N]

--since 7_17 keeps only dump dates >= July 17 (inclusive). Omit to scan every dated dump.
--half-life-days 7 RECENCY-WEIGHTS instead of cutting: each game counts
0.5 ** (age_days / N), aged from the newest dump date (an episode's date = the earliest
dump it appears in). "games" then holds the decayed weight (float; every consumer feeds
it to weighted sampling / popularity ranking, so floats are fine) and "games_raw" keeps
the true count. This keeps the FULL list diversity while the field shares track the
current meta.
"""

import argparse
import datetime
import glob
import json
import multiprocessing as mp
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Every dated dump folder under kaggle_dumps (each named M_DD, e.g. 6_25 or 7_04),
# same source the metagame explorer scans -- so the corpus tracks the current meta.
DUMP_ROOT = ROOT / "data" / "decks" / "kaggle_dumps"
DATE_DIR_RE = re.compile(r"\d{1,2}_\d{1,2}")
OUT_PATH = ROOT / "data" / "decks" / "corpus.json"


def parse_file(path):
    """-> (episode_id, [(username, canonical-deck-tuple), ...]) or None on any error."""
    try:
        with open(path, encoding="utf-8") as file:
            episode = json.load(file)
        names = episode["info"]["TeamNames"]
        out = []
        for player in (0, 1):
            deck = episode["steps"][1][player]["action"]
            if isinstance(deck, list) and len(deck) == 60:
                out.append((names[player], tuple(sorted(deck))))
        return (episode["info"]["EpisodeId"], out)
    except Exception:
        return None


def _date_key(name):
    month, day = name.split("_")
    return (int(month), int(day))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", default=None,
                        help="earliest dump date to include, M_DD inclusive (e.g. 7_17)")
    parser.add_argument("--half-life-days", type=float, default=None,
                        help="recency-weight games by 0.5**(age_days/N) instead of cutting")
    args = parser.parse_args()
    archives = sorted((path for path in DUMP_ROOT.iterdir()
                       if path.is_dir() and DATE_DIR_RE.fullmatch(path.name)),
                      key=lambda path: _date_key(path.name))
    if args.since:
        archives = [path for path in archives
                    if _date_key(path.name) >= _date_key(args.since)]
    files, file_dates = [], []
    for archive in archives:
        for path in glob.glob(str(archive / "*.json")):
            files.append(path)
            file_dates.append(_date_key(archive.name))
    print(f"parsing {len(files)} replays from {len(archives)} dump date(s): "
          f"{', '.join(a.name for a in archives)}", flush=True)
    with mp.Pool() as pool:
        results = pool.map(parse_file, files, chunksize=20)

    newest = max(file_dates, default=(1, 1))

    def game_weight(date_key):
        if not args.half_life_days:
            return 1.0
        age_days = (datetime.date(2026, *newest) - datetime.date(2026, *date_key)).days
        return 0.5 ** (age_days / args.half_life_days)

    # Archives iterate in ascending date order, so an episode's first sighting is its
    # EARLIEST dump date -- the best proxy for when it was actually played.
    seen_episodes = set()
    games_raw = Counter()        # (username, deck-tuple) -> true game count
    games_weighted = Counter()   # (username, deck-tuple) -> recency-decayed weight
    for date_key, result in zip(file_dates, results):
        if result is None:
            continue
        episode_id, parsed = result
        if episode_id in seen_episodes:   # same replay across overlapping dumps
            continue
        seen_episodes.add(episode_id)
        for username, deck in parsed:
            games_raw[(username, deck)] += 1
            games_weighted[(username, deck)] += game_weight(date_key)

    records = []
    for (username, deck), count in games_raw.items():
        records.append({
            "username": username,
            "cards": dict(Counter(deck)),   # {card_id: count}
            "games": (round(games_weighted[(username, deck)], 3)
                      if args.half_life_days else count),
            "games_raw": count,
        })
    records.sort(key=lambda record: -record["games"])

    OUT_PATH.write_text(json.dumps(records, indent=1), encoding="utf-8")
    raw_games = sum(record["games_raw"] for record in records)
    weighted_games = sum(record["games"] for record in records)
    print(f"{len(files)} files -> {raw_games} player-decks "
          f"(weighted {weighted_games:.0f}) -> {len(records)} unique "
          f"(username, deck) records -> {OUT_PATH}")


if __name__ == "__main__":
    main()
