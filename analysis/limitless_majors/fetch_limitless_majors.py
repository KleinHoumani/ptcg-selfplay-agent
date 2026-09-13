"""Tabulate the most recent Regional + International Championships from limitlesstcg.com.

For each event (masters division) this collects the Day 2 standings, the top 8 and the
winner by deck archetype and writes, under data/decks/irl/majors/:

  events.csv       one row per event (date, players, format, Day 2 size, winner)
  standings.csv    one row per Day 2 player (event, placing, archetype, variant)
  archetypes.csv   wins / top 8s / Day 2s per archetype, plus a Day 2 column per event
  variants.csv     the same per deck variant (Dragapult Dusknoir, Dragapult Blaziken, ...)
  summary.md       the tables as markdown; the archetype table shows the top 10

Decks are ranked by their average share of the Day 2 field per event (the mean over events
of Day 2 count / Day 2 size), so a small event weighs as much as a large one.

  ./.venv/Scripts/python.exe scripts/fetch_limitless_majors.py            # last 10 events
  ./.venv/Scripts/python.exe scripts/fetch_limitless_majors.py --count 12

Day 2: the standings page of limitlesstcg.com lists exactly the players whose decklist was
published, which is the Day 2 field. The size is cross-checked against the per-deck Day 2
counts that labs.limitlesstcg.com embeds in its decks page, which cover the whole field
(a mismatch is printed). Archetype = the parent deck limitless uses on the Statistics tab
(Dragapult), taken from the newest event when a grouping changed; variant = the label on
the standings row (Dragapult Dusknoir). Pages are cached under data/decks/irl/_cache/,
like fetch_limitless_decks.py, so re-runs are free.
"""

import argparse
import collections
import csv
import html
import re

from fetch_limitless_decks import BASE, OUT_DIR, get, parse_standings

MAJORS_DIR = OUT_DIR / "majors"
TOP_CUT = 8
SUMMARY_ROWS = 10
EVENT_ROW_RE = re.compile(
    r'<tr data-date="(?P<date>[^"]*)" data-country="(?P<country>[^"]*)"\s+'
    r'data-name="(?P<name>[^"]*)" data-format="(?P<format>[^"]*)"\s+'
    r'data-players="(?P<players>[^"]*)" data-winner="[^"]*">'
    r'.*?<a href="/tournaments/(?P<id>\d+)">', re.S)
# data-points is empty (not 0) for decks that scored no points.
STAT_ROW_RE = re.compile(
    r'<tr data-count="\d+" data-points="\d*">.*?<a href="/decks/(?P<id>\d+)">(?P<name>[^<]*)</a>',
    re.S)
LABS_RE = re.compile(r'href="https://labs\.limitlesstcg\.com/(\d+)/standings"')
LABS_DECK_RE = re.compile(r'"players":[0-9]+,"day2s":(?P<day2s>[0-9]+)')


def list_majors(count):
    """The `count` most recent Regional + International events, newest first."""
    events = []
    for kind in ("regional", "international"):
        page = get(f"{BASE}/tournaments?type={kind}&show=100")
        for match in EVENT_ROW_RE.finditer(page):
            events.append({
                "id": int(match.group("id")),
                "kind": kind,
                "date": match.group("date"),
                "name": html.unescape(match.group("name")),
                "country": match.group("country"),
                "players": int(match.group("players") or 0) or None,
                "format": match.group("format"),
            })
    events.sort(key=lambda event: (event["date"], event["id"]), reverse=True)
    return events[:count]


def fetch_event(event):
    """Attach the Day 2 standings, the parent-deck names and the labs Day 2 count."""
    event_id = event["id"]
    page = get(f"{BASE}/tournaments/{event_id}", cache_name=f"standings_{event_id}.html")
    meta, players = parse_standings(page)
    stats = get(f"{BASE}/tournaments/{event_id}/statistics",
                cache_name=f"statistics_{event_id}.html")
    labs = LABS_RE.search(page)
    event.update({
        "format": meta["format"],               # the set range, e.g. TEF-POR
        "format_name": meta["format_name"],
        "standings": players,
        "day2": len(players),
        "parents": {int(match.group("id")): html.unescape(match.group("name"))
                    for match in STAT_ROW_RE.finditer(stats)},
        "labs_id": labs.group(1) if labs else None,
        "day2_labs": count_labs_day2(labs.group(1)) if labs else None,
    })
    return event


def count_labs_day2(labs_id):
    """Sum of the per-deck Day 2 counts embedded (as escaped JSON) in the labs decks page."""
    page = get(f"https://labs.limitlesstcg.com/{labs_id}/decks",
               cache_name=f"labs_{labs_id}_decks.html")
    payload = page.replace('\\"', '"')
    return sum(int(match.group("day2s")) for match in LABS_DECK_RE.finditer(payload))


def assign_archetypes(events):
    """Name every standings row by its parent deck, using the newest grouping for each id."""
    parents = {}
    for event in reversed(events):
        parents.update(event["parents"])
    for event in events:
        for player in event["standings"]:
            player["variant"] = player["archetype"] or "Unknown"
            player["archetype"] = parents.get(player["deck_id"], player["variant"])


def aggregate(events, key):
    """-> {deck: Counter(wins, top8, day2, <event id>: day2 count)} for `key` in a standings row."""
    table = collections.defaultdict(collections.Counter)
    for event in events:
        for player in event["standings"]:
            row = table[player[key]]
            row["day2"] += 1
            row[event["id"]] += 1
            row["top8"] += player["placing"] <= TOP_CUT
            row["wins"] += player["placing"] == 1
    return table


def average_share(row, events):
    """Mean over events of the share of that event's Day 2 field, as a percentage."""
    return 100 * sum(row[event["id"]] / event["day2"] for event in events) / len(events)


def ranked(table, events):
    return sorted(table.items(),
                  key=lambda item: (-average_share(item[1], events), -item[1]["wins"], item[0]))


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def markdown_table(header, rows):
    lines = ["| " + " | ".join(str(cell) for cell in header) + " |",
             "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def event_rows(events):
    rows = []
    for event in events:
        winner = event["standings"][0]
        top8 = "; ".join(player["variant"] for player in event["standings"][:TOP_CUT])
        rows.append([event["id"], event["kind"], event["date"], event["name"], event["country"],
                     event["players"], event["format"], event["format_name"], event["day2"],
                     event["day2_labs"], event["labs_id"], winner["player"], winner["variant"],
                     winner["archetype"], top8])
    return rows


def standings_rows(events):
    return [[event["id"], event["name"], event["date"], event["format"], player["placing"],
             player["player"], player["country"], player["archetype"], player["variant"],
             player["decklist_id"]]
            for event in events for player in event["standings"]]


def deck_rows(table, events, parents=None):
    rows = []
    for deck, row in ranked(table, events):
        prefix = [deck] if parents is None else [deck, parents[deck]]
        rows.append(prefix + [row["wins"], row["top8"], row["day2"],
                              f"{average_share(row, events):.1f}"]
                    + [row[event["id"]] for event in events])
    return rows


def summary_rows(table, events, parents=None, limit=None):
    """Markdown cells as count (percent): wins of the events, top 8s of the top-8 slots,
    Day 2s with the average Day 2 share per event."""
    rows = []
    for deck, row in ranked(table, events)[:limit]:
        prefix = [deck] if parents is None else [deck, parents[deck]]
        rows.append(prefix + [
            f"{row['wins']} ({100 * row['wins'] / len(events):.1f}%)",
            f"{row['top8']} ({100 * row['top8'] / (TOP_CUT * len(events)):.1f}%)",
            f"{row['day2']} ({average_share(row, events):.1f}%)",
        ])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10, help="number of most recent events")
    parser.add_argument("--formats", default=None,
                        help="comma-separated set ranges to keep, e.g. TEF-POR,TEF-CRI; "
                             "events in other formats are skipped and the scan stops at "
                             "the first older format once a match was found")
    args = parser.parse_args()
    formats = set(args.formats.split(",")) if args.formats else None

    events = []
    for event in list_majors(100):
        fetch_event(event)
        if formats and event["format"] not in formats:
            if events:
                break                           # past the rotation: nothing older matches
            continue
        events.append(event)
        check = "" if event["day2_labs"] in (None, event["day2"]) else (
            f"  MISMATCH: labs counts {event['day2_labs']} Day 2 players")
        print(f"{event['date']} {event['name']}: {event['players']} players, "
              f"{event['day2']} Day 2, {event['format']} {event['format_name']}{check}",
              flush=True)
        if len(events) == args.count:
            break
    assign_archetypes(events)

    archetypes = aggregate(events, "archetype")
    variants = aggregate(events, "variant")
    parents = {player["variant"]: player["archetype"]
               for event in events for player in event["standings"]}
    event_columns = [f"{event['date']} {event['name']}" for event in events]
    deck_header = ["wins", "top8", "day2", "day2_avg_share_pct"] + event_columns

    MAJORS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(MAJORS_DIR / "events.csv",
              ["id", "kind", "date", "name", "country", "players", "format", "format_name",
               "day2", "day2_labs", "labs_id", "winner", "winner_variant", "winner_archetype",
               "top8_variants"], event_rows(events))
    write_csv(MAJORS_DIR / "standings.csv",
              ["event_id", "event", "date", "format", "placing", "player", "country",
               "archetype", "variant", "decklist_id"], standings_rows(events))
    write_csv(MAJORS_DIR / "archetypes.csv", ["archetype"] + deck_header,
              deck_rows(archetypes, events))
    write_csv(MAJORS_DIR / "variants.csv", ["variant", "archetype"] + deck_header,
              deck_rows(variants, events, parents))

    slots = TOP_CUT * len(events)
    percent_note = (f"Percentages: 1st = share of the {len(events)} events, Top 8 = share of "
                    f"the {slots} top-8 slots, Day 2 = average share of the Day 2 field per "
                    "event.\n")
    summary = [
        f"# Last {len(events)} Regional + International Championships (masters)\n",
        "Source: limitlesstcg.com. Day 2 = players with a published decklist "
        "(cross-checked against the labs.limitlesstcg.com Day 2 counts).\n",
        "## Events\n",
        markdown_table(["Date", "Event", "Players", "Format", "Day 2", "Winner", "Winning deck"],
                       [[e["date"], e["name"], e["players"], e["format_name"], e["day2"],
                         e["standings"][0]["player"], e["standings"][0]["variant"]]
                        for e in events]),
        f"\n## Archetypes (top {SUMMARY_ROWS} by average Day 2 share)\n",
        percent_note,
        markdown_table(["Archetype", "1st", "Top 8", "Day 2"],
                       summary_rows(archetypes, events, limit=SUMMARY_ROWS)),
        "\n## Variants\n",
        percent_note,
        markdown_table(["Variant", "Archetype", "1st", "Top 8", "Day 2"],
                       summary_rows(variants, events, parents)),
        "",
    ]
    (MAJORS_DIR / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(f"-> {MAJORS_DIR}")


if __name__ == "__main__":
    main()
