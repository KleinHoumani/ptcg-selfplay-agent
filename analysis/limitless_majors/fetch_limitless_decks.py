"""Scrape a real-world (in-person) tournament's decklists from limitlesstcg.com.

This builds a SEPARATE corpus from the Kaggle one (`data/decks/corpus.json`, produced by
parse_top_episode_decks.py). Nothing here touches that file -- IRL lists land under
`data/decks/irl/` and are keyed by card NAME/SET/NUMBER, not by cabt engine card id.

  ./.venv/Scripts/python.exe scripts/fetch_limitless_decks.py 518
  ./.venv/Scripts/python.exe scripts/fetch_limitless_decks.py 518 --division SR

Tournament ids come from https://limitlesstcg.com/tournaments (e.g. 518 = NAIC 2026,
New Orleans; 559 = Regional Indianapolis). The masters division is the base standings
page; --division JR / SR fetch the junior / senior sub-pages.

Note: limitlesstcg.com hosts in-person event coverage as HTML only -- the documented
play.limitlesstcg.com JSON API covers the ONLINE platform's tournaments, and does not
include IRL events. Hence the HTML scrape.

Decklist pages are cached under `data/decks/irl/_cache/` so re-runs are free.
"""

import argparse
import html
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "decks" / "irl"
CACHE_DIR = OUT_DIR / "_cache"
BASE = "https://limitlesstcg.com"
HEADERS = {"User-Agent": "pokemon-tcg-ai research scraper (contact: kleinhoumani@gmail.com)"}

ROW_RE = re.compile(
    r'<tr data-rank="(?P<rank>\d+)"\s+data-name="(?P<name>[^"]*)"\s+'
    r'data-country="(?P<country>[^"]*)"\s+data-deck="(?P<deck>[^"]*)"')
PLAYER_ID_RE = re.compile(r'href="/players/(\d+)"')
DECKLIST_ID_RE = re.compile(r'href="/decks/list/(\d+)"')
DECK_ID_RE = re.compile(r'href="/decks/(\d+)(?:\?variant=(\d+))?"')
HEADING_RE = re.compile(r'<div class="infobox-heading">\s*(?P<name>[^<]*?)\s*<', re.S)
LINE_RE = re.compile(r'<div class="infobox-line">(?P<line>.{0,1200})', re.S)
DATE_RE = re.compile(r'^\s*(?P<date>[^<•\n]+?)\s*$', re.M)
PLAYERS_RE = re.compile(r'(?P<players>\d[\d,]*)\s*\n\s*Players')
FORMAT_RE = re.compile(r'format=(?P<format>[^"&]+)">(?P<format_name>[^<]*)</a>')
TITLE_RE = re.compile(r'<div class="decklist-title">\s*([^<]*?)\s*<')
# One pass over the decklist body: every hit is either a section heading
# (Pokemon / Trainer / Energy) or a card entry belonging to the latest heading.
CARD_RE = re.compile(
    r'<div class="decklist-column-heading">(?P<heading>[^<(]+)'
    r'|<div class="decklist-card" data-set="(?P<set>[^"]*)" data-number="(?P<number>[^"]*)"'
    r'[^>]*?>.*?<span class="card-count">(?P<count>\d+)</span>\s*'
    r'<span class="card-name">(?P<cardname>[^<]*)</span>',
    re.S)


def get(url, cache_name=None):
    """Fetch a URL as text, optionally memoised on disk under _cache/."""
    if cache_name:
        cached = CACHE_DIR / cache_name
        if cached.exists():
            return cached.read_text(encoding="utf-8")
    for attempt in range(4):
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    response.encoding = "utf-8"
    if cache_name:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (CACHE_DIR / cache_name).write_text(response.text, encoding="utf-8")
    return response.text


def parse_standings(page):
    """-> (tournament metadata dict, [player record, ...]) from a standings page."""
    heading = HEADING_RE.search(page)
    line = LINE_RE.search(page)
    block = line.group("line") if line else ""
    date = DATE_RE.search(block)
    players_count = PLAYERS_RE.search(block)
    deck_format = FORMAT_RE.search(block)
    meta = {
        "name": html.unescape(heading.group("name")) if heading else None,
        "date": date.group("date") if date else None,
        "players": int(players_count.group("players").replace(",", ""))
                   if players_count else None,
        "format": deck_format.group("format") if deck_format else None,
        "format_name": html.unescape(deck_format.group("format_name"))
                       if deck_format else None,
    }
    # Each row's markup ends where the next row begins; slice on the row starts.
    starts = [match.start() for match in ROW_RE.finditer(page)]
    matches = list(ROW_RE.finditer(page))
    players = []
    for index, match in enumerate(matches):
        end = starts[index + 1] if index + 1 < len(starts) else len(page)
        chunk = page[match.start():end]
        player_id = PLAYER_ID_RE.search(chunk)
        decklist_id = DECKLIST_ID_RE.search(chunk)
        deck = DECK_ID_RE.search(chunk)
        players.append({
            "placing": int(match.group("rank")),
            "player": html.unescape(match.group("name")),
            "player_id": int(player_id.group(1)) if player_id else None,
            "country": match.group("country") or None,
            "archetype": html.unescape(match.group("deck")) or None,
            "deck_id": int(deck.group(1)) if deck else None,
            "deck_variant": int(deck.group(2)) if deck and deck.group(2) else None,
            "decklist_id": int(decklist_id.group(1)) if decklist_id else None,
        })
    return meta, players


def parse_decklist(page):
    """-> (archetype title, [{count, name, set, number, category}, ...])."""
    title = TITLE_RE.search(page)
    category = None
    cards = []
    for match in CARD_RE.finditer(page):
        if match.group("heading"):
            # "Pokémon" -> "pokemon" so every category key stays ascii.
            category = match.group("heading").strip().lower().replace("é", "e")
            continue
        cards.append({
            "count": int(match.group("count")),
            "name": html.unescape(match.group("cardname")),
            "set": match.group("set"),
            "number": match.group("number"),
            "category": category,
        })
    return (html.unescape(title.group(1)) if title else None), cards


def fetch_decklist(decklist_id):
    page = get(f"{BASE}/decks/list/{decklist_id}", cache_name=f"list_{decklist_id}.html")
    title, cards = parse_decklist(page)
    return decklist_id, title, cards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tournament", type=int, help="limitlesstcg.com tournament id")
    parser.add_argument("--division", default="masters", choices=["masters", "JR", "SR"])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", default=None, help="output path (default data/decks/irl/<id>_<division>.json)")
    args = parser.parse_args()

    url = f"{BASE}/tournaments/{args.tournament}"
    if args.division != "masters":
        url += f"/{args.division}"
    print(f"fetching standings: {url}", flush=True)
    meta, players = parse_standings(get(url))
    meta.update({
        "id": args.tournament,
        "division": args.division,
        "source": url,
        "fetched": time.strftime("%Y-%m-%d"),
    })
    listed = [player for player in players if player["decklist_id"]]
    unique = sorted({player["decklist_id"] for player in listed})
    print(f"{meta['name']} -- {len(players)} standings rows, {len(listed)} with a "
          f"decklist ({len(unique)} unique lists)", flush=True)

    decklists = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for done, (decklist_id, title, cards) in enumerate(
                pool.map(fetch_decklist, unique), start=1):
            decklists[decklist_id] = (title, cards)
            if done % 50 == 0 or done == len(unique):
                print(f"  {done}/{len(unique)} decklists", flush=True)

    incomplete = []
    for player in players:
        title, cards = decklists.get(player["decklist_id"], (None, []))
        player["deck_name"] = title
        player["cards"] = cards
        player["total_cards"] = sum(card["count"] for card in cards)
        if player["decklist_id"] and player["total_cards"] != 60:
            incomplete.append((player["placing"], player["total_cards"]))

    out_path = Path(args.out) if args.out else (
        OUT_DIR / f"{args.tournament}_{args.division}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"tournament": meta, "players": players}, indent=1,
                                   ensure_ascii=False), encoding="utf-8")
    if incomplete:
        print(f"WARNING: {len(incomplete)} lists are not 60 cards: {incomplete[:10]}")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
