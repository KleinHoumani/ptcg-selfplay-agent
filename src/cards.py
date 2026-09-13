import json
from pathlib import Path

_DATA = Path(__file__).resolve().parent.parent / "data"

CARDS = {card["cardId"]: card
         for card in json.loads((_DATA / "cards/cards.json").read_text(encoding="utf-8"))}

ATTACKS = {attack["attackId"]: attack
           for attack in json.loads((_DATA / "cards/attacks.json").read_text(encoding="utf-8"))}


def get_card(card_id):
    return CARDS.get(card_id)


def get_attack(attack_id):
    return ATTACKS.get(attack_id)

