"""Structured effect features, baked from data/cards/card_effects.yaml by
scripts/build_card_effect_features.py into card_effect_features.npz. Loaded here with numpy
only (no PyYAML -- the eval env has none), the same way text embeddings are loaded.

`effect_features(card_id)` -> the whole-card effect vector (per-tag magnitudes + provides
one-hot + optional/when/per flags), or zeros for cards with no tagged effects -- used for
board/hand tokens. `attack_effect_features(attack_id)` / `skill_effect_features(card_id, index)`
-> the same-shaped vector for ONE specific attack / ability -- used in the option features and
the per-attack/per-ability capability tokens so a card's individual actions stay distinct.
"""

import numpy as np
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "data" / "cards" / "card_effect_features.npz"

_data = np.load(_PATH)
_FEATURES = {int(card_id): row.astype(np.float32) for card_id, row in zip(_data["ids"], _data["features"])}
_ATTACK_FEATURES = {int(attack_id): row.astype(np.float32)
                    for attack_id, row in zip(_data["attack_ids"], _data["attack_features"])}
_SKILL_FEATURES = {(int(card_id), int(index)): row.astype(np.float32)
                   for card_id, index, row in zip(_data["skill_card_ids"], _data["skill_indices"],
                                                   _data["skill_features"])}
EFFECT_FEATURE_DIM = int(_data["features"].shape[1])

_MISSING = np.zeros(EFFECT_FEATURE_DIM, dtype=np.float32)
_MISSING.flags.writeable = False


def effect_features(card_id):
    """The whole-card effect vector (zeros if the card has no tagged effects)."""
    return _FEATURES.get(card_id, _MISSING)


def attack_effect_features(attack_id):
    """One specific attack's effect vector (zeros if the attack has no tagged effects or
    attack_id is None) -- the per-attack counterpart to effect_features, so an option to use a
    particular attack carries that attack's own effects instead of the whole-card union."""
    if attack_id is None:
        return _MISSING
    return _ATTACK_FEATURES.get(attack_id, _MISSING)


def skill_effect_features(card_id, index):
    """One specific ability's effect vector (zeros if that skill has no tagged effects) -- keyed
    by (card_id, skill index), for the per-ability capability tokens."""
    return _SKILL_FEATURES.get((card_id, index), _MISSING)
