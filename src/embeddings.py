import numpy as np
from pathlib import Path

# Static, id-keyed text-embedding table, loaded once at import (like src/cards.py).
#
# EMBEDDINGS_PATH is the single swap point: the PCA-compressed, net-ready table. To
# change the feed dim, re-run scripts/embeddings/compress_embeddings.py <dim> and point
# here at the new emb_<dim>.npz -- the code below is dimension-agnostic. (raw.npz is the
# 4096-d source of truth, too wide to feed the CPU net, so never point at it for
# training/shipping.)
_DATA = Path(__file__).resolve().parent.parent / "data"
EMBEDDINGS_PATH = _DATA / "cards" / "embeddings" / "qwen3-embedding-8b" / "emb_64.npz"

_table = np.load(EMBEDDINGS_PATH)
DIMENSION = int(_table["card_vecs"].shape[1])

# Returned for any id not in the table -- empty-effect-text attacks/abilities (which
# were never embedded) and unknown ids alike. A shared, read-only "no text" sentinel.
_MISSING = np.zeros(DIMENSION, dtype=np.float32)

_card = {int(card_id): _table["card_vecs"][row]
         for row, card_id in enumerate(_table["card_ids"])}
_attack = {int(attack_id): _table["attack_vecs"][row]
           for row, attack_id in enumerate(_table["attack_ids"])}
_ability = {(int(card_id), int(index)): _table["ability_vecs"][row]
            for row, (card_id, index)
            in enumerate(zip(_table["ability_card_ids"], _table["ability_indices"]))}


def card_embedding(card_id):
    """Card-level document embedding (name + all text). Zero sentinel if absent."""
    return _card.get(card_id, _MISSING)


def attack_embedding(attack_id):
    """Per-attack effect-text embedding. Zero sentinel for empty/unknown."""
    return _attack.get(attack_id, _MISSING)


def ability_embedding(card_id, skill_index):
    """Per-ability effect-text embedding, keyed by (card id, skill index)."""
    return _ability.get((card_id, skill_index), _MISSING)
