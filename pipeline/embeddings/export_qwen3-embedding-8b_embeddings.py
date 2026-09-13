import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "embeddings" / "qwen3-embedding-8b"
CARDS_PATH = PROJECT_ROOT / "data" / "cards" / "cards.json"
ATTACKS_PATH = PROJECT_ROOT / "data" / "cards" / "attacks.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "cards" / "embeddings" / MODEL_DIR.name
OUTPUT_PATH = OUTPUT_DIR / "raw.npz"

BATCH_SIZE = 32
MAX_SEQ_LENGTH = 512


def labelled(name, text):
    """A skill/attack rendered as '<name>: <effect text>' for the embedder."""
    return f"{name}: {text}"


def card_document(card, attacks_by_id):
    """Compose a card's full text: name + every non-empty ability and attack text."""
    parts = [card["name"]]
    for skill in card["skills"]:
        if skill["text"]:
            parts.append(labelled(skill["name"], skill["text"]))
    for attack_id in card["attacks"]:
        attack = attacks_by_id[attack_id]
        if attack["text"]:
            parts.append(labelled(attack["name"], attack["text"]))
    return "\n".join(parts)


def last_token(hidden_states, attention_mask):
    """Hidden state of each row's final real token (the appended <|endoftext|>).

    Handles both padding sides: with left padding (what we use) the final token is the
    last column for every row; with right padding we index by sequence length. Getting
    this wrong silently pools a padding token for shorter sequences.
    """
    if attention_mask[:, -1].all():
        return hidden_states[:, -1]
    last_index = attention_mask.sum(dim=1) - 1
    rows = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[rows, last_index]


@torch.inference_mode()
def embed_all(texts, tokenizer, model):
    """Embed texts as documents (no instruction), L2-normalized, returned as float32."""
    vectors = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        tokens = tokenizer(batch, padding=True, truncation=True,
                           max_length=MAX_SEQ_LENGTH, return_tensors="pt").to(model.device)
        hidden = model(**tokens).last_hidden_state
        pooled = F.normalize(last_token(hidden, tokens["attention_mask"]), p=2, dim=1)
        vectors.append(pooled.float().cpu().numpy())
        print(f"  {min(start + BATCH_SIZE, len(texts))}/{len(texts)}", end="\r")
    print()
    return np.concatenate(vectors)


def main():
    cards = json.loads(CARDS_PATH.read_text(encoding="utf-8"))
    attacks = json.loads(ATTACKS_PATH.read_text(encoding="utf-8"))
    attacks_by_id = {attack["attackId"]: attack for attack in attacks}

    # Collect the texts to embed, each tagged with the id the encoder looks up with.
    # Empty effect text (vanilla attacks, text-less cards) is skipped -- the encoder
    # uses a zero/sentinel vector for "no effect text" rather than embedding "".
    card_ids, card_texts = [], []
    ability_card_ids, ability_indices, ability_texts = [], [], []
    attack_ids, attack_texts = [], []
    seen_attacks = set()

    for card in cards:
        card_ids.append(card["cardId"])
        card_texts.append(card_document(card, attacks_by_id))

        for index, skill in enumerate(card["skills"]):
            if skill["text"]:
                ability_card_ids.append(card["cardId"])
                ability_indices.append(index)
                ability_texts.append(labelled(skill["name"], skill["text"]))

        for attack_id in card["attacks"]:
            if attack_id in seen_attacks:
                continue
            seen_attacks.add(attack_id)
            attack = attacks_by_id[attack_id]
            if attack["text"]:
                attack_ids.append(attack_id)
                attack_texts.append(labelled(attack["name"], attack["text"]))

    # Embed every unique string once; identical texts (shared attacks, reprinted
    # abilities) reuse the same vector.
    unique_texts = sorted({*card_texts, *ability_texts, *attack_texts})
    row_of = {text: row for row, text in enumerate(unique_texts)}

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
    else:
        device, dtype = "cpu", torch.float32
    print(f"loading {MODEL_DIR.name} on {device} ({dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), padding_side="left")
    model = AutoModel.from_pretrained(str(MODEL_DIR), torch_dtype=dtype).to(device).eval()

    print(f"embedding {len(unique_texts)} unique texts...")
    started = time.time()
    matrix = embed_all(unique_texts, tokenizer, model)

    def vectors_for(texts):
        if not texts:
            return np.empty((0, matrix.shape[1]), dtype=np.float32)
        return matrix[[row_of[text] for text in texts]]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT_PATH,
        card_ids=np.array(card_ids, dtype=np.int32),
        card_vecs=vectors_for(card_texts),
        attack_ids=np.array(attack_ids, dtype=np.int32),
        attack_vecs=vectors_for(attack_texts),
        ability_card_ids=np.array(ability_card_ids, dtype=np.int32),
        ability_indices=np.array(ability_indices, dtype=np.int32),
        ability_vecs=vectors_for(ability_texts),
    )

    size_mb = OUTPUT_PATH.stat().st_size / 1e6
    print(f"dim {matrix.shape[1]} | cards {len(card_ids)} "
          f"attacks {len(attack_ids)} abilities {len(ability_card_ids)}")
    print(f"wrote {OUTPUT_PATH} ({size_mb:.1f} MB) in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
