import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EMB_DIR = ROOT / "data" / "cards" / "embeddings" / "qwen3-embedding-8b"
RAW_PATH = EMB_DIR / "raw.npz"
DIM = int(sys.argv[1]) if len(sys.argv) > 1 else 64

raw = np.load(RAW_PATH)
stacked = np.concatenate([raw["card_vecs"], raw["attack_vecs"], raw["ability_vecs"]], axis=0)

mean = stacked.mean(axis=0)
# SVD-based PCA: rows of Vt are the principal axes, S**2 is proportional to variance.
_, singular, axes = np.linalg.svd(stacked - mean, full_matrices=False)
cumulative = np.cumsum(singular ** 2) / np.sum(singular ** 2)

print(f"vectors {stacked.shape[0]}  source dim {stacked.shape[1]}")
for k in (8, 16, 32, 64, 128, 256, DIM):
    if k <= len(cumulative):
        print(f"  dim {k:>4}: cumulative explained variance {cumulative[k - 1] * 100:5.1f}%")

components = axes[:DIM]   # [DIM, source_dim]


def project(vectors):
    return ((vectors - mean) @ components.T).astype(np.float32)


np.savez(EMB_DIR / f"pca_{DIM}.npz",
         mean=mean.astype(np.float32), components=components.astype(np.float32))
np.savez(
    EMB_DIR / f"emb_{DIM}.npz",
    card_ids=raw["card_ids"], card_vecs=project(raw["card_vecs"]),
    attack_ids=raw["attack_ids"], attack_vecs=project(raw["attack_vecs"]),
    ability_card_ids=raw["ability_card_ids"], ability_indices=raw["ability_indices"],
    ability_vecs=project(raw["ability_vecs"]),
)
print(f"\nwrote pca_{DIM}.npz + emb_{DIM}.npz  "
      f"(dim {DIM}, {cumulative[DIM - 1] * 100:.1f}% variance retained)")
