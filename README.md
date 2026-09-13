# Pokémon TCG AI Battle Challenge: a self-play agent

Code behind my entry to the Kaggle **Pokémon TCG AI Battle Challenge**. The Sylveon
submission finished **18th** in the Simulation category (score 1162.4); a second submission
with Dragapult ex, built with the same pipeline, scored 1039.1.

- Simulation category: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle
- Strategy category: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy

## The agent in short

- **Encoder.** Every card in play or in hand becomes a 593-dimensional token (a learned
  card-id embedding, a PCA-64 Qwen3-8B embedding of the card text, live state such as HP,
  energy and status). A 125-dimensional global vector carries prizes, hand sizes, turn and
  stadium; each legal option becomes a 2048-dimensional row. A seen-card tracker deduces
  which of our own cards sit in the prizes.
- **Model.** A Set Transformer (width 128, 4 layers, 8 heads, ff 256, 1,001,987
  parameters) with a pointer head over the legal options and a value head. Training-only
  auxiliary heads exist for diagnostics.
- **Training.** PPO with GAE (gamma 1, lambda 0.95), win/loss reward only, from random
  weights. Seat 0 always plays our deck; seat 1 draws an opponent archetype uniformly from
  16 pools of real ladder decklists, then a list uniformly inside the pool. 512 games per
  iteration, 20% of games against frozen past versions, fp32. The Dragapult base ran 1035
  iterations, the Sylveon base 1000.
- **Matchup fine-tunes and router.** 15 specialists per deck, each a 400-iteration
  fine-tune of the base against one archetype pool. At runtime a naive Bayes posterior over
  the pool decklists, updated from the cards the opponent reveals, swaps a specialist in at
  0.8 confidence and out below 0.7.
- **Search.** Every decision runs a determinized PUCT search on the game engine to the end
  of our turn, with the pointer head as priors and the value head at the leaves. Hidden
  cards are sampled, coin flips are 50/50 chance nodes, and shuffle-draw Supporters are
  sampled chance nodes. The budget follows the 600 s time bank; under 40 s the raw policy
  plays.
- **Action rules.** A few hand-written masks remove mistakes found by reading replays
  (counter cap, deck-out guards, a Pokégear gate for Sylveon, damage placement and
  stadium rules for Dragapult). Only the counter cap was active during training.

## Layout

```
agent/        the deployed Kaggle agent: main.py (entry, encoder glue, rules), turn_search.py,
              router.py, search_client.py. See agent/README.md for the runtime files it needs.
src/          the encoder (game/state_encoder.py, built on the stage modules beside it:
              encode.py, encode_full.py, encode_rich.py, encode_history.py, encode_details.py,
              encode_selection.py, encode_inflight.py), the model (models/transformer.py),
              the seen-card tracker (decks/card_knowledge.py), the prize solver and helpers.
              Shared by the agent and the trainer.
training/     train_ppo.py (PPO self-play, probes, fine-tuning), aux_head_labels.py (auxiliary
              labels), eval_panel.py (the probe opponents), verify_field_pools.py (gate for the
              opponent pools), train_entry.py (entry point that pins the multiprocessing start
              method), and launch/ with train_base.sh, finetune.sh and finetune_all.sh.
pipeline/     replays -> decklist corpus (parse_top_episode_decks.py), corpus -> archetype
              pools (build_matchup_pools.py), card text -> PCA-64 embeddings (embeddings/).
tests/        the unit matrices for the action rules (tests/rule_tests/).
decks/        the two submitted 60-card lists, as engine card ids and by card name.
analysis/     the Limitless TCG scrape behind the deck-choice table in the writeup.
```

## Setup

The code expects these inputs from the competition, placed at the paths shown:

- `cg/`: the cabt engine SDK (compiled engine plus Python wrappers). The agent and the
  trainer both drive it.
- `data/cards/`: `cards.json`, `attacks.json`, the card-effect feature table, and the
  text embeddings (`embeddings/emb_64.npz`, built from the card text with
  `pipeline/embeddings/`).
- `data/decks/`: the decklist corpus and the archetype pools, built from the Kaggle
  replays by `pipeline/`.
- `data/official_samples/agents/`: the four official sample agents the probe plays
  against (see `training/eval_panel.py`).
- `ppo.pt` and `models/<archetype>.pt`: the trained weights, produced by the training
  steps below and converted to the bundle format.

`src/cards.py` and `src/embeddings.py` load the card data at import time, so `data/cards/`
must be in place before anything in `src/` imports.

## Running the agent

Kaggle executes `main.py` from the bundle directory with the working directory set to the
bundle. The bundle layout is:

```
main.py  turn_search.py  router.py  search_client.py  deck.csv
src/               this repository's src/
cg/                the engine SDK
data/cards/        cards.json, attacks.json, card_effect_features.npz, embeddings/emb_64.npz
ppo.pt             base weights: fp16 state dict plus a metadata envelope (dims, encoding,
                   action rules, search options)
models/<archetype>.pt   the 15 specialists, same format
router_data.json   every archetype's decklists and ladder priors from build_matchup_pools.py,
                   the thresholds, and the specialist manifest
```

The agent never crashes on purpose: any failure inside a decision falls back to a legal
random move, and it plays the raw policy when the time bank drops under 40 s. Inference is
CPU only (the evaluation hardware was 2 vCPUs, no GPU).

## Training

Requirements: Python 3.13, PyTorch with CUDA, NumPy, the engine (`CG_DLL` points at the
engine library; `train_ppo.py` defaults to `engine_src/build/cg.dll`), the card data, and
the decklist corpus plus pools from step 1 (`data/decks/corpus.json` and
`data/decks/matchups/`). Runs write their checkpoints and metrics to `runs/<run-name>/`.
The base runs used an RTX 4090 with 16 CPU workers, about 85 s per iteration for
Dragapult and 48 s for Sylveon.

1. Build the opponent field: `pipeline/parse_top_episode_decks.py` (replays to corpus),
   then `pipeline/build_matchup_pools.py` (corpus to `data/decks/matchups/<archetype>/`),
   then `training/verify_field_pools.py` (gate).
2. Train the base model: `bash training/launch/train_base.sh <run> decks/dragapult_ex.csv 1035`
   (1000 iterations for Sylveon). Running it again resumes the run. On Windows, run the
   command below directly in PowerShell. The script runs:

   ```
   python training/train_ppo.py --run-name <run> --prize-labels truth \
     --deck-mode focus --focus-deck decks/dragapult_ex.csv --field-pools data/decks/matchups \
     --d-model 128 --num-layers 4 --num-heads 8 --ff-dim 256 --action-rules counter_cap \
     --workers 16 --games-per-worker 2 --games-per-iter 512 --epochs 2 \
     --resign-threshold 0.95 --resign-persist 6 --gpu-server --no-worker-model \
     --lr 2.5e-4 --lr-anneal-start 700 --lr-anneal-end 1035 --lr-final 0 \
     --ent-coef 0.01 --ent-anneal-start 700 --ent-anneal-end 1000 --ent-final 2e-3 \
     --kl-stop 0.5 --aux-weight 0.1 --attack-aux-weight 0.5 \
     --iterations 1035 --snapshot-every 25 --probe-every 25 --ckpt-every 5 --seed-base <seed>
   ```

3. Fine-tune one specialist per archetype:
   `bash training/launch/finetune.sh <run> decks/dragapult_ex.csv <archetype>`, or
   `finetune_all.sh <run> decks/dragapult_ex.csv` for all 15. Each specialist resumes the
   finished base with `--field-pools data/decks/matchups_<archetype>` for 400 more
   iterations, `--lr 2.5e-4` flat for 320 of them then linear to 0, and `--ent-coef 0.002`
   with no entropy anneal.
4. Evaluate a checkpoint with the probe used during training:
   `python training/train_ppo.py --probe-only --probe-ckpt <checkpoint> --probe-games 200`
   plus the same deck flags as the run. The probe plays the four official sample agents
   and a random agent; the sample agents must sit under `data/official_samples/agents/`
   (see `training/eval_panel.py`). The launch scripts run the same probe every 25
   iterations when those agents are present and skip it otherwise.
5. Run the rule matrices with `python tests/rule_tests/test_<rule>.py`; they import
   `src/` and `agent/turn_search.py` from this repository.
