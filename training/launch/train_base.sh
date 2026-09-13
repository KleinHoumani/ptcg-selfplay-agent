#!/usr/bin/env bash
# Train a base model from scratch with PPO self-play.
#
#   bash training/launch/train_base.sh <run-name> <deck.csv> [iterations] [seed]
#
#   bash training/launch/train_base.sh dragapult_base decks/dragapult_ex.csv 1035
#   bash training/launch/train_base.sh sylveon_base   decks/sylveon.csv     1000
#
# Run it from the repository root with the engine (CG_DLL), data/cards/ and
# data/decks/matchups/ in place (see README.md). Checkpoints and metrics land in
# runs/<run-name>/; running the same command again resumes the run.
#
# The schedule is the one both submitted bases used: learning rate flat at 2.5e-4 until
# iteration 700, then linear to 0 at the last iteration; entropy bonus annealed from 0.01
# to 0.002 over iterations 700 to 1000. The label invariant checks are assertions only;
# they are switched off because healing cards in the opponent field trip them.
set -euo pipefail

RUN="${1:?usage: train_base.sh <run-name> <deck.csv> [iterations] [seed]}"
DECK="${2:?usage: train_base.sh <run-name> <deck.csv> [iterations] [seed]}"
ITERATIONS="${3:-1035}"
SEED="${4:-20260812}"
WORKERS="${WORKERS:-16}"
PYTHON="${PYTHON:-python}"
ENT_ANNEAL_END=$(( ITERATIONS < 1000 ? ITERATIONS : 1000 ))
# The periodic probe plays the official sample agents (training/eval_panel.py); skip it
# when they are not installed so a run never dies at iteration 25.
if [ -f data/official_samples/agents/dragapult_ex.py ]; then PROBE_EVERY=25; else PROBE_EVERY=0; fi

[ -f "$DECK" ] || { echo "deck list missing: $DECK"; exit 1; }
[ -d data/decks/matchups ] || { echo "opponent pools missing: data/decks/matchups (see README.md)"; exit 1; }

ARGS=(
    --run-name "$RUN"
    --prize-labels truth
    --deck-mode focus
    --focus-deck "$DECK"
    --field-pools data/decks/matchups
    --d-model 128 --num-layers 4 --num-heads 8 --ff-dim 256
    --action-rules counter_cap
    --workers "$WORKERS"
    --games-per-worker 2
    --games-per-iter 512
    --epochs 2
    --resign-threshold 0.95 --resign-persist 6
    --gpu-server --no-worker-model
    --lr 2.5e-4
    --lr-anneal-start 700 --lr-anneal-end "$ITERATIONS" --lr-final 0
    --ent-coef 0.01
    --ent-anneal-start 700 --ent-anneal-end "$ENT_ANNEAL_END" --ent-final 2e-3
    --kl-stop 0.5
    --aux-weight 0.1 --attack-aux-weight 0.5
    --iterations "$ITERATIONS"
    --snapshot-every 25 --probe-every "$PROBE_EVERY" --ckpt-every 5 --full-ckpt-every 25
    --seed-base "$SEED"
    --no-label-invariant-checks
)

CHECKPOINT="runs/$RUN/ppo_latest.pt"
if [ -f "$CHECKPOINT" ]; then
    echo "resuming $RUN from $CHECKPOINT"
    ARGS+=(--resume "$CHECKPOINT")
fi

exec "$PYTHON" training/train_entry.py "${ARGS[@]}"
