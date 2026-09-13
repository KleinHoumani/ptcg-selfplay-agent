#!/usr/bin/env bash
# Fine-tune a finished base model against ONE opponent archetype (a matchup specialist).
#
#   bash training/launch/finetune.sh <base-run> <deck.csv> <archetype> [seed]
#
#   bash training/launch/finetune.sh dragapult_base decks/dragapult_ex.csv grimmsnarl
#
# Every specialist is an independent resume of the same base checkpoint: 400 more
# iterations against that archetype's pool only, learning rate flat at 2.5e-4 for 320 of
# them then linear to 0 over the last 80, entropy bonus flat at 0.002. The result lands in
# runs/<base-run>_ft_<archetype>/ and becomes models/<archetype>.pt in
# the bundle. Archetype names are the pool directories under data/decks/matchups/.
set -euo pipefail

BASE="${1:?usage: finetune.sh <base-run> <deck.csv> <archetype> [seed]}"
DECK="${2:?usage: finetune.sh <base-run> <deck.csv> <archetype> [seed]}"
ARCH="${3:?usage: finetune.sh <base-run> <deck.csv> <archetype> [seed]}"
SEED="${4:-$RANDOM}"
WORKERS="${WORKERS:-16}"
PYTHON="${PYTHON:-python}"

RUN="${BASE}_ft_${ARCH}"
# The periodic probe plays the official sample agents (training/eval_panel.py); skip it
# when they are not installed so a run never dies at iteration 25.
if [ -f data/official_samples/agents/dragapult_ex.py ]; then PROBE_EVERY=25; else PROBE_EVERY=0; fi
BASE_DIR="runs/$BASE"
[ -f "$DECK" ] || { echo "deck list missing: $DECK"; exit 1; }
[ -f "$BASE_DIR/ppo_latest.pt" ] || { echo "base checkpoint missing: $BASE_DIR/ppo_latest.pt"; exit 1; }
[ -d "data/decks/matchups/$ARCH" ] || { echo "no pool for archetype '$ARCH' under data/decks/matchups/"; exit 1; }

# The base's last logged iteration anchors the schedule: flat until +320, zero at +400.
BASE_ITERATIONS=$("$PYTHON" - "$BASE_DIR/metrics.jsonl" <<'PY'
import json, sys
print(max(json.loads(line)["iteration"] for line in open(sys.argv[1]) if line.strip()))
PY
)
ANNEAL_START=$((BASE_ITERATIONS + 320))
TARGET=$((BASE_ITERATIONS + 400))

# A pools root that holds only this archetype, so seat 1 always draws from its lists.
POOL="data/decks/matchups_$ARCH"
if [ ! -d "$POOL/$ARCH" ]; then
    mkdir -p "$POOL"
    cp -r "data/decks/matchups/$ARCH" "$POOL/"
fi

ARGS=(
    --run-name "$RUN"
    --prize-labels truth
    --deck-mode focus
    --focus-deck "$DECK"
    --field-pools "$POOL"
    --d-model 128 --num-layers 4 --num-heads 8 --ff-dim 256
    --action-rules counter_cap
    --workers "$WORKERS"
    --games-per-worker 2
    --games-per-iter 512
    --epochs 2
    --resign-threshold 0.95 --resign-persist 6
    --gpu-server --no-worker-model
    --lr 2.5e-4
    --lr-anneal-start "$ANNEAL_START" --lr-anneal-end "$TARGET" --lr-final 0
    --ent-coef 0.002
    --kl-stop 0.5
    --aux-weight 0.1 --attack-aux-weight 0.5
    --iterations "$TARGET"
    --snapshot-every 25 --probe-every "$PROBE_EVERY" --ckpt-every 5 --full-ckpt-every 25
    --seed-base "$SEED"
    --no-label-invariant-checks
)

OWN_CHECKPOINT="runs/$RUN/ppo_latest.pt"
if [ -f "$OWN_CHECKPOINT" ]; then
    echo "resuming $RUN from $OWN_CHECKPOINT"
    ARGS+=(--resume "$OWN_CHECKPOINT")
else
    echo "starting $RUN from $BASE_DIR/ppo_latest.pt (iteration $BASE_ITERATIONS), target $TARGET"
    ARGS+=(--resume "$BASE_DIR/ppo_latest.pt")
fi

exec "$PYTHON" training/train_entry.py "${ARGS[@]}"
