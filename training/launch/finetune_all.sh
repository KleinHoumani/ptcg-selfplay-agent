#!/usr/bin/env bash
# Train the 15 matchup specialists of a finished base model, one after another.
#
#   bash training/launch/finetune_all.sh <base-run> <deck.csv>
#
#   bash training/launch/finetune_all.sh dragapult_base decks/dragapult_ex.csv
#
# Each archetype runs through finetune.sh with a fixed seed offset, so a re-run resumes
# whichever specialist was interrupted and skips nothing.
set -euo pipefail

BASE="${1:?usage: finetune_all.sh <base-run> <deck.csv>}"
DECK="${2:?usage: finetune_all.sh <base-run> <deck.csv>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SEED_BASE="${SEED_BASE:-20260812}"

ARCHETYPES=(alakazam archaludon basic_box crustle cynthia_garchomp dragapult festival_lead
            grimmsnarl hydrapple lopunny lucario ns_zoroark slowking starmie team_rocket)

offset=0
for ARCH in "${ARCHETYPES[@]}"; do
    offset=$((offset + 101))
    bash "$HERE/finetune.sh" "$BASE" "$DECK" "$ARCH" $((SEED_BASE + offset))
done
