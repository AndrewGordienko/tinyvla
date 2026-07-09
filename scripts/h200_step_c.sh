#!/usr/bin/env bash
# Step C at scale: the targeted-data DAgger loop (recovery data for the states
# the policy actually drifts into — the near-miss-grasp failure mode).
#
# Each round: rebuild dataset from pool -> train from base -> score every command
# closed-loop (3 episodes each) -> grow the pool at the worst commands
# (curriculum expert demos + DAgger relabels). Watch "mean success by round".
#
# Run on the GPU box from the repo root, AFTER step B decides the delta question:
#   bash scripts/h200_step_c.sh                 # absolute actions
#   DELTA=1 bash scripts/h200_step_c.sh         # if B2 (delta) won
#   WARM=1 bash scripts/h200_step_c.sh          # warm-start rounds >1 from the
#                                               # previous round (WARM_STEPS each,
#                                               # default 3000) — ~2x faster loop,
#                                               # slight departure from batch-DAgger
# Tunables: ROUNDS, STEPS, BATCH, WORKERS, DEVICE, SEED_PER, CURR_PER, DAGGER_PER.
set -euo pipefail
cd "$(dirname "$0")/.."

export MUJOCO_GL="${MUJOCO_GL:-egl}"
ROUNDS="${ROUNDS:-6}"
STEPS="${STEPS:-6000}"
BATCH="${BATCH:-64}"
WORKERS="${WORKERS:-16}"
DEVICE="${DEVICE:-cuda}"
SEED_PER="${SEED_PER:-60}"       # 60 x 6 commands = 360-episode seed pool
CURR_PER="${CURR_PER:-20}"
DAGGER_PER="${DAGGER_PER:-20}"

DELTA_FLAG=""
SUFFIX="abs"
if [ "${DELTA:-0}" = "1" ]; then DELTA_FLAG="--delta-actions"; SUFFIX="delta"; fi
WARM_FLAGS=""
if [ "${WARM:-0}" = "1" ]; then WARM_FLAGS="--warm-start --warm-steps ${WARM_STEPS:-3000}"; fi

python -m tinyvla.dagger_loop \
  --rounds "$ROUNDS" --steps "$STEPS" --batch-size "$BATCH" --num-workers "$WORKERS" \
  --commands 0,1,2,3,6,7 \
  --seed-per "$SEED_PER" --curriculum-per "$CURR_PER" --dagger-per "$DAGGER_PER" --worst-k 3 \
  --collect-workers 6 \
  --n-action-steps 10 $DELTA_FLAG $WARM_FLAGS \
  --closed-loop-cap 220 --closed-loop-episodes 3 \
  --device "$DEVICE" \
  --pool "data/datasets/dagger_pool_$SUFFIX" \
  --work "data/checkpoints/dagger_run_$SUFFIX" \
  2>&1 | tee "data/checkpoints/dagger_run_$SUFFIX.log"

echo "DONE. Round-by-round: data/checkpoints/dagger_run_$SUFFIX/dagger_history.json"
echo "Teacher for distillation: last round's best_closed_loop directory."
