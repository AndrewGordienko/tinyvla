#!/usr/bin/env bash
# Step B at scale: 400-episode dataset + both levers, honestly measured.
#
# Generates TWO datasets with identical trajectories (deterministic expert, same
# seeds): absolute actions and delta actions. Both are generated CONCURRENTLY and
# stored as IMAGE datasets (--no-videos): video costs a torchcodec seek+decode per
# random training sample; images are a cheap PNG read — much higher dataloader
# throughput at batch 64. Then trains one run on each with n_action_steps=10,
# judging by multi-episode closed-loop success (3 x 6 = 18 rollouts per point).
#
# Run on the GPU box from the repo root:
#   bash scripts/h200_step_b.sh
# Tunables via env vars: SHARDS, EPS_PER_SHARD, STEPS, BATCH, WORKERS, DEVICE.
set -euo pipefail
cd "$(dirname "$0")/.."

export MUJOCO_GL="${MUJOCO_GL:-egl}"
CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu)
SHARDS="${SHARDS:-$(( (CORES - 2) / 2 > 4 ? (CORES - 2) / 2 : 4 ))}"   # two gens run at once
EPS_PER_SHARD="${EPS_PER_SHARD:-$(( 400 / SHARDS + 1 ))}"
STEPS="${STEPS:-20000}"
BATCH="${BATCH:-64}"
WORKERS="${WORKERS:-16}"
DEVICE="${DEVICE:-cuda}"
CL="--closed-loop-every 1000 --closed-loop-commands 0,1,2,3,6,7 \
    --closed-loop-cap 220 --closed-loop-episodes 3 --save-best-closed-loop"

ABS_ROOT=data/datasets/pickplace_abs_400
DELTA_ROOT=data/datasets/pickplace_delta_400

echo "=== 1/3 generate abs + delta datasets concurrently (2 x $SHARDS shards x $EPS_PER_SHARD eps) ==="
if [ ! -d "$ABS_ROOT" ]; then
  python scripts/gen_dataset.py \
    --shards "$SHARDS" --eps-per-shard "$EPS_PER_SHARD" --no-videos \
    --out-repo local/pickplace_abs_400 --out-root "$ABS_ROOT" \
    --shard-dir data/datasets/_shards_abs > data/datasets/_gen_abs.log 2>&1 &
  GEN_ABS=$!
fi
if [ ! -d "$DELTA_ROOT" ]; then
  python scripts/gen_dataset.py \
    --shards "$SHARDS" --eps-per-shard "$EPS_PER_SHARD" --no-videos --delta-actions \
    --out-repo local/pickplace_delta_400 --out-root "$DELTA_ROOT" \
    --shard-dir data/datasets/_shards_delta > data/datasets/_gen_delta.log 2>&1 &
  GEN_DELTA=$!
fi
[ -n "${GEN_ABS:-}" ] && { wait "$GEN_ABS" || { tail -5 data/datasets/_gen_abs.log; exit 1; }; }
[ -n "${GEN_DELTA:-}" ] && { wait "$GEN_DELTA" || { tail -5 data/datasets/_gen_delta.log; exit 1; }; }
rm -rf data/datasets/_shards_abs data/datasets/_shards_delta   # shards are merged; free the disk

echo "=== 2/3 train B1: n_action_steps=10, absolute actions ==="
python -m tinyvla.train \
  --repo-id local/pickplace_abs_400 --root "$ABS_ROOT" \
  --steps "$STEPS" --batch-size "$BATCH" --num-workers "$WORKERS" --device "$DEVICE" \
  --n-action-steps 10 --save-every 2000 \
  --output data/checkpoints/b1_nas10_abs $CL \
  2>&1 | tee data/checkpoints/b1_nas10_abs.log

echo "=== 3/3 train B2: n_action_steps=10 + delta actions ==="
python -m tinyvla.train \
  --repo-id local/pickplace_delta_400 --root "$DELTA_ROOT" \
  --steps "$STEPS" --batch-size "$BATCH" --num-workers "$WORKERS" --device "$DEVICE" \
  --n-action-steps 10 --delta-actions --save-every 2000 \
  --output data/checkpoints/b2_nas10_delta $CL \
  2>&1 | tee data/checkpoints/b2_nas10_delta.log

echo "DONE. Compare 'closed-loop success' lines (18 rollouts/point):"
grep -h "closed-loop" data/checkpoints/b1_nas10_abs.log data/checkpoints/b2_nas10_delta.log || true
echo "Best checkpoints: data/checkpoints/{b1_nas10_abs,b2_nas10_delta}/best_closed_loop"
