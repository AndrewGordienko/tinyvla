#!/usr/bin/env bash
# Frozen command-0 SmolVLA teacher campaign. Run from a fresh checkout on H200
# after copying data/datasets/command0_multiview_32 and data/models/smolvla_base.
set -euo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL="${MUJOCO_GL:-egl}"
DATA_ROOT="${DATA_ROOT:-data/datasets/command0_multiview_32}"
OUTPUT_DIR="${OUTPUT_DIR:-data/checkpoints/smolvla_teacher_command0}"
DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-8000}"
TOTAL_STEPS="${TOTAL_STEPS:-$STEPS}"
STOP_AFTER="${STOP_AFTER:-}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_EVERY="${SAVE_EVERY:-500}"
EVAL_EVERY="${EVAL_EVERY:-500}"
RESUME="${RESUME:-}"
PYTHON="${PYTHON:-python}"
VERSIONED_CHECKPOINTS="${VERSIONED_CHECKPOINTS:-1}"
SCHEDULER="${SCHEDULER:-none}"
FIXED_BATCH="${FIXED_BATCH:-0}"
FIXED_NOISE="${FIXED_NOISE:-0}"
LR="${LR:-1e-4}"

test -f "$DATA_ROOT/meta/info.json"
test -f "$DATA_ROOT/action_semantics.json"
test -f data/models/smolvla_base/model.safetensors
mkdir -p "$OUTPUT_DIR"
CMD=("$PYTHON" -m tinyvla.train \
  --repo-id local/command0_multiview_32 --root "$DATA_ROOT" \
  --output "$OUTPUT_DIR" --steps "$STEPS" --total-steps "$TOTAL_STEPS" --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" --device "$DEVICE" --trainable expert \
  --n-action-steps 10 --scheduler "$SCHEDULER" --lr "$LR" \
  --save-every "$SAVE_EVERY" --log-every 25 \
  --closed-loop-every "$EVAL_EVERY" --closed-loop-commands 0 \
  --closed-loop-cap 220 --closed-loop-episodes 4 \
  --save-best-closed-loop)
if [ -n "$RESUME" ]; then CMD+=(--resume "$RESUME"); fi
if [ "$VERSIONED_CHECKPOINTS" = "1" ]; then CMD+=(--versioned-checkpoints); fi
if [ "$FIXED_BATCH" = "1" ]; then CMD+=(--fixed-batch); fi
if [ "$FIXED_NOISE" = "1" ]; then CMD+=(--fixed-noise); fi
if [ -n "$STOP_AFTER" ]; then CMD+=(--stop-after "$STOP_AFTER"); fi
exec "${CMD[@]}"
