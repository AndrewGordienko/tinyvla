#!/usr/bin/env bash
# Frozen command-0 SmolVLA teacher campaign. Run from a fresh checkout on H200
# after copying data/datasets/command0_multiview_32 and data/models/smolvla_base.
set -euo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL="${MUJOCO_GL:-egl}"
DATA_ROOT="${DATA_ROOT:-data/datasets/command0_multiview_32}"
OUT="${OUT:-data/checkpoints/smolvla_teacher_command0}"
DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-8000}"
BATCH="${BATCH:-32}"
WORKERS="${WORKERS:-8}"

test -f "$DATA_ROOT/meta/info.json"
test -f "$DATA_ROOT/action_semantics.json"
test -f data/models/smolvla_base/model.safetensors
mkdir -p "$OUT"
exec python -m tinyvla.train \
  --repo-id local/command0_multiview_32 --root "$DATA_ROOT" \
  --output "$OUT" --steps "$STEPS" --batch-size "$BATCH" \
  --num-workers "$WORKERS" --device "$DEVICE" --trainable expert \
  --n-action-steps 10 --scheduler none --lr 1e-4 \
  --save-every 500 --log-every 25 \
  --closed-loop-every 500 --closed-loop-commands 0 \
  --closed-loop-cap 220 --closed-loop-episodes 4 \
  --save-best-closed-loop
