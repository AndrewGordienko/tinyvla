#!/usr/bin/env bash
# Unfreeze-at-scale experiment (run AFTER step B — highest cost, lowest prior).
#
# Tests whether unfreezing the VLM brain helps at scale, done safely:
#   --trainable brain_visual  unfreezes vision+connector+text (avoids the silent
#                             no-op where freeze_vision_encoder=False alone does
#                             nothing because train_expert_only re-freezes it)
#   --backbone-lr-scale 0.1   backbone moves 10x slower than the expert
#   --warmup-steps 500        early expert gradients are noise; don't let them
#                             wreck pretrained features at full LR
# Judged by the same multi-episode closed-loop metric as B — compare against the
# frozen B winner; small-scale evidence says frozen wins, this settles it.
#
#   bash scripts/h200_unfreeze.sh                # absolute-action dataset
#   DELTA=1 bash scripts/h200_unfreeze.sh        # delta dataset (if B2 won)
set -euo pipefail
cd "$(dirname "$0")/.."

export MUJOCO_GL="${MUJOCO_GL:-egl}"
STEPS="${STEPS:-20000}"
BATCH="${BATCH:-64}"
WORKERS="${WORKERS:-16}"
DEVICE="${DEVICE:-cuda}"
TRAINABLE="${TRAINABLE:-brain_visual}"

DELTA_FLAG=""; ROOT=data/datasets/pickplace_abs_400; REPO=local/pickplace_abs_400; SUFFIX=abs
if [ "${DELTA:-0}" = "1" ]; then
  DELTA_FLAG="--delta-actions"; ROOT=data/datasets/pickplace_delta_400
  REPO=local/pickplace_delta_400; SUFFIX=delta
fi

OUT="data/checkpoints/unfreeze_${TRAINABLE}_${SUFFIX}"
python -m tinyvla.train \
  --repo-id "$REPO" --root "$ROOT" \
  --steps "$STEPS" --batch-size "$BATCH" --num-workers "$WORKERS" --device "$DEVICE" \
  --n-action-steps 10 $DELTA_FLAG \
  --trainable "$TRAINABLE" --backbone-lr-scale 0.1 --warmup-steps 500 \
  --closed-loop-every 1000 --closed-loop-commands 0,1,2,3,6,7 \
  --closed-loop-cap 220 --closed-loop-episodes 3 --save-best-closed-loop \
  --output "$OUT" \
  2>&1 | tee "$OUT.log"

echo "DONE. Compare closed-loop lines vs the frozen B run on the SAME dataset."
