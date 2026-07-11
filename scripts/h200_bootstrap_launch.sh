#!/usr/bin/env bash
# Idempotent Vast.ai bootstrap + durable 500-step SmolVLA pilot launcher.
# Usage: SSH_PORT=12345 SSH_HOST=sshX.vast.ai bash scripts/h200_bootstrap_launch.sh
set -euo pipefail
cd "$(dirname "$0")/.."

: "${SSH_HOST:?set SSH_HOST to the current Vast.ai hostname}"
: "${SSH_PORT:?set SSH_PORT to the current Vast.ai SSH port}"
REMOTE="root@${SSH_HOST}"
PORT="$SSH_PORT"
COMMIT="df441e8"
REMOTE_ROOT="${REMOTE_ROOT:-/root/tinyvla}"
DATA_ROOT="data/datasets/command0_multiview_32"
BASE_ROOT="data/models/smolvla_base"
DATA_SHA="686808ab96fed5d3005b8cbf8d0351d7cb66f9e77d61f8827913f367f270fbd7"
BASE_SHA="b818911fd8cbf2bc7fd7a3752d0c25f3316b3399c2ecc86d2788ec9d3b49dafd"

SSH=(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p "$PORT")
RSYNC=(rsync -az --info=progress2 -e "ssh -o StrictHostKeyChecking=no -p $PORT")

"${SSH[@]}" "$REMOTE" "mkdir -p '$REMOTE_ROOT'"
"${SSH[@]}" "$REMOTE" "if [ ! -d '$REMOTE_ROOT/.git' ]; then git clone https://github.com/AndrewGordienko/tinyvla.git '$REMOTE_ROOT'; fi"
"${RSYNC[@]}" --exclude='.git' ./ "$REMOTE:$REMOTE_ROOT/"
"${SSH[@]}" "$REMOTE" bash -s -- "$REMOTE_ROOT" "$COMMIT" "$DATA_SHA" "$BASE_SHA" <<'REMOTE_BOOTSTRAP'
set -euo pipefail
ROOT="$1"; COMMIT="$2"; EXPECTED_DATA="$3"; EXPECTED_BASE="$4"
DATA_ROOT="data/datasets/command0_multiview_32"; BASE_ROOT="data/models/smolvla_base"
cd "$ROOT"
test "$(git status --porcelain)" = ""
git fetch origin exp/deployable-temporal-controller
git checkout --detach "$COMMIT"
test "$(git rev-parse HEAD)" = "$COMMIT"
command -v nvidia-smi >/dev/null
GPU_LINE="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)"
echo "$GPU_LINE" | grep -qi 'H200' || { echo "wrong GPU: $GPU_LINE" >&2; exit 2; }
python3 --version
if [ ! -x .venv/bin/python ]; then python3 -m venv .venv; fi
.venv/bin/pip install -e '.[lerobot,test]'
DATA_SHA="$(.venv/bin/python -c "from tinyvla.runtime import sha256_tree; print(sha256_tree('$ROOT/$DATA_ROOT'))")"
BASE_SHA="$(.venv/bin/python -c "from tinyvla.runtime import sha256_tree; print(sha256_tree('$ROOT/$BASE_ROOT'))")"
test "$DATA_SHA" = "$EXPECTED_DATA" || { echo "dataset hash mismatch: $DATA_SHA" >&2; exit 3; }
test "$BASE_SHA" = "$EXPECTED_BASE" || { echo "base hash mismatch: $BASE_SHA" >&2; exit 4; }
OUT="$ROOT/data/checkpoints/smolvla_teacher_command0"
test ! -e "$OUT" || { echo "refusing to reuse existing output: $OUT" >&2; exit 5; }
mkdir -p "$ROOT/artifacts/h200_teacher_pilot"
MUJOCO_GL=egl .venv/bin/python -m tinyvla.audit --model "$ROOT/$BASE_ROOT" --base-checkpoint \
  --repo-id local/command0_multiview_32 --root "$ROOT/$DATA_ROOT" --device cuda --batch-size 32 \
  --trainable expert > "$ROOT/artifacts/h200_teacher_pilot/cuda_smoke.json" 2>&1
cat > "$ROOT/artifacts/h200_teacher_pilot/launch_metadata.json" <<META
{"pid":null,"commit":"$COMMIT","git_sha":"$(git rev-parse HEAD)","gpu":$(printf '%s' "$GPU_LINE" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))'),"dataset_sha":"$DATA_SHA","base_sha":"$BASE_SHA","start_time":"$(date -u +%Y-%m-%dT%H:%M:%SZ)","command":"DEVICE=cuda TOTAL_STEPS=8000 STOP_AFTER=500 BATCH_SIZE=32 NUM_WORKERS=8 LR=3e-5 SAVE_EVERY=100 EVAL_EVERY=100 OUTPUT_DIR=data/checkpoints/smolvla_teacher_command0 MUJOCO_GL=egl bash scripts/h200_smolvla_teacher_command0.sh"}
META
tmux new-session -d -s smolvla_teacher "cd '$ROOT' && MUJOCO_GL=egl DEVICE=cuda TOTAL_STEPS=8000 STOP_AFTER=500 BATCH_SIZE=32 NUM_WORKERS=8 LR=3e-5 SAVE_EVERY=100 EVAL_EVERY=100 OUTPUT_DIR=data/checkpoints/smolvla_teacher_command0 bash scripts/h200_smolvla_teacher_command0.sh 2>&1 | tee artifacts/h200_teacher_pilot/train.log"
PID="$(tmux list-panes -t smolvla_teacher -F '#{pane_pid}' | head -1)"
.venv/bin/python - <<PY
import json
p='$ROOT/artifacts/h200_teacher_pilot/launch_metadata.json'; x=json.load(open(p)); x['pid']=int('$PID'); open(p,'w').write(json.dumps(x,indent=2)+'\n')
PY
echo "launched tmux=smolvla_teacher pid=$PID"
REMOTE_BOOTSTRAP
