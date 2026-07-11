# SmolVLA command-0 teacher campaign

The deployable command-0 demonstrations were rebuilt with both simulator cameras:

```bash
MUJOCO_GL=glfw .venv/bin/python -m tinyvla.collect \
  --episodes 32 --seed 2000 --commands 0 --cameras front,wrist --no-videos \
  --repo-id local/command0_multiview_32 \
  --root data/datasets/command0_multiview_32
```

The local audit is written to `artifacts/smolvla_command0_build_20260711/`.
It records dataset and sample hashes, 32 episode boundaries, absolute action
semantics, 2,180 frames, 25 Hz control, 256×256 images, and synchronized front
and wrist observations. All examples are marked `source_provenance=demonstration`;
recovery examples have not yet been added.

Before a GPU campaign, run the real base-checkpoint smoke test:

```bash
MUJOCO_GL=egl .venv/bin/python scripts/smolvla_teacher_smoke.py \
  --model data/models/smolvla_base \
  --repo-id local/command0_multiview_32 \
  --root data/datasets/command0_multiview_32 \
  --device cuda \
  --out artifacts/smolvla_command0_build_20260711/teacher_smoke_save_reload.json
```

The frozen H200 teacher command is:

```bash
MUJOCO_GL=egl bash scripts/h200_smolvla_teacher_command0.sh
```

It trains the real base checkpoint with the corrected loss, expert/state/projector
parameters, 10-step replanning, incremental checkpoints every 500 updates, and
four canonical command-0 closed-loop evaluations every 500 updates. Select
`best_closed_loop`, never the last checkpoint. The historical H200 endpoint was
unreachable during this audit (SSH connection refused), so no long job was
launched or represented as complete.

The recovery/DAgger campaign remains gated on this teacher pilot. Its dataset
must retain provenance and sample at least 50% original demonstrations in every
aggregate round before it is allowed to run.

## MPS rehearsal

The bounded local rehearsal uses the same script and production DataLoader:

```bash
PYTHON=.venv/bin/python PYTORCH_ENABLE_MPS_FALLBACK=1 MUJOCO_GL=glfw \
DEVICE=mps STEPS=10 BATCH_SIZE=1 NUM_WORKERS=0 SAVE_EVERY=10 EVAL_EVERY=0 \
VERSIONED_CHECKPOINTS=1 SCHEDULER=config FIXED_BATCH=1 FIXED_NOISE=1 \
OUTPUT_DIR=data/checkpoints/smolvla_mps_rehearsal \
bash scripts/h200_smolvla_teacher_command0.sh

PYTHON=.venv/bin/python PYTORCH_ENABLE_MPS_FALLBACK=1 MUJOCO_GL=glfw \
DEVICE=mps STEPS=25 BATCH_SIZE=1 NUM_WORKERS=0 SAVE_EVERY=25 EVAL_EVERY=25 \
VERSIONED_CHECKPOINTS=1 SCHEDULER=config FIXED_BATCH=1 FIXED_NOISE=1 \
OUTPUT_DIR=data/checkpoints/smolvla_mps_rehearsal \
RESUME=data/checkpoints/smolvla_mps_rehearsal \
bash scripts/h200_smolvla_teacher_command0.sh
```

`training_state.pt` now preserves optimizer, scheduler, RNG, and global step;
`checkpoint_step_10` and `checkpoint_step_25` are immutable rehearsal snapshots.
The verification artifact includes hashes, parameter updates, exact action
save/reload comparison, optimizer/scheduler steps, finite rollout action range,
and a short video. The flow-matching fixed-batch loss is reported honestly as a
numerical diagnostic; it is not a behavioral acceptance gate.

The fixed-tuple LR diagnostic is in `artifacts/fixed_tuple_lr_diagnostic_20260711/diagnostic.json`.
It hashes raw indices, processed tensors, padding/language fields, fixed noise and
timestep, target velocity, and normalization statistics at steps 0/10/25. The
selected rehearsal/teacher LR is `3e-5`: it reduced eval-mode fixed-tuple loss
from `1.194` to `0.0152` with exact save/reload action equality. `1e-4` reached a
similar endpoint but is more aggressive; `1e-5` converged more slowly.

The bounded CUDA pilot command, once H200 access is restored, is:

```bash
DEVICE=cuda TOTAL_STEPS=8000 STOP_AFTER=500 BATCH_SIZE=32 NUM_WORKERS=8 \
LR=3e-5 SAVE_EVERY=100 EVAL_EVERY=100 \
OUTPUT_DIR=data/checkpoints/smolvla_teacher_command0 \
bash scripts/h200_smolvla_teacher_command0.sh
```

Promote to the long run only after its held-fixed loss, finite bounded actions,
stage metric, and CUDA save/resume checks pass.
