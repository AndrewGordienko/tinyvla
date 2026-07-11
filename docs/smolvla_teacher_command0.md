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
