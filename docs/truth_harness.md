# Truth harness

H200 training, compression, recovery, and DAgger are frozen until the local gates
in this document pass. The authoritative environment is Python 3.11 with the
exact dependency versions in `pyproject.toml`, including LeRobot 0.4.4 and the
repository-owned corrected SmolVLA loss.

## Install and test

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e '.[lerobot,test]'
MUJOCO_GL=glfw .venv/bin/pytest -q
```

The runtime records the Git SHA, Python/platform details, pinned package
versions, action representation, and saved `n_action_steps`. It rejects
checkpoint/dataset action mismatches, unexplained tensor gaps, shape mismatches,
and compact vocabularies which do not cover the active dataset instructions.

## Four-scene gate

Generate four command-0 scenes and train only on those scenes:

```bash
MUJOCO_GL=glfw .venv/bin/python -m tinyvla.collect \
  --episodes 4 --commands 0 --seed 4242 --no-videos \
  --repo-id local/truth_gate_command0_4 \
  --root artifacts/truth_harness/datasets/command0_4

.venv/bin/python -m tinyvla.train \
  --repo-id local/truth_gate_command0_4 \
  --root artifacts/truth_harness/datasets/command0_4 \
  --output artifacts/truth_harness/checkpoints/command0_overfit \
  --steps 500 --batch-size 4 --n-action-steps 5 --device mps --seed 4242
```

Then run both the exact-scene overfit check and the deterministic 20-scene
held-out check:

```bash
.venv/bin/python -m tinyvla.gates \
  --model artifacts/truth_harness/checkpoints/command0_overfit \
  --held-out 20 --device mps
```

The gate requires at least 95% memorized-scene success and 80% held-out success.
Commands 0–3, stacking commands 4–5, and two-step commands 6–7 are always
reported as separate groups.
