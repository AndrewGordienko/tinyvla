# tinyvla

> **Result (2026-07-21):** a **291M** distilled SmolVLA student runs locally on an
> Apple M5 Pro (MPS) and retains roughly **70–85%** of the 450M teacher's held-out
> pick-place success (varies with the scene set; confidence intervals overlap), at
> 0.65× the parameters and — after a bf16 cast — **557 MB** on disk (vs 1.1 GB) and
> ~896 MB live MPS memory. Exact numbers, per-command breakdown, 95% CIs, latency,
> and checkpoint SHA256:
> [results/champion_2026-07-21/BENCHMARK_REPORT.md](results/champion_2026-07-21/BENCHMARK_REPORT.md).
>
> **Honest caveats:** absolute success is **not yet deployable** — student ~33%
> (95% CI 27–41%) vs teacher ~42% on hard held-out scenes. Failures are dominated by
> *reaching* and *transport*, i.e. a data-distribution gap, not model capacity. The
> MuJoCo grasp is kinematic scaffolding, not contact-valid physics. Evaluation was
> historically frozen pending the local gates in
> [docs/truth_harness.md](docs/truth_harness.md); those gates now pass and the
> numbers above are reproducible from this repo.

Small, hackable SO-101 MuJoCo and SmolVLA utilities. The shape is deliberately
closer to tinygrad/PyTorch: importable code lives in `tinyvla/`, commands run via
package modules or installed console scripts, and local datasets/checkpoints stay
out of source.

## Champion: 291M distilled student

| Model | Params | On-disk | MPS mem | Replan | Held-out success |
|---|---:|---:|---:|---:|---:|
| Teacher (450M SmolVLA) | 450M | 1.1 GB | 1199 MB | 225 ms | 42% (95% CI 38–47%), 8 cmds |
| **Student (291M, bf16)** | 292M | **557 MB** | 896 MB | 194 ms | 33% (95% CI 27–41%), cmds 1/3/4 |

Held-out over two fresh seeds (n=400 teacher / 180 student), Apple M5 Pro / MPS.
Full per-command breakdown, failure-by-stage, and checkpoint hashes:
[results/champion_2026-07-21/BENCHMARK_REPORT.md](results/champion_2026-07-21/BENCHMARK_REPORT.md).

**Recipe:** prune the task-trained teacher to 12 VLM layers (`paired_even`), then
teacher-distil on clean expert demos (`recover.py --objective mixed_action_expert`,
teacher/expert weights 1.0/0.5) at **lr 3e-6** — higher LR collapses; select
checkpoints on **held-out** seeds (single-seed gates overfit). Distillation stops
helping below 12 layers (a 10-layer/260M model needs a two-phase expert-stabilise →
distil and still drops to ~21%). Finally cast the fp32 weights (the SigLIP vision
tower) to bf16 for a −26% footprint with no measurable accuracy change.

**Live demo** — talk to the arm, toggle 450M ↔ 291M live:

```bash
MUJOCO_GL=glfw .venv/bin/python -m tinyvla.live_demo   # http://localhost:8010
```

**Reproduce the benchmark:**

```bash
MUJOCO_GL=glfw .venv/bin/python -m tinyvla.eval \
  --model artifacts/checkpoints/student291_champion_bf16 \
  --commands 1,3,4 --per-command 30 --seed 3001 --device mps \
  --output artifacts/benchmarks/student_s3001.json
.venv/bin/python scripts/benchmark_report.py    # -> results/.../BENCHMARK_REPORT.md
```

## Layout

- `tinyvla/env.py` — tiny Gym-style `SO101Env` wrapper.
- `tinyvla/task.py` — SO-101 reach task and scripted IK expert.
- `tinyvla/collect.py` — scripted-expert LeRobot dataset collection.
- `tinyvla/train.py`, `tinyvla/eval.py`, `tinyvla/benchmark.py` — policy loops.
- `tinyvla/smolvla_pruned.py` — loader for pruned SmolVLA checkpoints.
- `data/` — local datasets and source checkpoints.
- `examples/` — optional source-checkout shims for common commands.
- `scripts/` — one-off maintenance tools, such as checkpoint pruning.
- `artifacts/` — generated preview images and other disposable outputs.
- `SO-ARM100/` — upstream SO-101 MuJoCo model assets.

The root is intentionally boring: package, config, docs, assets, and coarse
project folders only.

## The arm

6 position-controlled actuators, matching the SmolVLA action space:
`shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`.

## Install

```bash
python3 -m pip install -e .
```

Install LeRobot separately when using dataset or policy commands.

## Quickstart

```bash
python3 -m tinyvla.env --render       # save a still frame to artifacts/so101_frame.png
mjpython -m tinyvla.env --viewer      # interactive viewer on macOS
python3 -m tinyvla.server             # live MJPEG stream on localhost:8000
python3 -m tinyvla.dashboard          # benchmark dashboard on localhost:8765
```

```python
from tinyvla import SO101Env

env = SO101Env()
obs = env.reset()                       # obs = [qpos(6), qvel(6)]
obs, reward, done, info = env.step(env.action_space_sample())
```

The tucked-away examples work from a source checkout too:

```bash
python3 examples/so101_env.py --render
python3 examples/train.py --steps 2
```

## SmolVLA pruning

To produce an action-only checkpoint that drops the language-generation head and
compacts the token embedding to the task vocabulary in a local LeRobot dataset:

```bash
python3 scripts/prune_smolvla.py \
  --source data/models/smolvla_base \
  --dest data/models/smolvla_headless_vocab_so101 \
  --compact-vocab \
  --dataset-root data/datasets/so101_reach
```

Load that pruned checkpoint with:

```python
from tinyvla import load_pruned_smolvla

policy = load_pruned_smolvla("data/models/smolvla_headless_vocab_so101", device="mps")
```

The compact vocab is tied to the task text used to build it. Re-run the pruning
script when adding new instructions.

## Benchmarks

Use `python3 -m tinyvla.benchmark` to compare checkpoints on the same SO-101 reach dataset.
The offline benchmark reports flow-matching loss and, by default, action-chunk
MAE/RMSE against the scripted expert.

Fast loss-only smoke test:

```bash
python3 -m tinyvla.benchmark \
  --model base=data/models/smolvla_base \
  --model pruned=data/models/smolvla_headless_vocab_so101 \
  --device cpu \
  --batch-size 1 \
  --batches 1 \
  --no-action-metric \
  --output artifacts/benchmarks/base_vs_pruned_smoke.json
```

A more useful offline comparison:

```bash
python3 -m tinyvla.benchmark \
  --model base=data/models/smolvla_base \
  --model pruned=data/models/smolvla_headless_vocab_so101 \
  --device mps \
  --batch-size 2 \
  --batches 16 \
  --output artifacts/benchmarks/base_vs_pruned_offline.json
```

For a true held-out benchmark, collect a separate dataset with a different seed
and point the benchmark at it:

```bash
python3 -m tinyvla.collect \
  --episodes 40 \
  --seed 1000 \
  --repo-id local/so101_reach_holdout \
  --root data/datasets/so101_reach_holdout

python3 -m tinyvla.benchmark \
  --model base=data/models/smolvla_base \
  --model pruned=data/models/smolvla_headless_vocab_so101 \
  --repo-id local/so101_reach_holdout \
  --root data/datasets/so101_reach_holdout \
  --device mps \
  --batches 16 \
  --output artifacts/benchmarks/base_vs_pruned_holdout.json
```

Closed-loop simulator rollouts are slower but measure actual task success:

```bash
python3 -m tinyvla.benchmark \
  --model pruned=data/models/smolvla_headless_vocab_so101 \
  --closed-loop \
  --episodes 20 \
  --device mps \
  --output artifacts/benchmarks/pruned_closed_loop.json
```

View benchmark JSON files locally:

```bash
python3 -m tinyvla.dashboard
```

Then open `http://127.0.0.1:8765`.

If that port is busy, choose another:

```bash
python3 -m tinyvla.dashboard --port 8767
```

Interview-facing recorded controller demo (deterministic four-scene replay):

```bash
MUJOCO_GL=glfw .venv/bin/python -m tinyvla.interview_demo --port 8768
```

See [docs/interview_project.md](docs/interview_project.md) for the honest
results table, reproduction commands, and current promotion status.

For pruning checks, compare a candidate against the 450M teacher:

```bash
python3 -m tinyvla.benchmark \
  --teacher teacher=data/models/smolvla_base \
  --model pruned=data/models/smolvla_headless_vocab_so101 \
  --device mps \
  --batches 16 \
  --output artifacts/benchmarks/pruned_vs_450m_teacher.json
```

The default pass gate is normalized `teacher_action_mae <= 0.01` and
`teacher_action_max_abs <= 0.10`. For deeper cuts, keep this teacher check, then
also require held-out task metrics and closed-loop success not to regress much.
