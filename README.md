# tinyvla

> **Truth-harness freeze:** do not run H200 training, compression, recovery, or
> DAgger until the local gates in [docs/truth_harness.md](docs/truth_harness.md)
> pass. All SmolVLA work uses the pinned LeRobot 0.4.4 environment and the
> repository-owned corrected padded-action loss.

Small, hackable SO-101 MuJoCo and SmolVLA utilities. The shape is deliberately
closer to tinygrad/PyTorch: importable code lives in `tinyvla/`, commands run via
package modules or installed console scripts, and local datasets/checkpoints stay
out of source.

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
