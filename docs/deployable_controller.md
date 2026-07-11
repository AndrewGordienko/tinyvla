# Deployable temporal controller (experiment branch)

Goal: reach **4/4 on the four memorized command-0 scenes using only deployable
observations** (no simulator-only state), then broaden layouts, then train the
450M SmolVLA, then compress. Small deployable encoder (ResNet-18-class), not
parameter scaling — the point is to prove the observation + control formulation.

Staged, gated build (do NOT launch long multi-seed runs until the 64-sample
supervised gate AND the one-seed four-scene gate pass):

1. Observation audit ✅ (below)
2. Supervised 64-sample overfit gate (+ shuffled-frame / blank-image controls)
3. Architecture ladder (single-frame → 4-frame stack → +multi-view → +action chunks)
4. Action-chunk labelling via snapshot-restore expert rollout (chunks 1/4/8)
5. One-seed four-scene gate (≥3/4, pref 4/4; stage completion + videos)
6. Only then: 3 seeds, ≥32 layouts, held-out distance bins, A/B/C controls

Deployable state everywhere: `qpos6` (incl. gripper joint position) + `qvel6` +
`prev_action6`. **No grasped bit.**

## Commands and gates

The default command runs only the supervised gate. It does not collect DAgger
states or evaluate held-out layouts:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 MUJOCO_GL=glfw .venv/bin/python -m scripts.deployable_controller \
  --configs single_frame,temporal,multiview \
  --output artifacts/truth_harness/deployable_supervised_gate.json
```

The JSON records normalized full-model and image-only overfit error plus the
image-only blank-image, swapped-image, and (for temporal variants) shuffled-frame
prediction changes. A temporal run must pass before any DAgger command runs.

The one-seed four-scene experiment is explicitly separate and is blocked by that
saved temporal result. It emits approach, grasp, transport, release, and final
success independently, and writes all successful plus one representative failing
rollout video. It does **not** expand layouts or seeds:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 MUJOCO_GL=glfw .venv/bin/python -m scripts.deployable_controller \
  --four-scene --architecture temporal --seed 0 --rounds 4 --replan-actions 1 \
  --gate-json artifacts/truth_harness/deployable_supervised_gate.json \
  --output artifacts/truth_harness/deployable_temporal_seed0.json
```

For the chunk rung, set `--architecture multiview_chunk --action-chunk 4` (or
`8`) after the temporal gate passes. Chunks are labelled at **each learner
state** by snapshotting that state, rolling the reactive expert forward for the
whole chunk, then restoring the learner simulator. They are never made from a
later learner trajectory. Evaluation replans after 1–4 predicted actions.

## 1. Observation audit (`scripts/observation_audit.py`)

| property | value |
|---|---|
| cameras in env model | **`front`, `wrist`** (no "top" camera exists) |
| views recorded in dataset | **`observation.images.front` only** (256×256×3) |
| SmolVLA base configured cameras | camera1/2/3, but the dataset override feeds it **`front` only** |
| current CNN views | `front` only |
| control / frame rate | **25 Hz** (sim 500 Hz, 20 substeps); fps 25 |
| image latency | **0 steps** — synchronous sim render (no capture lag) |
| dataset resolution | 256×256×3; CNN downsamples to 84×84 |
| simulator-only state in policy | **none** |

**Cube occlusion during grasp (front view, red-cube pixel fraction, 4 scenes):**

| | pre-grasp | at-grasp min | drop ratio |
|---|---|---|---|
| front | 0.00252 | 0.00060 | **0.24** (≈76% occluded) |

The target cube is small (~0.25% of pixels ≈ 17 px at 84², ~4 px at grasp) and is
**~76% occluded by the gripper/arm during grasp** in the front view. The wrist view
sees it well pre-grasp in 3/4 scenes but also occludes at contact. This is direct
evidence that a single instantaneous frame cannot encode contact/grasp/phase →
motivates **temporal frames** (infer grasp from motion before occlusion) and a
**wrist view** (better pre-grasp localization).

### Implication for the ladder
- Temporal frames should help most around grasp/contact (phase inference).
- A wrist view should help pre-grasp approach localization.
- A stronger encoder / the raw 256² (vs 84²) may help given how few cube pixels exist.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 MUJOCO_GL=glfw .venv/bin/python -m scripts.observation_audit
```
