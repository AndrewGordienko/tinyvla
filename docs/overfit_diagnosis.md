# Four-scene overfit failure — diagnosis (in progress)

The command-0 four-scene overfit gate fails (0/4). **The cause is not yet
established.** The failure is currently localized to **approach accuracy,
gripper timing, or their interaction** — this document tracks the experiments
that will actually discriminate between them. Nothing below rules out frozen
visual representation, chunk/queue staleness, or gripper calibration until the
causal counterfactual and corrected-metric tests are complete.

Corrections to the first-pass diagnosis (which overstated the evidence):
- Old "Gate B round-trip" only compared **stored statistics**, never a real
  physical→preprocess→postprocess→physical round-trip. Rebuilt as a numerical
  round-trip below.
- Old Gates C/P computed action error across the **whole chunk without masking
  `action_is_pad`** and without separating the executed prefix `0:n_action_steps`
  from unused future predictions. Raw radians were compared across differently
  ranged actuators, and out-of-range predictions were called failures even though
  the environment **clips before simulation**. All corrected below.
- "Frozen representation ruled out" was a logic error: the action expert being
  trainable does not mean the frozen VLM has the spatial precision required.
- "Flow matching smears a bimodal switch" is a **hypothesis**, not a demonstrated
  mechanism.

Every gate is reproducible via `scripts/diagnose_overfit.py` and
`scripts/hybrid_rollout.py` against the pinned environment.

Dataset under test: `artifacts/truth_harness/datasets/command0_4`
(4 command-0 scenes, absolute actions, fps 25, 283 frames).
Checkpoint under test: `artifacts/truth_harness/checkpoints/command0_overfit_500`
(base 450M, 500 steps, batch 4, lr 1e-4, `n_action_steps=5`, seed 4242).

## Gate results (corrected)

| Gate | Question | Result | Verdict |
|------|----------|--------|---------|
| **A** expert replay | Do stored dataset actions + reset + replay reproduce success? | **4/4**; min ee→cube ≈ 0.7 cm | data / action representation / reset / replay **correct** |
| **B** round-trip (numerical) | Does physical → normalize → postprocess recover the physical action? | per-dim max err **6e-8** (< tol 1e-4), pad-masked | normalization genuinely **invertible** |
| trainability | Is the representation frozen? | base ships VLM frozen; **~100M expert trainable** by default | not a *fully* frozen model — but the frozen **visual** features are **not** ruled out (see below) |
| **C** single-batch | Can the pipeline drive ONE fixed batch's flow loss → 0? | flow loss **1.82 → ~0.04** | training path **healthy**, no convergence bug |
| **P** per-dimension (corrected) | Which dim carries error on the *executed prefix*, pad-masked, range-normalized? | range-norm MAE: arm **0.02–0.06**, gripper **0.28**; gripper closed-F1 **0.70** | gripper is the noisiest dim open-loop — but see counterfactuals |

### Causal counterfactual rollouts (the decisive test)

Executed-action components swapped between the learned policy and the scripted
reactive expert, on the four memorized scenes, canonical 4 cm radius unless noted.
`scripts/hybrid_rollout.py`.

| Condition | success | grasped | lifted | mean min ee→cube |
|-----------|:-------:|:-------:|:------:|:----------------:|
| **E** all expert (sanity) | **4/4** | 4 | 4 | 0.013 |
| **C** expert arm + **learned gripper** | **4/4** | 4 | 4 | 0.013 |
| **A** full learned | 1/4 | 3 | 2 | 0.041 |
| **D** learned arm + thresholded learned gripper | 1/4 | 3 | 2 | 0.039 |
| **A @ 5 cm** (non-canonical diagnostic) | 1/4 | 4 | 2 | 0.041 |
| **B** learned arm + expert gripper | 0/4 | 0 | 0 | 0.026 |
| n_action_steps = 1 / 5 / 10 (cond A) | 0 / 1 / 2 of 4 | 4/3/4 | 1/2/3 | 0.035 / 0.041 / 0.037 |

Per-scene breakdown of full learned (A): scene1 success; scene0 grasped+lifted but
**place-fail** (arm fails to carry to bin); scenes 2 & 3 **approach-fail** (arm
never reaches < 4 cm, min 4.9–5.0 cm). The gripper closes correctly whenever the
arm gets near (e.g. gripper cmd 0.28 when ee < 4 cm in scene0).

### CPU vs MPS (`scripts/device_check.py`)

Open-loop chunk divergence on identical seeded inputs: **mean 7e-4 rad**, max
**0.036 rad on the gripper in one scene**. Negligible for the arm; enough to flip
a borderline grasp at the 4 cm rim, but not the root cause.

## Mechanistic conclusion (evidence-based)

**The primary blocker is the learned ARM trajectory (approach and carry/place),
not the gripper and not queue staleness.**

- **The learned gripper is adequate.** Condition C (expert arm + learned gripper)
  succeeds **4/4** — with a correctly positioned arm, the learned gripper closes,
  grasps, and places every time. The high open-loop gripper error (Gate P) does
  **not** break the task, because near the cube the policy has many timesteps and
  only needs to close once inside the radius.
- **The learned arm is the deficit.** Full learned (A) reaches only ~4–5 cm and
  completes 1/4; two scenes stall at ~5 cm (never enters the 4 cm radius) and one
  is picked but not delivered. The 5 cm diagnostic does **not** rescue it (still
  1/4), so it is not a few-mm threshold issue — the arm trajectory is broadly
  imprecise across approach *and* placement.
- **Queue staleness is not the cause.** `n_action_steps = 10` (≥ `1`) is no worse
  than fully closed-loop; more stale actions slightly *helps*.
- Condition B (0/4) is partly a **controller-mismatch artifact**: the reactive
  expert only closes its gripper when *its* geometry says ee is within 2 cm xy,
  which the learned arm rarely satisfies — so B is a weaker signal than C. It is
  still consistent with an arm-localization deficit.

**Not yet ruled out — and now the leading mechanism to test:** whether the arm
imprecision comes from **under-optimization** (500 steps; Gate C shows arm error
keeps falling with training) or from **insufficient frozen visual features**
(the VLM/vision encoder is frozen, which may cap spatial precision). These two
are what the next experiments must separate. Do **not** claim frozen
representation is ruled out.

The **grasp radius is not the bug** — do not loosen `GRASP_RADIUS`; the 5 cm run
is a labelled diagnostic only.

## Exact next experiment

Separate under-optimization from insufficient frozen visual features — do not
just train longer blindly:

1. **Baseline controls first (cheapest, most decisive).**
   - Privileged-state MLP: cube xy + robot state → action chunk. If it overfits
     the 4 scenes, the task and dynamics are learnable from clean state.
   - Small image CNN + MLP behavioural-cloning policy. If it overfits from
     *images* but SmolVLA cannot, the fault is specifically SmolVLA's (frozen)
     visual adaptation, not the task or observations.
2. **Trainability sweep (separately, not mixed), short-trajectory first:**
   `expert` only vs `expert + connector` vs `expert + last-2 text/VLM layers` vs
   full model with a small backbone LR. Record trainable-param counts and
   identical-seed results. This directly tests the frozen-visual-features
   hypothesis for the arm.
3. **Only if the evidence points to under-optimization:** run the 2000-step
   command-0 sweep (checkpoints at 50/100/250/500/1000/2000, deterministic eval)
   and re-check the arm approach/place error.

Only after the four scenes reach 4/4 at the canonical 4 cm radius should the
450M teacher be established on commands 0–3 (fixed seeds, separate
selection/test scenes, ≥3 seeds, per-command CIs). Compression / recovery /
DAgger / H200 stay frozen until then.

## Milestone 2: baselines + frozen-feature probe (perception vs control)

### Privileged-state MLP baseline (`scripts/baseline_mlp.py`)

75k-param MLP, one-step closed-loop; inputs = robot qpos/qvel + ee xyz + target
cube xyz + destination xyz + grasped + previous action (28-dim); output = 6
physical actuator targets; trained on the four command-0 demonstrations.

- Fits the 283 demonstration pairs to **loss 1e-5**, yet reaches only **1/4**
  closed-loop at 4 cm (approach 3/4, grasp 3/4, lift 3/4, carry 2/4, **place 1/4**).

### Frozen-feature spatial probe (`scripts/frozen_probe.py`)

300 randomized command-0 scenes (arm at home), held-out split by position. Decode
the red cube's xyz (linear + small MLP) from each representation:

| representation | probe | eucl mean | median | <1 cm | <2 cm | <4 cm |
|---|---|---|---|---|---|---|
| raw pixels 64² (tiny CNN) | cnn | 6.44 | 5.47 | 0% | 12% | 27% |
| frozen vision-encoder (768d, mean-pool) | mlp | 1.62 | 1.24 | 37% | 75% | 93% |
| frozen connector (960d, mean-pool) | mlp | 1.57 | 1.24 | 38% | 75% | 93% |

### Narrow, honest conclusion (supersedes "under-optimization vs frozen features")

- **SmolVLA's frozen features encode the cube position** — a *linear* probe on the
  pooled connector tokens localizes to ~1.6 cm mean (93% within 4 cm), and the
  expert consumes the richer *un-pooled* 64-token grid, so this is a lower bound.
  Perception is **not** the primary bottleneck; **do not unfreeze vision on this
  evidence**.
- **What is actually proven:** *one-step BC on these four stateful-expert
  demonstrations is not closed-loop stable even with perfect privileged spatial
  state.* This does **not** yet prove generic covariate shift is the sole cause.
- **Major unresolved confound:** the demonstrations come from the stateful
  `expert_action()`, whose target depends on hidden `phase`/`phase_t`, which the
  policy never observes. Around approach/close/lift/drop transitions, nearly
  identical observations can have different correct actions — i.e. **label
  ambiguity / partial observability**, distinct from ordinary covariate shift.

### Probe caveats (do not over-read)

- Mean-pooling can destroy positional info still present in the token grid — a
  spatial-token head is the fair test (TODO).
- Cube z is nearly constant in ungrasped initial scenes, so xyz overstates
  accuracy; report x/y separately.
- Initial-home-pose frames do not test approach/grasp/carry states where control
  fails. The raw-pixel CNN here is a weak learner (64², 180 samples), not a valid
  upper bound.

### Next: separate Markovity from covariate shift (before any SmolVLA sweep)

1. Nearest-neighbour ambiguity audit in privileged-state space (conflicting
   actions for near-identical states, around each phase transition); report **max**
   physical action error, not just mean.
2. Oracle-phase privileged MLP (add one-hot phase + phase_t + step_idx). If 1/4 →
   4/4, the demonstrations are partially observable for the policy.
3. Stateless **reactive-expert** dataset (regenerate the four scenes with
   `reactive_action()`), retrain the identical MLP with no phase. If it passes,
   the stateful generator's labels were the problem.
4. Small perturbation-recovery diagnostic (reactive-expert relabelling), original
   vs original+recovery.
5. Teacher-forced vs closed-loop deviation (where the first irreversible error
   occurs).
6. Corrected probe: spatial-token heads (not mean-pool) over dynamic states
   (approach/grasp/carry), x/y separate.

The trainability sweep and 2000-step run stay deferred until this is resolved.
Compression / recovery / distillation / full DAgger / H200 remain frozen.

## Reproduce

```bash
MUJOCO_GL=glfw .venv/bin/python -m scripts.diagnose_overfit --gate A   # expert replay
PYTORCH_ENABLE_MPS_FALLBACK=1 MUJOCO_GL=glfw .venv/bin/python -m scripts.diagnose_overfit --gate B   # round-trip
PYTORCH_ENABLE_MPS_FALLBACK=1 MUJOCO_GL=glfw .venv/bin/python -m scripts.diagnose_overfit --gate C   # single-batch (slow)
PYTORCH_ENABLE_MPS_FALLBACK=1 MUJOCO_GL=glfw .venv/bin/python -m scripts.diagnose_overfit --gate P   # corrected per-dim
PYTORCH_ENABLE_MPS_FALLBACK=1 MUJOCO_GL=glfw .venv/bin/python -m scripts.hybrid_rollout   # counterfactuals + nstep
PYTORCH_ENABLE_MPS_FALLBACK=1 MUJOCO_GL=glfw .venv/bin/python -m scripts.device_check     # CPU vs MPS
```
