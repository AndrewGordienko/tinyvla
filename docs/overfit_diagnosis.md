# Four-scene overfit failure — diagnosis

The command-0 four-scene overfit gate fails (0/4). This is a **systematic
localization of the cause**, not a "train longer" guess. Every gate is
reproducible via `scripts/diagnose_overfit.py` against the pinned environment.

Dataset under test: `artifacts/truth_harness/datasets/command0_4`
(4 command-0 scenes, absolute actions, fps 25, 283 frames).
Checkpoint under test: `artifacts/truth_harness/checkpoints/command0_overfit_500`
(base 450M, 500 steps, batch 4, lr 1e-4, `n_action_steps=5`, seed 4242).

## Gate results

| Gate | Question | Result | Verdict |
|------|----------|--------|---------|
| **A** expert replay | Do stored dataset actions + reset + replay reproduce success? | **4/4**; min ee→cube ≈ 0.7 cm; grasp fires ~t22 | data / action representation / reset / replay **correct** |
| **B** normalization | Do checkpoint action stats == dataset stats (train vs eval)? | max abs delta **2e-8** | normalization **consistent**; no train/eval stats drift |
| trainability | Is the representation frozen? | base ships VLM frozen; **~100M expert trainable** under default `checkpoint` mode | not a frozen-representation problem; already effectively expert-only |
| **C** single-batch | Can the pipeline drive ONE fixed batch's flow loss → 0? | flow loss **1.82 → ~0.04**; arm MAE ↓; **gripper stays worst dim** | training path **healthy**; gripper hard even to memorize |
| **P** per-dimension | Which action dimension carries the error on memorized frames? | arm joints MAE **0.04–0.09 rad**; **gripper MAE 0.26–0.30 rad, pred range [−2.8, 2.08]** vs valid [−0.17, 1.2] | **gripper is the dominant failure** |

## Mechanistic conclusion

The blocker is **not** data, action semantics, normalization, loading, or a
frozen backbone. It is a combination, dominated by the **gripper**:

1. **Gripper prediction (primary).** The gripper target is a near-**bimodal
   switch** (open ≈ 1.2, closed ≈ −0.17). Flow matching regresses a continuous
   velocity field and smears this switch: gripper open-loop MAE is 4–8× the arm
   joints, and integrated predictions overshoot far outside the valid range
   ([−2.8, 2.08]). It stays the max-error dimension even when overfitting a
   single batch (Gate C), so it is structurally hard here, not just under-trained.
2. **Approach accuracy (secondary).** Arm-joint MAE ≈ 0.04–0.09 rad puts the
   end-effector at ~4 cm closest approach — right on the `GRASP_RADIUS = 0.04`
   rim. Even a correct gripper close can miss by a few mm.

Together: the arm arrives at the cube rim but the gripper does not reliably
close at the grasp instant, so `_update_grasp()` never fires → cube never lifted
→ 0/4, with closest approach pinned at ~4.1 cm. This matches every observed number.

The **grasp radius is not the bug** — do not loosen `GRASP_RADIUS`. A 5 cm
diagnostic is acceptable to separate approach error from grasp error, but the
canonical gate stays at 4 cm.

## Exact next experiment

Attack the gripper directly, then re-run the gate. In priority order:

1. **More optimization with the gripper in mind.** Train command-0 four scenes
   to 2000+ steps, saving at 50/100/250/500/1000/2000 and evaluating each
   deterministically (Gate E). Gate C shows the objective does descend; the arm
   is already close, so additional steps + the items below should cross the
   grasp threshold.
2. **Reduce stale-chunk drift at the grasp instant:** compare
   `n_action_steps = 1, 5, 10` with identically-seeded flow noise. The gripper
   switch is time-sensitive; executing 5–10 stale queued actions can walk past
   the close moment.
3. **Trainability sweep (separately, not mixed):** `expert` only vs
   `expert + connector` vs `expert + last-2 text layers` vs full model with a
   small backbone LR. Record trainable-param counts and results independently.
4. **Baseline controls (Gate F):** a privileged-state MLP (cube pos + robot
   state → action) and a small CNN+MLP BC baseline. If these overfit 4 scenes
   and SmolVLA cannot, the fault is specifically SmolVLA adaptation of the
   bimodal gripper, not the task or data.

Only after the four scenes reach 4/4 at the canonical 4 cm radius should the
450M teacher be established on commands 0–3 (fixed seeds, separate
selection/test scenes, ≥3 seeds, per-command CIs). Compression stays frozen
until then.

## Reproduce

```bash
MUJOCO_GL=glfw .venv/bin/python -m scripts.diagnose_overfit --gate A
PYTORCH_ENABLE_MPS_FALLBACK=1 MUJOCO_GL=glfw .venv/bin/python -m scripts.diagnose_overfit --gate B
PYTORCH_ENABLE_MPS_FALLBACK=1 MUJOCO_GL=glfw .venv/bin/python -m scripts.diagnose_overfit --gate C   # slow: trains one batch
PYTORCH_ENABLE_MPS_FALLBACK=1 MUJOCO_GL=glfw .venv/bin/python -m scripts.diagnose_overfit --gate P
```
