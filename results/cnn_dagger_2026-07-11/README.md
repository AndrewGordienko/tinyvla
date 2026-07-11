# Image/proprio CNN DAgger — privileged vs deployable (2026-07-11, M5)

Machine-readable results for the two observation conditions, kept **separate**.
Full narrative + tables: `docs/overfit_diagnosis.md` Milestone 5.

| file | condition | policy state | schema |
|------|-----------|--------------|--------|
| `privileged_grasp.json` | UPPER BOUND | image + qpos6 + qvel6 + prev6 + **grasped1 (sim-only)** | Experiment I (transfer) + II (on-policy DAgger curve) |
| `deployable_no_grasp.json` | deployable gate | image + qpos6 (incl. gripper joint pos) + qvel6 + prev6 | A/B/C controls (demos-only / perturbation / DAgger), per-scene held-out + layout-distance bins |

## Headline (3 seeds, command 0, 4 memorized + 20 held-out, 4 cm, learner-only)

| condition | control | memorized /12 | held-out /60 |
|---|---|---|---|
| privileged-grasp | DAgger (final) | 8/12 (67%) | 21/60 (35%) |
| deployable-no-grasp | demos-only | 0/12 | 0/60 |
| deployable-no-grasp | perturbation | 0/12 | 0/60 |
| deployable-no-grasp | **DAgger** | **3/12 (25%)** | **6/60 (10%)** |

- DAgger strictly beats demos-only and perturbation on held-out **without** sim-only
  state (6 vs 0 vs 0) — on-policy coverage is genuinely helping.
- But the grasped bit was a large crutch (8/12→3/12 memorized, 21/60→6/60 held-out).
- Deployable held-out wins concentrate in the <0.05 m layout-distance bin; the
  generator only produces layouts <0.1 m from training, so this set does not test
  far generalization.

**Decision:** DAgger meets the directional rule but the deployable controller does
not solve the task. Do NOT promote SmolVLA DAgger / H200. Next: a deployable
temporal + multi-view + action-chunk controller that first reaches 4/4 on the four
memorized scenes without privileged state.

## Reproduce

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 MUJOCO_GL=glfw .venv/bin/python -m scripts.controlled_dagger_cnn                 # privileged (upper bound)
PYTORCH_ENABLE_MPS_FALLBACK=1 MUJOCO_GL=glfw .venv/bin/python -m scripts.controlled_dagger_cnn --exclude-grasp # deployable
```
