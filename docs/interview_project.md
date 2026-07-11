# tinyvla interview project

## One-command demo

From a fresh checkout with the pinned environment installed:

```bash
MUJOCO_GL=glfw .venv/bin/python -m tinyvla.interview_demo --port 8768
```

Open `http://127.0.0.1:8768`. Choose a deterministic memorized scene and press
Replay. The page shows the model, parameter count, artifact SHA, command, and
approach/grasp/transport/release status. The current champion is a recorded
round-0 controller result; its neural weights were not persisted by the
interrupted recovery process, and the UI says so explicitly.

## Results

| Controller | Result | Trust status |
|---|---:|---|
| Scripted expert | 4/4 | reference |
| 450M SmolVLA baseline | 0/4 local command-0 gate | trustworthy negative; not promoted |
| 163M pruned attempt | invalid compact vocabulary | superseded/invalid |
| Privileged MLP DAgger | 4/4 across 3 seeds | privileged upper bound |
| Single-frame deployable CNN | 3/12 memorized, 6/60 held-out | trustworthy diagnostic |
| Temporal multi-view controller, round 0 | **2/4** | current best learned result |
| Temporal multi-view aggregate round | 0/4 | regression; not promoted |

The round-0 videos and machine-readable metrics live under
`artifacts/truth_harness/deployable_multiview_seed0_short_2026-07-11.*`.

## Technical story

The original goal was to compress a working 450M SmolVLA toward 150–300M. That
goal was deliberately paused after audits found misleading evaluation: padded
action loss used nonexistent dimensions, pruned checkpoints loaded non-strictly,
and train/eval paths reconstructed policies differently. The corrected harness
now pins the environment, validates tensor coverage and action semantics, and
requires reproducible checkpoint reloads.

The causal experiments then established ordinary closed-loop distribution shift:
privileged state fits and perturbation controls were insufficient, while true
DAgger improved coverage. The privileged grasp bit inflated results, so the next
controller uses only deployable joint state, previous action, front/wrist views,
four-frame temporal context, and short action chunks.

The temporal visual model passes the deterministic 1/8/64-sample supervised gate
with exact save/reload. In closed loop, round 0 reaches 2/4; aggregate recovery
training regressed to 0/4 because the learner fit was under-trained and recovery
examples overwhelmed nominal demonstrations. The next technically justified step
is to persist round-0 weights, warm-start each aggregate round, rehearse at least
50% original demonstrations, and select checkpoints by the fixed four-scene gate.
Only a stable 3/4+ deployable controller justifies one bounded 450M DAgger job;
only a working teacher justifies one conservative 250–300M candidate. The 163M
target is intentionally deferred.
