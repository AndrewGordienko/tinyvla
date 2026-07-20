# tinyvla interview project

## One-command demo

From a fresh checkout with the pinned environment installed, drive the live arm
(talk to it; toggle the 450M teacher and 291M student on identical scenes):

```bash
MUJOCO_GL=glfw .venv/bin/python -m tinyvla.live_demo   # http://localhost:8010
```

Every frame is live closed-loop inference from the exact loaded checkpoint (not a
replay). The panel reports parameter count, checkpoint SHA, per-step latency, the
action-chunk queue, and the approach/grasp/transport/release stage. The default
student is the 291M bf16 champion (`student291_champion_bf16`).

## Results

Held-out closed-loop success (Apple M5 Pro / MPS, two fresh seeds; full report with
per-command breakdown, 95% CIs, latency, and checkpoint SHA256 in
[../results/champion_2026-07-21/BENCHMARK_REPORT.md](../results/champion_2026-07-21/BENCHMARK_REPORT.md)):

| Controller | Params | Held-out success | Status |
|---|---:|---:|---|
| Scripted expert | — | reference | oracle |
| 450M SmolVLA teacher (task-trained) | 450M | 42% (95% CI 38–47%), 8 cmds | ceiling |
| **291M distilled student (champion)** | 291M | 33% (95% CI 27–41%), cmds 1/3/4 | current best learned |
| 291M student, bf16 | 292M | 33% (no measurable drop), **557 MB** | deployable footprint |
| 260M / 10-layer | 260M | ~21% | below the distillation cliff |
| 163M pruned attempt | 163M | invalid compact vocabulary | superseded |

On the student's three trained commands the teacher scores ~46% and the student
~33% (retains ~72% on these seeds; ~80% combining all held-out evals, CIs overlap).

## Technical story

The goal: compress a working 450M SmolVLA while keeping its performance, and drive a
talk-to-the-arm demo locally on an Apple M5 Pro (MPS) — no H100/H200.

**What worked.** Prune the task-trained teacher to 12 VLM layers (`paired_even`,
291M), then teacher-distil on clean scripted-expert demos
(`recover.py --objective mixed_action_expert`, teacher/expert 1.0/0.5) at lr 3e-6.
This lifted the student from ~21% to ~33% held-out — a real +57% — recovering most
of the teacher. A bf16 cast of the fp32 SigLIP vision tower then cut the footprint
26% (749→557 MB) with no measurable accuracy drop. Data collection uses parallel
MuJoCo "arms" (`recovery_shard_collect.py`), ~2.8× wall-clock on this Mac.

**What broke, and the fixes.** Higher LR (1e-5) makes distillation peak early then
collapse to 0%; lr 3e-6 is stable. The single-seed gate silently overfits — a
checkpoint that gated 53% on seed 999 dropped to 27% on a fresh seed — so selection
and reporting use held-out seeds with ≥45 rollouts and confidence intervals.
Distillation stops helping below 12 layers: a raw 10-layer prune explodes under
distillation, needs a two-phase expert-stabilise → distil, and still tops out ~21%.
More data did not beat the teacher — a student cannot exceed the teacher it distils
from.

**The honest ceiling.** The teacher itself is only ~42% on hard held-out scenes
(~67% on easy ones — this sim has large scene variance). Failure analysis shows
failures cluster at *reaching* (never gets within 5 cm of the cube) and *transport*
(grasps then drops), not fine placement — a data-distribution gap, not capacity. So
the next levers are targeted recovery data on the failing object poses (or a
stronger teacher), and moving the bf16 champion onto a physical SO-101; another
point on the sim success curve is not. The MuJoCo grasp remains kinematic
scaffolding, not contact-valid physics.
