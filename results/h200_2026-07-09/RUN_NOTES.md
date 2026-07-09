# H200 run notes — 2026-07-09 (Step B at scale, IN FLIGHT)

Living notes for the run currently executing on the vast.ai H200 NVL. Written so
the state survives a push/pull to another machine. The July 6 results this builds
on are in [../h200_2026-07-06/RESULTS.md](../h200_2026-07-06/RESULTS.md); the
plan and all levers are in [docs/h100_recipe.md](../../docs/h100_recipe.md).

## What is running

`scripts/h200_step_b.sh`, launched 2026-07-09 ~16:52 UTC, daemonized with `nohup`
(parent PID 1 — survives SSH drops; no tmux needed). Params:
`SHARDS=8 EPS_PER_SHARD=50 STEPS=20000 BATCH=64 WORKERS=16 DEVICE=cuda`.

Sequence (each stage gated on the previous):
1. **DONE** — generated `pickplace_abs_400` and `pickplace_delta_400`
   concurrently: 400 eps / 32,787 frames each, IMAGE datasets (`--no-videos`),
   **identical trajectories** (deterministic expert, same seeds) so abs-vs-delta
   is a clean A/B. Delta verified: action mean ≈ 0, min per-joint std 0.0015
   (wrist_roll — small but finite, so MEAN_STD normalization is safe, no NaNs).
2. **RUNNING** — b1: `n_action_steps=10`, absolute actions, 20k steps.
3. **QUEUED** — b2: `n_action_steps=10` + `--delta-actions`, same steps
   (auto-chains when b1 finishes).

Both judged by **closed-loop success over 18 rollouts** (6 commands × 3 episodes,
`--closed-loop-episodes 3`, deterministic per-rollout scene seeds) every 1000
steps, keeping `<output>/best_closed_loop`.

## Results so far (update as evals land)

| step | b1 (abs) closed-loop |
|---|---|
| 1000 | **7/18 (39%)**, min_dist 0.035 — new best, saved |

Calibration: July 6 (120 eps, 6-rollout evals) peaked at 33% baseline / 50% with
the n_action_steps lever. b1 hit 39% after 1000/20000 steps on the
harder-to-fluke 18-rollout metric.

Throughput: ~3.5–4.6 it/s at batch 64 (train-only vs including eval pauses).
~4–5x the July throughput — mostly the FastChunkDataset dataloader fix (~70x on
the chunk query; see `tinyvla/fast_dataset.py` docstring) + image datasets +
TF32/cudnn.benchmark. ETA: b1 done ~18:30 UTC, b2 done ~20:00 UTC (~3 h total).

## The box

- vast.ai instance 44036111, 1x H200 NVL (143 GB), `ssh -p 36111 root@ssh9.vast.ai`.
  It is the July instance re-rented: `/root/tinyvla/.venv` already had torch
  2.10+cu128 / lerobot 0.4.4 / datasets 4.8.5 / transformers 4.57.6 / mujoco
  3.10.0 — **exact match to the locally tested versions, no install was needed**.
- Code got there via **rsync, not GitHub** (working tree was uncommitted):
  `rsync -az -e "ssh -p 36111" --exclude .git --exclude .venv --exclude datasets
  --exclude checkpoints --exclude outputs --exclude artifacts --exclude results
  ./ root@ssh9.vast.ai:/root/tinyvla/`
- Checking on it:
  `ssh -p 36111 root@ssh9.vast.ai "tail -5 /root/tinyvla/data/checkpoints/b1_nas10_abs.log"`
  (b2 log: `b2_nas10_delta.log`; main log: `/root/tinyvla/logs/h200_step_b_*.log`)
- Gotchas learned the hard way:
  - `nproc` reports 256 (host cores) but vast allocates 32 — never let scripts
    autoscale shard/worker counts from nproc on this box; pass SHARDS/WORKERS.
  - **One driver at a time.** Two sessions + the user all operated the box at
    once; an `rm -rf` raced live shard writers and destroyed the first attempt's
    logs. Coordinate before touching `data/`.
  - Disk is 32 GB total, ~9 GB free mid-run. Old July artifacts
    (`data/ckpt_A`, `data/ckpt_B`, 2.4 GB) are still there — deletable if tight
    (they're reproducible; results recorded in results/h200_2026-07-06).
  - **Billing kills runs, not crashes**: balance was negative at launch; vast
    stops (and may delete) instances at $0. Keep credits ahead of ~$2–3/hr.

## When it finishes — decision tree

1. Compare peak + final closed-loop between b1 and b2:
   `grep -h "closed-loop" data/checkpoints/b*_*.log`
   Winner decides `DELTA=` for everything downstream.
2. Pull artifacts off the box before stopping it (checkpoints are NOT in git):
   `rsync -az -e "ssh -p 36111" root@ssh9.vast.ai:/root/tinyvla/data/checkpoints/b1_nas10_abs/best_closed_loop ./checkpoints/...`
   plus both `.log` files into `results/h200_2026-07-09/`.
3. Next run: **Step C DAgger loop** — `bash scripts/h200_step_c.sh` (add
   `DELTA=1` if b2 won; `WARM=1 WARM_STEPS=3000` halves loop time but validate
   one round against cold-start first).
4. Unfreeze experiment (`scripts/h200_unfreeze.sh`) only after B/C — lowest
   prior, highest cost (see freeze findings in the recipe).

## Code state at launch

All of today's work (multi-episode eval, unfreeze/warmup/discriminative-LR,
delta collection, FastChunkDataset, early-stop rollouts, parallel collection,
launch scripts) was **uncommitted** when rsynced to the box — commit and push
before working from another computer, or the box and laptop will drift.
