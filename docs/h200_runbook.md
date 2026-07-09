# H200 Step B runbook

Last updated: 2026-07-09 17:15 UTC.

This note captures the live H200 run state and the reasoning around it so the
repo can be pushed/pulled to another machine without losing the thread.

## Ground rules

- Do not use GitHub to move code onto the Vast box unless explicitly deciding to
  change that workflow. The current box was updated from this local workspace
  with direct SSH/rsync/scp.
- Remote login:

```bash
ssh -p 36111 root@ssh9.vast.ai
```

- Remote repo path:

```bash
/root/tinyvla
```

- This Vast instance is not using a persistent workspace volume. If the instance
  is destroyed/recycled, `/root/tinyvla`, datasets, and checkpoints can be lost.
  Keep the instance funded/running while long jobs are active.

## Current run

Launched on the H200 from `/root/tinyvla`:

```bash
PATH=/root/tinyvla/.venv/bin:$PATH \
MUJOCO_GL=egl \
SHARDS=8 EPS_PER_SHARD=50 STEPS=20000 BATCH=64 WORKERS=16 DEVICE=cuda \
bash scripts/h200_step_b.sh
```

The run is detached and continues after the local laptop disconnects. It is also
visible in a tmux monitor:

```bash
tmux attach -t tinyvla_run
```

Detach from tmux with `Ctrl-b`, then `d`.

The main log is:

```bash
/root/tinyvla/logs/h200_step_b_20260709_165235.log
```

The Step B script does three things:

1. Generate two 400-episode image datasets concurrently:
   - `data/datasets/pickplace_abs_400`
   - `data/datasets/pickplace_delta_400`
2. Train B1 on absolute actions with `n_action_steps=10`.
3. Train B2 on delta actions with `n_action_steps=10`.

As of the timestamp above:

- Dataset generation completed successfully for both abs and delta.
- Shards were merged and removed to free disk.
- B1 training is running:

```bash
python -m tinyvla.train \
  --repo-id local/pickplace_abs_400 \
  --root data/datasets/pickplace_abs_400 \
  --steps 20000 \
  --batch-size 64 \
  --num-workers 16 \
  --device cuda \
  --n-action-steps 10 \
  --save-every 2000 \
  --output data/checkpoints/b1_nas10_abs \
  --closed-loop-every 1000 \
  --closed-loop-commands 0,1,2,3,6,7 \
  --closed-loop-cap 220 \
  --closed-loop-episodes 3 \
  --save-best-closed-loop
```

- First B1 closed-loop eval completed at step 1000:

```text
step 1000/20000 closed-loop success 7/18 (39%) min_dist 0.035 final_dist 0.088
new best closed-loop success 39% -> saved data/checkpoints/b1_nas10_abs/best_closed_loop
```

- Training loss is decreasing normally, from about `0.148` at step 25 to around
  `0.02` near step 1000.
- Step 2000 saved the regular B1 checkpoint and completed another eval:

```text
step 2000/20000 closed-loop success 6/18 (33%) min_dist 0.034 final_dist 0.084
```

- The step 1000 checkpoint remains the current best B1 closed-loop checkpoint.
- GPU utilization has been active during training, around 80-90% after warmup.
- Disk had about `7.7G` free after the step 2000 checkpoint/eval.

## How to check health

From the remote box:

```bash
cd /root/tinyvla

tail -f data/checkpoints/b1_nas10_abs.log
tail -f data/checkpoints/b2_nas10_delta.log

grep -h "closed-loop" data/checkpoints/b1_nas10_abs.log data/checkpoints/b2_nas10_delta.log

ps -eo pid,ppid,stat,etime,cmd | grep -E "[h]200_step_b|[b]1_nas10_abs|[b]2_nas10_delta|tinyvla.train"

nvidia-smi
df -h /root/tinyvla
du -sh data/datasets data/checkpoints
```

Healthy signs:

- Step logs continue every 25 optimizer steps.
- Loss should trend down or stay low, but do not judge success from offline loss
  alone.
- Closed-loop evals appear every 1000 steps.
- Best checkpoints are saved when closed-loop success improves.
- GPU utilization should be nonzero during optimizer steps; it can dip during
  closed-loop rollout or checkpoint save.

Watch-outs:

- Disk is limited. Checkpoint saves are the moments most likely to expose a disk
  problem.
- If the run stops, check the log tails and search for errors before restarting:

```bash
grep -RniE "traceback|error|exception|cuda out of memory|killed" logs data/checkpoints/*.log
```

- Do not blindly rerun `scripts/h200_step_b.sh` after a partial training run.
  The dataset generation stage will skip existing datasets, but the training
  stages do not resume optimizer state and may overwrite checkpoint directories.

## Why this run matters

This run is the current "is the training loop actually behaving?" check after
performance fell apart in the smaller/student setup.

The immediate levers under test are:

- `n_action_steps=10`, to reduce long open-loop drift.
- Delta actions, to test whether learning action deltas is more stable than
  absolute joint targets.
- Closed-loop success every 1000 steps, because offline loss/MAE can look good
  while rollout still fails.

The first B1 closed-loop result (`39%`) is a good sign that the pipeline is not
broken: data loading, policy training, MuJoCo rollout, success scoring, and
best-checkpoint saving all worked. It is not yet proof that the final model is
good enough. We need the full B1 curve and the B2 delta-action comparison.

## Next decisions after Step B finishes

1. Compare B1 vs B2 by closed-loop success, not just loss:

```bash
grep -h "closed-loop" data/checkpoints/b1_nas10_abs.log data/checkpoints/b2_nas10_delta.log
```

2. Prefer the best closed-loop checkpoint:

```bash
data/checkpoints/b1_nas10_abs/best_closed_loop
data/checkpoints/b2_nas10_delta/best_closed_loop
```

3. If both B1 and B2 are weak, the next work should be recovery-data oriented:
   collect or synthesize examples around failure states/gaps instead of only
   scaling the same successful expert distribution.

4. For the student/teacher recovery path, keep these checks in mind:
   - Warm up the expert/head before unfreezing the brain/backbone.
   - Propagate the same sampled noise through teacher and student when comparing
     action chunks.
   - Audit inside the "brain" layers to find which module causes the error jump.
   - Turn on the visual backbone carefully, with warmup and a lower LR scale.
   - Judge distillation by teacher-action MAE/RMSE and closed-loop success
     together, not by one metric alone.

## Local code notes

Relevant files changed or added during this phase:

- `scripts/gen_dataset.py`: unique local shard repo ids so concurrent abs/delta
  generation does not collide in LeRobot/HF metadata.
- `scripts/h200_step_b.sh`: scaled Step B dataset generation and B1/B2 training.
- `tinyvla/train.py`: trainability modes, closed-loop selection, warmup, and
  best closed-loop checkpoint saving.
- `tinyvla/recover.py`: trainability modes and teacher-action noise propagation.
- `tinyvla/trainability.py`: shared trainable parameter mode definitions.
- `tinyvla/benchmark.py`: preserves saved runtime config such as
  `n_action_steps` when loading checkpoints.
