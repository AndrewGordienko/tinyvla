# H100 run recipe

The plumbing changes (2026-07-06) make the H100 runs **measured and selected**
correctly, and add the three performance levers we identified. This is the order
to run things so each step's result gates the next.

## What changed vs. before

- **Closed-loop-in-the-loop eval** wired into `tinyvla.train` and `tinyvla.recover`
  (`--closed-loop-every N`). Select checkpoints by **closed-loop success**, not
  offline loss — they are decoupled (offline loss can be anti-correlated with
  rollout success).
- **`--n-action-steps`** lever. Base default is 50 (executes 50 actions open-loop
  before replanning → compounding drift). Use **10**.
- **`--delta-actions`** lever. Model predicts joint deltas (`action - state`);
  fixes the "hold current pose gets low loss" pathology (our actions ≈ state).
- **Targeted-data engine** (`tinyvla.dagger`, `tinyvla.dagger_loop`): grow the
  dataset where the policy is weakest — curriculum (more expert demos) + DAgger
  (relabel the policy's own drifted states with a reactive expert).

Linux/H100 note: set `MUJOCO_GL=egl`. Use `--num-workers 8-16` and (on CUDA)
`train.py` already runs bf16 autocast.

## Step A — baseline reality-check (establishes the TRUE number)

We never measured closed-loop success correctly before. Do one full-data finetune
and read the closed-loop metric — this is the honest baseline to beat.

```bash
# full dataset (hundreds of episodes); parallel shards then aggregate
MUJOCO_GL=egl python scripts/gen_dataset.py --episodes 400   # or tinyvla.collect

MUJOCO_GL=egl python -m tinyvla.train \
  --repo-id local/so101_pickplace --root data/datasets/so101_pickplace \
  --steps 20000 --batch-size 64 --num-workers 16 --device cuda \
  --output data/checkpoints/base_finetune \
  --closed-loop-every 1000 --closed-loop-commands 0,1,2,3,6,7 --closed-loop-cap 220
```

Read the `closed-loop success` lines, not the loss. That success rate is the real
baseline.

## Step B — add the two cheap levers

```bash
# n_action_steps=10 + delta actions. Regenerate the dataset as delta, or pass
# --delta-actions to a delta dataset (the dagger builder can emit one directly).
MUJOCO_GL=egl python -m tinyvla.train \
  --repo-id local/so101_pickplace --root data/datasets/so101_pickplace \
  --steps 20000 --batch-size 64 --num-workers 16 --device cuda \
  --n-action-steps 10 \
  --output data/checkpoints/levers_finetune \
  --closed-loop-every 1000 --closed-loop-commands 0,1,2,3,6,7 --closed-loop-cap 220
```

Compare closed-loop success vs. Step A. `n_action_steps=10` is the most likely
single win (attacks the open-loop drift directly).

## Step C — targeted-data DAgger loop (the compounding-error fix)

```bash
MUJOCO_GL=egl python -m tinyvla.dagger_loop \
  --rounds 6 --steps 6000 --batch-size 64 \
  --commands 0,1,2,3,6,7 \
  --seed-per 60 --curriculum-per 20 --dagger-per 20 --worst-k 3 \
  --n-action-steps 10 \
  --device cuda \
  --pool data/datasets/dagger_pool --work data/checkpoints/dagger_run
```

Each round: trains, scores every command closed-loop, finds the worst, and grows
the pool there (curriculum + DAgger). Watch `mean success by round` climb. Add
`--delta-actions` to combine with the delta lever.

Stacking (commands 4,5) is excluded by default: the reactive DAgger labeler is
weak there (precise placement on a 24 mm cube). Keep them out of DAgger or supply
scripted-expert demos for them only.

## Step D — student–teacher distillation to the small model (163M)

Use the best closed-loop 450M checkpoint from B/C as the teacher. The student
must be **loadable** — the local pruned checkpoints are bare `model.safetensors`
with no `config.json`/sidecars; regenerate them (`scripts/prune_smolvla*.py`) so
they load via `load_pruned_smolvla`.

```bash
MUJOCO_GL=egl python -m tinyvla.recover \
  --student data/checkpoints/smolvla_cut_full \
  --teacher data/checkpoints/dagger_run/round_06 \
  --objective mixed_action_expert --trainable all \
  --repo-id local/so101_pickplace --root data/datasets/so101_pickplace \
  --steps 15000 --batch-size 64 --device cuda \
  --n-action-steps 10 \
  --closed-loop-every 1000 --closed-loop-commands 0,1,2,3,6,7 --closed-loop-cap 220 \
  --save-best-closed-loop \
  --output data/checkpoints/student_distilled
```

Judge the student by **its own** closed-loop success (`--save-best-closed-loop`
keeps the best), not by how well it matches the teacher offline. `--trainable all`
unfreezes the student's brain so the teacher's generated samples actually move it
(with the brain frozen they did nothing — the original failure).

## Notes / future

- Training restarts from base each DAgger round (batch-DAgger). Warm-starting
  from the previous round's checkpoint would be faster; not yet implemented.
- The DAgger pool stores images uncompressed-ish in `.npz`; fine at this scale,
  revisit if the pool gets very large.
- Everything selects on closed-loop success — that is the one metric we trust.
