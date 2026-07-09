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
- **`--closed-loop-episodes N`** (2026-07-09, train/recover/dagger_loop): N rollouts
  per command per eval with deterministic per-rollout scene seeds. The 2026-07-06
  H200 runs used 1 rollout/command = success quantized to 17% steps (3/6 vs 2/6 is
  ONE episode); use **3** whenever two numbers will be compared.
- **Unfreeze knobs** (2026-07-09, train + dagger_loop): `--trainable
  expert|brain|brain_visual|all` (avoids the silent no-op where
  `freeze_vision_encoder=False` alone does nothing), `--backbone-lr-scale 0.1`
  (VLM backbone LR = 0.1 x expert LR), `--warmup-steps 500` (linear warmup).
- **Delta collection** (2026-07-09): `tinyvla.collect --delta-actions` and
  `scripts/gen_dataset.py --delta-actions` write delta datasets directly (+ the
  `delta_actions.json` marker). Same seeds -> identical trajectories as an
  absolute collection, so abs-vs-delta is a clean A/B.
- **Launch scripts**: `scripts/h200_step_b.sh` (400-ep data + both levers),
  `scripts/h200_step_c.sh` (DAgger at scale), `scripts/h200_unfreeze.sh`
  (gated unfreeze experiment — run last).

Linux/H100 note: set `MUJOCO_GL=egl`. Use `--num-workers 8-16` and (on CUDA)
`train.py` already runs bf16 autocast.

## Performance pass (2026-07-09) — the box costs money

- **`FastChunkDataset` (~70x dataloader win, the big one).** LeRobot's
  delta-timestamps query falls back to ROW-first indexing, PNG-decoding the
  embedded image column ~50x per sample just to read the action chunk: measured
  43 ms/sample -> 0.6 ms/sample (bit-identical tensors) by projecting to the
  column first. Used by train/recover/benchmark/audit automatically.
- **Image datasets for training** (`--no-videos` in collect/gen_dataset): video
  costs a torchcodec seek+decode per random sample; parquet-embedded PNG reads
  are ~1 ms. Step B generates image datasets. (Also dodges the broken local
  torchcodec.)
- **Eval/DAgger rollouts early-stop** on sustained success (same dwell=8 rule as
  the collector) — roughly halves closed-loop eval and DAgger rollout time.
- **TF32 + cudnn.benchmark + no per-step `.item()` sync** in train.py on CUDA.
- **Parallel expert collection** (`--collect-workers`, one process per command)
  for DAgger seed/curriculum pools; threaded PNG writer in dataset builds.
- **`--warm-start`** in dagger_loop: rounds >1 initialise from the previous
  round's checkpoint (`--warm-steps`, e.g. 3000 vs 6000) — ~2x faster loop,
  slight departure from pure batch-DAgger; validate once against a cold run.
- Checkpoints save every 2000 steps in the scripts (each save writes 1.2 GB).

## Step A — baseline reality-check (establishes the TRUE number)

**DONE 2026-07-06 on an H200** (120 eps): baseline (`n_action_steps=50`) peaked
2/6 (33%); the `n_action_steps=10` lever peaked 3/6 (50%) — see
`results/h200_2026-07-06/RESULTS.md`. Caveat: single-rollout eval (17% steps);
step B re-measures with `--closed-loop-episodes 3`.

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

## Step B — 400-episode data + both levers (one command)

```bash
bash scripts/h200_step_b.sh
```

Generates two 400-episode datasets with identical trajectories (absolute + delta),
then trains B1 (`n_action_steps=10`, absolute) and B2 (`n_action_steps=10` +
`--delta-actions`), both judged by 18-rollout closed-loop evals with
`best_closed_loop` retention. Whichever wins decides `DELTA=` for steps C and the
unfreeze experiment.

## Step C — targeted-data DAgger loop (the compounding-error fix)

```bash
bash scripts/h200_step_c.sh              # absolute actions
DELTA=1 bash scripts/h200_step_c.sh      # if B2 (delta) won step B
```

(Expands to `python -m tinyvla.dagger_loop --rounds 6 --steps 6000 --batch-size 64
--num-workers 16 --commands 0,1,2,3,6,7 --seed-per 60 --curriculum-per 20
--dagger-per 20 --worst-k 3 --n-action-steps 10 --closed-loop-episodes 3 ...`.)

Each round: trains, saves/selects the best closed-loop checkpoint, scores every
command closed-loop, finds the worst, and grows the pool there (curriculum +
DAgger). Watch `mean success by round` climb. Add `--delta-actions` to combine
with the delta lever. The teacher path to use downstream is the final round's
`best_closed_loop` directory, or the `checkpoint` field in `dagger_history.json`.

Stacking (commands 4,5) is excluded by default: the reactive DAgger labeler is
weak there (precise placement on a 24 mm cube). Keep them out of DAgger or supply
scripted-expert demos for them only.

## Optional — unfreeze-at-scale experiment (run AFTER B/C, not before)

Small-scale evidence says frozen wins (val 0.137 frozen vs 0.147 +vision vs
0.176 +vision+text, and 8–12x slower); this settles it at scale on the honest
metric. Uses `--trainable brain_visual` (dodges the `freeze_vision_encoder=False`
silent no-op), backbone at 0.1x LR, 500-step warmup:

```bash
bash scripts/h200_unfreeze.sh            # DELTA=1 / TRAINABLE=brain|all to vary
```

Compare its closed-loop trajectory against the frozen B run on the same dataset.
If frozen still wins, the backbone question is closed — spend GPU time on data.

## Step D — audit the 163M student before distillation

Check whether the student's action expert, VLM brain, and visual path are
trainable and whether images influence the loss. Run these on the pruned student
before recovery:

```bash
MUJOCO_GL=egl python -m tinyvla.audit \
  --model data/checkpoints/smolvla_cut_full \
  --repo-id local/so101_pickplace --root data/datasets/so101_pickplace \
  --device cuda --batch-size 4 --trainable brain \
  > artifacts/audits/student_brain.json

MUJOCO_GL=egl python -m tinyvla.audit \
  --model data/checkpoints/smolvla_cut_full \
  --repo-id local/so101_pickplace --root data/datasets/so101_pickplace \
  --device cuda --batch-size 4 --trainable brain_visual \
  > artifacts/audits/student_brain_visual.json
```

Use `brain` first if pixel gradients and `fixed_noise_loss_image_delta` are
non-zero. Use `brain_visual` if the image path is effectively dead. Use `all`
only if the narrower modes cannot recover.

## Step E — student–teacher distillation to the small model (163M)

Use the best closed-loop 450M checkpoint from B/C as the teacher. The student
must be **loadable** — the local pruned checkpoints are bare `model.safetensors`
with no `config.json`/sidecars; regenerate them (`scripts/prune_smolvla*.py`) so
they load via `load_pruned_smolvla`.

```bash
MUJOCO_GL=egl python -m tinyvla.recover \
  --student data/checkpoints/smolvla_cut_full \
  --teacher data/checkpoints/dagger_run/round_06/best_closed_loop \
  --objective mixed_action_expert --trainable brain \
  --repo-id local/so101_pickplace --root data/datasets/so101_pickplace \
  --steps 15000 --batch-size 64 --num-workers 16 --device cuda \
  --n-action-steps 10 \
  --closed-loop-every 1000 --closed-loop-commands 0,1,2,3,6,7 --closed-loop-cap 220 \
  --save-best-closed-loop \
  --output data/checkpoints/student_distilled
```

Judge the student by **its own** closed-loop success (`--save-best-closed-loop`
keeps the best), not by how well it matches the teacher offline. The recovery
script now propagates the teacher action noise into the student's flow loss. Keep
architecture search paused until this 163M recovery path beats the 450M teacher's
closed-loop gap enough to justify the smaller model.

## Notes / future

- Training restarts from base each DAgger round (batch-DAgger). Warm-starting
  from the previous round's checkpoint would be faster; not yet implemented.
- The DAgger pool stores images uncompressed-ish in `.npz`; fine at this scale,
  revisit if the pool gets very large.
- Everything selects on closed-loop success — that is the one metric we trust.
