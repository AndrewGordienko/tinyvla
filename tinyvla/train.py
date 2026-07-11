"""Fine-tune SmolVLA (smolvla_base) on the scripted-expert reach dataset.

This is a minimal, self-contained training loop that reuses LeRobot's policy and
processor factories while the repository-owned runtime enforces the pinned
environment, strict loading, action semantics, and corrected action loss.

  - loads smolvla_base pretrained weights
  - overrides the input/output normalizers with OUR dataset's stats
  - trains the action expert to imitate the expert reaches
  - saves the fine-tuned policy + processors to artifacts/checkpoints/

Run:  python3 -m tinyvla.train --steps 2000 --batch-size 8
      python3 -m tinyvla.train --steps 2      # smoke test
"""
from __future__ import annotations

import argparse
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader

# Import datasets before policies to avoid LeRobot's policy/dataset import cycle.
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import cycle

from .paths import CHECKPOINTS_ROOT, DATASETS_ROOT, MODELS_ROOT
from .eval_closedloop import evaluate_closed_loop, format_metrics
from .fast_dataset import FastChunkDataset
from .determinism import make_generator, seed_everything, seed_worker
from .runtime import load_runtime, save_runtime
from .trainability import TRAINABLE_MODES, group_for_param, set_trainable

BACKBONE_GROUPS = ("vision_encoder", "vision_connector", "vlm_text")

BASE = MODELS_ROOT / "smolvla_base"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="local/so101_pickplace")
    ap.add_argument("--root", default=str(DATASETS_ROOT / "so101_pickplace"))
    ap.add_argument("--output", default=str(CHECKPOINTS_ROOT / "smolvla_pickplace"))
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--stop-after", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--episodes", default=None,
                    help="Optional comma-separated dataset episode indices (for local overfit gates).")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--versioned-checkpoints", action="store_true",
                    help="Also retain immutable checkpoint_step_N directories for rehearsal/audit runs.")
    ap.add_argument("--fixed-batch", action="store_true",
                    help="Reuse the first production DataLoader batch every step (numerical rehearsal only).")
    ap.add_argument("--fixed-noise", action="store_true",
                    help="Use one fixed flow-matching noise/timestep with --fixed-batch (rehearsal only).")
    ap.add_argument("--num-workers", type=int, default=0, help="dataloader workers (use 8-16 on a GPU box)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--closed-loop-every", type=int, default=0,
                    help="Run a closed-loop rollout eval every N steps (0=off). "
                         "Judge/select checkpoints by this, not offline loss.")
    ap.add_argument("--closed-loop-commands", default="0,1,2,3")
    ap.add_argument("--closed-loop-cap", type=int, default=180)
    ap.add_argument("--closed-loop-seed", type=int, default=100)
    ap.add_argument("--closed-loop-episodes", type=int, default=1,
                    help="Rollouts per command per eval. 1 rollout/command quantizes success to "
                         "1/len(commands) steps — use >=3 when comparing runs/levers.")
    ap.add_argument("--n-action-steps", type=int, default=None,
                    help="Actions executed per replan (<=chunk_size). Base default is 50 "
                         "(50-step open-loop -> compounding drift). Try 10 for tighter closed-loop.")
    ap.add_argument("--delta-actions", action="store_true", default=None,
                    help="Legacy assertion only. Semantics are detected from dataset/checkpoint markers.")
    ap.add_argument("--save-best-closed-loop", action="store_true",
                    help="Keep the best closed-loop checkpoint in <output>/best_closed_loop.")
    ap.add_argument("--trainable", choices=TRAINABLE_MODES, default="checkpoint",
                    help="Optionally override checkpoint trainability for brain/vision experiments.")
    ap.add_argument("--backbone-lr-scale", type=float, default=1.0,
                    help="LR multiplier for the pretrained VLM backbone (vision/connector/text) "
                         "relative to --lr. Use ~0.1 when unfreezing (--trainable brain/brain_visual/all) "
                         "so the expert adapts fast while the backbone moves gently.")
    ap.add_argument("--warmup-steps", type=int, default=0,
                    help="Linear LR warmup over this many steps (0=off), used only with "
                         "--scheduler linear. Recommended (~500) when unfreezing the backbone: "
                         "early expert gradients are noise and will wreck pretrained features "
                         "at full LR.")
    ap.add_argument("--scheduler", choices=("config", "linear", "none"), default="config",
                    help="LR schedule: 'config' (default) = SmolVLA's own cosine-decay-with-warmup "
                         "preset, which LeRobot auto-scales to --steps when the run is shorter than "
                         "the configured 30k decay horizon; 'linear' = legacy --warmup-steps linear "
                         "warmup then flat; 'none' = constant LR.")
    ap.add_argument("--init-from", default=None,
                    help="Warm-start from this checkpoint dir instead of smolvla_base "
                         "(e.g. the previous DAgger round) — later rounds then need fewer steps.")
    ap.add_argument("--resume", default=None,
                    help="Resume optimizer/scheduler/RNG/global-step state from a training output directory.")
    ap.add_argument("--allow-legacy-semantics", action="store_true",
                    help="Treat a dataset/checkpoint with no action_semantics marker as "
                         "'absolute' actions. Off by default: unmarked legacy artifacts error "
                         "so an unknown action representation is never assumed silently.")
    args = ap.parse_args()
    planned_steps = args.total_steps if args.total_steps is not None else args.steps
    run_until = min(planned_steps, args.stop_after) if args.stop_after is not None else planned_steps
    if run_until <= 0 or run_until > planned_steps:
        raise SystemExit("invalid stop/total step configuration")
    cl_commands = [int(x) for x in args.closed_loop_commands.split(",") if x != ""]
    episode_indices = [int(x) for x in args.episodes.split(",")] if args.episodes else None
    seed_everything(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True   # fp32 matmuls (norms etc.) on tensor cores
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True          # fixed shapes -> autotune convs once
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)

    # One canonical runtime owns checkpoint reconstruction, processors, action
    # semantics, strict load auditing, vocabulary coverage, and corrected loss.
    src = Path(args.init_from) if args.init_from else (Path(args.resume) if args.resume else BASE)
    runtime = load_runtime(
        src,
        meta=meta,
        dataset_root=args.root,
        device=device,
        stats_source="dataset",
        base_checkpoint=args.init_from is None and args.resume is None,
        allow_legacy_semantics=args.allow_legacy_semantics,
    )
    policy = runtime.policy
    preprocessor, postprocessor = runtime.preprocessor, runtime.postprocessor
    if args.delta_actions is not None and args.delta_actions != runtime.delta_actions:
        raise SystemExit(
            f"--delta-actions contradicts detected {runtime.action_semantics} dataset/checkpoint semantics"
        )
    if args.n_action_steps is not None:
        policy.config.n_action_steps = args.n_action_steps
        policy.reset()

    # each sample needs an action chunk of `chunk_size` future steps.
    # FastChunkDataset fixes a ~70x dataloader slowdown in the chunk query
    # (lerobot row-first fallback decodes the image column per chunk row).
    delta_timestamps = {"action": [i / meta.fps for i in range(policy.config.chunk_size)]}
    ds = FastChunkDataset(
        args.repo_id, root=args.root, episodes=episode_indices, delta_timestamps=delta_timestamps
    )
    trainable_params = set_trainable(policy, args.trainable)

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                    persistent_workers=(args.num_workers > 0), drop_last=True,
                    worker_init_fn=seed_worker, generator=make_generator(args.seed))
    fixed_batch = next(iter(dl)) if args.fixed_batch else None
    trainable_parameters = [p for p in policy.parameters() if p.requires_grad]
    if not trainable_parameters:
        raise SystemExit(f"no trainable parameters for --trainable {args.trainable}")
    # Use SmolVLA's own optimizer recipe (betas=(0.9, 0.95), weight_decay~1e-10,
    # eps=1e-8, grad-clip=10) rather than PyTorch's AdamW defaults (betas=(0.9, 0.999),
    # weight_decay=0.01). On a small-data overfit that difference is material and can
    # be the difference between fitting four scenes and not.
    opt_preset = policy.config.get_optimizer_preset()
    grad_clip_norm = opt_preset.grad_clip_norm
    adamw_kwargs = dict(betas=opt_preset.betas, eps=opt_preset.eps,
                        weight_decay=opt_preset.weight_decay)
    if args.backbone_lr_scale != 1.0:
        backbone, head = [], []
        for name, p in policy.named_parameters():
            if p.requires_grad:
                (backbone if group_for_param(name) in BACKBONE_GROUPS else head).append(p)
        opt = torch.optim.AdamW([
            {"params": head, "lr": args.lr},
            {"params": backbone, "lr": args.lr * args.backbone_lr_scale},
        ], **adamw_kwargs)
        print(f"discriminative LR: head {sum(p.numel() for p in head)/1e6:.0f}M @ {args.lr:g}, "
              f"backbone {sum(p.numel() for p in backbone)/1e6:.0f}M @ {args.lr * args.backbone_lr_scale:g}")
    else:
        opt = torch.optim.AdamW(trainable_parameters, lr=args.lr, **adamw_kwargs)
    print(f"optimizer AdamW betas={opt_preset.betas} eps={opt_preset.eps:g} "
          f"weight_decay={opt_preset.weight_decay:g} grad_clip={grad_clip_norm:g} | "
          f"scheduler={args.scheduler}")
    if args.scheduler == "config":
        # LeRobot's CosineDecayWithWarmupScheduler auto-scales warmup/decay when
        # --steps is shorter than the configured 30k decay horizon, so short overfit
        # runs still warm up and decay proportionally instead of at a flat peak LR.
        sched = policy.config.get_scheduler_preset().build(opt, num_training_steps=planned_steps)
    elif args.scheduler == "linear" and args.warmup_steps > 0:
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup_steps))
    else:
        sched = None

    start_step = 0
    if args.resume:
        prior_meta = Path(args.resume) / "runtime_metadata.json"
        if prior_meta.exists():
            prior = json.loads(prior_meta.read_text())
            expected = {"planned_steps": planned_steps, "lr": args.lr, "scheduler": args.scheduler,
                        "batch_size": args.batch_size, "seed": args.seed}
            mismatches = {k: (prior.get(k), v) for k, v in expected.items() if prior.get(k) != v}
            if mismatches:
                raise SystemExit(f"resume configuration mismatch: {mismatches}")
        resume_path = Path(args.resume) / "training_state.pt"
        if not resume_path.exists():
            raise SystemExit(f"resume state missing: {resume_path}")
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        opt.load_state_dict(state["optimizer"])
        if sched is not None and state.get("scheduler") is not None:
            sched.load_state_dict(state["scheduler"])
        start_step = int(state["global_step"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        print(f"resumed global_step={start_step} from {resume_path}")

    policy.train()
    print(f"training SmolVLA ({trainable_params/1e6:.0f}M trainable params, "
          f"mode={args.trainable}) on {device} | "
          f"{meta.total_episodes} eps, {meta.total_frames} frames | "
          f"chunk={policy.config.chunk_size}, batch={args.batch_size}, steps={args.steps}, "
          f"seed={args.seed}, actions={runtime.action_semantics}")

    # bf16 autocast on CUDA (big speedup on H100; no GradScaler needed for bf16)
    amp = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device.type == "cuda" else nullcontext())

    t0 = time.time()
    running = torch.zeros((), device=device)   # accumulate on-device; .item() every step = a GPU sync
    best_cl = -1.0
    def save_progress(step: int) -> None:
        metadata = {
          "repo_id": args.repo_id, "dataset_root": str(Path(args.root).resolve()),
            "step": step, "steps_this_run": run_until, "planned_steps": planned_steps,
            "lr": args.lr, "scheduler": args.scheduler, "batch_size": args.batch_size,
            "init_from": args.init_from,
            "resume": args.resume, "episodes": episode_indices,
        }
        save_runtime(runtime, args.output, seed=args.seed, extra_metadata=metadata)
        training_state = {
            "optimizer": opt.state_dict(), "scheduler": sched.state_dict() if sched is not None else None,
            "global_step": step, "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        torch.save(training_state, Path(args.output) / "training_state.pt")
        if args.versioned_checkpoints:
            versioned = Path(args.output) / f"checkpoint_step_{step}"
            save_runtime(runtime, versioned, seed=args.seed, extra_metadata=metadata)
            torch.save(training_state, versioned / "training_state.pt")

    batches = iter(cycle(dl)) if fixed_batch is None else None
    fixed_noise = fixed_time = None
    if args.fixed_noise:
        if not args.fixed_batch:
            raise SystemExit("--fixed-noise requires --fixed-batch")
        torch.manual_seed(args.seed + 991)
        fixed_noise = torch.randn((args.batch_size, policy.config.chunk_size, policy.config.max_action_dim), device=device)
        fixed_time = torch.rand(args.batch_size, device=device).to(dtype=torch.float32) * 0.999 + 0.001
    for step in range(start_step + 1, run_until + 1):
        batch = fixed_batch if fixed_batch is not None else next(batches)
        batch = preprocessor(batch)
        with amp:
            loss, _ = policy.forward(batch, noise=fixed_noise, time=fixed_time) if fixed_noise is not None else policy.forward(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters, grad_clip_norm)
        opt.step()
        if sched is not None:
            sched.step()
        opt.zero_grad()

        running += loss.detach()
        if step % args.log_every == 0:
            dt = time.time() - t0
            print(f"  step {step:5d}/{args.steps}  loss {running.item()/args.log_every:.4f}  "
                  f"{step/dt:.2f} it/s")
            running.zero_()
        if step % args.save_every == 0 or step == args.steps:
            save_progress(step)
            print(f"  saved checkpoint to {args.output} (step {step})")

        if args.closed_loop_every and (step % args.closed_loop_every == 0 or step == args.steps):
            cl = evaluate_closed_loop(
                policy, preprocessor, postprocessor,
                device=device, commands=cl_commands,
                cap=args.closed_loop_cap, seed=args.closed_loop_seed,
                delta_actions=runtime.delta_actions, episodes=args.closed_loop_episodes,
            )
            print(f"  step {step:5d}/{args.steps}  closed-loop {format_metrics(cl)}")
            if args.save_best_closed_loop and cl["success_rate"] > best_cl:
                best_cl = cl["success_rate"]
                best_path = Path(args.output) / "best_closed_loop"
                save_runtime(runtime, best_path, seed=args.seed, extra_metadata={
                    "repo_id": args.repo_id, "dataset_root": str(Path(args.root).resolve()),
                    "step": step, "steps_this_run": args.steps, "init_from": args.init_from,
                    "selection": "best_closed_loop", "episodes": episode_indices,
                })
                print(f"    new best closed-loop success {cl['success_rate']:.0%} -> saved {best_path}")

    print(f"done in {(time.time()-t0)/60:.1f} min -> {args.output}")


if __name__ == "__main__":
    main()
