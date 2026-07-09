"""Fine-tune SmolVLA (smolvla_base) on the scripted-expert reach dataset.

This is a minimal, self-contained training loop that reuses LeRobot's policy and
processor factories (the official `lerobot-train` CLI's config parser is broken
under Python 3.14, so we build the pieces directly).

  - loads smolvla_base pretrained weights
  - overrides the input/output normalizers with OUR dataset's stats
  - trains the action expert to imitate the expert reaches
  - saves the fine-tuned policy + processors to artifacts/checkpoints/

Run:  python3 -m tinyvla.train --steps 2000 --batch-size 8
      python3 -m tinyvla.train --steps 2      # smoke test
"""
from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# NOTE: import datasets before policies to avoid a circular import in lerobot 0.5.1
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import cycle
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

from .paths import CHECKPOINTS_ROOT, DATASETS_ROOT
from .eval_closedloop import evaluate_closed_loop, format_metrics
from .fast_dataset import FastChunkDataset
from .trainability import TRAINABLE_MODES, group_for_param, set_trainable

BACKBONE_GROUPS = ("vision_encoder", "vision_connector", "vlm_text")

BASE = "lerobot/smolvla_base"


def save_checkpoint(policy, preprocessor, postprocessor, output: str | Path, *, delta_actions: bool) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(output)
    preprocessor.save_pretrained(output)
    postprocessor.save_pretrained(output)
    if delta_actions:
        (output / "delta_actions.json").write_text('{"delta_actions": true}\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="local/so101_pickplace")
    ap.add_argument("--root", default=str(DATASETS_ROOT / "so101_pickplace"))
    ap.add_argument("--output", default=str(CHECKPOINTS_ROOT / "smolvla_pickplace"))
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=500)
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
    ap.add_argument("--delta-actions", action="store_true",
                    help="Dataset stores joint deltas (action-state); closed-loop eval adds the live "
                         "pose back. Set this to match a delta-actions dataset.")
    ap.add_argument("--save-best-closed-loop", action="store_true",
                    help="Keep the best closed-loop checkpoint in <output>/best_closed_loop.")
    ap.add_argument("--trainable", choices=TRAINABLE_MODES, default="checkpoint",
                    help="Optionally override checkpoint trainability for brain/vision experiments.")
    ap.add_argument("--backbone-lr-scale", type=float, default=1.0,
                    help="LR multiplier for the pretrained VLM backbone (vision/connector/text) "
                         "relative to --lr. Use ~0.1 when unfreezing (--trainable brain/brain_visual/all) "
                         "so the expert adapts fast while the backbone moves gently.")
    ap.add_argument("--warmup-steps", type=int, default=0,
                    help="Linear LR warmup over this many steps (0=off). Recommended (~500) when "
                         "unfreezing the backbone: early expert gradients are noise and will wreck "
                         "pretrained features at full LR.")
    ap.add_argument("--init-from", default=None,
                    help="Warm-start from this checkpoint dir instead of smolvla_base "
                         "(e.g. the previous DAgger round) — later rounds then need fewer steps.")
    args = ap.parse_args()
    cl_commands = [int(x) for x in args.closed_loop_commands.split(",") if x != ""]

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True   # fp32 matmuls (norms etc.) on tensor cores
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True          # fixed shapes -> autotune convs once
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)

    # policy config -> load pretrained weights (smolvla_base, or --init-from checkpoint)
    src = args.init_from or BASE
    cfg = SmolVLAConfig(pretrained_path=src, device=args.device)
    if args.n_action_steps is not None:
        cfg.n_action_steps = args.n_action_steps

    # each sample needs an action chunk of `chunk_size` future steps.
    # FastChunkDataset fixes a ~70x dataloader slowdown in the chunk query
    # (lerobot row-first fallback decodes the image column per chunk row).
    delta_timestamps = {"action": [i / meta.fps for i in range(cfg.chunk_size)]}
    ds = FastChunkDataset(args.repo_id, root=args.root, delta_timestamps=delta_timestamps)

    policy = make_policy(cfg=cfg, ds_meta=meta)
    policy.to(device)
    trainable_params = set_trainable(policy, args.trainable)

    # processors: reuse the base tokenizer/image pipeline, but normalize with OUR stats
    norm_feats = {**policy.config.input_features, **policy.config.output_features}
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=src,
        preprocessor_overrides={
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": meta.stats,
                "features": norm_feats,
                "norm_map": policy.config.normalization_mapping,
            },
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "stats": meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        },
    )

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                    persistent_workers=(args.num_workers > 0), drop_last=True)
    trainable_parameters = [p for p in policy.parameters() if p.requires_grad]
    if not trainable_parameters:
        raise SystemExit(f"no trainable parameters for --trainable {args.trainable}")
    if args.backbone_lr_scale != 1.0:
        backbone, head = [], []
        for name, p in policy.named_parameters():
            if p.requires_grad:
                (backbone if group_for_param(name) in BACKBONE_GROUPS else head).append(p)
        opt = torch.optim.AdamW([
            {"params": head, "lr": args.lr},
            {"params": backbone, "lr": args.lr * args.backbone_lr_scale},
        ])
        print(f"discriminative LR: head {sum(p.numel() for p in head)/1e6:.0f}M @ {args.lr:g}, "
              f"backbone {sum(p.numel() for p in backbone)/1e6:.0f}M @ {args.lr * args.backbone_lr_scale:g}")
    else:
        opt = torch.optim.AdamW(trainable_parameters, lr=args.lr)
    sched = (torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup_steps))
             if args.warmup_steps > 0 else None)

    policy.train()
    print(f"training SmolVLA ({trainable_params/1e6:.0f}M trainable params, "
          f"mode={args.trainable}) on {device} | "
          f"{meta.total_episodes} eps, {meta.total_frames} frames | "
          f"chunk={cfg.chunk_size}, batch={args.batch_size}, steps={args.steps}")

    # bf16 autocast on CUDA (big speedup on H100; no GradScaler needed for bf16)
    amp = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device.type == "cuda" else nullcontext())

    t0 = time.time()
    running = torch.zeros((), device=device)   # accumulate on-device; .item() every step = a GPU sync
    best_cl = -1.0
    for step, batch in zip(range(1, args.steps + 1), cycle(dl)):
        batch = preprocessor(batch)
        with amp:
            loss, _ = policy.forward(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters, 10.0)
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
            save_checkpoint(policy, preprocessor, postprocessor, args.output,
                            delta_actions=args.delta_actions)
            print(f"  saved checkpoint to {args.output} (step {step})")

        if args.closed_loop_every and (step % args.closed_loop_every == 0 or step == args.steps):
            cl = evaluate_closed_loop(
                policy, preprocessor, postprocessor,
                device=device, commands=cl_commands,
                cap=args.closed_loop_cap, seed=args.closed_loop_seed,
                delta_actions=args.delta_actions, episodes=args.closed_loop_episodes,
            )
            print(f"  step {step:5d}/{args.steps}  closed-loop {format_metrics(cl)}")
            if args.save_best_closed_loop and cl["success_rate"] > best_cl:
                best_cl = cl["success_rate"]
                best_path = Path(args.output) / "best_closed_loop"
                save_checkpoint(policy, preprocessor, postprocessor, best_path,
                                delta_actions=args.delta_actions)
                print(f"    new best closed-loop success {cl['success_rate']:.0%} -> saved {best_path}")

    print(f"done in {(time.time()-t0)/60:.1f} min -> {args.output}")


if __name__ == "__main__":
    main()
