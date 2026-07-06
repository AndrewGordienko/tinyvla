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
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import cycle
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

from .paths import CHECKPOINTS_ROOT, DATASETS_ROOT
from .eval_closedloop import evaluate_closed_loop, format_metrics

BASE = "lerobot/smolvla_base"


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
    ap.add_argument("--n-action-steps", type=int, default=None,
                    help="Actions executed per replan (<=chunk_size). Base default is 50 "
                         "(50-step open-loop -> compounding drift). Try 10 for tighter closed-loop.")
    ap.add_argument("--delta-actions", action="store_true",
                    help="Dataset stores joint deltas (action-state); closed-loop eval adds the live "
                         "pose back. Set this to match a delta-actions dataset.")
    args = ap.parse_args()
    cl_commands = [int(x) for x in args.closed_loop_commands.split(",") if x != ""]

    device = torch.device(args.device)
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)

    # policy config -> load pretrained smolvla_base, run on our device
    cfg = SmolVLAConfig(pretrained_path=BASE, device=args.device)
    if args.n_action_steps is not None:
        cfg.n_action_steps = args.n_action_steps

    # each sample needs an action chunk of `chunk_size` future steps
    delta_timestamps = {"action": [i / meta.fps for i in range(cfg.chunk_size)]}
    ds = LeRobotDataset(args.repo_id, root=args.root, delta_timestamps=delta_timestamps)

    policy = make_policy(cfg=cfg, ds_meta=meta)
    policy.to(device)

    # processors: reuse the base tokenizer/image pipeline, but normalize with OUR stats
    norm_feats = {**policy.config.input_features, **policy.config.output_features}
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=BASE,
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
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    policy.train()
    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"training SmolVLA ({n_params/1e6:.0f}M trainable params) on {device} | "
          f"{meta.total_episodes} eps, {meta.total_frames} frames | "
          f"chunk={cfg.chunk_size}, batch={args.batch_size}, steps={args.steps}")

    # bf16 autocast on CUDA (big speedup on H100; no GradScaler needed for bf16)
    amp = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device.type == "cuda" else nullcontext())

    t0 = time.time()
    running = 0.0
    for step, batch in zip(range(1, args.steps + 1), cycle(dl)):
        batch = preprocessor(batch)
        with amp:
            loss, _ = policy.forward(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        opt.step()
        opt.zero_grad()

        running += loss.item()
        if step % args.log_every == 0:
            dt = time.time() - t0
            print(f"  step {step:5d}/{args.steps}  loss {running/args.log_every:.4f}  "
                  f"{step/dt:.2f} it/s")
            running = 0.0
        if step % args.save_every == 0 or step == args.steps:
            policy.save_pretrained(args.output)
            preprocessor.save_pretrained(args.output)
            postprocessor.save_pretrained(args.output)
            if args.delta_actions:
                (Path(args.output) / "delta_actions.json").write_text('{"delta_actions": true}\n')
            print(f"  saved checkpoint to {args.output} (step {step})")

        if args.closed_loop_every and (step % args.closed_loop_every == 0 or step == args.steps):
            cl = evaluate_closed_loop(
                policy, preprocessor, postprocessor,
                device=device, commands=cl_commands,
                cap=args.closed_loop_cap, seed=args.closed_loop_seed,
                delta_actions=args.delta_actions,
            )
            print(f"  step {step:5d}/{args.steps}  closed-loop {format_metrics(cl)}")

    print(f"done in {(time.time()-t0)/60:.1f} min -> {args.output}")


if __name__ == "__main__":
    main()
