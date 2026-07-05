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

import torch
from torch.utils.data import DataLoader

# NOTE: import datasets before policies to avoid a circular import in lerobot 0.5.1
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import cycle
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

from .paths import CHECKPOINTS_ROOT, DATASETS_ROOT

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
    args = ap.parse_args()

    device = torch.device(args.device)
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)

    # policy config -> load pretrained smolvla_base, run on our device
    cfg = SmolVLAConfig(pretrained_path=BASE, device=args.device)

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

    t0 = time.time()
    running = 0.0
    for step, batch in zip(range(1, args.steps + 1), cycle(dl)):
        batch = preprocessor(batch)
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
            print(f"  saved checkpoint to {args.output} (step {step})")

    print(f"done in {(time.time()-t0)/60:.1f} min -> {args.output}")


if __name__ == "__main__":
    main()
