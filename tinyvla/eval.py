"""Closed-loop evaluation of the fine-tuned SmolVLA policy on the reach task.

The policy sees ONLY what it was trained on — the camera image, the joint state,
and the language instruction — and drives the arm. No privileged cube position.
We measure how often it reaches the cube, and save a filmstrip.

Run:  python3 -m tinyvla.eval --episodes 20
      python3 -m tinyvla.eval --episodes 6 --film
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
import mujoco

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # datasets before policies
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

from .task import SO101ReachTask, INSTRUCTION
from .collect import IMG, EP_LEN
from .paths import ARTIFACTS_ROOT, CHECKPOINTS_ROOT, DATASETS_ROOT

BASE = "lerobot/smolvla_base"  # source of the tokenizer/image processor pipeline


def build_obs(env, renderer, device):
    renderer.update_scene(env.data, camera="front")
    img = renderer.render()                                   # (H,W,3) uint8
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (3,H,W) [0,1]
    state = torch.from_numpy(env.data.qpos[:6].copy().astype(np.float32))
    return {
        "observation.state": state.unsqueeze(0).to(device),
        "observation.images.front": img.unsqueeze(0).to(device),
        "task": [INSTRUCTION],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(CHECKPOINTS_ROOT / "smolvla_reach"))
    ap.add_argument("--root", default=str(DATASETS_ROOT / "so101_reach"))
    ap.add_argument("--repo-id", default="local/so101_reach")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--steps", type=int, default=EP_LEN)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--film", action="store_true", help="save a filmstrip of episode 0")
    ap.add_argument("--seed", type=int, default=999)
    args = ap.parse_args()

    device = torch.device(args.device)
    # load like train.py (SmolVLAPolicy.from_pretrained's config parser is broken on py3.14)
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    cfg = SmolVLAConfig(pretrained_path=args.model, device=args.device)
    policy = make_policy(cfg=cfg, ds_meta=meta).to(device).eval()

    norm_feats = {**policy.config.input_features, **policy.config.output_features}
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=BASE,
        preprocessor_overrides={
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": meta.stats, "features": norm_feats,
                "norm_map": policy.config.normalization_mapping},
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "stats": meta.stats, "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping},
        },
    )

    env = SO101ReachTask(seed=args.seed)
    renderer = mujoco.Renderer(env.model, height=IMG, width=IMG)
    big = mujoco.Renderer(env.model, height=480, width=640)

    succ = 0
    frames = []
    for ep in range(args.episodes):
        env.reset()
        policy.reset()                                  # clear the action-chunk queue
        for t in range(args.steps):
            obs = build_obs(env, renderer, device)
            obs = preprocessor(obs)
            with torch.inference_mode():
                action = policy.select_action(obs)
            action = postprocessor(action).squeeze(0).cpu().numpy()
            env.step(action)
            if args.film and ep == 0 and t % 6 == 0:
                big.update_scene(env.data, camera="front")
                frames.append(big.render())
        ok = env.success()
        succ += ok
        print(f"  ep {ep:2d}: {'reach' if ok else 'MISS '}  "
              f"final dist {np.linalg.norm(env.ee_pos()-env.target_pos()):.3f}")

    print(f"\nSmolVLA closed-loop success: {succ}/{args.episodes} = {100*succ/args.episodes:.0f}%")

    if frames:
        from PIL import Image
        out = ARTIFACTS_ROOT / "eval_filmstrip.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.concatenate(frames, axis=1)).save(out)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
