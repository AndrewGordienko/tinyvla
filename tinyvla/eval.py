"""Closed-loop evaluation of the fine-tuned SmolVLA policy on the SO-101 task.

The policy sees ONLY what it was trained on — the camera image, the joint state,
and the language instruction — and drives the arm. No privileged cube/destination.
We run every supported command and report per-command success, and (optionally)
save a filmstrip montage (one row per command).

Run:  python3 -m tinyvla.eval --per-command 5 --film --device cuda
"""
from __future__ import annotations

import argparse
import os
import numpy as np
import torch
import mujoco

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # datasets before policies
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

from .task import SO101PickPlaceTask, COMMANDS
from .collect import IMG, CAMERAS
from .paths import ARTIFACTS_ROOT, CHECKPOINTS_ROOT, DATASETS_ROOT

BASE = "lerobot/smolvla_base"


def build_obs(env, renderer, device):
    state = torch.from_numpy(env.data.qpos[:6].copy().astype(np.float32))
    obs = {
        "observation.state": state.unsqueeze(0).to(device),
        "task": [env.instruction],                                 # varies per command
    }
    for cam in CAMERAS:
        renderer.update_scene(env.data, camera=cam)
        img = torch.from_numpy(renderer.render()).permute(2, 0, 1).float() / 255.0
        obs[f"observation.images.{cam}"] = img.unsqueeze(0).to(device)
    return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(CHECKPOINTS_ROOT / "smolvla_pickplace"))
    ap.add_argument("--root", default=str(DATASETS_ROOT / "so101_pickplace"))
    ap.add_argument("--repo-id", default="local/so101_pickplace")
    ap.add_argument("--per-command", type=int, default=5)
    ap.add_argument("--base-steps", type=int, default=120, help="closed-loop steps per grasp phase")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--film", action="store_true", help="save a filmstrip montage (one row per command)")
    ap.add_argument("--seed", type=int, default=999)
    args = ap.parse_args()

    device = torch.device(args.device)
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    cfg = SmolVLAConfig(pretrained_path=args.model, device=args.device)
    policy = make_policy(cfg=cfg, ds_meta=meta).to(device).eval()

    norm_feats = {**policy.config.input_features, **policy.config.output_features}
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=BASE,
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

    env = SO101PickPlaceTask(seed=args.seed)
    renderer = mujoco.Renderer(env.model, height=IMG, width=IMG)
    big = mujoco.Renderer(env.model, height=360, width=480) if args.film else None

    total_ok = total = 0
    rows = []
    print(f"\nclosed-loop eval | {args.per_command} episodes/command | model={args.model}\n")
    for ci, spec in enumerate(COMMANDS):
        horizon = args.base_steps * len(spec["steps"])
        ok = 0
        for ep in range(args.per_command):
            env.reset(command=ci)
            policy.reset()
            grab_film = args.film and ep == 0
            film = []
            for t in range(horizon):
                obs = preprocessor(build_obs(env, renderer, device))
                with torch.inference_mode():
                    action = policy.select_action(obs)
                action = postprocessor(action).squeeze(0).cpu().numpy()
                env.step(action)
                if grab_film and t % max(1, horizon // 7) == 0:
                    big.update_scene(env.data, camera="front"); film.append(big.render())
            ok += env.success()
            if grab_film:
                big.update_scene(env.data, camera="front"); film.append(big.render())
                rows.append(np.concatenate(film, axis=1))
        total_ok += ok; total += args.per_command
        print(f"  [{ok}/{args.per_command}] {spec['instruction']}")

    print(f"\nOVERALL closed-loop success: {total_ok}/{total} = {100*total_ok/total:.0f}%")

    if rows:
        from PIL import Image
        w = min(r.shape[1] for r in rows)
        rows = [r[:, :w] for r in rows]
        out = ARTIFACTS_ROOT / "eval_montage.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.concatenate(rows, axis=0)).save(out)
        print(f"saved {out}")

    os._exit(0)   # skip noisy EGL context teardown on headless boxes


if __name__ == "__main__":
    main()
