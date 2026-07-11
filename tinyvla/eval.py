"""Closed-loop evaluation of the fine-tuned SmolVLA policy on the SO-101 task.

The policy sees ONLY what it was trained on — the camera image, the joint state,
and the language instruction — and drives the arm. No privileged cube/destination.
We run every supported command and report per-command success, and (optionally)
save a filmstrip montage (one row per command).

Run:  python3 -m tinyvla.eval --per-command 5 --film --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
import mujoco

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # datasets before policies
from .task import SO101PickPlaceTask, COMMANDS
from .collect import IMG, CAMERAS
from .paths import ARTIFACTS_ROOT, CHECKPOINTS_ROOT, DATASETS_ROOT
from .determinism import seed_everything
from .eval_closedloop import evaluate_closed_loop
from .runtime import experiment_metadata, load_runtime


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
    ap.add_argument("--commands", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--delta-actions", action="store_true", default=None,
                    help="Legacy assertion only; semantics are loaded automatically.")
    ap.add_argument("--output", default=str(ARTIFACTS_ROOT / "evaluations" / "latest.json"))
    args = ap.parse_args()
    commands = [int(value) for value in args.commands.split(",") if value]
    seed_everything(args.seed)

    device = torch.device(args.device)
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    runtime = load_runtime(
        args.model, meta=meta, dataset_root=args.root, device=device, stats_source="checkpoint"
    )
    if args.delta_actions is not None and args.delta_actions != runtime.delta_actions:
        raise SystemExit(
            f"--delta-actions contradicts detected {runtime.action_semantics} runtime semantics"
        )
    policy = runtime.policy.eval()
    preprocessor, postprocessor = runtime.preprocessor, runtime.postprocessor

    metrics = evaluate_closed_loop(
        policy,
        preprocessor,
        postprocessor,
        device=device,
        commands=commands,
        cap=args.base_steps * 2,
        seed=args.seed,
        delta_actions=runtime.delta_actions,
        episodes=args.per_command,
    )
    result = {
        "model": str(args.model),
        "dataset": {"repo_id": args.repo_id, "root": args.root},
        "action_semantics": runtime.action_semantics,
        "load_report": runtime.load_report,
        "metrics": metrics,
        "experiment": experiment_metadata(seed=args.seed),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))

    if args.film:
        env = SO101PickPlaceTask(seed=args.seed)
        renderer = mujoco.Renderer(env.model, height=IMG, width=IMG)
        big = mujoco.Renderer(env.model, height=360, width=480)
        rows = []
        for ci in commands:
            spec = COMMANDS[ci]
            horizon = args.base_steps * len(spec["steps"])
            env.rng = np.random.default_rng(args.seed + ci)
            env.reset(command=ci)
            policy.reset()
            film = []
            for t in range(horizon):
                obs = preprocessor(build_obs(env, renderer, device))
                with torch.inference_mode():
                    action = policy.select_action(obs)
                action = postprocessor(action).squeeze(0).cpu().numpy()
                if runtime.delta_actions:
                    action = action + env.data.qpos[:6].astype(action.dtype)
                env.step(action)
                if t % max(1, horizon // 7) == 0:
                    big.update_scene(env.data, camera="front")
                    film.append(big.render())
            big.update_scene(env.data, camera="front")
            film.append(big.render())
            rows.append(np.concatenate(film, axis=1))
        renderer.close()
        big.close()
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
