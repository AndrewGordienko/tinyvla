"""Render canonical command-0 scenes and compare renderer/action divergence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from scripts.canonical_dagger_round1 import CAMERAS, IMG, build_obs_from_frames
from tinyvla.runtime import load_runtime
from tinyvla.task import COMMANDS, SO101PickPlaceTask


def render_set(output):
    env = SO101PickPlaceTask(seed=100)
    renderers = {camera: mujoco.Renderer(env.model, height=IMG, width=IMG) for camera in CAMERAS}
    result = {}
    for episode in range(4):
        seed = 100 + 1009 * episode
        env.rng = np.random.default_rng(seed); env.reset(command=0)
        result[f"state_{episode}"] = env.data.qpos[:6].copy().astype(np.float32)
        for camera in CAMERAS:
            renderers[camera].update_scene(env.data, camera=camera)
            result[f"{camera}_{episode}"] = renderers[camera].render().copy()
    np.savez_compressed(output, **result)


def ssim(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5); mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    var_a = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a * mu_a
    var_b = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b * mu_b
    cov = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    return float(np.mean(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) /
                         ((mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2))))


def action_chunks(frames, checkpoint, dataset, device):
    meta = LeRobotDatasetMetadata("local/command0_multiview_32", root=dataset)
    runtime = load_runtime(checkpoint, meta=meta, dataset_root=dataset, device=device, stats_source="checkpoint")
    env = SO101PickPlaceTask(seed=100); outputs = []
    for episode in range(4):
        env.reset(command=0); env.data.qpos[:6] = frames[f"state_{episode}"]; mujoco.mj_forward(env.model, env.data)
        images = [frames[f"{camera}_{episode}"] for camera in CAMERAS]
        obs = runtime.preprocessor(build_obs_from_frames(env, images, COMMANDS[0]["instruction"], device))
        generator = torch.Generator(device="cpu").manual_seed(4242 + episode)
        noise = torch.randn((1, runtime.policy.config.chunk_size, runtime.policy.config.max_action_dim), generator=generator).to(device)
        with torch.inference_mode():
            chunk = runtime.postprocessor(runtime.policy.predict_action_chunk(obs, noise=noise))
        outputs.append(chunk.squeeze(0).cpu().numpy()[:, :6])
    return outputs


def compare(args):
    left, right = np.load(args.left), np.load(args.right)
    frame_rows = []
    for episode in range(4):
        for camera in CAMERAS:
            a, b = left[f"{camera}_{episode}"], right[f"{camera}_{episode}"]
            diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
            frame_rows.append({"episode": episode, "camera": camera, "mae": float(diff.mean()),
                               "p99": float(np.percentile(diff, 99)), "max": int(diff.max()),
                               "changed_pixel_fraction": float(np.any(diff != 0, axis=2).mean()), "ssim": ssim(a, b)})
    result = {"frames": frame_rows}
    if args.checkpoint:
        left_actions = action_chunks(left, args.checkpoint, args.dataset, args.device)
        right_actions = action_chunks(right, args.checkpoint, args.dataset, args.device)
        rows = []
        for episode, (a, b) in enumerate(zip(left_actions, right_actions)):
            diff = np.abs(a - b)
            rows.append({"episode": episode, "mae": float(diff.mean()), "max": float(diff.max()),
                         "physical_action_range_left": [float(a.min()), float(a.max())],
                         "physical_action_range_right": [float(b.min()), float(b.max())]})
        result["action_chunks"] = rows
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    render = sub.add_parser("render"); render.add_argument("--output", required=True)
    compare_parser = sub.add_parser("compare"); compare_parser.add_argument("--left", required=True)
    compare_parser.add_argument("--right", required=True); compare_parser.add_argument("--output", required=True)
    compare_parser.add_argument("--checkpoint"); compare_parser.add_argument("--dataset", default="data/datasets/command0_multiview_32")
    compare_parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    render_set(args.output) if args.command == "render" else compare(args)
