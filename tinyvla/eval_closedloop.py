"""Reusable closed-loop evaluation for SmolVLA policies on the sim pick-place task.

Judge checkpoints by CLOSED-LOOP success, not offline flow-matching loss: the two
are decoupled (a checkpoint can have the best offline loss and the worst rollout).
Returns graded metrics (min/final end-effector -> target-cube distance) so partial
progress is visible even at 0% success.

Used inline during training (tinyvla.train / tinyvla.recover) so every run reports
the metric that actually matters, and as a library for tinyvla.benchmark.
"""
from __future__ import annotations

import numpy as np
import torch
import mujoco

from .task import SO101PickPlaceTask, COMMANDS

IMG = 256
DEFAULT_COMMANDS = (0, 1, 2, 3)


def build_obs(env: SO101PickPlaceTask, renderer, instruction: str, device, camera: str = "front"):
    renderer.update_scene(env.data, camera=camera)
    img = torch.from_numpy(renderer.render()).permute(2, 0, 1).float() / 255.0
    state = torch.from_numpy(env.data.qpos[:6].copy().astype(np.float32))
    return {
        "observation.state": state.unsqueeze(0).to(device),
        f"observation.images.{camera}": img.unsqueeze(0).to(device),
        "task": [instruction],
    }


def evaluate_closed_loop(
    policy,
    preprocessor,
    postprocessor,
    *,
    device,
    commands=DEFAULT_COMMANDS,
    cap: int = 180,
    seed: int = 100,
    camera: str = "front",
    img: int = IMG,
    delta_actions: bool = False,
) -> dict:
    """Roll out `policy` on each command and return graded closed-loop metrics.

    If ``delta_actions`` is set, the model predicts joint deltas relative to the
    current pose; we add the live ``qpos[:6]`` back to recover absolute targets
    before stepping the sim (mirror of the delta transform applied to the data).

    Restores the policy's train/eval mode on exit so it is safe to call between
    optimizer steps.
    """
    was_training = policy.training
    policy.eval()
    env = SO101PickPlaceTask(seed=seed)
    renderer = mujoco.Renderer(env.model, height=img, width=img)
    successes, min_dists, final_dists = [], [], []
    try:
        for i, ci in enumerate(commands):
            env.reset(command=ci)
            policy.reset()
            torch.manual_seed(seed + i)
            dmin = float("inf")
            for _ in range(cap):
                obs = preprocessor(build_obs(env, renderer, COMMANDS[ci]["instruction"], device, camera))
                with torch.inference_mode():
                    action = policy.select_action(obs)
                action = postprocessor(action).squeeze(0).cpu().numpy()
                if delta_actions:
                    action = action + env.data.qpos[:6].astype(action.dtype)
                env.step(action)
                dmin = min(dmin, float(np.linalg.norm(env.ee_pos() - env.cube_pos())))
            successes.append(int(env.success()))
            min_dists.append(dmin)
            final_dists.append(float(np.linalg.norm(env.ee_pos() - env.cube_pos())))
    finally:
        renderer.close()
        if was_training:
            policy.train()
    n = len(commands)
    return {
        "n": n,
        "successes": int(np.sum(successes)),
        "success_rate": float(np.mean(successes)) if n else 0.0,
        "mean_min_dist": float(np.mean(min_dists)) if n else None,
        "mean_final_dist": float(np.mean(final_dists)) if n else None,
    }


def evaluate_per_command(policy, preprocessor, postprocessor, *, device,
                         commands=DEFAULT_COMMANDS, cap: int = 180, seed: int = 100,
                         camera: str = "front", img: int = IMG,
                         delta_actions: bool = False) -> dict:
    """Per-command closed-loop metrics -> {command_index: metrics}. Used by the
    curriculum to find which commands the policy is worst at."""
    out = {}
    for ci in commands:
        out[ci] = evaluate_closed_loop(
            policy, preprocessor, postprocessor, device=device, commands=[ci],
            cap=cap, seed=seed, camera=camera, img=img, delta_actions=delta_actions,
        )
    return out


def worst_commands(per_cmd: dict, k: int) -> list[int]:
    """Rank commands worst-first by (success_rate asc, mean_final_dist desc) and
    return the k worst command indices — the gaps to generate more data for."""
    ranked = sorted(per_cmd.items(),
                    key=lambda kv: (kv[1]["success_rate"], -kv[1]["mean_final_dist"]))
    return [ci for ci, _ in ranked[:k]]


def format_metrics(m: dict) -> str:
    return (f"success {m['successes']}/{m['n']} ({m['success_rate']:.0%})  "
            f"min_dist {m['mean_min_dist']:.3f}  final_dist {m['mean_final_dist']:.3f}")
