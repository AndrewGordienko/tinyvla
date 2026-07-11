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
from .determinism import preserve_rng_state

IMG = 256
DEFAULT_COMMANDS = (0, 1, 2, 3)


def build_obs(env: SO101PickPlaceTask, renderer, instruction: str, device, camera: str = "front"):
    renderers = renderer if isinstance(renderer, dict) else {camera: renderer}
    cameras = list(renderers) if isinstance(renderer, dict) else [camera]
    images = {}
    for cam in cameras:
        renderers[cam].update_scene(env.data, camera=cam)
        images[f"observation.images.{cam}"] = torch.from_numpy(renderers[cam].render()).permute(2, 0, 1).float() / 255.0
    state = torch.from_numpy(env.data.qpos[:6].copy().astype(np.float32))
    return {
        "observation.state": state.unsqueeze(0).to(device),
        **{key: image.unsqueeze(0).to(device) for key, image in images.items()},
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
    episodes: int = 1,
    stop_on_success: bool = True,
    dwell: int = 8,
    positions_by_rollout: dict[tuple[int, int], dict[str, np.ndarray]] | None = None,
) -> dict:
    """Roll out `policy` on each command and return graded closed-loop metrics.

    If ``delta_actions`` is set, the model predicts joint deltas relative to the
    current pose; we add the live ``qpos[:6]`` back to recover absolute targets
    before stepping the sim (mirror of the delta transform applied to the data).

    ``episodes`` rolls out each command that many times under different (but
    deterministic) scene draws — success over len(commands)*episodes rollouts.
    A single 6-command sweep quantizes to 17% steps, too coarse to compare runs;
    use episodes>=3 whenever two numbers will be compared. The env RNG is
    reseeded per rollout from (seed, command, episode), so every checkpoint and
    every run sees the exact same scenes.

    Restores the policy's train/eval mode on exit so it is safe to call between
    optimizer steps.
    """
    was_training = policy.training
    policy.eval()
    env = SO101PickPlaceTask(seed=seed)
    configured_cameras = list(getattr(getattr(policy, "config", None), "image_features", {}) or {})
    cameras = [key.removeprefix("observation.images.") for key in configured_cameras]
    if not cameras:
        cameras = [camera]
    renderers = {cam: mujoco.Renderer(env.model, height=img, width=img) for cam in cameras}
    successes, min_dists, final_dists = [], [], []
    per_command_raw: dict[int, dict[str, list[float]]] = {
        int(ci): {"successes": [], "min_dists": [], "final_dists": []} for ci in commands
    }
    with preserve_rng_state():
        try:
            for ci in commands:
                for ep in range(episodes):
                    rollout_seed = seed + 1009 * ep + ci
                    env.rng = np.random.default_rng(rollout_seed)
                    positions = (positions_by_rollout or {}).get((int(ci), ep))
                    env.reset(command=ci, positions=positions)
                    policy.reset()
                    torch.manual_seed(rollout_seed)
                    dmin = float("inf")
                    hold = 0
                    for _ in range(cap):
                        obs = preprocessor(build_obs(env, renderers, COMMANDS[ci]["instruction"], device, camera))
                        with torch.inference_mode():
                            action = policy.select_action(obs)
                        action = postprocessor(action).squeeze(0).cpu().numpy()
                        if delta_actions:
                            action = action + env.data.qpos[:6].astype(action.dtype)
                        env.step(action)
                        # Distance to the ACTIVE sub-task's cube, derived from scene
                        # state — for two-step commands a learned policy never advances
                        # step_idx, so env.cube_pos() (target_color) would keep measuring
                        # the first cube even while the policy delivers the second.
                        active_color = env.active_subtask()[0]
                        dmin = min(dmin, float(np.linalg.norm(env.ee_pos() - env.cube_pos(active_color))))
                        hold = hold + 1 if env.success() else 0
                        if stop_on_success and hold >= dwell:
                            break
                    succeeded = int(env.success())
                    final_dist = float(np.linalg.norm(env.ee_pos() - env.cube_pos(env.active_subtask()[0])))
                    successes.append(succeeded)
                    min_dists.append(dmin)
                    final_dists.append(final_dist)
                    row = per_command_raw[int(ci)]
                    row["successes"].append(succeeded)
                    row["min_dists"].append(dmin)
                    row["final_dists"].append(final_dist)
        finally:
            for renderer in renderers.values():
                renderer.close()
            policy.reset()
            if was_training:
                policy.train()
    n = len(commands) * episodes
    per_command = {
        str(ci): {
            "instruction": COMMANDS[ci]["instruction"],
            "n": len(values["successes"]),
            "successes": int(np.sum(values["successes"])),
            "success_rate": float(np.mean(values["successes"])) if values["successes"] else 0.0,
            "mean_min_dist": float(np.mean(values["min_dists"])) if values["min_dists"] else None,
            "mean_final_dist": float(np.mean(values["final_dists"])) if values["final_dists"] else None,
        }
        for ci, values in per_command_raw.items()
    }
    result = {
        "n": n,
        "successes": int(np.sum(successes)),
        "success_rate": float(np.mean(successes)) if n else 0.0,
        "mean_min_dist": float(np.mean(min_dists)) if n else None,
        "mean_final_dist": float(np.mean(final_dists)) if n else None,
        "per_command": per_command,
    }
    groups = {"single_step_0_3": (0, 1, 2, 3), "stacking_4_5": (4, 5), "two_step_6_7": (6, 7)}
    result["groups"] = {}
    for name, members in groups.items():
        rows = [per_command[str(ci)] for ci in members if str(ci) in per_command]
        if rows:
            group_n = sum(row["n"] for row in rows)
            group_successes = sum(row["successes"] for row in rows)
            result["groups"][name] = {
                "n": group_n,
                "successes": group_successes,
                "success_rate": group_successes / group_n if group_n else 0.0,
            }
    return result


def evaluate_per_command(policy, preprocessor, postprocessor, *, device,
                         commands=DEFAULT_COMMANDS, cap: int = 180, seed: int = 100,
                         camera: str = "front", img: int = IMG,
                         delta_actions: bool = False, episodes: int = 1) -> dict:
    """Per-command closed-loop metrics -> {command_index: metrics}. Used by the
    curriculum to find which commands the policy is worst at."""
    out = {}
    for ci in commands:
        out[ci] = evaluate_closed_loop(
            policy, preprocessor, postprocessor, device=device, commands=[ci],
            cap=cap, seed=seed, camera=camera, img=img, delta_actions=delta_actions,
            episodes=episodes,
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
