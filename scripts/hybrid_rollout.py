"""Causal counterfactual rollouts on the memorized command-0 scenes.

Discriminates arm-localization vs gripper-timing vs interaction vs queue
staleness WITHOUT retraining, by swapping components of the executed action
between the learned policy and the scripted reactive expert (evaluated from the
current simulator state). Also runs the n_action_steps sweep and a clearly
labelled non-canonical 5 cm diagnostic.

Conditions (all at the canonical 4 cm radius unless noted):
  A  full learned policy
  B  learned arm (0:5) + expert gripper (5)
  C  expert arm (0:5) + learned gripper (5)
  D  learned arm + learned gripper thresholded to canonical open/closed
  E  all expert (reactive) — sanity, must pass 4/4
  A@5cm  full learned policy with a non-canonical 5 cm grasp radius (diagnostic)

Interpretation:
  B fails                -> arm/localization/controller alone explains failure
  B passes, C fails      -> gripper timing is primary
  B and C pass, A fails  -> interaction / queued-action timing
  only A@5cm passes      -> approach localization is the blocker
  D passes               -> continuous gripper output / calibration is the problem

Every executed step is instrumented (see TRACE_FIELDS) and full traces are saved.

Usage:
  PYTORCH_ENABLE_MPS_FALLBACK=1 MUJOCO_GL=glfw .venv/bin/python -m scripts.hybrid_rollout \
    --model artifacts/truth_harness/checkpoints/command0_overfit_500 --device mps
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import mujoco

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from tinyvla import task as task_module
from tinyvla.task import SO101PickPlaceTask, COMMANDS, GRIP_GRAB, GRIP_OPEN, GRIP_CLOSED, CUBE_Z
from tinyvla.eval_closedloop import build_obs, IMG
from tinyvla.runtime import load_runtime
from tinyvla.determinism import preserve_rng_state, seed_everything

CANONICAL_RADIUS = 0.04
LIFT_Z = CUBE_Z + 0.03  # cube counted "lifted" when raised >3 cm off the table


def _target_color(command: int) -> str:
    return COMMANDS[command]["steps"][0][0]


def _compose(condition: str, learned: np.ndarray, expert: np.ndarray) -> np.ndarray:
    """Return the pre-clip 6-dim executed action for a condition."""
    a = learned.copy()
    if condition == "A":          # full learned
        return learned.copy()
    if condition == "B":          # learned arm + expert gripper
        a[:5] = learned[:5]; a[5] = expert[5]; return a
    if condition == "C":          # expert arm + learned gripper
        a[:5] = expert[:5]; a[5] = learned[5]; return a
    if condition == "D":          # learned arm + thresholded learned gripper
        a[:5] = learned[:5]
        a[5] = GRIP_CLOSED if learned[5] < GRIP_GRAB else GRIP_OPEN
        return a
    if condition == "E":          # all expert
        return expert.copy()
    raise ValueError(condition)


def rollout(policy, pre, post, scene: dict, *, device, n_action_steps: int,
            radius: float, condition: str, cap: int, seed: int, camera: str = "front") -> dict:
    command = int(scene["command"])
    color = _target_color(command)
    positions = {c: np.asarray(v, dtype=float) for c, v in scene["positions"].items()}

    # grasp radius is a module global consulted inside env._update_grasp()
    saved_radius = task_module.GRASP_RADIUS
    task_module.GRASP_RADIUS = radius
    env = SO101PickPlaceTask()
    renderer = mujoco.Renderer(env.model, height=IMG, width=IMG)
    cs = policy.config.chunk_size
    max_dim = policy.config.max_action_dim
    ctrl_lo, ctrl_hi = env.ctrl_range[:, 0], env.ctrl_range[:, 1]

    trace = []
    try:
        env.rng = np.random.default_rng(seed)
        env.reset(command=command, positions=positions)
        policy.reset()
        queue: list[np.ndarray] = []
        replan_idx = 0
        dmin = float("inf")
        grasp_fired_t = -1
        for t in range(cap):
            newly = not queue
            if newly:  # explicit chunk queue so "queued vs new" is unambiguous
                noise_seed = seed * 100003 + replan_idx
                gen = torch.Generator(device="cpu").manual_seed(noise_seed)
                noise = torch.randn((1, cs, max_dim), generator=gen).to(device)
                obs = pre(build_obs(env, renderer, env.instruction, device, camera))
                with torch.inference_mode():
                    chunk = post(policy.predict_action_chunk(obs, noise=noise))
                chunk = chunk.squeeze(0).cpu().numpy()[:, :6]
                queue = [chunk[i] for i in range(min(n_action_steps, cs))]
                replan_idx += 1
            else:
                noise_seed = -1
            q_index = len(queue) - 1
            learned = np.asarray(queue.pop(0), dtype=float)
            expert = np.asarray(env.reactive_action(gain=0.25, max_dq=0.03), dtype=float)
            composed = _compose(condition, learned, expert)
            clipped = np.clip(composed, ctrl_lo, ctrl_hi)

            ee = env.ee_pos(); cube = env.cube_pos(color)
            dist = float(np.linalg.norm(ee - cube))
            dmin = min(dmin, dist)
            env.step(clipped)
            if grasp_fired_t < 0 and env.grasped is not None:
                grasp_fired_t = t
            trace.append({
                "t": t,
                "ee_cube_dist": round(dist, 4),
                "action_preclip": [round(float(x), 4) for x in composed],
                "action_clipped": [round(float(x), 4) for x in clipped],
                "qpos6": [round(float(x), 4) for x in env.data.qpos[:6]],
                "gripper_qpos": round(float(env.data.qpos[5]), 4),
                "gripper_cmd_clipped": round(float(clipped[5]), 4),
                "crossed_GRIP_GRAB": bool(clipped[5] < GRIP_GRAB),
                "inside_radius": bool(dist < radius),
                "grasp_fired": env.grasped is not None,
                "grasped": env.grasped,
                "cube_lifted": bool(env.cube_pos(color)[2] > LIFT_Z),
                "queue_index": q_index,
                "newly_predicted": newly,
                "noise_seed": noise_seed,
            })
        success = bool(env.success())
    finally:
        renderer.close()
        task_module.GRASP_RADIUS = saved_radius

    lifted = any(r["cube_lifted"] for r in trace)
    grasped_ever = grasp_fired_t >= 0
    return {
        "condition": condition, "command": command, "n_action_steps": n_action_steps,
        "radius": radius, "success": success, "min_ee_cube_dist": round(dmin, 4),
        "grasp_fired_t": grasp_fired_t, "grasped_ever": grasped_ever, "cube_lifted": lifted,
        "instruction": scene["instruction"], "positions": scene["positions"], "trace": trace,
    }


def _summ(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "successes": sum(r["success"] for r in rows),
        "grasped_any": sum(r["grasped_ever"] for r in rows),
        "lifted_any": sum(r["cube_lifted"] for r in rows),
        "mean_min_ee_cube_dist": round(float(np.mean([r["min_ee_cube_dist"] for r in rows])), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/truth_harness/checkpoints/command0_overfit_500")
    ap.add_argument("--repo-id", default="local/truth_gate_command0_4")
    ap.add_argument("--root", default="artifacts/truth_harness/datasets/command0_4")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--cap", type=int, default=160)
    ap.add_argument("--trace-dir", default="artifacts/truth_harness/hybrid_traces")
    ap.add_argument("--output", default="artifacts/truth_harness/hybrid_summary.json")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    root = Path(args.root)
    scenes = json.loads((root / "scene_manifest.json").read_text())["scenes"]
    meta = LeRobotDatasetMetadata(args.repo_id, root=str(root))
    r = load_runtime(args.model, meta=meta, dataset_root=str(root), device=device, stats_source="checkpoint")
    policy, pre, post = r.policy.eval(), r.preprocessor, r.postprocessor
    n_default = int(policy.config.n_action_steps)

    trace_dir = Path(args.trace_dir); trace_dir.mkdir(parents=True, exist_ok=True)
    summary = {"model": args.model, "n_action_steps_default": n_default,
               "canonical_radius": CANONICAL_RADIUS, "conditions": {}, "nstep_sweep": {}}

    # ---- counterfactual battery at the canonical radius (+ 5 cm diagnostic on A)
    battery = [
        ("A", CANONICAL_RADIUS), ("B", CANONICAL_RADIUS), ("C", CANONICAL_RADIUS),
        ("D", CANONICAL_RADIUS), ("E", CANONICAL_RADIUS), ("A", 0.05),
    ]
    with preserve_rng_state():
        for cond, radius in battery:
            key = f"{cond}@{int(radius*100)}cm"
            rows = []
            for si, scene in enumerate(scenes):
                res = rollout(policy, pre, post, scene, device=device, n_action_steps=n_default,
                              radius=radius, condition=cond, cap=args.cap, seed=args.seed + si)
                rows.append(res)
                (trace_dir / f"{key}_scene{si}.json").write_text(json.dumps(res, indent=2))
            summary["conditions"][key] = _summ(rows)
            s = summary["conditions"][key]
            print(f"{key:8s}  success {s['successes']}/{s['n']}  grasped {s['grasped_any']}  "
                  f"lifted {s['lifted_any']}  mean_min_dist {s['mean_min_ee_cube_dist']}")

        # ---- n_action_steps sweep on condition A (no retraining)
        for n in (1, 5, 10):
            rows = []
            for si, scene in enumerate(scenes):
                res = rollout(policy, pre, post, scene, device=device, n_action_steps=n,
                              radius=CANONICAL_RADIUS, condition="A", cap=args.cap, seed=args.seed + si)
                rows.append(res)
            summary["nstep_sweep"][str(n)] = _summ(rows)
            s = summary["nstep_sweep"][str(n)]
            print(f"nstep={n:<2d}  success {s['successes']}/{s['n']}  grasped {s['grasped_any']}  "
                  f"lifted {s['lifted_any']}  mean_min_dist {s['mean_min_ee_cube_dist']}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {args.output} and per-rollout traces to {trace_dir}/")


if __name__ == "__main__":
    main()
