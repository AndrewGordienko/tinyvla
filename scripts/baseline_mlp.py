"""Privileged-state MLP behavioural-cloning control for the four command-0 scenes.

A control, NOT a SmolVLA replacement: it shows whether the task + dynamics are
learnable from a clean privileged state (cube/ee/dest positions + robot state).
One-step closed-loop prediction (no action chunks). If this cannot reach 4/4 at
the canonical 4 cm radius, the problem is the task / dataset / controller, not
SmolVLA. Reports approach / grasp / lift / carry / place separately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from tinyvla.task import SO101PickPlaceTask, COMMANDS, HOME_QPOS, CUBE_Z, GRASP_RADIUS, BIN_XY
from tinyvla.runtime import detect_action_semantics
from tinyvla.determinism import seed_everything

FEAT_DIM = 28  # qpos6 + qvel6 + ee3 + cube3 + dest3 + grasped1 + prev_action6
LIFT_Z = CUBE_Z + 0.03


def _target_color(command: int) -> str:
    return COMMANDS[command]["steps"][0][0]


def features(env, prev_action: np.ndarray) -> np.ndarray:
    color = env.target_color
    dxy = env._dest_xy(env.target_dest, color)
    drop_z = env._drop_z(env.target_dest, color)
    return np.concatenate([
        env.data.qpos[:6], env.data.qvel[:6], env.ee_pos(), env.cube_pos(color),
        [dxy[0], dxy[1], drop_z], [1.0 if env.grasped is not None else 0.0], prev_action,
    ]).astype(np.float32)


def _episode_actions(ds: LeRobotDataset) -> dict[int, np.ndarray]:
    hf = ds.hf_dataset.with_format("numpy")
    a, e, f = np.asarray(hf["action"]), np.asarray(hf["episode_index"]).astype(int), np.asarray(hf["frame_index"]).astype(int)
    return {int(ep): a[e == ep][np.argsort(f[e == ep])] for ep in sorted(set(e.tolist()))}


class MLP(nn.Module):
    def __init__(self, d_in=FEAT_DIM, d_out=6, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


def build_training_data(repo_id, root, scenes, delta):
    """Replay stored demonstrations to reconstruct (privileged feature, action) pairs."""
    ds = LeRobotDataset(repo_id, root=str(root))
    ep_actions = _episode_actions(ds)
    env = SO101PickPlaceTask()
    X, Y = [], []
    for scene in scenes:
        ep, command = int(scene["episode"]), int(scene["command"])
        positions = {c: np.asarray(v, float) for c, v in scene["positions"].items()}
        env.reset(command=command, positions=positions)
        prev = HOME_QPOS.astype(np.float32)
        for a in ep_actions[ep]:
            a = np.asarray(a, dtype=np.float32)
            step_action = a + env.data.qpos[:6].astype(np.float32) if delta else a
            X.append(features(env, prev))
            Y.append(step_action.astype(np.float32))
            env.step(step_action)
            prev = step_action
    return np.asarray(X), np.asarray(Y)


def rollout(model, mu_x, sd_x, mu_y, sd_y, scene, cap, radius):
    command = int(scene["command"])
    color = _target_color(command)
    positions = {c: np.asarray(v, float) for c, v in scene["positions"].items()}
    env = SO101PickPlaceTask()
    lo, hi = env.ctrl_range[:, 0], env.ctrl_range[:, 1]
    env.reset(command=command, positions=positions)
    prev = HOME_QPOS.astype(np.float32)
    dmin, grasp_t, lifted, carried = float("inf"), -1, False, False
    for t in range(cap):
        x = (features(env, prev) - mu_x) / sd_x
        with torch.inference_mode():
            y = model(torch.from_numpy(x).float().unsqueeze(0)).squeeze(0).numpy()
        action = np.clip(y * sd_y + mu_y, lo, hi).astype(np.float32)
        dmin = min(dmin, float(np.linalg.norm(env.ee_pos() - env.cube_pos(color))))
        env.step(action)
        prev = action
        if grasp_t < 0 and env.grasped is not None:
            grasp_t = t
        if env.cube_pos(color)[2] > LIFT_Z:
            lifted = True
        if np.linalg.norm(env.cube_pos(color)[:2] - BIN_XY) < 0.05:
            carried = True
    return {"command": command, "success": bool(env.success()), "min_ee_cube_dist": round(dmin, 4),
            "approach_within_4cm": bool(dmin < radius), "grasp_fired": grasp_t >= 0,
            "grasp_t": grasp_t, "lifted": lifted, "carried_near_bin": carried}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="local/truth_gate_command0_4")
    ap.add_argument("--root", default="artifacts/truth_harness/datasets/command0_4")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cap", type=int, default=200)
    ap.add_argument("--output", default="artifacts/truth_harness/baseline_mlp.json")
    args = ap.parse_args()
    seed_everything(args.seed)
    root = Path(args.root)
    scenes = json.loads((root / "scene_manifest.json").read_text())["scenes"]
    delta = detect_action_semantics(root) == "delta"

    X, Y = build_training_data(args.repo_id, root, scenes, delta)
    mu_x, sd_x = X.mean(0), X.std(0) + 1e-6
    mu_y, sd_y = Y.mean(0), Y.std(0) + 1e-6
    Xn = torch.from_numpy((X - mu_x) / sd_x).float()
    Yn = torch.from_numpy((Y - mu_y) / sd_y).float()

    model = MLP()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.MSELoss()
    curve = []
    for step in range(1, args.steps + 1):
        opt.zero_grad()
        loss = lossf(model(Xn), Yn)
        loss.backward(); opt.step()
        if step % 500 == 0 or step == 1:
            curve.append({"step": step, "loss": round(float(loss), 6)})

    rollouts = [rollout(model, mu_x, sd_x, mu_y, sd_y, s, args.cap, GRASP_RADIUS) for s in scenes]
    n = len(rollouts)
    result = {
        "baseline": "privileged_state_mlp", "params": sum(p.numel() for p in model.parameters()),
        "feat_dim": FEAT_DIM, "train_pairs": int(len(X)), "steps": args.steps,
        "final_loss": curve[-1]["loss"], "train_curve": curve,
        "grasp_radius": GRASP_RADIUS,
        "successes": sum(r["success"] for r in rollouts), "n": n,
        "approach": sum(r["approach_within_4cm"] for r in rollouts),
        "grasp": sum(r["grasp_fired"] for r in rollouts),
        "lift": sum(r["lifted"] for r in rollouts),
        "carry": sum(r["carried_near_bin"] for r in rollouts),
        "place": sum(r["success"] for r in rollouts),
        "rollouts": rollouts,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in
                      ("baseline", "params", "train_pairs", "final_loss",
                       "successes", "n", "approach", "grasp", "lift", "carry", "place")}, indent=2))
    for r in rollouts:
        print(" ", r)


if __name__ == "__main__":
    main()
