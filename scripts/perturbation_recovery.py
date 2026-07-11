"""Perturbation-recovery clincher for the covariate-shift hypothesis.

The Markovity controls showed the four-scene BC failure is off-trajectory
compounding, not hidden-state ambiguity. This confirms it causally: relabel
lightly-perturbed (off-manifold) states with the STATELESS reactive expert and
test whether adding that recovery coverage turns rollout from failing to passing.

  demos_only    : train on the four reactive trajectories only
  demos+recovery: add reactive-labelled perturbed states at 3 magnitudes

If demos+recovery passes while demos_only fails, classic covariate shift is
confirmed and the fix is recovery-state coverage (the four clean trajectories are
insufficient), not unfreezing vision or training SmolVLA longer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import mujoco

from tinyvla.task import SO101PickPlaceTask, COMMANDS, HOME_QPOS, GRASP_RADIUS
from tinyvla.determinism import seed_everything
from scripts.baseline_mlp import features, MLP, rollout, FEAT_DIM

EP_LEN, DWELL = 220, 8
MAGS = [0.02, 0.05, 0.10]     # arm-qpos perturbation std (rad)


def _snapshot(env):
    return {"qpos": env.data.qpos.copy(), "qvel": env.data.qvel.copy(),
            "grasped": env.grasped,
            "off_pos": getattr(env, "_off_pos", None),
            "off_quat": getattr(env, "_off_quat", None)}


def _restore(env, snap):
    env.data.qpos[:] = snap["qpos"]; env.data.qvel[:] = snap["qvel"]
    env.grasped = snap["grasped"]
    if snap["off_pos"] is not None:
        env._off_pos = snap["off_pos"]; env._off_quat = snap["off_quat"]
    mujoco.mj_forward(env.model, env.data)


def reactive_demos(scenes):
    """Reactive trajectories: (features, action) pairs + full state snapshots."""
    env = SO101PickPlaceTask()
    X, Y, snaps, cmds = [], [], [], []
    for scene in scenes:
        command = int(scene["command"])
        positions = {c: np.asarray(v, float) for c, v in scene["positions"].items()}
        env.reset(command=command, positions=positions)
        prev = HOME_QPOS.astype(np.float32)
        dwell = 0
        for _ in range(EP_LEN):
            a = env.reactive_action(gain=0.25, max_dq=0.03).astype(np.float32)
            X.append(features(env, prev)); Y.append(a)
            snaps.append(_snapshot(env)); cmds.append(command)
            env.step(a); prev = a
            dwell = dwell + 1 if env.success() else 0
            if dwell >= DWELL:
                break
    return np.asarray(X, np.float32), np.asarray(Y, np.float32), snaps, cmds


def recovery_set(snaps, cmds, mags, k_per, seed):
    """Perturb each snapshot, relabel with the reactive expert (stateless)."""
    rng = np.random.default_rng(seed)
    env = SO101PickPlaceTask()
    X, Y, MG = [], [], []
    for snap, command in zip(snaps, cmds):
        for mag in mags:
            for _ in range(k_per):
                _restore(env, snap)
                env.steps = COMMANDS[command]["steps"]; env.step_idx = 0
                env.data.qpos[:5] += rng.normal(0, mag, 5)          # arm only (gripper left
                # untouched so we never flip the grasp label at a recovery state)
                mujoco.mj_forward(env.model, env.data)
                if env.grasped is not None:
                    env._carry(); mujoco.mj_forward(env.model, env.data)  # keep grasp consistent
                prev = env.data.qpos[:6].astype(np.float32)
                a = env.reactive_action(gain=0.25, max_dq=0.03).astype(np.float32)
                X.append(features(env, prev)); Y.append(a); MG.append(mag)
    return np.asarray(X, np.float32), np.asarray(Y, np.float32), np.asarray(MG, np.float32)


def train(X, Y, steps, lr, seed):
    seed_everything(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    ymu, ysd = Y.mean(0), Y.std(0) + 1e-6
    xt = torch.from_numpy((X - mu) / sd).float(); yt = torch.from_numpy((Y - ymu) / ysd).float()
    model = MLP(d_in=FEAT_DIM); opt = torch.optim.Adam(model.parameters(), lr=lr); lossf = nn.MSELoss()
    bs = min(256, len(xt))
    for _ in range(steps):
        idx = torch.randint(0, len(xt), (bs,))
        opt.zero_grad(); loss = lossf(model(xt[idx]), yt[idx]); loss.backward(); opt.step()
    return model, (mu, sd, ymu, ysd), float(loss)


def _summ(rollouts):
    return {"successes": sum(r["success"] for r in rollouts), "n": len(rollouts),
            "grasp": sum(r["grasp_fired"] for r in rollouts),
            "lift": sum(r["lifted"] for r in rollouts),
            "carry": sum(r["carried_near_bin"] for r in rollouts),
            "place": sum(r["success"] for r in rollouts),
            "mean_min_dist": round(float(np.mean([r["min_ee_cube_dist"] for r in rollouts])), 4)}


def action_err_by_mag(model, norm, Xr, Yr, MG):
    mu, sd, ymu, ysd = norm
    with torch.inference_mode():
        pred = model(torch.from_numpy((Xr - mu) / sd).float()).numpy() * ysd + ymu
    err = np.abs(pred - Yr).mean(1)
    return {f"mag_{m}": round(float(err[MG == m].mean()), 4) for m in sorted(set(MG.tolist()))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="artifacts/truth_harness/datasets/command0_4")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cap", type=int, default=200)
    ap.add_argument("--k-per", type=int, default=3)
    ap.add_argument("--output", default="artifacts/truth_harness/perturbation_recovery.json")
    args = ap.parse_args()
    root = Path(args.root)
    scenes = json.loads((root / "scene_manifest.json").read_text())["scenes"]

    Xd, Yd, snaps, cmds = reactive_demos(scenes)
    Xr, Yr, MG = recovery_set(snaps, cmds, MAGS, args.k_per, args.seed)

    m_demo, n_demo, l_demo = train(Xd, Yd, args.steps, args.lr, args.seed)
    # balance demo:recovery ~1:1 by oversampling the on-manifold demonstrations so
    # the augmented model keeps its on-trajectory precision.
    reps = max(1, len(Xr) // max(1, len(Xd)))
    Xaug = np.concatenate([np.tile(Xd, (reps, 1)), Xr])
    Yaug = np.concatenate([np.tile(Yd, (reps, 1)), Yr])
    m_aug, n_aug, l_aug = train(Xaug, Yaug, args.steps, args.lr, args.seed)

    roll_demo = [rollout(m_demo, *n_demo, s, args.cap, GRASP_RADIUS) for s in scenes]
    roll_aug = [rollout(m_aug, *n_aug, s, args.cap, GRASP_RADIUS) for s in scenes]

    out = {
        "demo_pairs": int(len(Xd)), "recovery_pairs": int(len(Xr)), "magnitudes": MAGS,
        "demo_oversample_reps": int(reps),
        "demos_only": {"final_loss": round(l_demo, 6), **_summ(roll_demo),
                       "action_err_on_recovery_by_mag": action_err_by_mag(m_demo, n_demo, Xr, Yr, MG)},
        "demos_plus_recovery": {"final_loss": round(l_aug, 6), **_summ(roll_aug),
                                "action_err_on_recovery_by_mag": action_err_by_mag(m_aug, n_aug, Xr, Yr, MG)},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
