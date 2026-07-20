"""Diagnose the exact release/place failure of the deployed student.

Runs the deployed student closed-loop on the scoped commands and, per episode,
records the mechanism-level facts that determine release success:
  - did it grasp + carry?               (env.grasped)
  - closest xy the cube got to the dest  (positioning)
  - gripper joint qpos[5] over time      (does it ever open past GRIP_RELEASE?)
  - the gripper value AT closest approach (did it try to release over the dest?)

This separates a POSITIONING failure (never gets the cube over the dest) from a
GRIPPER-RELEASE failure (gets there but never opens the gripper).
"""
from __future__ import annotations
import argparse, json
import numpy as np, torch, mujoco
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from tinyvla.task import SO101PickPlaceTask, COMMANDS, GRIP_RELEASE, GRIP_OPEN
from tinyvla.eval_closedloop import build_obs
from tinyvla.runtime import load_runtime
from tinyvla.paths import CHECKPOINTS_ROOT, DATASETS_ROOT

SCOPED = [1, 3, 4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(CHECKPOINTS_ROOT / "student291_recover_brain_v1" / "best_closed_loop"))
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--cap", type=int, default=240)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    device = torch.device(args.device)
    meta = LeRobotDatasetMetadata("local/so101_pickplace", root=str(DATASETS_ROOT / "so101_pickplace"))
    rt = load_runtime(args.model, meta=meta, dataset_root=str(DATASETS_ROOT / "so101_pickplace"),
                      device=device, stats_source="checkpoint")
    policy = rt.policy.eval()
    env = SO101PickPlaceTask(seed=7)
    renderer = mujoco.Renderer(env.model, height=256, width=256)
    print(f"GRIP_RELEASE={GRIP_RELEASE}  GRIP_OPEN={GRIP_OPEN}\n")

    summary = {"grasped": 0, "positioned": 0, "opened_ever": 0, "opened_over_dest": 0,
               "success": 0, "n": 0}
    for ci in SCOPED:
        for ep in range(args.episodes):
            env.rng = np.random.default_rng(999 + 1009 * ep + ci)
            env.reset(command=ci); policy.reset()
            color, dest = env.steps[0]
            dxy = env._dest_xy(dest, color)
            min_xy = float("inf"); max_grip = -9; grip_at_closest = None; ever_grasped = False
            for _ in range(args.cap):
                obs = rt.preprocessor(build_obs(env, renderer, COMMANDS[ci]["instruction"], device, "front"))
                with torch.inference_mode():
                    a = rt.postprocessor(policy.select_action(obs)).squeeze(0).cpu().numpy()
                env.step(a)
                ever_grasped |= env.grasped is not None
                grip = float(env.data.qpos[5])
                max_grip = max(max_grip, grip)
                cube_xy = env.cube_pos(color)[:2]
                d = float(np.linalg.norm(cube_xy - dxy[:2]))
                if d < min_xy:
                    min_xy = d; grip_at_closest = grip
                if env.success():
                    break
            ok = bool(env.success())
            positioned = min_xy <= 0.03          # cube got within 3cm xy of dest
            opened = max_grip > GRIP_RELEASE
            opened_over = positioned and grip_at_closest is not None and grip_at_closest > GRIP_RELEASE
            summary["n"] += 1
            summary["grasped"] += ever_grasped
            summary["positioned"] += positioned
            summary["opened_ever"] += opened
            summary["opened_over_dest"] += opened_over
            summary["success"] += ok
            print(f"cmd{ci} ep{ep}: success={ok!s:5} grasped={ever_grasped!s:5} "
                  f"min_xy_to_dest={min_xy:.3f} positioned={positioned!s:5} "
                  f"max_grip={max_grip:.2f} opened={opened!s:5} grip@closest={grip_at_closest:.2f}")
    n = summary["n"]
    print(f"\n=== {n} episodes")
    for k in ("grasped", "positioned", "opened_ever", "opened_over_dest", "success"):
        print(f"  {k:18s}: {summary[k]}/{n}  ({summary[k]/n*100:.0f}%)")
    print("\nInterpretation:")
    if summary["positioned"] and not summary["opened_over_dest"]:
        print("  -> GRIPPER-RELEASE failure: cube reaches the dest but the gripper rarely opens over it.")
    elif not summary["positioned"]:
        print("  -> POSITIONING failure: cube rarely gets within 3cm xy of the dest.")


if __name__ == "__main__":
    main()
