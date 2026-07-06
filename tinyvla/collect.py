"""Record scripted-expert reach rollouts into a LeRobot dataset.

Each timestep stores exactly what SmolVLA consumes:
  - observation.state          : 6 joint angles (radians)
  - observation.images.front   : 256x256 RGB from the fixed camera
  - action                     : 6 joint targets the expert commanded
  - task                       : the language instruction

Convention: frame t holds the observation at t and the action applied at t
(standard behaviour-cloning alignment).

Run:  python3 -m tinyvla.collect --episodes 60
"""
from __future__ import annotations

import argparse
import os
import shutil
import numpy as np
import mujoco

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from .paths import DATASETS_ROOT
from .task import SO101ReachTask, JOINT_NAMES, COMMANDS

IMG = 256                    # square camera resolution recorded to the dataset
EP_LEN = 220                 # max steps per episode (variable-length; 2-step sort ~150)
DWELL = 8                    # extra frames recorded after the command succeeds
EXPERT = dict(gain=0.25, max_dq=0.03)   # natural pace: ~70-140 frame episodes, full chunk coverage

CAMERAS = ["front"]              # front-only (wrist cam removed: it hurt closed-loop)

FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
    "action": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
    **{f"observation.images.{cam}": {"dtype": "video", "shape": (IMG, IMG, 3),
                                     "names": ["height", "width", "channels"]}
       for cam in CAMERAS},
}


def render_cam(env, renderer, cam):
    renderer.update_scene(env.data, camera=cam)
    return renderer.render()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--repo-id", default="local/so101_pickplace")
    ap.add_argument("--root", default=str(DATASETS_ROOT / "so101_pickplace"))
    ap.add_argument("--seed", type=int, default=100)
    args = ap.parse_args()

    if os.path.exists(args.root):
        shutil.rmtree(args.root)

    env = SO101ReachTask(seed=args.seed)
    renderer = mujoco.Renderer(env.model, height=IMG, width=IMG)

    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=int(env.control_hz),
        features=FEATURES,
        root=args.root,
        robot_type="so101",
        use_videos=True,
    )

    n_success = 0
    for ep in range(args.episodes):
        env.reset(command=ep % len(COMMANDS))        # round-robin over all commands
        dwell = 0
        for t in range(EP_LEN):
            state = env.data.qpos[:6].copy().astype(np.float32)
            images = {f"observation.images.{cam}": render_cam(env, renderer, cam) for cam in CAMERAS}
            action = env.expert_action(**EXPERT).astype(np.float32)
            ds.add_frame({
                "observation.state": state,
                **images,
                "action": action,
                "task": env.instruction,             # one of the 8 supported commands
            })
            env.step(action)
            dwell = dwell + 1 if env.success() else 0
            if dwell >= DWELL:                        # stop shortly after success
                break
        n_success += env.success()
        ds.save_episode()
        if (ep + 1) % 20 == 0:
            print(f"  episode {ep + 1}/{args.episodes}  (running success {n_success}/{ep + 1})")

    print(f"\nDone. {args.episodes} episodes, expert success {n_success}/{args.episodes}, "
          f"{ds.num_frames} frames")
    print(f"Dataset written to {args.root}  ({args.episodes * EP_LEN} frames)")


if __name__ == "__main__":
    main()
