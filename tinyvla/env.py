"""Minimal MuJoCo environment for the SO-101 arm (the SO-ARM100/101 used by
LeRobot / SmolVLA-450M).

The arm has 6 position-controlled actuators matching the SmolVLA action space:
    shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper

Usage:
    from tinyvla import SO101Env
    env = SO101Env()
    obs = env.reset()
    obs, reward, done, info = env.step(env.action_space_sample())

CLI:
    python3 -m tinyvla.env --render      # save a frame to artifacts/so101_frame.png
    mjpython -m tinyvla.env --viewer     # interactive viewer (macOS needs mjpython)
"""
from __future__ import annotations

import numpy as np
import mujoco

from .paths import ARTIFACTS_ROOT, SO101_SCENE

JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]


class SO101Env:
    """A tiny Gym-style wrapper around the SO-101 MuJoCo model."""

    def __init__(self, scene_path: str = str(SO101_SCENE), control_hz: float = 50.0):
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.control_hz = control_hz
        # how many physics substeps per env.step()
        self.n_substeps = max(1, int(round((1.0 / control_hz) / self.model.opt.timestep)))
        # actuator control ranges -> action bounds
        self.ctrl_range = self.model.actuator_ctrlrange.copy()  # (nu, 2)
        self.nu = self.model.nu
        self.reset()

    # -- core API ---------------------------------------------------------
    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self._obs()

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=float),
                         self.ctrl_range[:, 0], self.ctrl_range[:, 1])
        self.data.ctrl[:] = action
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        return self._obs(), 0.0, False, {}

    def _obs(self) -> np.ndarray:
        # joint positions + velocities, matching the SmolVLA proprio state
        return np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()])

    # -- helpers ----------------------------------------------------------
    def action_space_sample(self) -> np.ndarray:
        lo, hi = self.ctrl_range[:, 0], self.ctrl_range[:, 1]
        return np.random.uniform(lo, hi)

    def render(self, width: int = 640, height: int = 480,
               azimuth: float = 160, elevation: float = -20,
               distance: float = 0.9, lookat=(0, 0, 0.1)) -> np.ndarray:
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance
        cam.lookat[:] = lookat
        renderer.update_scene(self.data, cam)
        img = renderer.render()
        renderer.close()
        return img


def _run_viewer():
    import mujoco.viewer
    env = SO101Env()
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(env.model, env.data)
            viewer.sync()


def _run_render(out=str(ARTIFACTS_ROOT / "so101_frame.png")):
    from PIL import Image
    from pathlib import Path

    env = SO101Env()
    env.data.qpos[:] = np.array([0.2, -0.4, 0.6, 0.3, 0.0, 0.5])
    mujoco.mj_forward(env.model, env.data)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(env.render()).save(out)
    print(f"saved {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--viewer", action="store_true", help="interactive viewer (use mjpython on macOS)")
    p.add_argument("--render", action="store_true", help="save a still frame")
    args = p.parse_args()
    if args.viewer:
        _run_viewer()
    else:
        _run_render()
