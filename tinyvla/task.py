"""SO-101 language-conditioned manipulation task + scripted expert.

Two cubes (red + blue) rest on a tabletop with two destinations: a bin (box) and
a plate. A natural-language command decides what to do. The supported command set
mirrors SmolVLA's SO-100 datasets (pick-place, stacking) plus a sorting-style
multi-step command:

  - "Pick up the {colour} cube and place it in the box."
  - "Put the {colour} cube on the plate."
  - "Put the {colour} cube on top of the {other} cube."      (stacking)
  - "Put the {c1} cube in the box and the {c2} cube on the plate."   (2-step sort)

A scripted state-machine expert executes each command: grasp -> lift -> carry ->
release, once per step. Cubes are real dynamic bodies (they fall under gravity and
settle at the destination) but are carried kinematically while grasped -- the
standard, reliable way to script manipulation demos. Cubes never collide with the
arm, so the approach can't knock them.

Because the target and destination are chosen independently of the layout, the
policy can only succeed by reading the instruction.
"""
from __future__ import annotations

import numpy as np
import mujoco

from .paths import ARTIFACTS_ROOT, SO101_TASK

COLORS = ["red", "blue"]

def other_color(color: str) -> str:
    return "blue" if color == "red" else "red"


# -- command set -------------------------------------------------------------
def _cmd(instruction, steps):
    return {"instruction": instruction, "steps": steps}

def _build_commands():
    cmds = []
    for c in COLORS:
        cmds.append(_cmd(f"Pick up the {c} cube and place it in the box.", [(c, "box")]))
    for c in COLORS:
        cmds.append(_cmd(f"Put the {c} cube on the plate.", [(c, "plate")]))
    for c in COLORS:
        cmds.append(_cmd(f"Put the {c} cube on top of the {other_color(c)} cube.", [(c, "stack")]))
    for c in COLORS:
        o = other_color(c)
        cmds.append(_cmd(f"Put the {c} cube in the box and the {o} cube on the plate.",
                         [(c, "box"), (o, "plate")]))
    return cmds

COMMANDS = _build_commands()

def instruction_for(color: str, goal: str = "box") -> str:
    """Back-compat helper for single-step commands."""
    if goal == "stack":
        return f"Put the {color} cube on top of the {other_color(color)} cube."
    if goal == "plate":
        return f"Put the {color} cube on the plate."
    return f"Pick up the {color} cube and place it in the box."

INSTRUCTION = COMMANDS[0]["instruction"]

# -- constants ---------------------------------------------------------------
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
HOME_QPOS = np.array([0.0, -1.2, 0.6, 1.2, 0.0, 1.2])   # gripper starts OPEN
EE_LOCAL = np.array([0.0045, 0.0001, -0.0382])          # fingertip / grasp centre

GRIP_OPEN = 1.2
GRIP_CLOSED = -0.17

CUBE_X = (0.18, 0.24)
CUBE_Y = (-0.085, 0.085)
CUBE_Z = 0.087          # table top (0.075) + half cube (0.012)
MIN_SEP = 0.06

BIN_XY = np.array([0.13, -0.075])
BIN_INNER = 0.036
PLATE_XY = np.array([0.13, 0.075])
PLATE_R = 0.045
PLATE_TOP = 0.081
SAFE_Z = 0.185
DROP_Z_BIN = 0.135


class SO101PickPlaceTask:
    def __init__(self, control_hz: float = 25.0, seed: int | None = None):
        self.model = mujoco.MjModel.from_xml_path(str(SO101_TASK))
        self.data = mujoco.MjData(self.model)
        self.control_hz = control_hz
        self.n_substeps = max(1, int(round((1.0 / control_hz) / self.model.opt.timestep)))
        self.rng = np.random.default_rng(seed)

        self.gripper_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        self.cube_bid = {c: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"cube_{c}")
                         for c in COLORS}
        self.cube_qadr = {c: self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"cube_{c}_free")] for c in COLORS}
        self.cube_dofadr = {c: self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"cube_{c}_free")] for c in COLORS}
        self.arm_qadr = np.array([
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)]
            for j in ARM_JOINTS])
        self.arm_dof = np.array([
            self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)]
            for j in ARM_JOINTS])
        self.ctrl_range = self.model.actuator_ctrlrange.copy()
        self.nu = self.model.nu

        self.steps = COMMANDS[0]["steps"]
        self.instruction = COMMANDS[0]["instruction"]
        self.step_idx = 0
        self.grasped = None

    # -- scene ------------------------------------------------------------
    def _sample_xy(self):
        return np.array([self.rng.uniform(*CUBE_X), self.rng.uniform(*CUBE_Y)])

    def _set_cube(self, color, xy, z=CUBE_Z, quat=(1, 0, 0, 0)):
        a = self.cube_qadr[color]
        self.data.qpos[a:a + 3] = [xy[0], xy[1], z]
        self.data.qpos[a + 3:a + 7] = quat
        self.data.qvel[self.cube_dofadr[color]:self.cube_dofadr[color] + 6] = 0

    def reset(self, command=None, positions=None):
        """command: index into COMMANDS, a command dict, or None (random)."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:6] = HOME_QPOS
        self.data.ctrl[:] = HOME_QPOS
        self.grasped = None
        self.phase = 0
        self.phase_t = 0
        self.step_idx = 0

        if command is None:
            command = COMMANDS[self.rng.integers(len(COMMANDS))]
        elif isinstance(command, int):
            command = COMMANDS[command]
        self.steps = command["steps"]
        self.instruction = command["instruction"]

        if positions is None:
            xy = {COLORS[0]: self._sample_xy()}
            while True:
                cand = self._sample_xy()
                if np.linalg.norm(cand - xy[COLORS[0]]) >= MIN_SEP:
                    xy[COLORS[1]] = cand
                    break
            positions = xy
        for c in COLORS:
            self._set_cube(c, positions[c])
        mujoco.mj_forward(self.model, self.data)
        return self.observation()

    # -- current step -----------------------------------------------------
    @property
    def target_color(self):
        return self.steps[min(self.step_idx, len(self.steps) - 1)][0]

    @property
    def target_dest(self):
        return self.steps[min(self.step_idx, len(self.steps) - 1)][1]

    def cube_pos(self, color=None):
        return self.data.xpos[self.cube_bid[color or self.target_color]].copy()

    def _dest_xy(self, dest, color):
        if dest == "stack":
            return self.cube_pos(other_color(color))[:2]
        if dest == "plate":
            return PLATE_XY
        return BIN_XY

    def _drop_z(self, dest, color):
        if dest == "stack":
            return self.cube_pos(other_color(color))[2] + 0.028
        if dest == "plate":
            return PLATE_TOP + 0.032
        return DROP_Z_BIN

    def _at_dest(self, color, dest):
        c = self.cube_pos(color)
        if dest == "stack":
            o = self.cube_pos(other_color(color))
            return np.linalg.norm(c[:2] - o[:2]) < 0.02 and c[2] > o[2] + 0.015
        if dest == "plate":
            return np.linalg.norm(c[:2] - PLATE_XY) < PLATE_R and c[2] < 0.11
        return np.linalg.norm(c[:2] - BIN_XY) < BIN_INNER and c[2] < 0.105

    def ee_pos(self):
        R = self.data.xmat[self.gripper_bid].reshape(3, 3)
        return self.data.xpos[self.gripper_bid] + R @ EE_LOCAL

    def observation(self):
        return {
            "state": self.data.qpos[:6].copy().astype(np.float32),
            "ee": self.ee_pos(),
            "cube": self.cube_pos(self.target_color),
            "instruction": self.instruction,
            "target_color": self.target_color,
            "target_dest": self.target_dest,
            "step_idx": self.step_idx,
            "grasped": self.grasped,
        }

    def success(self):
        return all(self._at_dest(c, d) for c, d in self.steps)

    # -- grasp (kinematic carry) ------------------------------------------
    def _grasp(self, color):
        R_g = self.data.xmat[self.gripper_bid].reshape(3, 3)
        p_g = self.data.xpos[self.gripper_bid]
        p_c = self.cube_pos(color)
        q_g = np.zeros(4); mujoco.mju_mat2Quat(q_g, self.data.xmat[self.gripper_bid])
        a = self.cube_qadr[color]
        q_c = self.data.qpos[a + 3:a + 7].copy()
        neg = np.zeros(4); mujoco.mju_negQuat(neg, q_g)
        self._off_quat = np.zeros(4); mujoco.mju_mulQuat(self._off_quat, neg, q_c)
        self._off_pos = R_g.T @ (p_c - p_g)
        self.grasped = color

    def _carry(self):
        c = self.grasped
        R_g = self.data.xmat[self.gripper_bid].reshape(3, 3)
        p_g = self.data.xpos[self.gripper_bid]
        q_g = np.zeros(4); mujoco.mju_mat2Quat(q_g, self.data.xmat[self.gripper_bid])
        a = self.cube_qadr[c]
        self.data.qpos[a:a + 3] = p_g + R_g @ self._off_pos
        q = np.zeros(4); mujoco.mju_mulQuat(q, q_g, self._off_quat)
        self.data.qpos[a + 3:a + 7] = q
        self.data.qvel[self.cube_dofadr[c]:self.cube_dofadr[c] + 6] = 0

    def step(self, action):
        action = np.clip(action, self.ctrl_range[:, 0], self.ctrl_range[:, 1])
        self.data.ctrl[:] = action
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
            if self.grasped is not None:
                self._carry(); mujoco.mj_forward(self.model, self.data)
        return self.observation()

    def render(self, cam="front", width=640, height=480):
        r = mujoco.Renderer(self.model, height=height, width=width)
        r.update_scene(self.data, camera=cam)
        img = r.render(); r.close()
        return img

    # -- IK ---------------------------------------------------------------
    def _ik_action(self, target, grip, gain=0.5, damping=0.08, max_dq=0.06):
        err = target - self.ee_pos()
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jacp, None, self.ee_pos(), self.gripper_bid)
        J = jacp[:, self.arm_dof]
        dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(3), err) * gain
        dq = np.clip(dq, -max_dq, max_dq)
        action = self.data.qpos[:6].copy()
        action[:5] = self.data.qpos[self.arm_qadr] + dq
        action[5] = grip
        return np.clip(action, self.ctrl_range[:, 0], self.ctrl_range[:, 1])

    # -- scripted state-machine expert (runs each step in sequence) -------
    def expert_action(self, gain=0.5, max_dq=0.06):
        color, dest = self.steps[self.step_idx]
        cube = self.cube_pos(color)
        above_cube = np.array([cube[0], cube[1], SAFE_Z])
        at_cube = cube.copy()
        dxy = self._dest_xy(dest, color)
        above_dest = np.array([dxy[0], dxy[1], SAFE_Z])
        drop = np.array([dxy[0], dxy[1], self._drop_z(dest, color)])
        ee = self.ee_pos()
        kw = dict(gain=gain, max_dq=max_dq)

        def near(a, b, tol):
            return np.linalg.norm(a - b) < tol

        self.phase_t += 1
        p = self.phase
        if p == 0:
            act = self._ik_action(above_cube, GRIP_OPEN, **kw)
            if near(ee, above_cube, 0.02):
                self.phase, self.phase_t = 1, 0
        elif p == 1:
            act = self._ik_action(at_cube, GRIP_OPEN, **kw)
            if near(ee, at_cube, 0.015):
                self.phase, self.phase_t = 2, 0
        elif p == 2:
            act = self._ik_action(at_cube, GRIP_CLOSED, **kw)
            if self.phase_t >= 5:
                if self.grasped is None:
                    self._grasp(color)
                if self.phase_t >= 8:
                    self.phase, self.phase_t = 3, 0
        elif p == 3:
            act = self._ik_action(above_cube, GRIP_CLOSED, **kw)
            if ee[2] > SAFE_Z - 0.02:
                self.phase, self.phase_t = 4, 0
        elif p == 4:
            act = self._ik_action(above_dest, GRIP_CLOSED, **kw)
            if near(ee[:2], above_dest[:2], 0.02):
                self.phase, self.phase_t = 5, 0
        elif p == 5:
            act = self._ik_action(drop, GRIP_CLOSED, **kw)
            if near(ee, drop, 0.02):
                self.phase, self.phase_t = 6, 0
        else:  # release, settle, then advance to the next step (if any)
            act = self._ik_action(drop, GRIP_OPEN, **kw)
            if self.grasped is not None and self.phase_t >= 3:
                self.grasped = None
            if self.phase_t >= 8 and self.step_idx < len(self.steps) - 1:
                self.step_idx += 1
                self.phase, self.phase_t = 0, 0
        return act


# backward-compatible alias
SO101ReachTask = SO101PickPlaceTask


if __name__ == "__main__":
    from PIL import Image
    env = SO101PickPlaceTask(seed=0)
    strips = []
    for ci in [0, 2, 4, 6]:
        env.reset(command=ci)
        frames = []
        for t in range(140):
            env.step(env.expert_action())
            if t % 20 == 0:
                frames.append(env.render())
        frames.append(env.render())
        print(f'"{env.instruction}"  success={env.success()}')
        strips.append(np.concatenate(frames, axis=1))
    out = ARTIFACTS_ROOT / "commands_strip.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(strips, axis=0)).save(out)
    print("saved", out)
