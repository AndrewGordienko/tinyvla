"""Targeted-data engine: an episode pool + collectors + a LeRobot-dataset builder.

Why a pool instead of appending to a LeRobot dataset directly? LeRobot's on-disk
append/resume path is fragile (streaming video encoders, finalize semantics). We
instead accumulate episodes as compressed .npz files in a *pool* directory, and
rebuild a fresh LeRobot dataset from the whole pool before each training round.
Simple, reproducible, and both the initial expert demos and later DAgger rounds
just drop more .npz files in.

Producers:
  - collect_expert_episodes : scripted state-machine expert demos for given
    commands (initial data + command-curriculum top-ups).
  - dagger_collect          : roll out the CURRENT policy, and at every state it
    drifts into, record the reactive (stateless) expert's action as the label —
    classic DAgger, which targets the compounding-error gap directly.

Consumer:
  - build_lerobot_dataset   : materialise the whole pool as an image (non-video)
    LeRobot dataset, optionally storing joint *deltas* (action - state).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import mujoco
import torch

from .task import SO101PickPlaceTask, COMMANDS, JOINT_NAMES

IMG = 256
CAMERA = "front"


# --------------------------------------------------------------------------- #
# pool                                                                         #
# --------------------------------------------------------------------------- #
def _next_index(pool: Path) -> int:
    pool.mkdir(parents=True, exist_ok=True)
    existing = sorted(pool.glob("ep_*.npz"))
    if not existing:
        return 0
    return int(existing[-1].stem.split("_")[1]) + 1


def save_episode_to_pool(pool: Path, states, actions, images, instruction: str,
                         command_index: int, source: str) -> Path:
    """Append one episode (T frames) to the pool as a compressed .npz."""
    pool = Path(pool)
    idx = _next_index(pool)
    path = pool / f"ep_{idx:06d}.npz"
    np.savez_compressed(
        path,
        state=np.asarray(states, dtype=np.float32),
        action=np.asarray(actions, dtype=np.float32),
        images=np.asarray(images, dtype=np.uint8),
        instruction=np.asarray(instruction),
        command_index=np.asarray(command_index, dtype=np.int64),
        source=np.asarray(source),
    )
    return path


def pool_episodes(pool: Path) -> list[Path]:
    return sorted(Path(pool).glob("ep_*.npz"))


def pool_summary(pool: Path) -> dict:
    counts: dict = {}
    per_cmd: dict = {}
    frames = 0
    for p in pool_episodes(pool):
        d = np.load(p, allow_pickle=False)
        src = str(d["source"])
        ci = int(d["command_index"])
        counts[src] = counts.get(src, 0) + 1
        per_cmd[ci] = per_cmd.get(ci, 0) + 1
        frames += int(d["action"].shape[0])
    return {"episodes": len(pool_episodes(pool)), "frames": frames,
            "by_source": counts, "by_command": per_cmd}


# --------------------------------------------------------------------------- #
# producers                                                                    #
# --------------------------------------------------------------------------- #
def _render(env, renderer):
    renderer.update_scene(env.data, camera=CAMERA)
    return renderer.render()


def collect_expert_episodes(pool: Path, commands, n_per_command: int, *, seed: int = 100,
                            source: str = "expert", cap: int = 220, dwell: int = 8,
                            gain: float = 0.25, max_dq: float = 0.03) -> int:
    """Scripted-expert demos for each command index in `commands`. Returns #episodes."""
    env = SO101PickPlaceTask(seed=seed)
    renderer = mujoco.Renderer(env.model, height=IMG, width=IMG)
    n = 0
    for ci in commands:
        for _ in range(n_per_command):
            env.reset(command=ci)
            states, actions, images = [], [], []
            hold = 0
            for _ in range(cap):
                states.append(env.data.qpos[:6].copy().astype(np.float32))
                images.append(_render(env, renderer))
                a = env.expert_action(gain=gain, max_dq=max_dq).astype(np.float32)
                actions.append(a)
                env.step(a)
                hold = hold + 1 if env.success() else 0
                if hold >= dwell:
                    break
            save_episode_to_pool(pool, states, actions, images,
                                 COMMANDS[ci]["instruction"], ci, source)
            n += 1
    renderer.close()
    return n


def dagger_collect(pool: Path, policy, preprocessor, postprocessor, commands, n_per_command: int,
                   *, device, cap: int = 200, seed: int = 500, delta_actions: bool = False,
                   gain: float = 0.25, max_dq: float = 0.03) -> int:
    """Roll out `policy`; label every visited state with the reactive expert (DAgger).

    The env is STEPPED with the policy's action (so we visit the policy's own state
    distribution), but the stored target is the reactive expert's action for that
    state. `delta_actions` only affects how closed-loop add-back is applied to the
    policy's action during rollout; stored labels are always absolute (the builder
    converts to deltas if requested).
    """
    from .eval_closedloop import build_obs
    was_training = policy.training
    policy.eval()
    env = SO101PickPlaceTask(seed=seed)
    renderer = mujoco.Renderer(env.model, height=IMG, width=IMG)
    n = 0
    try:
        for ci in commands:
            for k in range(n_per_command):
                env.reset(command=ci)
                policy.reset()
                torch.manual_seed(seed + 1000 * ci + k)
                states, actions, images = [], [], []
                for _ in range(cap):
                    state = env.data.qpos[:6].copy().astype(np.float32)
                    image = _render(env, renderer)
                    label = env.reactive_action(gain=gain, max_dq=max_dq).astype(np.float32)
                    states.append(state); images.append(image); actions.append(label)
                    # advance the sim with the POLICY's action (visit its state dist)
                    obs = preprocessor(build_obs(env, renderer, COMMANDS[ci]["instruction"], device, CAMERA))
                    with torch.inference_mode():
                        pa = policy.select_action(obs)
                    pa = postprocessor(pa).squeeze(0).cpu().numpy()
                    if delta_actions:
                        pa = pa + env.data.qpos[:6].astype(pa.dtype)
                    env.step(pa)
                save_episode_to_pool(pool, states, actions, images,
                                     COMMANDS[ci]["instruction"], ci, "dagger")
                n += 1
    finally:
        renderer.close()
        if was_training:
            policy.train()
    return n


# --------------------------------------------------------------------------- #
# consumer                                                                     #
# --------------------------------------------------------------------------- #
def build_lerobot_dataset(pool: Path, repo_id: str, root: Path, *,
                          delta_actions: bool = False, fps: int = 25) -> Path:
    """Materialise the whole pool as an image (non-video) LeRobot dataset.

    If `delta_actions`, the stored action is (action - state) so the model learns
    joint deltas; a delta_actions.json marker is written for the eval add-back.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    features = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
        "action": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
        f"observation.images.{CAMERA}": {"dtype": "image", "shape": (IMG, IMG, 3),
                                         "names": ["height", "width", "channels"]},
    }
    ds = LeRobotDataset.create(repo_id=repo_id, fps=fps, features=features, root=root,
                               robot_type="so101", use_videos=False)
    eps = pool_episodes(pool)
    if not eps:
        raise SystemExit(f"pool {pool} is empty")
    for p in eps:
        d = np.load(p, allow_pickle=False)
        state = d["state"]; action = d["action"]; images = d["images"]
        instruction = str(d["instruction"])
        stored_action = (action - state) if delta_actions else action
        for t in range(action.shape[0]):
            ds.add_frame({
                "observation.state": state[t],
                "action": stored_action[t].astype(np.float32),
                f"observation.images.{CAMERA}": images[t],
                "task": instruction,
            })
        ds.save_episode()
    if delta_actions:
        (root / "delta_actions.json").write_text('{"delta_actions": true}\n')
    (root / "pool_meta.json").write_text(json.dumps(
        {"pool": str(pool), "delta_actions": delta_actions, **pool_summary(pool)}, indent=2) + "\n")
    return root
