"""Twin-environment, exact-state command-0 DAgger collector.

The learner is read-only while labels are generated in a second environment.
This is deliberately strict: a clone or replay mismatch aborts collection.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from tinyvla.eval_closedloop import build_obs
from tinyvla.runtime import load_runtime, sha256_tree
from tinyvla.task import COMMANDS, SO101PickPlaceTask, SAFE_Z

IMG = 256
CHUNK = 10
CAMERAS = ("front", "wrist")


def _hash(*values) -> str:
    d = hashlib.sha256()
    for value in values:
        if isinstance(value, np.ndarray):
            d.update(str(value.dtype).encode()); d.update(str(value.shape).encode()); d.update(value.tobytes())
        elif isinstance(value, (bytes, bytearray)):
            d.update(value)
        else:
            d.update(repr(value).encode())
    return d.hexdigest()


def _rng_state(env):
    return copy.deepcopy(env.rng.bit_generator.state)


def snapshot(env):
    """Capture full physics plus controls/forces and Python task state."""
    full = np.zeros(mujoco.mj_stateSize(env.model, mujoco.mjtState.mjSTATE_FULLPHYSICS), dtype=np.float64)
    mujoco.mj_getState(env.model, env.data, full, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    return {
        "fullphysics": full,
        "ctrl": env.data.ctrl.copy(), "qacc": env.data.qacc.copy(),
        "qacc_warmstart": env.data.qacc_warmstart.copy(),
        "qfrc_applied": env.data.qfrc_applied.copy(), "xfrc_applied": env.data.xfrc_applied.copy(),
        "mocap_pos": env.data.mocap_pos.copy(), "mocap_quat": env.data.mocap_quat.copy(),
        "userdata": env.data.userdata.copy(),
        "grasped": copy.deepcopy(env.grasped), "_off_pos": copy.deepcopy(getattr(env, "_off_pos", None)),
        "_off_quat": copy.deepcopy(getattr(env, "_off_quat", None)), "phase": env.phase,
        "phase_t": env.phase_t, "step_idx": env.step_idx, "steps": copy.deepcopy(env.steps),
        "instruction": env.instruction, "rng": _rng_state(env),
    }


def restore(env, state):
    mujoco.mj_setState(env.model, env.data, state["fullphysics"], mujoco.mjtState.mjSTATE_FULLPHYSICS)
    env.data.ctrl[:] = state["ctrl"]; env.data.qacc[:] = state["qacc"]
    env.data.qacc_warmstart[:] = state["qacc_warmstart"]
    env.data.qfrc_applied[:] = state["qfrc_applied"]; env.data.xfrc_applied[:] = state["xfrc_applied"]
    env.data.mocap_pos[:] = state["mocap_pos"]; env.data.mocap_quat[:] = state["mocap_quat"]
    env.data.userdata[:] = state["userdata"]
    env.grasped = copy.deepcopy(state["grasped"]); env.phase = state["phase"]; env.phase_t = state["phase_t"]
    env.step_idx = state["step_idx"]; env.steps = copy.deepcopy(state["steps"]); env.instruction = state["instruction"]
    if state["_off_pos"] is None:
        if hasattr(env, "_off_pos"): del env._off_pos
        if hasattr(env, "_off_quat"): del env._off_quat
    else:
        env._off_pos = state["_off_pos"].copy(); env._off_quat = state["_off_quat"].copy()
    env.rng.bit_generator.state = copy.deepcopy(state["rng"])
    mujoco.mj_forward(env.model, env.data)


def state_hash(state):
    return _hash(state["fullphysics"], state["ctrl"], state["qacc"], state["qacc_warmstart"],
                 state["qfrc_applied"], state["xfrc_applied"], state["mocap_pos"], state["mocap_quat"],
                 state["userdata"], state["grasped"], state["_off_pos"], state["_off_quat"],
                 state["phase"], state["phase_t"], state["step_idx"], state["steps"], state["instruction"],
                 json.dumps(state["rng"], sort_keys=True))


def clone(src, dst):
    state = snapshot(src); restore(dst, state)
    if state_hash(state) != state_hash(snapshot(dst)):
        raise RuntimeError("oracle clone physical/task-state mismatch")
    return state


def render(env, renderers):
    images = []
    for camera in CAMERAS:
        renderers[camera].update_scene(env.data, camera=camera)
        images.append(renderers[camera].render().copy())
    return images


def oracle_chunk(oracle, state):
    restore(oracle, state)
    actions = []
    for _ in range(CHUNK):
        action = oracle.expert_action(gain=.25, max_dq=.03).astype(np.float32)
        actions.append(action.copy()); oracle.step(action)
    return np.asarray(actions)


def phase_name(env):
    if env.grasped is None and env.phase <= 1: return "approach"
    if env.grasped is not None and env.data.qpos[5] < .6 and env.ee_pos()[2] <= SAFE_Z: return "grasp"
    if env.grasped is not None and env.ee_pos()[2] > SAFE_Z: return "transport"
    if env.grasped is None and env.phase >= 3: return "release"
    return str(env.phase)


def verify_phase_clones(learner, oracle):
    """Exercise clone/replay at semantic boundaries before records are accepted."""
    learner.rng = np.random.default_rng(2000); learner.reset(command=0)
    candidates = {"approach": snapshot(learner)}
    for _ in range(500):
        learner.step(learner.expert_action(gain=.5, max_dq=.06))
        if "grasp" not in candidates and learner.grasped is not None:
            candidates["grasp"] = snapshot(learner)
        if "transport" not in candidates and learner.grasped is not None and learner.ee_pos()[2] > SAFE_Z - 0.02:
            candidates["transport"] = snapshot(learner)
        if "release_before" not in candidates and learner.phase >= 5 and learner.grasped is not None:
            candidates["release_before"] = snapshot(learner)
        if "release" not in candidates and learner.phase >= 6 and learner.grasped is None:
            candidates["release"] = snapshot(learner); break
    required = ("approach", "grasp", "transport", "release_before", "release")
    missing = [name for name in required if name not in candidates]
    if missing:
        raise RuntimeError(f"phase clone smoke missing states: {missing}")
    learner_final_hash = state_hash(snapshot(learner))
    for name in required:
        state = candidates[name]
        first = oracle_chunk(oracle, state); second = oracle_chunk(oracle, state)
        if not np.array_equal(first, second):
            raise RuntimeError(f"{name}: independently restored oracle chunks differ")
        if learner_final_hash != state_hash(snapshot(learner)):
            raise RuntimeError(f"{name}: learner changed during oracle query")
    return {name: True for name in required}


def collect(args):
    meta = LeRobotDatasetMetadata("local/command0_multiview_32", root=args.dataset)
    runtime = load_runtime(args.teacher, meta=meta, dataset_root=args.dataset, device=args.device, stats_source="dataset")
    policy = runtime.policy; policy.eval()
    learner = SO101PickPlaceTask(seed=0); oracle = SO101PickPlaceTask(seed=0)
    renderers = {c: mujoco.Renderer(learner.model, height=IMG, width=IMG) for c in CAMERAS}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    phase_tests = verify_phase_clones(learner, oracle)
    (out / "clone_smoke.json").write_text(json.dumps({"phase_tests": phase_tests,
                                                        "learner_untouched": True,
                                                        "oracle_replay_deterministic": True}, indent=2) + "\n")
    records = []; seeds = [2000] if args.smoke else list(range(2000, 2064))
    for seed in seeds:
        learner.rng = np.random.default_rng(seed); learner.reset(command=0); policy.reset()
        for t in range(0, args.cap, args.interval):
            images = render(learner, renderers); before = snapshot(learner); before_hash = state_hash(before)
            obs = runtime.preprocessor(build_obs(learner, renderers, COMMANDS[0]["instruction"], args.device))
            chunk_a = oracle_chunk(oracle, before)
            chunk_b = oracle_chunk(oracle, before)
            if not np.array_equal(chunk_a, chunk_b):
                raise RuntimeError("independently restored oracle action chunks differ")
            after_hash = state_hash(snapshot(learner))
            if before_hash != after_hash:
                raise RuntimeError("live learner changed during oracle query")
            if any(not np.array_equal(a, b) for a, b in zip(images, render(learner, renderers))):
                raise RuntimeError("live learner rendered observation changed during oracle query")
            rec = {"source": "dagger", "scene_seed": seed, "timestep": t, "stage": phase_name(learner),
                   "teacher_sha": sha256_tree(args.teacher, patterns=("*.json", "*.safetensors")),
                   "observation_hash": _hash(*images, learner.data.qpos[:6].copy()),
                   "action_chunk_hash": _hash(chunk_a), "front": images[0], "wrist": images[1],
                   "state": learner.data.qpos[:6].copy().astype(np.float32), "action_chunk": chunk_a,
                   "instruction": COMMANDS[0]["instruction"], "learner_state_before": before_hash,
                   "learner_state_after": after_hash, "oracle_replay_identical": True,
                   "snapshot_restore_verified": True}
            records.append(rec)
            if args.smoke and len(records) >= args.smoke_states: break
            with torch.inference_mode():
                action = runtime.postprocessor(policy.select_action(obs)).squeeze(0).cpu().numpy()
            learner.step(action)
        if args.smoke and len(records) >= args.smoke_states: break
    if args.smoke and len(records) < args.smoke_states:
        raise RuntimeError(f"smoke collected only {len(records)} states")
    np.savez_compressed(out / "recovery_records.npz", records=np.asarray(records, dtype=object))
    manifest = {"source": "dagger", "records": len(records), "cameras": CAMERAS, "chunk_length": CHUNK,
                "dataset_hash": sha256_tree(args.dataset), "teacher_hash": sha256_tree(args.teacher, patterns=("*.json", "*.safetensors")),
                "privileged_policy_inputs": False, "sealed_heldout_excluded": True,
                "twin_environment": True, "learner_untouched": True, "oracle_state_api": "mjSTATE_FULLPHYSICS",
                "phase_clone_tests": phase_tests}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"records": len(records), "out": str(out), "manifest": manifest}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True); parser.add_argument("--dataset", default="data/datasets/command0_multiview_32")
    parser.add_argument("--out", required=True); parser.add_argument("--device", default="cuda")
    parser.add_argument("--cap", type=int, default=120); parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--smoke", action="store_true"); parser.add_argument("--smoke-states", type=int, default=4)
    collect(parser.parse_args())
