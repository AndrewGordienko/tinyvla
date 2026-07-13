"""Twin-environment, exact-state command-0 DAgger collector.

The learner is read-only while labels are generated in a second environment.
This is deliberately strict: a clone or replay mismatch aborts collection.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
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


def build_obs_from_frames(env, frames, instruction, device):
    """Build the exact policy input from the one render per camera."""
    if len(frames) != len(CAMERAS):
        raise ValueError("front and wrist frames are required")
    images = {}
    for camera, frame in zip(CAMERAS, frames):
        images[f"observation.images.{camera}"] = torch.from_numpy(frame).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    images["observation.state"] = torch.from_numpy(env.data.qpos[:6].copy().astype(np.float32)).unsqueeze(0).to(device)
    images["task"] = [instruction]
    return images


def cached_observation_hash(env, frames):
    return _hash(*frames, env.data.qpos[:6].copy())


def tensor_hash(value):
    if isinstance(value, torch.Tensor):
        return _hash(value.detach().cpu().numpy())
    if isinstance(value, dict):
        return _hash(*[key for key in sorted(value)], *[tensor_hash(value[key]) for key in sorted(value)])
    if isinstance(value, (list, tuple)):
        return _hash(*[tensor_hash(item) for item in value])
    return _hash(value)


def render_diagnostic(env, renderers):
    """Record EGL repeatability without making it an acceptance gate."""
    first = render(env, renderers); second = render(env, renderers)
    diffs = np.concatenate([np.abs(a.astype(np.int16) - b.astype(np.int16)).ravel() for a, b in zip(first, second)])
    changed = np.concatenate([(a != b).any(axis=2).ravel() for a, b in zip(first, second)])
    return {"max_difference": int(diffs.max(initial=0)), "mae": float(diffs.mean()),
            "p99_difference": float(np.percentile(diffs, 99)), "changed_pixel_fraction": float(changed.mean())}


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


def new_event_tracker():
    return {"ever_grasped": False, "ever_lifted": False, "released_after_grasp": False}


def update_event_tracker(env, events):
    events["ever_grasped"] = bool(events["ever_grasped"] or env.grasped is not None)
    events["ever_lifted"] = bool(events["ever_lifted"] or (events["ever_grasped"] and env.grasped is not None and float(env.ee_pos()[2]) > SAFE_Z - 0.02))
    events["released_after_grasp"] = bool(events["released_after_grasp"] or (events["ever_grasped"] and env.grasped is None))
    return events


def physical_stage(events, env):
    if events["released_after_grasp"]:
        return "release"
    if events["ever_lifted"] and env.grasped is not None:
        return "transport"
    if events["ever_grasped"] and env.grasped is not None:
        return "grasp"
    return "approach"


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


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path); tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as handle:
        json.dump(value, handle, indent=2); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_npz(path, **arrays):
    path = Path(path); tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays); handle.flush(); os.fsync(handle.fileno())
    os.replace(tmp, path)


def verify_shard(npz_path, json_path, expected_seed=None):
    metadata = json.loads(Path(json_path).read_text())
    if metadata["shard_sha256"] != file_sha256(npz_path):
        raise RuntimeError(f"shard hash mismatch: {npz_path}")
    with np.load(npz_path, allow_pickle=True) as archive:
        records = list(archive["records"])
    seed = int(metadata["scene_seed"])
    if expected_seed is not None and seed != expected_seed: raise RuntimeError("shard seed mismatch")
    if len(records) != 24 or metadata["records"] != 24: raise RuntimeError("shard record count mismatch")
    if [int(r["timestep"]) for r in records] != list(range(0, 120, 5)): raise RuntimeError("shard timestep mismatch")
    if any(int(r["scene_seed"]) != seed for r in records): raise RuntimeError("record seed mismatch")
    if any(r["learner_state_before"] != r["learner_state_after"] for r in records): raise RuntimeError("learner state mismatch")
    if any(not r["oracle_replay_identical"] for r in records): raise RuntimeError("oracle replay mismatch")
    return records, metadata


def code_commit():
    if os.environ.get("CODE_COMMIT"): return os.environ["CODE_COMMIT"]
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception: return "uncommitted"


def aggregate(out, seeds=range(2000, 2064)):
    out = Path(out); all_records = []
    for seed in seeds:
        records, _ = verify_shard(out / "shards" / f"seed_{seed}.npz", out / "shards" / f"seed_{seed}.json", seed)
        all_records.extend(records)
    if len(all_records) != len(list(seeds)) * 24: raise RuntimeError("aggregate record count mismatch")
    atomic_npz(out / "recovery_records.npz", records=np.asarray(all_records, dtype=object))
    manifest = {"records": len(all_records), "scenes": len(list(seeds)),
                "aggregate_sha256": file_sha256(out / "recovery_records.npz"),
                "source_shards": [f"seed_{seed}.npz" for seed in seeds]}
    atomic_json(out / "aggregate_manifest.json", manifest)
    print(json.dumps(manifest), flush=True)


def collect(args):
    meta = LeRobotDatasetMetadata("local/command0_multiview_32", root=args.dataset)
    runtime = load_runtime(args.teacher, meta=meta, dataset_root=args.dataset, device=args.device, stats_source="dataset")
    policy = runtime.policy; policy.eval()
    learner = SO101PickPlaceTask(seed=0); oracle = SO101PickPlaceTask(seed=0)
    renderers = {c: mujoco.Renderer(learner.model, height=IMG, width=IMG) for c in CAMERAS}
    out = Path(args.out); shards = out / "shards"; shards.mkdir(parents=True, exist_ok=True)
    learner.rng = np.random.default_rng(2000); learner.reset(command=0)
    pixel_diagnostic = render_diagnostic(learner, renderers)
    phase_tests = verify_phase_clones(learner, oracle)
    (out / "clone_smoke.json").write_text(json.dumps({"phase_tests": phase_tests,
                                                        "learner_untouched": True,
                                                        "oracle_replay_deterministic": True,
                                                        "egl_repeat_render_diagnostic": pixel_diagnostic}, indent=2) + "\n")
    seeds = [2000] if args.smoke else list(range(2000, 2064)); started = time.monotonic()
    completed = []; global_stages = {stage: 0 for stage in ("approach", "grasp", "transport", "release")}
    for seed in seeds:
        npz_path, json_path = shards / f"seed_{seed}.npz", shards / f"seed_{seed}.json"
        if npz_path.exists() and json_path.exists():
            _, metadata = verify_shard(npz_path, json_path, seed); completed.append(seed)
            for stage, count in metadata["stage_counts"].items(): global_stages[stage] += int(count)
            print(f"resume skip verified seed={seed} completed={len(completed)}/{len(seeds)}", flush=True)
    pending = [seed for seed in seeds if seed not in completed]
    if args.max_scenes: pending = pending[:args.max_scenes]
    for seed in pending:
        records = []
        learner.rng = np.random.default_rng(seed); learner.reset(command=0); policy.reset()
        events = new_event_tracker()
        for t in range(args.cap):
            images = render(learner, renderers); before = snapshot(learner); before_hash = state_hash(before)
            raw_observation_hash = cached_observation_hash(learner, images)
            obs = runtime.preprocessor(build_obs_from_frames(learner, images, COMMANDS[0]["instruction"], args.device))
            policy_observation_hash = tensor_hash(obs)
            if t % args.interval == 0:
                chunk_a = oracle_chunk(oracle, before)
                chunk_b = oracle_chunk(oracle, before)
                if not np.array_equal(chunk_a, chunk_b):
                    raise RuntimeError("independently restored oracle action chunks differ")
                after_hash = state_hash(snapshot(learner))
                if before_hash != after_hash:
                    raise RuntimeError("live learner changed during oracle query")
                rec = {"source": "dagger", "scene_seed": seed, "timestep": t, "stage": physical_stage(events, learner),
                       "teacher_sha": sha256_tree(args.teacher, patterns=("*.json", "*.safetensors")),
                       "observation_hash": raw_observation_hash, "raw_observation_hash": raw_observation_hash,
                       "policy_observation_hash": policy_observation_hash,
                       "action_chunk_hash": _hash(chunk_a), "front": images[0], "wrist": images[1],
                       "state": learner.data.qpos[:6].copy().astype(np.float32), "action_chunk": chunk_a,
                       "instruction": COMMANDS[0]["instruction"], "learner_state_before": before_hash,
                       "learner_state_after": after_hash, "oracle_replay_identical": True,
                       "snapshot_restore_verified": True, "event_tracker": copy.deepcopy(events)}
                records.append(rec)
                if args.smoke and len(records) >= args.smoke_states and args.smoke_short:
                    break
            with torch.inference_mode():
                action = runtime.postprocessor(policy.select_action(obs)).squeeze(0).cpu().numpy()
            learner.step(action)
            update_event_tracker(learner, events)
        if len(records) != 24: raise RuntimeError(f"seed {seed}: expected 24 records, got {len(records)}")
        stage_counts = {stage: sum(record["stage"] == stage for record in records) for stage in global_stages}
        atomic_npz(npz_path, records=np.asarray(records, dtype=object))
        shard_metadata = {"scene_seed": seed, "records": len(records), "timestamps": [r["timestep"] for r in records],
                          "stage_counts": stage_counts, "learner_action_count": args.cap,
                          "learner_hash_checks": all(r["learner_state_before"] == r["learner_state_after"] for r in records),
                          "oracle_replay_status": all(r["oracle_replay_identical"] for r in records),
                          "dataset_hash": sha256_tree(args.dataset),
                          "teacher_hash": sha256_tree(args.teacher, patterns=("*.json", "*.safetensors")),
                          "code_commit": code_commit(), "shard_sha256": file_sha256(npz_path)}
        atomic_json(json_path, shard_metadata); verify_shard(npz_path, json_path, seed)
        completed.append(seed)
        for stage, count in stage_counts.items(): global_stages[stage] += int(count)
        elapsed = time.monotonic() - started; rate = elapsed / max(1, len(completed)); eta = rate * (len(seeds) - len(completed))
        progress = {"status": "collecting" if len(completed) < len(seeds) else "complete",
                    "completed_scenes": len(completed), "total_scenes": len(seeds), "total_records": len(completed) * 24,
                    "stage_counts": global_stages, "elapsed_seconds": elapsed, "eta_seconds": eta,
                    "completed_seeds": sorted(completed), "last_seed": seed}
        atomic_json(out / "progress.json", progress)
        print(f"progress scenes={len(completed)}/{len(seeds)} records={len(completed)*24} stages={global_stages} elapsed={elapsed:.1f}s eta={eta:.1f}s", flush=True)
    if len(completed) == len(seeds): aggregate(out, seeds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher"); parser.add_argument("--dataset", default="data/datasets/command0_multiview_32")
    parser.add_argument("--out", required=True); parser.add_argument("--device", default="cuda")
    parser.add_argument("--cap", type=int, default=120); parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--smoke", action="store_true"); parser.add_argument("--smoke-states", type=int, default=4)
    parser.add_argument("--smoke-short", action="store_true")
    parser.add_argument("--max-scenes", type=int, default=0); parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate: aggregate(args.out)
    else:
        if not args.teacher: parser.error("--teacher is required for collection")
        collect(args)
