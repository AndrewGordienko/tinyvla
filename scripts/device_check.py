"""CPU vs MPS determinism check on the same checkpoint, scenes, and seeds.

The four-scene task sits right at the 4 cm grasp threshold, so small numerical
divergence between accelerators can flip borderline scenes (cf. the historical
CUDA 3/6 vs MPS 0/6 report). This measures open-loop action-chunk divergence on
identical seeded inputs for the memorized scenes and reports per-dimension and
gripper differences.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import mujoco

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from tinyvla.task import SO101PickPlaceTask
from tinyvla.eval_closedloop import build_obs, IMG
from tinyvla.runtime import load_runtime

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def _predict(model, root, meta, device, obs_raw_list, seed):
    dev = torch.device(device)
    r = load_runtime(model, meta=meta, dataset_root=str(root), device=dev, stats_source="checkpoint")
    pol, pre, post = r.policy.eval(), r.preprocessor, r.postprocessor
    cs, md = pol.config.chunk_size, pol.config.max_action_dim
    outs = []
    with torch.inference_mode():
        for i, obs in enumerate(obs_raw_list):
            gen = torch.Generator(device="cpu").manual_seed(seed + i)
            noise = torch.randn((1, cs, md), generator=gen).to(dev)
            pred = post(pol.predict_action_chunk(pre(dict(obs)), noise=noise))
            outs.append(pred.squeeze(0).cpu().numpy()[:, :6])
    del r
    return outs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/truth_harness/checkpoints/command0_overfit_500")
    ap.add_argument("--repo-id", default="local/truth_gate_command0_4")
    ap.add_argument("--root", default="artifacts/truth_harness/datasets/command0_4")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--output", default="artifacts/truth_harness/device_check.json")
    args = ap.parse_args()
    root = Path(args.root)
    meta = LeRobotDatasetMetadata(args.repo_id, root=str(root))
    scenes = json.loads((root / "scene_manifest.json").read_text())["scenes"]

    # build identical raw observations (same rendered image + state) for each scene start
    env = SO101PickPlaceTask()
    renderer = mujoco.Renderer(env.model, height=IMG, width=IMG)
    obs_list = []
    for s in scenes:
        pos = {c: np.asarray(v, float) for c, v in s["positions"].items()}
        env.reset(command=int(s["command"]), positions=pos)
        obs_list.append(build_obs(env, renderer, env.instruction, torch.device("cpu")))
    renderer.close()

    cpu = _predict(args.model, root, meta, "cpu", obs_list, args.seed)
    mps = _predict(args.model, root, meta, "mps", obs_list, args.seed)

    per_scene = []
    for i, (c, m) in enumerate(zip(cpu, mps)):
        diff = np.abs(c - m)
        per_scene.append({
            "scene": i,
            "max_abs_diff": round(float(diff.max()), 6),
            "mean_abs_diff": round(float(diff.mean()), 6),
            "per_dim_max": {JOINTS[d]: round(float(diff[:, d].max()), 6) for d in range(6)},
            "gripper_max_diff": round(float(diff[:, 5].max()), 6),
        })
    overall = {
        "max_abs_diff": round(max(p["max_abs_diff"] for p in per_scene), 6),
        "mean_abs_diff": round(float(np.mean([p["mean_abs_diff"] for p in per_scene])), 6),
        "gripper_max_diff": round(max(p["gripper_max_diff"] for p in per_scene), 6),
    }
    result = {"model": args.model, "overall": overall, "per_scene": per_scene,
              "note": "open-loop chunk divergence; task threshold is 4 cm so even ~0.02 rad can flip borderline scenes"}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"overall": overall, "per_scene_max": [p["max_abs_diff"] for p in per_scene]}, indent=2))


if __name__ == "__main__":
    main()
