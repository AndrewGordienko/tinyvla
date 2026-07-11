"""Observation audit for the deployable temporal-controller work.

Records what a deployable policy can actually see: which cameras exist, control /
frame rate, resolution, image latency, and — critically — whether the target cube
becomes OCCLUDED during grasp/carry in each view (motivating temporal frames and a
wrist view). No simulator-only state is used anywhere.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import mujoco

from tinyvla.task import SO101PickPlaceTask, GRASP_RADIUS
from scripts.controlled_dagger_mlp import _color

ROOT = Path("artifacts/truth_harness/datasets/command0_4")
IMG = 256


def red_fraction(img: np.ndarray) -> float:
    """Fraction of pixels that look like the red cube (R high, G/B low)."""
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    mask = (r > 130) & (g < 90) & (b < 90)
    return float(mask.mean())


def audit_scene(env, renderers, scene, cap=220, dwell=8):
    command = int(scene["command"]); color = _color(command)
    positions = {c: np.asarray(v, float) for c, v in scene["positions"].items()}
    env.reset(command=command, positions=positions)
    series = {cam: [] for cam in renderers}
    grasp_t, phase = None, []
    for t in range(cap):
        for cam, r in renderers.items():
            r.update_scene(env.data, camera=cam)
            series[cam].append(red_fraction(r.render()))
        # phase label from grasp/height
        if env.grasped is not None and grasp_t is None:
            grasp_t = t
        env.step(env.reactive_action(gain=0.25, max_dq=0.03))
        if env.success():
            break
    return series, grasp_t, t


def main() -> None:
    env = SO101PickPlaceTask(seed=0)
    m = env.model
    cams = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(m.ncam)]
    renderers = {c: mujoco.Renderer(m, height=IMG, width=IMG) for c in cams}

    scenes = json.loads((ROOT / "scene_manifest.json").read_text())["scenes"]
    per_scene = []
    for i, s in enumerate(scenes):
        series, grasp_t, end_t = audit_scene(env, renderers, s)
        row = {"scene": i, "grasp_step": grasp_t, "end_step": end_t}
        for cam in cams:
            arr = np.asarray(series[cam])
            row[f"{cam}_red_mean"] = round(float(arr.mean()), 5)
            # occlusion during grasp/carry window (grasp_t .. grasp_t+30)
            if grasp_t is not None:
                w = arr[grasp_t:grasp_t + 30]
                row[f"{cam}_red_at_grasp_min"] = round(float(w.min()) if len(w) else 0.0, 5)
                pre = arr[max(0, grasp_t - 20):grasp_t]
                row[f"{cam}_red_pre_grasp"] = round(float(pre.mean()) if len(pre) else 0.0, 5)
        per_scene.append(row)

    for r in renderers.values():
        r.close()

    audit = {
        "cameras_in_model": cams,
        "dataset_recorded_views": ["observation.images.front"],
        "smolvla_base_configured_cameras": ["camera1", "camera2", "camera3"],
        "smolvla_effective_views_here": ["observation.images.front"],
        "current_cnn_views": ["front"],
        "control_hz": env.control_hz,
        "sim_hz": round(1.0 / m.opt.timestep),
        "n_substeps": env.n_substeps,
        "fps": 25,
        "image_latency_steps": 0,
        "image_latency_note": "synchronous sim render: the frame is generated from the "
                              "current state at each control step (no real-world capture lag)",
        "dataset_resolution": [256, 256, 3],
        "cnn_input_resolution": [84, 84, 3],
        "no_simulator_only_state": True,
        "grasp_radius_m": GRASP_RADIUS,
        "per_scene_cube_visibility": per_scene,
    }
    # occlusion verdict
    front_pre = np.mean([r["front_red_pre_grasp"] for r in per_scene])
    front_grasp = np.mean([r["front_red_at_grasp_min"] for r in per_scene])
    audit["occlusion_summary"] = {
        "front_red_pre_grasp_mean": round(float(front_pre), 5),
        "front_red_at_grasp_min_mean": round(float(front_grasp), 5),
        "front_occlusion_drop_ratio": round(float(front_grasp / front_pre) if front_pre else 0.0, 3),
    }
    out = Path("artifacts/truth_harness/observation_audit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
