"""Systematic diagnosis of the four-scene command-0 overfit failure.

Runs staged gates that localize the failure to data / normalization / loading /
action semantics / chunk execution / grasp timing / model adaptation, instead of
blindly training longer. Each gate prints a JSON/verdict block.

Gates:
  A  expert-replay   replay stored dataset actions -> require 4/4 success (pure sim)
  B  roundtrip       checkpoint action-normalizer stats == dataset stats
  C  single-batch    overfit ONE fixed batch; does flow loss collapse + actions converge?
  P  perdim          per-action-dimension open-loop error of a trained checkpoint

Usage:
  MUJOCO_GL=glfw .venv/bin/python -m scripts.diagnose_overfit --gate A
  MUJOCO_GL=glfw .venv/bin/python -m scripts.diagnose_overfit --gate P \
      --model artifacts/truth_harness/checkpoints/command0_overfit_500
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# datasets before policies (lerobot import-cycle guard)
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.constants import ACTION

from tinyvla.task import SO101PickPlaceTask, COMMANDS, GRASP_RADIUS
from tinyvla.runtime import detect_action_semantics

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _episode_actions(ds: LeRobotDataset) -> dict[int, np.ndarray]:
    hf = ds.hf_dataset.with_format("numpy")
    actions = np.asarray(hf["action"])
    episodes = np.asarray(hf["episode_index"]).astype(int)
    frames = np.asarray(hf["frame_index"]).astype(int)
    out: dict[int, np.ndarray] = {}
    for ep in sorted(set(episodes.tolist())):
        mask = episodes == ep
        out[int(ep)] = actions[mask][np.argsort(frames[mask])]
    return out


def _manifest(root: Path) -> list[dict]:
    return json.loads((root / "scene_manifest.json").read_text())["scenes"]


def command_color(command: int) -> str:
    return COMMANDS[command]["steps"][0][0]


# --------------------------------------------------------------------------- #
# Gate A: expert replay (pure sim, no model)
# --------------------------------------------------------------------------- #
def gate_a_expert_replay(repo_id: str, root: Path) -> dict:
    semantics = detect_action_semantics(root)
    delta = semantics == "delta"
    ds = LeRobotDataset(repo_id, root=str(root))
    ep_actions = _episode_actions(ds)
    env = SO101PickPlaceTask()  # default control_hz=25 matches collection
    rollouts = []
    for scene in _manifest(root):
        ep, command = int(scene["episode"]), int(scene["command"])
        positions = {c: np.asarray(v, dtype=float) for c, v in scene["positions"].items()}
        env.reset(command=command, positions=positions)
        dmin, t_at_min, grip_at_min, grasp_fired_t = float("inf"), -1, None, -1
        for t, a in enumerate(ep_actions[ep]):
            a = np.asarray(a, dtype=float)
            env.step(a + env.data.qpos[:6] if delta else a)
            if grasp_fired_t < 0 and env.grasped is not None:
                grasp_fired_t = t
            d = float(np.linalg.norm(env.ee_pos() - env.cube_pos(command_color(command))))
            if d < dmin:
                dmin, t_at_min, grip_at_min = d, t, float(env.data.qpos[5])
        rollouts.append({
            "episode": ep, "command": command, "n_actions": int(len(ep_actions[ep])),
            "success": bool(env.success()), "min_ee_cube_dist": round(dmin, 4),
            "t_at_min": t_at_min, "grip_at_min": round(grip_at_min, 4),
            "grasp_fired_t": grasp_fired_t, "grasp_radius": GRASP_RADIUS,
        })
    n = sum(r["success"] for r in rollouts)
    return {"gate": "A_expert_replay", "action_semantics": semantics, "n": len(rollouts),
            "successes": n, "pass": n == len(rollouts), "rollouts": rollouts}


# --------------------------------------------------------------------------- #
# Gate B: normalization round-trip (checkpoint stats vs dataset stats)
# --------------------------------------------------------------------------- #
def gate_b_roundtrip(model: str, repo_id: str, root: Path, device: str) -> dict:
    from tinyvla.runtime import load_runtime
    meta = LeRobotDatasetMetadata(repo_id, root=str(root))
    r = load_runtime(model, meta=meta, dataset_root=str(root), device=device, stats_source="checkpoint")
    ckpt_stats = None
    for step in r.preprocessor.steps:
        st = getattr(step, "stats", None)
        if st and ACTION in st:
            ckpt_stats = st[ACTION]
            break
    ds_stats = meta.stats[ACTION]
    fields = {}
    max_delta = 0.0
    for key in ("mean", "std", "min", "max"):
        if ckpt_stats is not None and key in ckpt_stats and key in ds_stats:
            c = np.asarray(ckpt_stats[key]).ravel()
            d = np.asarray(ds_stats[key]).ravel()
            delta = float(np.abs(c - d).max())
            max_delta = max(max_delta, delta)
            fields[key] = {"max_abs_delta": delta}
    del r
    return {"gate": "B_roundtrip", "checkpoint": model, "stats_max_abs_delta": max_delta,
            "fields": fields, "pass": max_delta < 1e-6}


# --------------------------------------------------------------------------- #
# Gate C: single fixed-batch memorization
# --------------------------------------------------------------------------- #
def gate_c_single_batch(repo_id: str, root: Path, device: str, steps: int, lr: float) -> dict:
    import torch
    from torch.utils.data import DataLoader
    from tinyvla.fast_dataset import FastChunkDataset
    from tinyvla.runtime import load_runtime
    from tinyvla.determinism import seed_everything, make_generator
    from tinyvla.trainability import set_trainable
    from tinyvla.paths import MODELS_ROOT

    seed_everything(4242)
    dev = torch.device(device)
    meta = LeRobotDatasetMetadata(repo_id, root=str(root))
    r = load_runtime(MODELS_ROOT / "smolvla_base", meta=meta, dataset_root=str(root),
                     device=dev, stats_source="dataset", base_checkpoint=True)
    pol, pre, post = r.policy, r.preprocessor, r.postprocessor
    ntrain = set_trainable(pol, "checkpoint")
    cs = pol.config.chunk_size
    dt = {"action": [i / meta.fps for i in range(cs)]}
    ds = FastChunkDataset(repo_id, root=str(root), delta_timestamps=dt)
    raw = next(iter(DataLoader(ds, batch_size=4, shuffle=True,
                               generator=make_generator(4242), drop_last=True)))
    tgt = post(pre(dict(raw))[ACTION]).detach().cpu()[:, :, :6]
    noise = torch.randn((4, cs, pol.config.max_action_dim), device=dev)
    opt = torch.optim.AdamW([p for p in pol.parameters() if p.requires_grad], lr=lr)
    trace = []
    pol.train()
    for step in range(1, steps + 1):
        loss, _ = pol.forward(pre(dict(raw)))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 25 == 0 or step == 1:
            pol.eval()
            with torch.inference_mode():
                pred = post(pol.predict_action_chunk(pre(dict(raw)), noise=noise)).detach().cpu()[:, :, :6]
            pol.train()
            err = (pred - tgt).abs()
            trace.append({"step": step, "flow_loss": round(float(loss), 4),
                          "action_mae": round(float(err.mean()), 4),
                          "action_max_abs": round(float(err.max()), 4),
                          "gripper_mae": round(float(err[:, :, 5].mean()), 4)})
    del r
    last = trace[-1]
    return {"gate": "C_single_batch", "trainable_millions": round(ntrain / 1e6, 1),
            "chunk_size": cs, "lr": lr, "steps": steps, "trace": trace,
            "converged_flow_loss": last["flow_loss"], "final_action_mae": last["action_mae"],
            "final_gripper_mae": last["gripper_mae"]}


# --------------------------------------------------------------------------- #
# Gate P: per-dimension open-loop error of a trained checkpoint
# --------------------------------------------------------------------------- #
def gate_p_perdim(model: str, repo_id: str, root: Path, device: str, batches: int) -> dict:
    import torch
    from torch.utils.data import DataLoader
    from tinyvla.fast_dataset import FastChunkDataset
    from tinyvla.runtime import load_runtime
    from tinyvla.determinism import seed_everything, make_generator

    seed_everything(0)
    dev = torch.device(device)
    meta = LeRobotDatasetMetadata(repo_id, root=str(root))
    r = load_runtime(model, meta=meta, dataset_root=str(root), device=dev, stats_source="checkpoint")
    pol, pre, post = r.policy.eval(), r.preprocessor, r.postprocessor
    cs = pol.config.chunk_size
    dt = {"action": [i / meta.fps for i in range(cs)]}
    ds = FastChunkDataset(repo_id, root=str(root), delta_timestamps=dt)
    dl = DataLoader(ds, batch_size=4, shuffle=True, generator=make_generator(0), drop_last=True)
    errs, preds, tgts = [], [], []
    with torch.inference_mode():
        for bi, raw in enumerate(dl):
            if bi >= batches:
                break
            b = pre(dict(raw))
            noise = torch.randn((b[ACTION].shape[0], cs, pol.config.max_action_dim), device=dev)
            pred = post(pol.predict_action_chunk(b, noise=noise)).cpu().numpy()[:, :, :6]
            tgt = post(b[ACTION]).cpu().numpy()[:, :, :6]
            errs.append(np.abs(pred - tgt)); preds.append(pred); tgts.append(tgt)
    E, P, T = np.concatenate(errs), np.concatenate(preds), np.concatenate(tgts)
    per_dim = {JOINTS[d]: {"mae": round(float(E[:, :, d].mean()), 4),
                           "max_abs": round(float(E[:, :, d].max()), 4),
                           "tgt_range": [round(float(T[:, :, d].min()), 3), round(float(T[:, :, d].max()), 3)],
                           "pred_range": [round(float(P[:, :, d].min()), 3), round(float(P[:, :, d].max()), 3)]}
               for d in range(6)}
    closed = T[:, :, 5] < 0.5
    left_open = P[:, :, 5] > 0.5
    del r
    return {"gate": "P_perdim", "checkpoint": model, "overall_mae": round(float(E.mean()), 4),
            "overall_max_abs_dim": JOINTS[int(np.unravel_index(E.argmax(), E.shape)[2])],
            "per_dim": per_dim,
            "grasp_frames_frac": round(float(closed.mean()), 3),
            "gripper_left_open_at_grasp_frac": round(float(left_open[closed].mean()), 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True, choices=["A", "B", "C", "P"])
    ap.add_argument("--repo-id", default="local/truth_gate_command0_4")
    ap.add_argument("--root", default="artifacts/truth_harness/datasets/command0_4")
    ap.add_argument("--model", default="artifacts/truth_harness/checkpoints/command0_overfit_500")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batches", type=int, default=12)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    root = Path(args.root)

    if args.gate == "A":
        result = gate_a_expert_replay(args.repo_id, root)
    elif args.gate == "B":
        result = gate_b_roundtrip(args.model, args.repo_id, root, args.device)
    elif args.gate == "C":
        result = gate_c_single_batch(args.repo_id, root, args.device, args.steps, args.lr)
    else:
        result = gate_p_perdim(args.model, args.repo_id, root, args.device, args.batches)

    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n")


if __name__ == "__main__":
    main()
