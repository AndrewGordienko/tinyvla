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

from tinyvla.task import SO101PickPlaceTask, COMMANDS, GRASP_RADIUS, GRIP_GRAB
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
# Gate B: numerical action round-trip  physical -> normalize -> postprocess
# --------------------------------------------------------------------------- #
def gate_b_roundtrip(model: str, repo_id: str, root: Path, device: str,
                     tol: float = 1e-4, batches: int = 8) -> dict:
    """Verify physical action == postprocess(preprocess(physical action)) per dim.

    Comparing checkpoint stats to dataset stats (done separately below) is
    necessary but is NOT a round-trip. This runs real dataset samples through the
    actual preprocessor normalizer and postprocessor unnormalizer and requires the
    recovered physical action to match the original within `tol`, per action
    dimension, over valid (unpadded) timesteps only.
    """
    from torch.utils.data import DataLoader
    from tinyvla.fast_dataset import FastChunkDataset
    from tinyvla.runtime import load_runtime
    from tinyvla.determinism import make_generator

    meta = LeRobotDatasetMetadata(repo_id, root=str(root))
    r = load_runtime(model, meta=meta, dataset_root=str(root), device=device, stats_source="checkpoint")
    pre, post = r.preprocessor, r.postprocessor
    cs = r.policy.config.chunk_size
    dt = {"action": [i / meta.fps for i in range(cs)]}
    ds = FastChunkDataset(repo_id, root=str(root), delta_timestamps=dt)
    dl = DataLoader(ds, batch_size=4, shuffle=False, generator=make_generator(0), drop_last=True)

    per_dim_err = np.zeros(6)
    valid_count = 0
    pad_count = 0
    for bi, raw in enumerate(dl):
        if bi >= batches:
            break
        original = np.asarray(raw["action"])[:, :, :6]              # physical, from dataset
        pad = np.asarray(raw["action_is_pad"]).astype(bool)         # (B, cs)
        b = pre(dict(raw))
        recovered = post(b[ACTION]).detach().cpu().numpy()[:, :, :6]
        err = np.abs(recovered - original)                          # (B, cs, 6)
        valid = ~pad
        for d in range(6):
            per_dim_err[d] = max(per_dim_err[d], err[:, :, d][valid].max())
        valid_count += int(valid.sum())
        pad_count += int(pad.sum())
    del r
    per_dim = {JOINTS[d]: round(float(per_dim_err[d]), 8) for d in range(6)}
    max_err = float(per_dim_err.max())
    return {"gate": "B_roundtrip", "checkpoint": model, "tol": tol,
            "action_dim_order": JOINTS, "valid_timesteps": valid_count, "padded_timesteps": pad_count,
            "per_dim_max_abs_roundtrip_err": per_dim, "max_abs_roundtrip_err": round(max_err, 8),
            "pass": max_err < tol}


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
# Gate P: corrected per-dimension open-loop error of a trained checkpoint
#   - masks action_is_pad (never scores padded future timesteps)
#   - separates the executed prefix 0:n_action_steps from the unused tail
#   - reports timestep 0 and per-horizon error
#   - reports raw AND actuator-clipped predictions
#   - range-normalizes per-dim error (raw radians are not comparable)
#   - gripper: open/closed classification vs GRIP_GRAB, not raw radians
# --------------------------------------------------------------------------- #
def _mae_masked(err: np.ndarray, valid: np.ndarray) -> float:
    return float(err[valid].mean()) if valid.any() else float("nan")


def gate_p_perdim(model: str, repo_id: str, root: Path, device: str, batches: int) -> dict:
    import torch
    from torch.utils.data import DataLoader
    from tinyvla.fast_dataset import FastChunkDataset
    from tinyvla.runtime import load_runtime
    from tinyvla.determinism import seed_everything, make_generator

    seed_everything(0)
    dev = torch.device(device)
    env = SO101PickPlaceTask()
    lo, hi = env.ctrl_range[:, 0], env.ctrl_range[:, 1]
    rng6 = (hi - lo)[:6]
    meta = LeRobotDatasetMetadata(repo_id, root=str(root))
    r = load_runtime(model, meta=meta, dataset_root=str(root), device=dev, stats_source="checkpoint")
    pol, pre, post = r.policy.eval(), r.preprocessor, r.postprocessor
    cs = pol.config.chunk_size
    nstep = int(pol.config.n_action_steps)
    dt = {"action": [i / meta.fps for i in range(cs)]}
    ds = FastChunkDataset(repo_id, root=str(root), delta_timestamps=dt)
    dl = DataLoader(ds, batch_size=4, shuffle=True, generator=make_generator(0), drop_last=True)
    P_raw, P_clip, T, PAD = [], [], [], []
    with torch.inference_mode():
        for bi, raw in enumerate(dl):
            if bi >= batches:
                break
            b = pre(dict(raw))
            noise = torch.randn((b[ACTION].shape[0], cs, pol.config.max_action_dim), device=dev)
            pred = post(pol.predict_action_chunk(b, noise=noise)).cpu().numpy()[:, :, :6]
            P_raw.append(pred)
            P_clip.append(np.clip(pred, lo[:6], hi[:6]))
            T.append(np.asarray(raw["action"])[:, :, :6])           # physical target
            PAD.append(np.asarray(raw["action_is_pad"]).astype(bool))
    Praw, Pclip, Tt, Pad = (np.concatenate(x) for x in (P_raw, P_clip, T, PAD))
    valid = ~Pad                                                    # (N, cs)
    exec_mask = np.zeros_like(valid); exec_mask[:, :nstep] = True; exec_mask &= valid
    tail_mask = np.zeros_like(valid); tail_mask[:, nstep:] = True; tail_mask &= valid
    t0_mask = np.zeros_like(valid); t0_mask[:, 0] = True; t0_mask &= valid

    def dimtable(pred):
        err = np.abs(pred - Tt)
        out = {}
        for d in range(6):
            vmask = valid
            out[JOINTS[d]] = {
                "mae_all_valid": round(_mae_masked(err[:, :, d], vmask), 4),
                "mae_exec_prefix": round(_mae_masked(err[:, :, d], exec_mask), 4),
                "mae_tail": round(_mae_masked(err[:, :, d], tail_mask), 4),
                "mae_t0": round(_mae_masked(err[:, :, d], t0_mask), 4),
                "range_norm_mae_exec": round(_mae_masked(err[:, :, d], exec_mask) / float(rng6[d]), 4),
            }
        return out

    # per-horizon overall MAE (valid only), arm vs gripper
    horizon = []
    for k in range(cs):
        vk = valid[:, k]
        if not vk.any():
            continue
        earm = np.abs(Pclip[:, k, :5] - Tt[:, k, :5])[vk].mean()
        egrip = np.abs(Pclip[:, k, 5] - Tt[:, k, 5])[vk].mean()
        horizon.append({"h": k, "arm_mae": round(float(earm), 4), "gripper_mae": round(float(egrip), 4)})

    # gripper open/closed classification on the EXECUTED prefix (clipped pred)
    tgt_closed = (Tt[:, :, 5] < GRIP_GRAB) & exec_mask
    pred_closed = (Pclip[:, :, 5] < GRIP_GRAB) & exec_mask
    tp = int((tgt_closed & pred_closed).sum())
    fp = int((~tgt_closed & pred_closed & exec_mask).sum())
    fn = int((tgt_closed & ~pred_closed & exec_mask).sum())
    tn = int((~tgt_closed & ~pred_closed & exec_mask).sum())
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = (2 * prec * rec / (prec + rec)) if prec and rec else None
    should_close = int(tgt_closed.sum())
    should_open = int((~(Tt[:, :, 5] < GRIP_GRAB) & exec_mask).sum())
    del r
    return {
        "gate": "P_perdim", "checkpoint": model, "n_action_steps": nstep,
        "note": "raw radians not comparable across actuators; see range_norm_mae_exec and gripper classification",
        "per_dim_clipped": dimtable(Pclip),
        "per_dim_raw": dimtable(Praw),
        "gripper_classification_exec_prefix": {
            "threshold": GRIP_GRAB, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision_closed": round(prec, 3) if prec is not None else None,
            "recall_closed": round(rec, 3) if rec is not None else None,
            "f1_closed": round(f1, 3) if f1 is not None else None,
            "pct_should_close_but_open": round(100 * fn / should_close, 1) if should_close else None,
            "pct_should_open_but_closed": round(100 * fp / should_open, 1) if should_open else None,
        },
        "per_horizon": horizon,
    }


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
