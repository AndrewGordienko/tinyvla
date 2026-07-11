"""Deterministic 1 -> 8 -> 64 sample overfit diagnostic.

This is intentionally a *supervised-only* tool.  It writes an artifact before
optimization and checkpoints progress so an interrupted run is evidence, not a
silent missing JSON.  It neither collects DAgger data nor evaluates rollouts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.deployable_controller import (
    LADDER, STATE_DIM, DeployableController, _masked_mse, _normalize,
    collect_demos,
)
from tinyvla.determinism import seed_everything


def digest(value) -> str:
    a = np.ascontiguousarray(value)
    return hashlib.sha256(a.view(np.uint8)).hexdigest()


def write_artifact(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def data_audit(data: dict, chunk: int) -> dict:
    """Audit temporal alignment, masks, and contradictory supervision."""
    imgs, state, label, mask = data["imgs"], data["state"], data["label"], data["mask"]
    frame_idx, episode, tvt = data["frame_indices"], data["episode"], data["temporal_view_times"]
    frame_monotonic = bool(np.all(np.diff(frame_idx, axis=1) >= 0))
    time_monotonic = bool(np.all(np.diff(tvt[..., 0], axis=1) >= -1e-12))
    same_timestep_views = bool(np.all(np.ptp(tvt, axis=2) <= 1e-12))
    no_cross_episode = bool(np.all(episode >= 0))  # Windows are built inside each episode loop.
    expected_spacing = 1 / 25
    diffs = np.diff(tvt[..., 0], axis=1)
    # Repeated first frames are intentional start padding; non-zero intervals must be one control period.
    spacing_ok = bool(np.all(np.isclose(diffs[(diffs > 1e-12)], expected_spacing, atol=1e-8)))
    suffix_ok = bool(np.all(mask == np.maximum.accumulate(mask[:, ::-1], axis=1)[:, ::-1]))
    # Exact duplicate complete observations, and the largest valid-target difference within each group.
    groups: dict[str, list[int]] = {}
    for i in range(len(state)):
        groups.setdefault(digest(np.concatenate([imgs[i].reshape(-1), state[i]])), []).append(i)
    exact_conflicts = []
    for rows in groups.values():
        if len(rows) > 1:
            target_range = float(np.abs(label[rows] - label[rows[0]]).max())
            if target_range > 1e-4:
                exact_conflicts.append({"rows": rows, "max_action_difference": target_range})
    # Near-duplicate screen/proprio summaries: a conservative candidate search, not a claim that pixels are equal.
    thumb = imgs.mean(axis=(2, 4, 5))  # [N, T, V], invariant enough to flag suspicious pairs cheaply
    summary = np.concatenate([thumb.reshape(len(thumb), -1), state], axis=1)
    z = (summary - summary.mean(0)) / (summary.std(0) + 1e-6)
    near = []
    for i in range(len(z)):
        d = np.sqrt(np.mean((z[i + 1:] - z[i]) ** 2, axis=1))
        if len(d):
            jrel = int(d.argmin()); j = i + 1 + jrel
            action_delta = float(np.abs(label[i] - label[j]).max())
            if d[jrel] < 0.05 and action_delta > 0.1:
                near.append({"rows": [i, j], "summary_distance": float(d[jrel]), "max_action_difference": action_delta})
    return {
        "data_hashes": {"frames": digest(imgs), "proprio": digest(state), "temporal_order": digest(frame_idx),
                        "targets": digest(label), "target_mask": digest(mask)},
        "alignment": {"frame_indices_monotonic": frame_monotonic, "temporal_times_monotonic": time_monotonic,
                      "same_timestep_all_views": same_timestep_views, "no_cross_episode": no_cross_episode,
                      "control_spacing_s": expected_spacing, "spacing_ok": spacing_ok},
        "chunk_mask": {"shape": list(mask.shape), "suffix_mask_is_prefix_valid": suffix_ok,
                       "valid_tokens": int(mask.sum()), "padded_tokens": int(mask.size - mask.sum()), "chunk": chunk},
        "supervision_conflicts": {"exact_observation_conflicts": exact_conflicts,
                                  "near_observation_conflicts": near,
                                  "near_definition": "z-scored mean-RGB-per-temporal-view plus proprio RMS < 0.05; target max difference > 0.1 rad"},
    }


def normalized_batch(data: dict, indices: np.ndarray):
    im = data["imgs"][indices].astype(np.float32, copy=True)
    st = data["state"][indices].astype(np.float32, copy=True)
    lab = data["label"][indices].astype(np.float32, copy=True)
    mask = data["mask"][indices].astype(np.float32, copy=True)
    valid = lab.reshape(-1, 6)[mask.reshape(-1).astype(bool)]
    smu, ssd = st.mean(0), st.std(0) + 1e-6
    ymu, ysd = valid.mean(0), valid.std(0) + 1e-6
    return im, _normalize(st, smu, ssd), _normalize(lab, ymu, ysd), mask, {
        "state_mean": smu.tolist(), "state_std": ssd.tolist(), "action_mean": ymu.tolist(), "action_std": ysd.tolist(),
        "units": "radians/position targets, exactly as SO101PickPlaceTask.ctrl_range", "dimension_order": "shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper",
        "chunk": int(lab.shape[1]),
    }


def parameter_norm(parameters) -> float:
    total = None
    for parameter in parameters:
        value = parameter.detach().float().square().sum()
        total = value if total is None else total + value
    return 0.0 if total is None else float(torch.sqrt(total).cpu().item())


def train_rung(data, cfg, n, seed, device, steps, min_steps, checkpoint, progress, checkpoint_dir, lr, grad_clip_norm):
    indices = np.linspace(0, len(data["state"]) - 1, n).astype(np.int64)
    im, st, lab, mask, normalization = normalized_batch(data, indices)
    before_hash = {"frames": digest(im), "proprio": digest(st), "targets": digest(lab), "mask": digest(mask)}
    seed_everything(seed)
    dev = torch.device(device)
    model = DeployableController(cfg["n_frames"], len(cfg["views"]), STATE_DIM, cfg["chunk"]).to(dev)
    model.train()
    named = {name: p for name, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.Adam(named.values(), lr=lr)
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    parameter_check = {"model_train_mode": model.training, "all_intended_trainable_in_optimizer": optimizer_ids == {id(p) for p in named.values()},
                       "trainable_parameter_count": len(named), "optimizer_parameter_count": len(optimizer_ids),
                       "missing": [name for name, p in named.items() if id(p) not in optimizer_ids]}
    I, S, Y, M = (torch.from_numpy(x).to(dev) for x in (im, st, lab, mask))
    start = time.monotonic(); curve = []
    model.eval()
    with torch.no_grad():
        initial_pred = model(I, S)
        initial_loss = float(_masked_mse(initial_pred, Y, M).item())
    model.train()
    stopped_early = False
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        pred = model(I, S)
        loss = _masked_mse(pred, Y, M)
        if not model.training or not pred.requires_grad or not loss.requires_grad:
            raise RuntimeError("detached tensor or non-training model during overfit diagnostic")
        loss.backward()
        grad_norm = parameter_norm(p.grad for p in named.values() if p.grad is not None)
        clip_applied = grad_clip_norm is not None and grad_norm > grad_clip_norm
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(named.values(), grad_clip_norm)
        clipped_grad_norm = parameter_norm(p.grad for p in named.values() if p.grad is not None)
        old = [p.detach().clone() for p in named.values()]
        optimizer.step()
        update_norm = parameter_norm((p.detach() - b for p, b in zip(named.values(), old)))
        model.eval()
        with torch.no_grad():
            current = model(I, S)
            current_loss = float(_masked_mse(current, Y, M).item())
            per_dim = (((current - Y).square() * M.unsqueeze(-1)).sum((0, 1)) / M.sum()).cpu().tolist()
        model.train()
        curve.append({"step": step + 1, "loss": current_loss, "per_action_dimension_mse": per_dim,
                      "gradient_norm": grad_norm, "clipped_gradient_norm": clipped_grad_norm, "clip_applied": clip_applied, "parameter_update_norm": update_norm,
                      "learning_rate": optimizer.param_groups[0]["lr"], "runtime_s": time.monotonic() - start})
        if (step + 1) % checkpoint == 0 or step + 1 == steps:
            progress({"last_completed": {"rung": n, "step": step + 1}, "latest": curve[-1]})
            # A completed curve may stop once it has remained well inside the
            # stated gate for four checkpoints. This bounds a pure memorization
            # diagnostic without loosening any gate.
            threshold = 1e-6 if n == 1 else 1e-4
            recent = [curve[-1 - k * checkpoint] for k in range(4)] if len(curve) >= 1 + 3 * checkpoint else []
            if step + 1 >= min_steps and recent and all(x["loss"] < threshold for x in recent):
                stopped_early = True
                break
    after_hash = {"frames": digest(im), "proprio": digest(st), "targets": digest(lab), "mask": digest(mask)}
    checkpoint_path = None
    reload_loss = None
    model.eval()
    with torch.no_grad():
        current = model(I, S)
        final_loss = float(_masked_mse(current, Y, M).item())
        final_mae = float((((current - Y).abs() * M.unsqueeze(-1)).sum() / (M.sum() * 6)).item())
    if n == 64:
        checkpoint_path = Path(checkpoint_dir) / "supervised_overfit_64.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        # Keep the checkpoint weights-only-loadable: a NumPy array would require
        # an unsafe pickle global even though the model state itself is sound.
        torch.save({"state_dict": model.state_dict(), "normalization": normalization, "indices": indices.tolist()}, checkpoint_path)
        restored = DeployableController(cfg["n_frames"], len(cfg["views"]), STATE_DIM, cfg["chunk"]).to(dev)
        restored.load_state_dict(torch.load(checkpoint_path, map_location=dev, weights_only=True)["state_dict"])
        restored.eval()
        with torch.no_grad():
            reload_loss = float(_masked_mse(restored(I, S), Y, M).item())
    final = curve[-1]
    return {"n_samples": n, "indices": indices.tolist(), "steps_requested": steps, "steps_completed": len(curve), "stopped_early_converged": stopped_early, "initial_loss": initial_loss,
            "final_loss": final_loss, "final_normalized_mae": final_mae,
            "final_per_action_dimension_mse": (((current - Y).square() * M.unsqueeze(-1)).sum((0, 1)) / M.sum()).cpu().tolist(), "curve": curve,
            "runtime_s": time.monotonic() - start, "device": str(dev), "parameter_check": parameter_check,
            "input_hashes_before": before_hash, "input_hashes_after": after_hash, "inputs_unchanged": before_hash == after_hash,
            "normalization": normalization, "save_reload": {"path": None if checkpoint_path is None else str(checkpoint_path), "loss": reload_loss,
                                                                 "matches_final": None if reload_loss is None else bool(np.isclose(reload_loss, final_loss, atol=1e-9))}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--architecture", default="temporal", choices=tuple(LADDER))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--min-steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-clip-norm", type=float, help="Optional fixed global gradient clip; recorded at every step.")
    ap.add_argument("--cpu-steps", type=int, default=20)
    ap.add_argument("--checkpoint-interval", type=int, default=25)
    ap.add_argument("--backend-probe", action="store_true", help="Run the same small fixed batch on MPS and CPU, then stop.")
    ap.add_argument("--probe-samples", type=int, default=8, choices=(1, 8))
    ap.add_argument("--output", default="artifacts/truth_harness/deployable_overfit_ladder_2026-07-11.json")
    args = ap.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing evidence: {output}")
    cfg = dict(LADDER[args.architecture])
    if cfg["chunk"] != 1:
        raise SystemExit("This first diagnostic is intentionally the temporal chunk-1 gate.")
    scenes = json.loads((Path("artifacts/truth_harness/datasets/command0_4") / "scene_manifest.json").read_text())["scenes"]
    payload = {"status": "collecting", "kind": "deterministic_supervised_overfit_ladder", "config": cfg,
               "seed": args.seed, "device": args.device, "augmentation": False, "shuffle": False, "dropout": False,
               "scheduler": None, "learning_rate": args.lr, "gradient_clip_norm": args.grad_clip_norm,
               "rungs": {}, "checkpoint_dir": str(output.with_suffix("")) + "_checkpoints"}
    write_artifact(output, payload)
    print(json.dumps({"status": "collecting", "output": str(output)}), flush=True)
    data = collect_demos(scenes, cfg["n_frames"], cfg["views"], cfg["chunk"], args.seed)
    payload["data_audit"] = data_audit(data, cfg["chunk"])
    payload["status"] = "training"
    write_artifact(output, payload)
    def progress(update):
        payload.update(update); write_artifact(output, payload)
        print(json.dumps({"status": payload["status"], **update}), flush=True)
    if args.backend_probe:
        payload["backend_comparison"] = {}
        for backend in ("mps", "cpu"):
            payload["last_completed"] = {"backend": backend, "rung": args.probe_samples, "step": 0}; write_artifact(output, payload)
            payload["backend_comparison"][backend] = train_rung(
                data, cfg, args.probe_samples, args.seed, backend, args.steps, args.steps,
                args.checkpoint_interval, progress, payload["checkpoint_dir"], args.lr, args.grad_clip_norm)
            write_artifact(output, payload)
        payload["status"] = "complete_backend_probe"; write_artifact(output, payload)
        print(json.dumps({k: {"final_loss": v["final_loss"], "mae": v["final_normalized_mae"]}
                          for k, v in payload["backend_comparison"].items()}, indent=2), flush=True)
        return
    for n in (1, 8, 64):
        payload["last_completed"] = {"rung": n, "step": 0}; write_artifact(output, payload)
        result = train_rung(data, cfg, n, args.seed, args.device, args.steps, args.min_steps, args.checkpoint_interval, progress, payload["checkpoint_dir"], args.lr, args.grad_clip_norm)
        payload["rungs"][str(n)] = result; write_artifact(output, payload)
    # A deliberately small same-data CPU/MPS comparison; not a performance run.
    if args.device == "mps" and torch.backends.mps.is_available():
        payload["backend_comparison"] = {
            "mps_one_sample": train_rung(data, cfg, 1, args.seed, "mps", args.cpu_steps, args.cpu_steps, args.cpu_steps, progress, payload["checkpoint_dir"], args.lr, args.grad_clip_norm),
            "cpu_one_sample": train_rung(data, cfg, 1, args.seed, "cpu", args.cpu_steps, args.cpu_steps, args.cpu_steps, progress, payload["checkpoint_dir"], args.lr, args.grad_clip_norm),
        }
    one, eight, sixtyfour = (payload["rungs"][str(n)] for n in (1, 8, 64))
    payload["gates"] = {"one_sample_perfect": one["final_normalized_mae"] < 1e-3,
                        "eight_sample_near_zero": eight["final_normalized_mae"] < 1e-2,
                        "sixtyfour_stable": sixtyfour["final_normalized_mae"] < 1e-2 and sixtyfour["save_reload"]["matches_final"]}
    payload["status"] = "complete"; payload["last_completed"] = {"rung": 64, "step": sixtyfour["steps_completed"]}; write_artifact(output, payload)
    print(json.dumps({"gates": payload["gates"], "losses": {n: payload["rungs"][n]["final_loss"] for n in payload["rungs"]}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
