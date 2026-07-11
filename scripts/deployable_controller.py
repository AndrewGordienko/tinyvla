"""Deployable temporal/multi-view controller + supervised overfit gate.

Deployable observations only: image(s) + qpos6 (incl. gripper joint position) +
qvel6 + prev_action6. NO simulator-only state (no grasped bit, no cube/ee/dest).

Architecture ladder (isolates one factor at a time), shared ResNet-18-class
encoder:
  L1 single_frame   1 front frame
  L2 temporal       4 front frames (temporal stack)
  L3 multiview      4 frames x {front, wrist}
  L4 multiview+chunk L3 + action-chunk output

This module provides the data pipeline, models, and the SUPERVISED 64-SAMPLE GATE
(overfit to near-zero normalized action error + blank-image / shuffled-frame /
swapped-image controls proving the visual + temporal stream is used). The
closed-loop four-scene gate lives alongside once this gate passes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
import mujoco
from torchvision.models import resnet18

from tinyvla.task import SO101PickPlaceTask, HOME_QPOS
from tinyvla.determinism import seed_everything
from scripts.controlled_dagger_mlp import stage_of, _color

ROOT = Path("artifacts/truth_harness/datasets/command0_4")
IMG_RENDER, IMG_NET, EP_LEN, DWELL = 256, 96, 220, 8
STATE_DIM = 18  # qpos6 + qvel6 + prev6 ; deployable, no grasp bit

LADDER = {
    "single_frame": {"n_frames": 1, "views": ("front",), "chunk": 1},
    "temporal":     {"n_frames": 4, "views": ("front",), "chunk": 1},
    "multiview":    {"n_frames": 4, "views": ("front", "wrist"), "chunk": 1},
    "multiview_chunk": {"n_frames": 4, "views": ("front", "wrist"), "chunk": 8},
}


def snapshot(env: SO101PickPlaceTask) -> dict[str, Any]:
    """Capture every mutable state used by the simulated task.

    This is deliberately local to the deployable experiment.  It is used only
    to ask the *expert* for labels, never as a policy input.
    """
    return {
        "qpos": env.data.qpos.copy(), "qvel": env.data.qvel.copy(),
        "ctrl": env.data.ctrl.copy(), "time": float(env.data.time),
        "act": env.data.act.copy(), "grasped": env.grasped,
        "off_pos": None if not hasattr(env, "_off_pos") else env._off_pos.copy(),
        "off_quat": None if not hasattr(env, "_off_quat") else env._off_quat.copy(),
        "phase": env.phase, "phase_t": env.phase_t, "step_idx": env.step_idx,
    }


def restore(env: SO101PickPlaceTask, saved: dict[str, Any]) -> None:
    """Restore ``snapshot`` exactly enough that label generation is side-effect free."""
    env.data.qpos[:] = saved["qpos"]; env.data.qvel[:] = saved["qvel"]
    env.data.ctrl[:] = saved["ctrl"]; env.data.time = saved["time"]
    if env.data.act.size:
        env.data.act[:] = saved["act"]
    env.grasped = saved["grasped"]
    if saved["off_pos"] is not None:
        env._off_pos = saved["off_pos"].copy(); env._off_quat = saved["off_quat"].copy()
    elif hasattr(env, "_off_pos"):
        del env._off_pos; del env._off_quat
    env.phase, env.phase_t, env.step_idx = saved["phase"], saved["phase_t"], saved["step_idx"]
    mujoco.mj_forward(env.model, env.data)


def expert_chunk_from_snapshot(env: SO101PickPlaceTask, chunk: int) -> np.ndarray:
    """Label one learner observation by an independent reactive-expert rollout.

    The learner state is restored before returning.  In particular, this never
    pieces a chunk together from labels observed later on a learner rollout.
    """
    saved = snapshot(env)
    out = []
    try:
        for _ in range(chunk):
            action = env.reactive_action(gain=0.25, max_dq=0.03).astype(np.float32)
            out.append(action)
            env.step(action)
    finally:
        restore(env, saved)
    return np.stack(out)


# ---- observation ---------------------------------------------------------
def state_vec(env, prev):
    return np.concatenate([env.data.qpos[:6], env.data.qvel[:6], prev]).astype(np.float32)


def render_views(env, renderers, views):
    out = []
    for v in views:
        renderers[v].update_scene(env.data, camera=v)
        img = renderers[v].render()
        t = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
        t = nn.functional.interpolate(t, size=(IMG_NET, IMG_NET), mode="bilinear", align_corners=False)
        out.append(t.squeeze(0).numpy().astype(np.float32))
    return np.stack(out)  # [V, 3, H, W]


def _window(frames, t, n_frames):
    """Last n_frames frames ending at t, padding the start by repeating frame 0.
    frames: [T, V, 3, H, W]; returns [n_frames, V, 3, H, W] in temporal order."""
    idx = [max(0, t - k) for k in range(n_frames - 1, -1, -1)]
    return frames[idx]


def collect_demos(scenes, n_frames, views, chunk, seed=0):
    """Reactive-expert demos as deployable samples with temporal windows and
    expert action chunks (label[t] = the expert's own next `chunk` actions)."""
    env = SO101PickPlaceTask(seed=seed)
    renderers = {v: mujoco.Renderer(env.model, height=IMG_RENDER, width=IMG_RENDER) for v in views}
    samples_imgs, samples_state, samples_label, samples_stage = [], [], [], []
    for scene in scenes:
        command = int(scene["command"]); color = _color(command)
        positions = {c: np.asarray(v, float) for c, v in scene["positions"].items()}
        env.reset(command=command, positions=positions)
        prev = HOME_QPOS.astype(np.float32)
        frames, states, actions, stages = [], [], [], []
        dwell = 0
        for _ in range(EP_LEN):
            frames.append(render_views(env, renderers, views))
            states.append(state_vec(env, prev))
            a = env.reactive_action(gain=0.25, max_dq=0.03).astype(np.float32)
            actions.append(a); stages.append(stage_of(env, color))
            env.step(a); prev = a
            dwell = dwell + 1 if env.success() else 0
            if dwell >= DWELL:
                break
        frames = np.asarray(frames, np.float32); actions = np.asarray(actions, np.float32)
        T = len(frames)
        for t in range(T):
            samples_imgs.append(_window(frames, t, n_frames))
            samples_state.append(states[t])
            # expert chunk: next `chunk` actions along the expert trajectory, padded
            chunk_lab = actions[t:t + chunk]
            if len(chunk_lab) < chunk:
                chunk_lab = np.concatenate([chunk_lab, np.repeat(actions[-1:], chunk - len(chunk_lab), 0)])
            samples_label.append(chunk_lab)
            samples_stage.append(stages[t])
    for r in renderers.values():
        r.close()
    return {"imgs": np.asarray(samples_imgs, np.float32), "state": np.asarray(samples_state, np.float32),
            "label": np.asarray(samples_label, np.float32), "stage": samples_stage}


# ---- model ---------------------------------------------------------------
class SharedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        base = resnet18(weights=None)
        base.fc = nn.Identity()
        self.net = base
        self.out_dim = 512

    def forward(self, x):          # [B, 3, H, W] -> [B, 512]
        return self.net(x)


class DeployableController(nn.Module):
    def __init__(self, n_frames, n_views, state_dim, chunk, encoder=None):
        super().__init__()
        self.n_frames, self.n_views, self.chunk = n_frames, n_views, chunk
        self.enc = encoder or SharedEncoder()
        feat = self.enc.out_dim * n_frames * n_views
        self.smlp = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(feat + 128, 256), nn.ReLU(), nn.Linear(256, chunk * 6))

    def forward(self, imgs, st):    # imgs: [B, T, V, 3, H, W]
        B, T, V = imgs.shape[:3]
        f = self.enc(imgs.reshape(B * T * V, *imgs.shape[3:]))     # [B*T*V, 512]
        f = f.reshape(B, T * V * self.enc.out_dim)                 # temporal+view order preserved
        out = self.head(torch.cat([f, self.smlp(st)], dim=1))
        return out.reshape(B, self.chunk, 6)


# ---- supervised 64-sample gate ------------------------------------------
def _normalize(a, mu, sd):
    return (a - mu) / sd


def _train_overfit(cfg, I, S, Y, seed, device, steps, lr):
    seed_everything(seed)
    net = DeployableController(cfg["n_frames"], len(cfg["views"]), STATE_DIM, cfg["chunk"]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.MSELoss()
    net.train()
    for _ in range(steps):
        opt.zero_grad(); lossf(net(I, S), Y).backward(); opt.step()
    net.eval()
    return net


def supervised_gate(data, cfg, seed, device, steps=1200, n=64, lr=3e-4):
    """Overfit a fixed n-sample batch to near-zero normalized action error.

    Because a heavily over-parameterized model can memorize 64 (state->action)
    pairs from STATE ALONE, "prediction changes when the image changes" is not a
    reliable wiring proof. The decisive test is an IMAGE-ONLY overfit (state
    zeroed): if the visual pathway can drive the output to near-zero error on its
    own, the encoder's gradients flow and images reach the head. We also report
    the blank-image / swapped-image / shuffled-frame prediction sensitivities on
    the full model, and whether state alone suffices on this batch (diagnostic)."""
    dev = torch.device(device)
    N = len(data["state"])
    idx = np.linspace(0, N - 1, n).astype(int)     # deterministic batch spanning phases
    imgs = torch.from_numpy(data["imgs"][idx]).float()
    st_np, lab_np = data["state"][idx], data["label"][idx]
    smu, ssd = st_np.mean(0), st_np.std(0) + 1e-6
    lmu, lsd = lab_np.reshape(-1, 6).mean(0), lab_np.reshape(-1, 6).std(0) + 1e-6
    I = imgs.to(dev)
    S = torch.from_numpy(_normalize(st_np, smu, ssd)).float().to(dev)
    Y = torch.from_numpy(_normalize(lab_np, lmu, lsd)).float().to(dev)
    Sz = torch.zeros_like(S)          # state zeroed  -> image-only
    Iz = torch.zeros_like(I)          # image zeroed  -> state-only
    lossf = nn.MSELoss()

    net = _train_overfit(cfg, I, S, Y, seed, dev, steps, lr)          # full
    net_img = _train_overfit(cfg, I, Sz, Y, seed, dev, steps, lr)     # image-only (vision wiring)
    net_st = _train_overfit(cfg, Iz, S, Y, seed, dev, steps, lr)      # state-only (diagnostic)

    with torch.inference_mode():
        pred = net(I, S)
        full_overfit = float(lossf(pred, Y).item())
        full_norm_mae = float((pred - Y).abs().mean().item())
        mae_phys = float(np.abs(pred.cpu().numpy() * lsd + lmu - lab_np).mean())
        image_only_overfit = float(lossf(net_img(I, Sz), Y).item())
        image_only_norm_mae = float((net_img(I, Sz) - Y).abs().mean().item())
        state_only_overfit = float(lossf(net_st(Iz, S), Y).item())
        # Use the image-only model for the controls.  A full model can reasonably
        # use proprioception too; this makes the visual-pathway claim falsifiable.
        img_pred = net_img(I, Sz)
        d_blank = float((net_img(Iz, Sz) - img_pred).abs().mean().item())
        d_swap = float((net_img(torch.roll(I, 1, 0), Sz) - img_pred).abs().mean().item())
        if cfg["n_frames"] > 1:
            perm = torch.tensor([cfg["n_frames"] - 1] + list(range(cfg["n_frames"] - 1)))  # reverse-ish shift
            d_shuffle = float((net_img(I[:, perm], Sz) - img_pred).abs().mean().item())
        else:
            d_shuffle = None

    # Near-zero is assessed in normalized units.  The image-only fit and the
    # controls rule out an accidentally disconnected visual/temporal input.
    passed = (full_norm_mae < 0.05 and image_only_norm_mae < 0.10
              and d_blank > 1e-3 and d_swap > 1e-3
              and (d_shuffle is None or d_shuffle > 1e-3))
    return {
        "config": cfg, "n_samples": n, "steps": steps,
        "full_overfit_norm_mse": round(full_overfit, 6),
        "full_overfit_norm_mae": round(full_norm_mae, 6),
        "full_overfit_mae_phys_rad": round(mae_phys, 5),
        "image_only_overfit_norm_mse": round(image_only_overfit, 6),
        "image_only_overfit_norm_mae": round(image_only_norm_mae, 6),
        "state_only_overfit_norm_mse": round(state_only_overfit, 6),
        "pred_change_blank_image": round(d_blank, 5),
        "pred_change_swapped_image": round(d_swap, 5),
        "pred_change_shuffled_frames": None if d_shuffle is None else round(d_shuffle, 5),
        "vision_pathway_ok": image_only_overfit < 0.05,
        "temporal_order_sensitive": None if d_shuffle is None else d_shuffle > 0.02,
        "state_alone_sufficient": state_only_overfit < 0.01,
        "passed": bool(passed),
    }


def train_policy(data, cfg, seed, device, steps, lr=3e-4):
    """Train from scratch on the aggregate, with normalization fit to that aggregate."""
    seed_everything(seed)
    images = torch.from_numpy(np.asarray(data["imgs"], np.float32))
    states = np.asarray(data["state"], np.float32)
    labels = np.asarray(data["label"], np.float32)
    smu, ssd = states.mean(0), states.std(0) + 1e-6
    ymu, ysd = labels.reshape(-1, 6).mean(0), labels.reshape(-1, 6).std(0) + 1e-6
    st = torch.from_numpy(_normalize(states, smu, ssd)).float()
    y = torch.from_numpy(_normalize(labels, ymu, ysd)).float()
    dev = torch.device(device)
    net = DeployableController(cfg["n_frames"], len(cfg["views"]), STATE_DIM, cfg["chunk"]).to(dev)
    opt, lossf = torch.optim.Adam(net.parameters(), lr=lr), nn.MSELoss()
    bs = min(64, len(images))
    net.train()
    for _ in range(steps):
        idx = torch.randint(0, len(images), (bs,))
        opt.zero_grad()
        lossf(net(images[idx].to(dev), st[idx].to(dev)), y[idx].to(dev)).backward()
        opt.step()
    net.eval()
    return net, (smu, ssd, ymu, ysd)


def predict_chunk(net, norm, images, state, device):
    smu, ssd, ymu, ysd = norm
    with torch.inference_mode():
        out = net(torch.from_numpy(images).float().unsqueeze(0).to(device),
                  torch.from_numpy(_normalize(state, smu, ssd)).float().unsqueeze(0).to(device))
    return out.squeeze(0).cpu().numpy() * ysd + ymu


def _stage_metrics(env, color, grasped_before, seen_grasp):
    cube = env.cube_pos(color)
    dest = env._dest_xy(env.target_dest, color)
    return {
        "approach": float(np.linalg.norm(env.ee_pos() - cube)) < 0.04,
        "grasp": env.grasped == color,
        "transport": env.grasped == color and cube[2] > 0.117 and np.linalg.norm(cube[:2] - dest) < 0.05,
        "release": seen_grasp and grasped_before == color and env.grasped is None
                   and np.linalg.norm(cube[:2] - dest) < 0.05,
    }


def rollout(net, norm, cfg, scene, cap, device, collect=False, replan=1, video_path=None):
    """Learner-only rollout.  DAgger labels are expert chunks from snapshots."""
    command, color = int(scene["command"]), _color(int(scene["command"]))
    positions = {c: np.asarray(v, float) for c, v in scene["positions"].items()}
    env = SO101PickPlaceTask()
    renderers = {v: mujoco.Renderer(env.model, height=IMG_RENDER, width=IMG_RENDER) for v in cfg["views"]}
    env.reset(command=command, positions=positions)
    prev, frames, visited, movie = HOME_QPOS.astype(np.float32), [], [], []
    queued, queue_i = None, 0
    metrics = {"approach": False, "grasp": False, "transport": False, "release": False}
    seen_grasp = False
    try:
        for t in range(cap):
            view = render_views(env, renderers, cfg["views"])
            frames.append(view)
            window = _window(np.asarray(frames, np.float32), len(frames) - 1, cfg["n_frames"])
            state = state_vec(env, prev)
            if collect:
                # The snapshot call is intentionally immediately before the
                # label query; expert_chunk_from_snapshot restores this learner
                # state before we execute its predicted action.
                visited.append({"imgs": window, "state": state,
                                "label": expert_chunk_from_snapshot(env, cfg["chunk"]),
                                "stage": stage_of(env, color)})
            if queued is None or t % replan == 0:
                queued, queue_i = predict_chunk(net, norm, window, state, device), 0
            action = np.clip(queued[min(queue_i, len(queued) - 1)], env.ctrl_range[:, 0], env.ctrl_range[:, 1])
            queue_i += 1
            grasped_before = env.grasped
            env.step(action)
            prev = action.astype(np.float32)
            seen_grasp = seen_grasp or env.grasped == color
            now = _stage_metrics(env, color, grasped_before, seen_grasp)
            for key in metrics:
                metrics[key] = metrics[key] or now[key]
            if video_path is not None:
                movie.append(np.rint(view[0].transpose(1, 2, 0) * 255).astype(np.uint8))
    finally:
        for renderer in renderers.values():
            renderer.close()
    if video_path is not None and movie:
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(video_path, movie, fps=25)
    return {**{k: int(v) for k, v in metrics.items()}, "success": int(env.success())}, visited


def _summary(rows):
    return {key: sum(row[key] for row in rows) for key in ("approach", "grasp", "transport", "release", "success")} | {"n": len(rows)}


def four_scene_dagger(scenes, cfg, seed, device, rounds, steps, cap, replan, video_dir):
    """One-seed promotion experiment; deliberately no held-out or multi-seed work."""
    aggregate = collect_demos(scenes, cfg["n_frames"], cfg["views"], cfg["chunk"], seed)
    curve, final = [], None
    for rnd in range(rounds):
        net, norm = train_policy(aggregate, cfg, seed, device, steps)
        rows, additions = [], {"imgs": [], "state": [], "label": [], "stage": []}
        for i, scene in enumerate(scenes):
            row, visited = rollout(net, norm, cfg, scene, cap, device, collect=rnd < rounds - 1, replan=replan)
            rows.append(row)
            for key in additions:
                additions[key].extend(v[key] for v in visited)
        curve.append({"round": rnd, "train_size": len(aggregate["state"]), "new_learner_states": len(additions["state"]),
                      **_summary(rows), "per_scene": rows})
        final = (net, norm)
        if rnd < rounds - 1:
            for key in aggregate:
                aggregate[key] = np.concatenate([aggregate[key], np.asarray(additions[key])])
    # Videos only for the final model: each success and the first representative failure.
    net, norm = final
    videos, failure_saved = [], False
    final_rows = []
    for i, scene in enumerate(scenes):
        dry, _ = rollout(net, norm, cfg, scene, cap, device, replan=replan)
        save = bool(dry["success"]) or not failure_saved
        path = Path(video_dir) / f"seed{seed}_{'success' if dry['success'] else 'failure'}_scene{i}.mp4" if save else None
        row, _ = rollout(net, norm, cfg, scene, cap, device, replan=replan, video_path=path)
        if not row["success"]:
            failure_saved = True
        final_rows.append(row)
        if path is not None:
            videos.append(str(path))
    return {"seed": seed, "config": cfg, "replan_actions": replan, "rounds": curve,
            "final": _summary(final_rows), "promotion_pass": sum(r["success"] for r in final_rows) >= 3,
            "videos": videos}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="single_frame,temporal,multiview")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--output", default="artifacts/truth_harness/deployable_supervised_gate.json")
    ap.add_argument("--four-scene", action="store_true", help="Run the one-seed promotion gate after a passing supervised gate JSON.")
    ap.add_argument("--architecture", default="temporal", choices=tuple(LADDER))
    ap.add_argument("--gate-json", help="Passing supervised-gate JSON produced by this script.")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--cap", type=int, default=220)
    ap.add_argument("--replan-actions", type=int, default=1, choices=(1, 2, 3, 4))
    ap.add_argument("--action-chunk", type=int, choices=(1, 4, 8), help="Override the architecture chunk; 4/8 only after temporal gate passes.")
    ap.add_argument("--video-dir", default="artifacts/truth_harness/deployable_rollouts")
    args = ap.parse_args()
    scenes = json.loads((ROOT / "scene_manifest.json").read_text())["scenes"]
    if args.four_scene:
        if not args.gate_json:
            raise SystemExit("--four-scene requires --gate-json from a passing supervised run")
        gate = json.loads(Path(args.gate_json).read_text())
        if not gate.get("temporal", {}).get("passed", False):
            raise SystemExit("Temporal 64-sample supervised gate has not passed; DAgger is blocked.")
        cfg = dict(LADDER[args.architecture])
        if args.action_chunk:
            if args.action_chunk > 1 and not gate["temporal"]["passed"]:
                raise SystemExit("Chunk sizes 4/8 are blocked until temporal supervised gate passes.")
            cfg["chunk"] = args.action_chunk
        result = four_scene_dagger(scenes, cfg, args.seed, args.device, args.rounds, args.steps,
                                   args.cap, args.replan_actions, args.video_dir)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"final": result["final"], "promotion_pass": result["promotion_pass"], "videos": result["videos"]}, indent=2))
        return
    results = {}
    for name in args.configs.split(","):
        cfg = LADDER[name]
        data = collect_demos(scenes, cfg["n_frames"], cfg["views"], cfg["chunk"], seed=args.seed)
        res = supervised_gate(data, cfg, args.seed, args.device, steps=args.steps)
        results[name] = res
        print(f"[{name}] " + json.dumps({k: res[k] for k in (
            "full_overfit_norm_mse", "full_overfit_mae_phys_rad", "image_only_overfit_norm_mse",
            "pred_change_swapped_image", "pred_change_shuffled_frames",
            "vision_pathway_ok", "temporal_order_sensitive", "state_alone_sufficient", "passed")}))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
    all_pass = all(r["passed"] for r in results.values())
    print("SUPERVISED GATE:", "PASS" if all_pass else "FAIL")


if __name__ == "__main__":
    main()
