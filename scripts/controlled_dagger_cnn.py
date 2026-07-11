"""Image/state CNN control: does on-policy recovery survive partial observability?

Two experiments (command 0, four memorized scenes, canonical 4 cm, learner-only):
  I   transfer   train the CNN on the EXACT winning privileged-DAgger states
                 (rendered images from the exported snapshots) + reactive labels
  II  on-policy  the CNN runs its OWN DAgger loop (its own rollout distribution)

Inputs: front image + qpos + qvel + previous action + grasped flag (grasped is
privileged and labelled as such). Output: one-step absolute 6-D actuator target.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import mujoco

from tinyvla.task import SO101PickPlaceTask, HOME_QPOS
from tinyvla.determinism import seed_everything
from scripts.controlled_dagger_mlp import stage_of, expert_takeover, held_out_scenes, STAGES, _color
from scripts.perturbation_recovery import _restore

IMG_RENDER, IMG_NET, EP_LEN, DWELL, LIFT_Z = 256, 84, 220, 8, 0.117
STATE_DIM = 19  # qpos6 + qvel6 + prev6 + grasped1(privileged)


def render(env, renderer):
    renderer.update_scene(env.data, camera="front")     # MUST update before render
    img = renderer.render()
    small = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    return nn.functional.interpolate(small, size=(IMG_NET, IMG_NET), mode="bilinear",
                                     align_corners=False).squeeze(0).numpy().astype(np.float32)


def state_vec(env, prev):
    g = 1.0 if env.grasped is not None else 0.0
    return np.concatenate([env.data.qpos[:6], env.data.qvel[:6], prev, [g]]).astype(np.float32)


class ImageStatePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(), nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.smlp = nn.Sequential(nn.Linear(STATE_DIM, 128), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(64 + 128, 128), nn.ReLU(), nn.Linear(128, 6))

    def forward(self, img, st):
        return self.head(torch.cat([self.cnn(img), self.smlp(st)], dim=1))


def train_cnn(imgs, states, labels, seed, steps, device, lr=1e-3):
    seed_everything(seed)
    dev = torch.device(device)
    smu, ssd = states.mean(0), states.std(0) + 1e-6
    ymu, ysd = labels.mean(0), labels.std(0) + 1e-6
    I = torch.from_numpy(imgs).float()
    S = torch.from_numpy((states - smu) / ssd).float()
    Y = torch.from_numpy((labels - ymu) / ysd).float()
    net = ImageStatePolicy().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr); lossf = nn.MSELoss()
    bs = min(128, len(I))
    net.train()
    for _ in range(steps):
        idx = torch.randint(0, len(I), (bs,))
        opt.zero_grad()
        loss = lossf(net(I[idx].to(dev), S[idx].to(dev)), Y[idx].to(dev))
        loss.backward(); opt.step()
    net.eval()
    return net, (smu, ssd, ymu, ysd)


def cnn_predict(net, norm, img, st, device):
    smu, ssd, ymu, ysd = norm
    with torch.inference_mode():
        y = net(torch.from_numpy(img).float().unsqueeze(0).to(device),
                torch.from_numpy((st - smu) / ssd).float().unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
    return y * ysd + ymu


def cnn_rollout(net, norm, scene, cap, device, collect):
    command = int(scene["command"]); color = _color(command)
    positions = {c: np.asarray(v, float) for c, v in scene["positions"].items()}
    env = SO101PickPlaceTask()
    renderer = mujoco.Renderer(env.model, height=IMG_RENDER, width=IMG_RENDER)
    lo, hi = env.ctrl_range[:, 0], env.ctrl_range[:, 1]
    env.reset(command=command, positions=positions)
    prev = HOME_QPOS.astype(np.float32)
    dmin, grasp_t, lifted, carried, visited = float("inf"), -1, False, False, []
    for t in range(cap):
        img = render(env, renderer); st = state_vec(env, prev)
        action = np.clip(cnn_predict(net, norm, img, st, device), lo, hi).astype(np.float32)
        if collect:
            expert = env.reactive_action(gain=0.25, max_dq=0.03).astype(np.float32)
            visited.append({"img": img, "state": st, "expert": expert, "stage": stage_of(env, color),
                            "snapshot": {"qpos": env.data.qpos.copy(), "qvel": env.data.qvel.copy(),
                                         "grasped": env.grasped,
                                         "off_pos": getattr(env, "_off_pos", None),
                                         "off_quat": getattr(env, "_off_quat", None)}})
        dmin = min(dmin, float(np.linalg.norm(env.ee_pos() - env.cube_pos(color))))
        env.step(action); prev = action
        if grasp_t < 0 and env.grasped is not None:
            grasp_t = t
        if env.cube_pos(color)[2] > LIFT_Z:
            lifted = True
        if np.linalg.norm(env.cube_pos(color)[:2] - env._dest_xy(env.target_dest, color)) < 0.05:
            carried = True
    renderer.close()
    return {"success": bool(env.success()), "min_dist": round(dmin, 4), "grasp": grasp_t >= 0,
            "lift": lifted, "carry": carried}, visited


def render_from_export(npz, device):
    """Restore each exported snapshot and render its image + state + label."""
    d = np.load(npz)
    env = SO101PickPlaceTask()
    renderer = mujoco.Renderer(env.model, height=IMG_RENDER, width=IMG_RENDER)
    imgs, states, labels = [], [], []
    for i in range(len(d["sample_id"])):
        snap = {"qpos": d["qpos"][i], "qvel": d["qvel"][i],
                "grasped": d["grasped"][i] or None,
                "off_pos": None if np.isnan(d["off_pos"][i]).any() else d["off_pos"][i],
                "off_quat": None if np.isnan(d["off_quat"][i]).any() else d["off_quat"][i]}
        env.reset(command=int(d["command"][i]), positions={c: np.asarray([0.2, 0.0]) for c in ("red", "blue")})
        _restore(env, snap)
        imgs.append(render(env, renderer))
        prev = d["feat"][i][22:28]
        states.append(np.concatenate([d["qpos"][i][:6], d["qvel"][i][:6], prev,
                                      [1.0 if snap["grasped"] else 0.0]]).astype(np.float32))
        labels.append(d["label"][i])
    renderer.close()
    return np.asarray(imgs, np.float32), np.asarray(states, np.float32), np.asarray(labels, np.float32)


def reactive_demo_images(scenes):
    env = SO101PickPlaceTask()
    renderer = mujoco.Renderer(env.model, height=IMG_RENDER, width=IMG_RENDER)
    imgs, states, labels, stages = [], [], [], []
    for scene in scenes:
        command = int(scene["command"]); color = _color(command)
        positions = {c: np.asarray(v, float) for c, v in scene["positions"].items()}
        env.reset(command=command, positions=positions)
        prev = HOME_QPOS.astype(np.float32); dwell = 0
        for _ in range(EP_LEN):
            a = env.reactive_action(gain=0.25, max_dq=0.03).astype(np.float32)
            imgs.append(render(env, renderer)); states.append(state_vec(env, prev))
            labels.append(a); stages.append(stage_of(env, color))
            env.step(a); prev = a
            dwell = dwell + 1 if env.success() else 0
            if dwell >= DWELL:
                break
    renderer.close()
    return (np.asarray(imgs, np.float32), np.asarray(states, np.float32),
            np.asarray(labels, np.float32), stages)


def _summ(rolls):
    return {"success": sum(r["success"] for r in rolls), "n": len(rolls),
            "grasp": sum(r["grasp"] for r in rolls), "lift": sum(r["lift"] for r in rolls),
            "carry": sum(r["carry"] for r in rolls)}


def evaluate(net, norm, scenes, cap, device):
    return _summ([cnn_rollout(net, norm, s, cap, device, False)[0] for s in scenes])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="artifacts/truth_harness/datasets/command0_4")
    ap.add_argument("--export-dir", default="artifacts/truth_harness/dagger_dataset")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--cap", type=int, default=200)
    ap.add_argument("--stage-cap", type=int, default=500)
    ap.add_argument("--held-out", type=int, default=20)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--output", default="artifacts/truth_harness/controlled_dagger_cnn.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    dev = args.device
    scenes = json.loads((Path(args.root) / "scene_manifest.json").read_text())["scenes"]
    ho = held_out_scenes(args.held_out, 0)
    command0 = int(scenes[0]["command"])

    net_params = sum(p.numel() for p in ImageStatePolicy().parameters())
    Id, Sd, Yd, stg_d = reactive_demo_images(scenes)

    out = {"cnn_params": int(net_params), "img_net": IMG_NET, "seeds": seeds,
           "state_note": "grasped flag is privileged (1 bit not visible in a single frame)"}

    # ---- Experiment I: transfer from the exact winning privileged-DAgger aggregate
    expI = {}
    for seed in seeds:
        npz = Path(args.export_dir) / f"seed{seed}.npz"
        if not npz.exists():
            expI[str(seed)] = "missing_export"; continue
        Ie, Se, Ye = render_from_export(str(npz), dev)
        t0 = time.time()
        net, norm = train_cnn(Ie, Se, Ye, seed, args.steps, dev)
        expI[str(seed)] = {"train_size": int(len(Ie)), "train_s": round(time.time() - t0, 1),
                           "memorized": evaluate(net, norm, scenes, args.cap, dev),
                           "held_out": evaluate(net, norm, ho, args.cap, dev)}
    out["experiment_I_transfer"] = expI

    # ---- Experiment II: CNN-specific on-policy DAgger
    expII = {}
    for seed in seeds:
        aI = [x for x in Id]; aS = [x for x in Sd]; aY = [x for x in Yd]; aStg = list(stg_d)
        aSrc = ["demo"] * len(Id)
        curve = []
        for rnd in range(args.rounds):
            counts = {}; keep = []
            for i, s in enumerate(aStg):
                counts.setdefault(s, 0)
                if aSrc[i] == "demo" or counts[s] < args.stage_cap:
                    keep.append(i); counts[s] += 1
            net, norm = train_cnn(np.asarray([aI[i] for i in keep], np.float32),
                                  np.asarray([aS[i] for i in keep], np.float32),
                                  np.asarray([aY[i] for i in keep], np.float32), seed, args.steps, dev)
            rolls, visited = [], []
            for s in scenes:
                r, vis = cnn_rollout(net, norm, s, args.cap, dev, True)
                rolls.append(r); visited += vis
            # takeover from CNN-visited states (subsampled)
            tk = {st: {"n": 0, "rec": 0} for st in STAGES}
            for v in visited[::12]:
                tk[v["stage"]]["n"] += 1
                tk[v["stage"]]["rec"] += int(expert_takeover(v["snapshot"], command0, args.cap))
            curve.append({"round": rnd, "train_size": len(keep), "new_states": len(visited),
                          **_summ(rolls),
                          "takeover": {st: (round(tk[st]["rec"] / tk[st]["n"], 2) if tk[st]["n"] else None)
                                       for st in STAGES}})
            # Only aggregate for a round that will be trained on; the final round's
            # collection is never used (net/held-out use this round's keep), so skip it.
            if rnd < args.rounds - 1:
                for v in visited:
                    aI.append(v["img"]); aS.append(v["state"]); aY.append(v["expert"])
                    aStg.append(v["stage"]); aSrc.append("dagger")
        ho_final = evaluate(net, norm, ho, args.cap, dev)
        expII[str(seed)] = {"curve": curve, "held_out": ho_final}
    out["experiment_II_cnn_dagger"] = expII

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({
        "cnn_params": net_params,
        "expI_transfer": {s: (expI[s].get("memorized", {}).get("success") if isinstance(expI[s], dict) else expI[s])
                          for s in expI},
        "expI_heldout": {s: (expI[s].get("held_out", {}).get("success") if isinstance(expI[s], dict) else None)
                         for s in expI},
        "expII_success_by_round": {s: [c["success"] for c in expII[s]["curve"]] for s in expII},
        "expII_heldout": {s: expII[s]["held_out"]["success"] for s in expII},
    }, indent=2))


if __name__ == "__main__":
    main()
