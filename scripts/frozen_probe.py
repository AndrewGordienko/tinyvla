"""Frozen-feature spatial probe: does SmolVLA already SEE the cube accurately?

Generates a deterministic set of randomized command-0 scenes (both cubes random,
arm at home), renders the initial front image, and asks how well the RED cube's
xyz can be decoded from:
  - raw pixels via a small CNN (upper bound: is the info in the image at all?)
  - frozen SmolVLA vision-encoder tokens (mean-pooled)
  - frozen SmolVLA connector output = image tokens the action expert consumes
with a linear probe and a 2-layer MLP probe. Held-out split is by scene (distinct
random positions), never the original four, so probes cannot memorize identities.

Interpretation:
  raw CNN accurate but frozen features not -> frozen representation is the bottleneck
  frozen features accurate (linear/MLP)     -> info exists; expert/optim fails to use it
  neither accurate                          -> inspect rendering/resolution/variation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import mujoco

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from tinyvla.runtime import load_runtime
from tinyvla.paths import MODELS_ROOT
from tinyvla.task import SO101PickPlaceTask
from tinyvla.eval_closedloop import build_obs, IMG
from tinyvla.determinism import seed_everything


def gen_dataset(n: int, seed: int, model_path, meta, root, device):
    """Return raw images (N,3,64,64), frozen reps, and red-cube xyz targets."""
    dev = torch.device(device)
    r = load_runtime(model_path, meta=meta, dataset_root=str(root), device=dev,
                     base_checkpoint=True, stats_source="dataset")
    pol, pre = r.policy.eval(), r.preprocessor
    vlm = pol.model.vlm_with_expert
    vision_model = vlm.vlm.model.vision_model

    env = SO101PickPlaceTask(seed=seed)
    renderer = mujoco.Renderer(env.model, height=IMG, width=IMG)
    raw_imgs, enc_feats, con_feats, targets = [], [], [], []
    with torch.inference_mode():
        for i in range(n):
            env.rng = np.random.default_rng(seed + i)
            env.reset(command=0)                       # both cubes randomized; target = red
            renderer.update_scene(env.data, camera="front")  # MUST update before render
            raw = renderer.render()                    # (IMG, IMG, 3) uint8 of THIS scene
            small = torch.from_numpy(raw).permute(2, 0, 1).float().div(255.0)
            small = nn.functional.interpolate(small.unsqueeze(0), size=(64, 64),
                                              mode="bilinear", align_corners=False).squeeze(0)
            raw_imgs.append(small.numpy())
            b = pre(build_obs(env, renderer, env.instruction, dev))
            imgs, _ = pol.prepare_images(b)
            enc = vision_model(imgs[0]).last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()   # (768,)
            con = vlm.embed_image(imgs[0]).mean(dim=1).squeeze(0).cpu().numpy()                  # (960,)
            enc_feats.append(enc); con_feats.append(con)
            targets.append(env.cube_pos("red").astype(np.float32))
    renderer.close()
    del r
    return (np.asarray(raw_imgs, dtype=np.float32), np.asarray(enc_feats, dtype=np.float32),
            np.asarray(con_feats, dtype=np.float32), np.asarray(targets, dtype=np.float32))


def _metrics(pred, tgt):
    err = pred - tgt
    eucl = np.linalg.norm(err, axis=1)                 # meters
    xy = np.linalg.norm(err[:, :2], axis=1)
    return {
        "x_mae_cm": round(float(np.abs(err[:, 0]).mean()) * 100, 3),
        "y_mae_cm": round(float(np.abs(err[:, 1]).mean()) * 100, 3),
        "z_mae_cm": round(float(np.abs(err[:, 2]).mean()) * 100, 3),
        "euclidean_mean_cm": round(float(eucl.mean()) * 100, 3),
        "euclidean_median_cm": round(float(np.median(eucl)) * 100, 3),
        "xy_mean_cm": round(float(xy.mean()) * 100, 3),
        "pct_below_1cm": round(float((eucl < 0.01).mean()) * 100, 1),
        "pct_below_2cm": round(float((eucl < 0.02).mean()) * 100, 1),
        "pct_below_4cm": round(float((eucl < 0.04).mean()) * 100, 1),
    }


def fit_head(Xtr, Ytr, Xte, Yte, kind, steps=3000, lr=1e-3):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    tmu, tsd = Ytr.mean(0), Ytr.std(0) + 1e-6
    xtr = torch.from_numpy((Xtr - mu) / sd).float()
    ytr = torch.from_numpy((Ytr - tmu) / tsd).float()
    xte = torch.from_numpy((Xte - mu) / sd).float()
    d = Xtr.shape[1]
    if kind == "linear":
        head = nn.Linear(d, 3)
    else:
        head = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 3))
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    lossf = nn.MSELoss()
    for _ in range(steps):
        opt.zero_grad(); loss = lossf(head(xtr), ytr); loss.backward(); opt.step()
    with torch.inference_mode():
        pred = head(xte).numpy() * tsd + tmu
    return _metrics(pred, Yte)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.f = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.f(self.c(x))


def fit_cnn(Xtr, Ytr, Xte, Yte, steps=3000, lr=1e-3):
    tmu, tsd = Ytr.mean(0), Ytr.std(0) + 1e-6
    xtr = torch.from_numpy(Xtr).float(); ytr = torch.from_numpy((Ytr - tmu) / tsd).float()
    xte = torch.from_numpy(Xte).float()
    net = SmallCNN(); opt = torch.optim.Adam(net.parameters(), lr=lr); lossf = nn.MSELoss()
    bs = 64
    for step in range(steps):
        idx = torch.randint(0, len(xtr), (bs,))
        opt.zero_grad(); loss = lossf(net(xtr[idx]), ytr[idx]); loss.backward(); opt.step()
    with torch.inference_mode():
        pred = net(xte).numpy() * tsd + tmu
    return _metrics(pred, Yte)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--repo-id", default="local/truth_gate_command0_4")
    ap.add_argument("--root", default="artifacts/truth_harness/datasets/command0_4")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--output", default="artifacts/truth_harness/frozen_probe.json")
    args = ap.parse_args()
    seed_everything(args.seed)
    meta = LeRobotDatasetMetadata(args.repo_id, root=str(args.root))
    raw, enc, con, tgt = gen_dataset(args.n, args.seed, MODELS_ROOT / "smolvla_base",
                                     meta, args.root, args.device)
    n = len(tgt); ntr, nval = int(n * 0.6), int(n * 0.2)
    idx = np.random.default_rng(args.seed).permutation(n)
    tr, te = idx[:ntr], idx[ntr + nval:]               # train vs held-out (val reserved)

    result = {"probe": "frozen_spatial", "n": n, "n_train": len(tr), "n_heldout": len(te),
              "target": "red_cube_xyz_world",
              "note": "initial-scene (arm-at-home) red-cube localization; z ~constant so "
                      "xy is the meaningful axis; dynamic-state spatial-token probe is future work",
              "constant_mean_baseline": _metrics(np.tile(tgt[tr].mean(0), (len(te), 1)), tgt[te]),
              "representations": {}}
    result["representations"]["raw_pixels_cnn"] = {
        "dim": "3x64x64", "cnn": fit_cnn(raw[tr], tgt[tr], raw[te], tgt[te])}
    result["representations"]["frozen_vision_encoder_meanpool"] = {
        "dim": int(enc.shape[1]),
        "linear": fit_head(enc[tr], tgt[tr], enc[te], tgt[te], "linear"),
        "mlp": fit_head(enc[tr], tgt[tr], enc[te], tgt[te], "mlp")}
    result["representations"]["frozen_connector_meanpool"] = {
        "dim": int(con.shape[1]),
        "linear": fit_head(con[tr], tgt[tr], con[te], tgt[te], "linear"),
        "mlp": fit_head(con[tr], tgt[tr], con[te], tgt[te], "mlp")}
    # target spread for context (how hard is the localization)
    result["target_spread_cm"] = {ax: round(float(tgt[:, i].std()) * 100, 2)
                                   for i, ax in enumerate("xyz")}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
