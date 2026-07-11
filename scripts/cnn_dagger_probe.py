"""Single-round correctness + timing probe for scripts.controlled_dagger_cnn.

Verifies, before committing to a full 3-seed run, that:
  1. images + proprio inputs are valid (shape/range/finite)
  2. the state contains ONLY the requested signals; in --exclude-grasp mode it is
     exactly qpos6+qvel6+prev6 (all real-arm-measurable) with NO grasped bit and
     no cube/ee/dest leakage
  3. gradients and predicted actions are finite
  4. DAgger appends learner-visited states with reactive-expert labels
  5. held-out scenes are disjoint from the four training scenes
  6. MPS is actually used for the network

Then times one round to extrapolate the full runtime. Exit 0 == healthy.
Run with --exclude-grasp to probe the deployable condition.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from scripts import controlled_dagger_cnn as C
from scripts.controlled_dagger_mlp import held_out_scenes, STAGES
from tinyvla.task import SO101PickPlaceTask, HOME_QPOS

ROOT = Path("artifacts/truth_harness/datasets/command0_4")


def _fail(msg: str) -> None:
    print(f"PROBE FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude-grasp", action="store_true")
    args = ap.parse_args()
    include_grasp = not args.exclude_grasp
    sdim = C.state_dim(include_grasp)
    cond = "privileged_grasp" if include_grasp else "deployable_no_grasp"

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"condition = {cond} (state_dim={sdim}) | device = {dev}")
    if dev != "mps":
        print("WARNING: MPS not available; probe running on CPU")
    scenes = json.loads((ROOT / "scene_manifest.json").read_text())["scenes"]
    assert len(scenes) == 4, f"expected 4 training scenes, got {len(scenes)}"

    # ---- 1. images + proprio validity ------------------------------------
    Id, Sd, Yd, stg_d = C.reactive_demo_images(scenes, include_grasp)
    if Id.shape[1:] != (3, C.IMG_NET, C.IMG_NET) or not np.isfinite(Id).all():
        _fail(f"bad images {Id.shape} finite={np.isfinite(Id).all()}")
    if Id.min() < 0.0 or Id.max() > 1.0:
        _fail(f"pixels out of [0,1]: [{Id.min()},{Id.max()}]")
    if Sd.shape[1] != sdim or not np.isfinite(Sd).all():
        _fail(f"state shape {Sd.shape} != (*,{sdim}) / finite={np.isfinite(Sd).all()}")
    if not np.isfinite(Yd).all():
        _fail("non-finite labels")
    print(f"[1] images {Id.shape} in [{Id.min():.3f},{Id.max():.3f}]; states {Sd.shape}; labels {Yd.shape}  OK")

    # ---- 2. state composition + no leakage -------------------------------
    env = SO101PickPlaceTask()
    marker = np.array([0.219137, -0.061921])
    env.reset(command=0, positions={"red": marker, "blue": marker + np.array([0.07, 0.0])})
    prev = (HOME_QPOS + 0.01).astype(np.float32)  # distinctive prev action
    st = C.state_vec(env, prev, include_grasp)
    if st.shape[0] != sdim:
        _fail(f"state dim {st.shape[0]} != {sdim}")
    real = np.concatenate([env.data.qpos[:6], env.data.qvel[:6], prev]).astype(np.float32)
    if not np.allclose(st[:18], real, atol=0):
        _fail("first 18 state dims are not exactly qpos6+qvel6+prev6")
    if include_grasp:
        if st.shape[0] != 19 or st[18] not in (0.0, 1.0):
            _fail("grasp bit missing/invalid in privileged mode")
        note = "qpos6+qvel6+prev6 + grasped1 (PRIVILEGED)"
    else:
        note = "qpos6(incl gripper joint pos)+qvel6+prev6 — all real-arm-measurable; NO grasped bit"
    cube, ee = env.cube_pos("red"), env.ee_pos()
    leaked = [v for v in list(cube) + list(ee) if np.any(np.abs(st - v) < 1e-6)]
    if leaked:
        _fail(f"spatial values leaked into state: {leaked}")
    print(f"[2] state = {note}; cube/ee/dest NOT in state  OK")

    # ---- 5. held-out disjoint --------------------------------------------
    ho = held_out_scenes(20, 0)
    tl = [C.layout_vec(s) for s in scenes]
    gaps = sorted(C.nearest_train_dist(h, tl) for h in ho)
    if gaps[0] < 1e-3:
        _fail(f"held-out coincides with train (gap {gaps[0]:.2e})")
    print(f"[5] {len(ho)} held-out disjoint; layout-dist to nearest train: "
          f"min {gaps[0]:.3f} med {gaps[len(gaps)//2]:.3f} max {gaps[-1]:.3f} m  OK")

    # ---- 3 + 6. finite grads, MPS ----------------------------------------
    net = C.ImageStatePolicy(sdim).to(dev)
    if next(net.parameters()).device.type != dev:
        _fail(f"net not on {dev}")
    smu, ssd = Sd.mean(0), Sd.std(0) + 1e-6
    ymu, ysd = Yd.mean(0), Yd.std(0) + 1e-6
    I = torch.from_numpy(Id[:64]).float().to(dev)
    S = torch.from_numpy((Sd[:64] - smu) / ssd).float().to(dev)
    Y = torch.from_numpy((Yd[:64] - ymu) / ysd).float().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    out = net(I, S)
    if out.device.type != dev:
        _fail(f"forward on {out.device.type}, not {dev}")
    loss = nn.MSELoss()(out, Y); loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    if not all(torch.isfinite(g).all() for g in grads) or not torch.isfinite(loss).all():
        _fail("non-finite grads/loss")
    opt.step()
    print(f"[3+6] net+forward on {dev}, {len(grads)} grad tensors finite, loss {loss.item():.4f}  OK")

    # ---- 4. DAgger append -------------------------------------------------
    net_s, norm = C.train_cnn(Id, Sd, Yd, 0, 300, dev, sdim)
    roll, visited = C.cnn_rollout(net_s, norm, scenes[0], 40, dev, True, include_grasp)
    if len(visited) != 40:
        _fail(f"collect produced {len(visited)} states, expected 40")
    v0 = visited[0]
    if not (np.isfinite(v0["img"]).all() and np.isfinite(v0["state"]).all() and np.isfinite(v0["expert"]).all()):
        _fail("non-finite visited img/state/expert")
    if v0["state"].shape[0] != sdim:
        _fail(f"visited state dim {v0['state'].shape[0]} != {sdim}")
    if v0["stage"] not in STAGES:
        _fail(f"bad stage {v0['stage']}")
    env2 = SO101PickPlaceTask()
    env2.reset(command=0, positions={c: np.asarray(scenes[0]["positions"][c], float) for c in ("red", "blue")})
    exp0 = env2.reactive_action(gain=0.25, max_dq=0.03).astype(np.float32)
    if not np.allclose(exp0, v0["expert"], atol=1e-5):
        _fail("visited expert label != reactive_action at that state")
    acts = np.array([C.cnn_predict(net_s, norm, v["img"], v["state"], dev) for v in visited[:5]])
    if not np.isfinite(acts).all():
        _fail("non-finite predicted actions")
    print(f"[4] collected {len(visited)} learner-visited states, reactive-labelled, finite, "
          f"appends as 'dagger'  OK; roll success={roll['success']}")

    # ---- timing extrapolation --------------------------------------------
    STEPS, CAP, ROUNDS, SEEDS, HELD = 4000, 200, 5, 3, 20
    t0 = time.time(); C.train_cnn(Id, Sd, Yd, 0, 400, dev, sdim); tr_step = (time.time() - t0) / 400
    t0 = time.time(); C.cnn_rollout(net_s, norm, scenes[0], CAP, dev, True, include_grasp); roll_s = time.time() - t0
    _, vis = C.cnn_rollout(net_s, norm, scenes[0], CAP, dev, True, include_grasp)
    sub = vis[::12]
    t0 = time.time()
    for v in sub[:3]:
        C.expert_takeover(v["snapshot"], 0, CAP)
    tk = (time.time() - t0) / max(1, len(sub[:3]))
    train_full = tr_step * STEPS
    dagger_seed = ROUNDS * (train_full + 4 * roll_s + tk * len(sub) * 4)
    # A + B: 2 trains + (mem 4 + held 20) rollouts each; C eval: (4 + 20) rollouts
    controls_seed = 2 * (train_full + (4 + HELD) * roll_s) + (4 + HELD) * roll_s
    total = SEEDS * (dagger_seed + controls_seed)
    est = {"train_s_per_step": round(tr_step, 4), "train_full_s": round(train_full, 1),
           "rollout_s_cap200": round(roll_s, 1), "per_seed_min": round((dagger_seed + controls_seed) / 60, 1),
           "estimated_total_min": round(total / 60, 1)}
    print("[timing] " + json.dumps(est))
    print("PROBE OK")
    Path("artifacts/truth_harness/cnn_probe.json").write_text(
        json.dumps({"condition": cond, **est}, indent=2) + "\n")


if __name__ == "__main__":
    main()
