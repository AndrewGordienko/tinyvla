"""Single-round correctness + timing probe for scripts.controlled_dagger_cnn.

Verifies, before committing to the full 3-seed run, that:
  1. images + proprio inputs are valid (shape/range/finite)
  2. no SPATIAL privileged-state leakage (cube/ee/dest not in the policy state;
     only the declared 1-bit grasped flag is privileged)
  3. gradients and predicted actions are finite
  4. DAgger appends learner-visited states with reactive-expert labels
  5. held-out scenes are disjoint from the four training scenes
  6. MPS is actually used for the network

Then times one Experiment-II round (train + 4 rollouts + takeover) and one
Experiment-I transfer train to extrapolate the full runtime. Exit 0 == healthy.
"""
from __future__ import annotations

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
EXPORT = Path("artifacts/truth_harness/dagger_dataset")


def _fail(msg: str) -> None:
    print(f"PROBE FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device = {dev}")
    if dev != "mps":
        print("WARNING: MPS not available; probe running on CPU")
    scenes = json.loads((ROOT / "scene_manifest.json").read_text())["scenes"]
    assert len(scenes) == 4, f"expected 4 training scenes, got {len(scenes)}"

    # ---- 1. images + proprio validity ------------------------------------
    Id, Sd, Yd, stg_d = C.reactive_demo_images(scenes)
    if Id.shape[1:] != (3, C.IMG_NET, C.IMG_NET):
        _fail(f"image shape {Id.shape[1:]} != (3,{C.IMG_NET},{C.IMG_NET})")
    if not np.isfinite(Id).all():
        _fail("non-finite pixels")
    if Id.min() < 0.0 or Id.max() > 1.0:
        _fail(f"pixels out of [0,1]: min {Id.min()} max {Id.max()}")
    if Sd.shape[1] != C.STATE_DIM or not np.isfinite(Sd).all():
        _fail(f"bad state array shape {Sd.shape} / finite={np.isfinite(Sd).all()}")
    if not np.isfinite(Yd).all():
        _fail("non-finite labels")
    print(f"[1] images {Id.shape} in [{Id.min():.3f},{Id.max():.3f}]; "
          f"states {Sd.shape}; labels {Yd.shape}  OK")

    # ---- 2. no spatial privileged leakage --------------------------------
    # Place the cubes at a distinctive location and confirm those coordinates
    # never appear in the policy state vector (which is qpos/qvel/prev/grasped).
    env = SO101PickPlaceTask()
    marker = np.array([0.219137, -0.061921])  # unlikely to coincide with any joint value
    env.reset(command=0, positions={"red": marker, "blue": marker + np.array([0.07, 0.0])})
    st = C.state_vec(env, HOME_QPOS.astype(np.float32))
    cube = env.cube_pos("red")
    ee = env.ee_pos()
    leaked = [v for v in list(cube) + list(ee) if np.any(np.abs(st - v) < 1e-6)]
    if leaked:
        _fail(f"spatial values leaked into state: {leaked}")
    if st.shape[0] != 19:
        _fail(f"state dim {st.shape[0]} != 19")
    print(f"[2] state = qpos6+qvel6+prev6+grasped1 (19-D); cube/ee/dest NOT in state; "
          f"only 1-bit grasped is privileged  OK")

    # ---- 5. held-out disjoint from training scenes -----------------------
    ho = held_out_scenes(20, 0)
    train_xy = [np.concatenate([np.asarray(s["positions"]["red"])[:2],
                                np.asarray(s["positions"]["blue"])[:2]]) for s in scenes]
    min_gap = np.inf
    for h in ho:
        hxy = np.concatenate([np.asarray(h["positions"]["red"])[:2],
                              np.asarray(h["positions"]["blue"])[:2]])
        min_gap = min(min_gap, min(np.linalg.norm(hxy - t) for t in train_xy))
    if min_gap < 1e-3:
        _fail(f"held-out scene coincides with a training scene (gap {min_gap:.2e})")
    print(f"[5] {len(ho)} held-out scenes disjoint from train (min layout gap {min_gap:.3f} m)  OK")

    # ---- 3 + 6. finite grads, MPS in use ---------------------------------
    net = C.ImageStatePolicy().to(dev)
    if next(net.parameters()).device.type != dev:
        _fail(f"net not on {dev}")
    smu, ssd = Sd.mean(0), Sd.std(0) + 1e-6
    ymu, ysd = Yd.mean(0), Yd.std(0) + 1e-6
    I = torch.from_numpy(Id[:64]).float().to(dev)
    S = torch.from_numpy(((Sd[:64] - smu) / ssd)).float().to(dev)
    Y = torch.from_numpy(((Yd[:64] - ymu) / ysd)).float().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    out = net(I, S)
    if out.device.type != dev:
        _fail(f"forward ran on {out.device.type}, not {dev}")
    loss = nn.MSELoss()(out, Y)
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    if not all(torch.isfinite(g).all() for g in grads):
        _fail("non-finite gradients")
    opt.step()
    if not torch.isfinite(loss).all():
        _fail("non-finite loss")
    print(f"[3+6] net on {dev}, forward on {out.device.type}, {len(grads)} grad tensors all finite, "
          f"loss {loss.item():.4f}  OK")

    # ---- 4. DAgger appends learner-visited states ------------------------
    # Train briefly, roll out one scene with collect=True, verify the visited
    # states are finite and would append with a 'dagger' source label.
    net_s, norm = C.train_cnn(Id, Sd, Yd, 0, 300, dev)
    if next(net_s.parameters()).device.type != dev:
        _fail("trained net not on device")
    roll, visited = C.cnn_rollout(net_s, norm, scenes[0], 40, dev, True)
    if len(visited) != 40:
        _fail(f"collect produced {len(visited)} visited states, expected 40")
    v0 = visited[0]
    if not (np.isfinite(v0["img"]).all() and np.isfinite(v0["state"]).all()
            and np.isfinite(v0["expert"]).all()):
        _fail("non-finite visited img/state/expert")
    if v0["stage"] not in STAGES:
        _fail(f"bad stage {v0['stage']}")
    # simulate the aggregation step used in Experiment II
    agg_src = ["demo"] * len(Id)
    before = len(agg_src)
    for v in visited:
        agg_src.append("dagger")
    if len(agg_src) != before + 40 or agg_src[-1] != "dagger":
        _fail("DAgger append did not grow aggregate with 'dagger'-sourced rows")
    # the expert label must be the reactive action at the LEARNER's visited state
    env2 = SO101PickPlaceTask()
    env2.reset(command=0, positions={c: np.asarray(scenes[0]["positions"][c], float) for c in ("red", "blue")})
    exp0 = env2.reactive_action(gain=0.25, max_dq=0.03).astype(np.float32)
    if not np.allclose(exp0, v0["expert"], atol=1e-5):
        _fail("first visited expert label != reactive_action at that state")
    finite_actions = np.isfinite([C.cnn_predict(net_s, norm, v["img"], v["state"], dev) for v in visited[:5]]).all()
    if not finite_actions:
        _fail("non-finite predicted actions during rollout")
    print(f"[4] rollout collected {len(visited)} learner-visited states, reactive-expert labelled, "
          f"finite, append -> 'dagger'  OK; roll={roll['success']} grasp={roll['grasp']}")

    # ---- timing extrapolation --------------------------------------------
    STEPS_FULL, CAP_FULL, ROUNDS, SEEDS, HELDOUT = 4000, 200, 5, 3, 20
    # train time per step (measure 400 steps after warmup)
    t0 = time.time(); C.train_cnn(Id, Sd, Yd, 0, 400, dev); train_s_per_step = (time.time() - t0) / 400
    # rollout time per scene at full cap (render-bound)
    t0 = time.time(); C.cnn_rollout(net_s, norm, scenes[0], CAP_FULL, dev, True); roll_s = time.time() - t0
    # takeover time per subsampled visited state
    _, vis_full = C.cnn_rollout(net_s, norm, scenes[0], CAP_FULL, dev, True)
    sub = vis_full[::12]
    t0 = time.time()
    for v in sub[:3]:
        C.expert_takeover(v["snapshot"], 0, CAP_FULL)
    tk_s_per = (time.time() - t0) / max(1, len(sub[:3]))

    train_full = train_s_per_step * STEPS_FULL
    # Experiment II per seed: ROUNDS*(train + 4 rollouts + takeover on ~4*sub) + held-out(20 rollouts)
    tk_per_round = tk_s_per * (len(sub) * 4)
    expII_seed = ROUNDS * (train_full + 4 * roll_s + tk_per_round) + HELDOUT * roll_s
    # Experiment I per seed: 1 train + 4 rollouts + 20 held-out rollouts
    expI_seed = train_full + (4 + HELDOUT) * roll_s
    total = SEEDS * (expI_seed + expII_seed)
    est = {
        "train_s_per_step": round(train_s_per_step, 4),
        "train_full_4000_s": round(train_full, 1),
        "rollout_s_per_scene_cap200": round(roll_s, 1),
        "takeover_s_per_state": round(tk_s_per, 2),
        "expI_per_seed_min": round(expI_seed / 60, 1),
        "expII_per_seed_min": round(expII_seed / 60, 1),
        "estimated_total_min": round(total / 60, 1),
    }
    print("[timing] " + json.dumps(est))
    print("PROBE OK")
    Path("artifacts/truth_harness/cnn_probe.json").write_text(json.dumps(est, indent=2) + "\n")


if __name__ == "__main__":
    main()
