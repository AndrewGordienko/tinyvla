"""Separate hidden-state label ambiguity from ordinary covariate shift.

The privileged-state MLP fits the four command-0 demonstrations to ~0 loss yet is
not closed-loop stable. Those demos come from the STATEFUL expert_action() whose
target depends on hidden phase/phase_t, which the policy never sees. This script
runs the controls that decide whether the 1/4 failure is (a) non-Markov label
ambiguity, (b) ordinary off-trajectory compounding, or (c) both:

  1  ambiguity   nearest-neighbour audit in privileged-state space + MAX action err
  2  oracle      privileged MLP + one-hot phase + phase_t + step_idx (diagnostic)
  3  reactive    identical MLP trained on stateless reactive_action() labels

Same architecture / optimizer / updates / normalization / seed / evaluator across
2 and 3 so the only variable is the label source.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from tinyvla.task import SO101PickPlaceTask, COMMANDS, HOME_QPOS, GRASP_RADIUS, GRIP_GRAB
from tinyvla.determinism import seed_everything
from scripts.baseline_mlp import features, MLP, rollout, FEAT_DIM

EP_LEN, DWELL, N_PHASE = 220, 8, 7
PHASE_NAMES = {0: "approach", 1: "descend", 2: "close", 3: "lift", 4: "carry", 5: "descend2", 6: "release"}
TRANSITIONS = {"approach->descend": (0, 1), "descend->close": (1, 2), "close->lift": (2, 3),
               "carry->descend2": (4, 5), "descend2->release": (5, 6)}


def _target_color(command: int) -> str:
    return COMMANDS[command]["steps"][0][0]


def gen(scenes, kind: str):
    """Run an expert on the four scenes; record features, actions, phase context."""
    env = SO101PickPlaceTask()
    X, Y, PH, PT, SI, EP = [], [], [], [], [], []
    succ = []
    for scene in scenes:
        command = int(scene["command"])
        positions = {c: np.asarray(v, float) for c, v in scene["positions"].items()}
        env.reset(command=command, positions=positions)
        prev = HOME_QPOS.astype(np.float32)
        dwell = 0
        for _ in range(EP_LEN):
            ph, pt, si = env.phase, env.phase_t, env.step_idx
            act = (env.reactive_action(gain=0.25, max_dq=0.03) if kind == "reactive"
                   else env.expert_action(gain=0.25, max_dq=0.03)).astype(np.float32)
            X.append(features(env, prev)); Y.append(act)
            PH.append(ph); PT.append(pt); SI.append(si); EP.append(int(scene["episode"]))
            env.step(act); prev = act
            dwell = dwell + 1 if env.success() else 0
            if dwell >= DWELL:
                break
        succ.append(bool(env.success()))
    return (np.asarray(X, np.float32), np.asarray(Y, np.float32),
            np.asarray(PH), np.asarray(PT), np.asarray(SI), np.asarray(EP), succ)


def oracle_feats(X, PH, PT, SI):
    onehot = np.zeros((len(X), N_PHASE), np.float32)
    onehot[np.arange(len(X)), np.clip(PH, 0, N_PHASE - 1)] = 1.0
    extra = np.stack([PT / 50.0, SI.astype(np.float32)], axis=1)
    return np.concatenate([X, onehot, extra.astype(np.float32)], axis=1)


def train_mlp(X, Y, d_in, steps, lr, seed):
    seed_everything(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    ymu, ysd = Y.mean(0), Y.std(0) + 1e-6
    xt = torch.from_numpy((X - mu) / sd).float()
    yt = torch.from_numpy((Y - ymu) / ysd).float()
    model = MLP(d_in=d_in)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    for _ in range(steps):
        opt.zero_grad(); loss = lossf(model(xt), yt); loss.backward(); opt.step()
    with torch.inference_mode():
        pred = model(xt).numpy() * ysd + ymu
    max_err = np.abs(pred - Y).max(axis=0)
    return model, (mu, sd, ymu, ysd), float(loss), max_err


def oracle_rollout(model, norm, scene, cap, radius):
    """Closed-loop eval with a shadow stateful expert providing the oracle phase."""
    mu, sd, ymu, ysd = norm
    command = int(scene["command"]); color = _target_color(command)
    positions = {c: np.asarray(v, float) for c, v in scene["positions"].items()}
    env = SO101PickPlaceTask()
    lo, hi = env.ctrl_range[:, 0], env.ctrl_range[:, 1]
    env.reset(command=command, positions=positions)
    prev = HOME_QPOS.astype(np.float32)
    dmin, grasp_t, lifted = float("inf"), -1, False
    for t in range(cap):
        ph, pt, si = env.phase, env.phase_t, env.step_idx
        x = oracle_feats(features(env, prev)[None, :], np.array([ph]), np.array([pt]), np.array([si]))[0]
        with torch.inference_mode():
            y = model(torch.from_numpy((x - mu) / sd).float().unsqueeze(0)).squeeze(0).numpy()
        action = np.clip(y * ysd + ymu, lo, hi).astype(np.float32)
        _ = env.expert_action(gain=0.25, max_dq=0.03)   # advance shadow phase machine only
        dmin = min(dmin, float(np.linalg.norm(env.ee_pos() - env.cube_pos(color))))
        env.step(action); prev = action
        if grasp_t < 0 and env.grasped is not None:
            grasp_t = t
        if env.cube_pos(color)[2] > 0.117:
            lifted = True
    return {"command": command, "success": bool(env.success()), "min_ee_cube_dist": round(dmin, 4),
            "grasp_fired": grasp_t >= 0, "lifted": lifted}


def ambiguity_audit(X, Y, PH):
    """NN in normalized physical-state space (no prev action): conflicting labels?"""
    state = X[:, :22]                                   # qpos,qvel,ee,cube,dest,grasped
    mu, sd = state.mean(0), state.std(0) + 1e-6
    S = (state - mu) / sd
    n = len(S)
    d2 = ((S[:, None, :] - S[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nn_idx = d2.argmin(1)
    nn_dist = np.sqrt(d2[np.arange(n), nn_idx])
    grip_tgt = Y[:, 5]
    conflicts = []
    near = nn_dist < np.percentile(nn_dist, 25)        # closest quartile of neighbours
    grip_flip = ((grip_tgt < GRIP_GRAB) != (grip_tgt[nn_idx] < GRIP_GRAB))
    phase_diff = PH != PH[nn_idx]
    for i in np.where(near & grip_flip)[0][:6]:
        conflicts.append({"i_phase": PHASE_NAMES.get(int(PH[i])), "nn_phase": PHASE_NAMES.get(int(PH[nn_idx[i]])),
                          "state_dist": round(float(nn_dist[i]), 3),
                          "grip_i": round(float(grip_tgt[i]), 3), "grip_nn": round(float(grip_tgt[nn_idx[i]]), 3),
                          "action_dist": round(float(np.linalg.norm(Y[i] - Y[nn_idx[i]])), 3)})
    per_trans = {}
    for name, (a, b) in TRANSITIONS.items():
        m = ((PH == a) | (PH == b))
        if m.sum() > 1:
            sub = np.where(m)[0]
            gp = grip_tgt[sub]
            per_trans[name] = {"frames": int(m.sum()),
                               "gripper_target_spread": round(float(gp.max() - gp.min()), 3)}
    return {
        "n_frames": n,
        "mean_nn_state_dist": round(float(nn_dist.mean()), 4),
        "near_neighbour_gripper_flip_frac": round(float(grip_flip[near].mean()), 3),
        "near_neighbour_phase_differs_frac": round(float(phase_diff[near].mean()), 3),
        "mean_action_dist_near": round(float(np.mean([np.linalg.norm(Y[i] - Y[nn_idx[i]]) for i in np.where(near)[0]])), 4),
        "conflict_examples": conflicts,
        "transition_gripper_spread": per_trans,
    }


def _summ(rollouts):
    n = len(rollouts)
    return {"successes": sum(r["success"] for r in rollouts), "n": n,
            "grasp": sum(r["grasp_fired"] for r in rollouts),
            "lift": sum(r["lifted"] for r in rollouts),
            "mean_min_dist": round(float(np.mean([r["min_ee_cube_dist"] for r in rollouts])), 4),
            "per_scene": rollouts}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="artifacts/truth_harness/datasets/command0_4")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cap", type=int, default=200)
    ap.add_argument("--output", default="artifacts/truth_harness/markovity.json")
    args = ap.parse_args()
    root = Path(args.root)
    scenes = json.loads((root / "scene_manifest.json").read_text())["scenes"]

    Xs, Ys, PH, PT, SI, EP, _ = gen(scenes, "stateful")
    Xr, Yr, _, _, _, _, react_succ = gen(scenes, "reactive")

    out = {"reactive_expert_selfplay_success": f"{sum(react_succ)}/{len(react_succ)}"}
    out["ambiguity_audit"] = ambiguity_audit(Xs, Ys, PH)

    # base (stateful labels, state-only obs) — reproduce baseline
    base, nbase, lbase, maxb = train_mlp(Xs, Ys, FEAT_DIM, args.steps, args.lr, args.seed)
    out["stateful_base"] = {"final_loss": round(lbase, 7),
                            "max_abs_action_err": {k: round(float(v), 4) for k, v in
                                                   zip(["pan", "lift", "elbow", "wflex", "wroll", "grip"], maxb)},
                            **_summ([rollout(base, *nbase, s, args.cap, GRASP_RADIUS) for s in scenes])}

    # oracle phase (stateful labels + phase/phase_t/step_idx)
    Xo = oracle_feats(Xs, PH, PT, SI)
    omodel, onorm, oloss, omax = train_mlp(Xo, Ys, Xo.shape[1], args.steps, args.lr, args.seed)
    out["oracle_phase"] = {"final_loss": round(oloss, 7), "feat_dim": int(Xo.shape[1]),
                           "max_abs_action_err": {k: round(float(v), 4) for k, v in
                                                  zip(["pan", "lift", "elbow", "wflex", "wroll", "grip"], omax)},
                           **_summ([oracle_rollout(omodel, onorm, s, args.cap, GRASP_RADIUS) for s in scenes])}

    # reactive labels (stateless expert), state-only obs, identical everything
    rmodel, rnorm, rloss, rmax = train_mlp(Xr, Yr, FEAT_DIM, args.steps, args.lr, args.seed)
    out["reactive_labels"] = {"final_loss": round(rloss, 7), "train_pairs": int(len(Xr)),
                              "max_abs_action_err": {k: round(float(v), 4) for k, v in
                                                     zip(["pan", "lift", "elbow", "wflex", "wroll", "grip"], rmax)},
                              **_summ([rollout(rmodel, *rnorm, s, args.cap, GRASP_RADIUS) for s in scenes])}

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({
        "reactive_expert_selfplay": out["reactive_expert_selfplay_success"],
        "ambiguity": {k: out["ambiguity_audit"][k] for k in
                      ("near_neighbour_gripper_flip_frac", "near_neighbour_phase_differs_frac", "mean_action_dist_near")},
        "stateful_base": {k: out["stateful_base"][k] for k in ("final_loss", "successes", "grasp", "lift", "max_abs_action_err")},
        "oracle_phase": {k: out["oracle_phase"][k] for k in ("successes", "grasp", "lift", "max_abs_action_err")},
        "reactive_labels": {k: out["reactive_labels"][k] for k in ("successes", "grasp", "lift", "max_abs_action_err")},
    }, indent=2))


if __name__ == "__main__":
    main()
