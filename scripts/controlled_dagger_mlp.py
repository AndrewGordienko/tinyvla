"""Controlled privileged-MLP DAgger validation (diagnostic exception to the freeze).

Tests the covariate-shift fix directly: does aggregating the states the LEARNER
actually visits (labelled by the stateless reactive expert) turn the privileged
one-step MLP from failure into stable 4/4? Retrain-from-scratch each round (so
data, not extra optimizer steps, is the variable); learner-only evaluation.

Mandatory prerequisite: expert-takeover recoverability — can the reactive expert
finish from the learner's visited states? If not, its labels are not a valid
recovery oracle and DAgger cannot be expected to solve those states.

Controls at matched size / updates:
  A  reactive demos only (oversampled to the aggregate size)
  B  reactive demos + fixed random-perturbation recovery
  C  true on-policy DAgger aggregation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from tinyvla.task import SO101PickPlaceTask, COMMANDS, HOME_QPOS, GRASP_RADIUS
from tinyvla.determinism import seed_everything
from scripts.baseline_mlp import features, MLP, FEAT_DIM
from scripts.perturbation_recovery import _snapshot, _restore, recovery_set, MAGS

EP_LEN, DWELL, LIFT_Z = 220, 8, 0.117
STAGES = ["approach", "near_grasp", "lift", "carry", "dest_approach", "release"]


def _color(command):
    return COMMANDS[command]["steps"][0][0]


def stage_of(env, color):
    ee, cube = env.ee_pos(), env.cube_pos(color)
    if env.grasped != color:
        if np.linalg.norm(ee[:2] - cube[:2]) > 0.02:
            return "approach"
        return "near_grasp"
    dxy = env._dest_xy(env.target_dest, color)
    near = np.linalg.norm(ee[:2] - dxy) < 0.03
    if near and cube[2] < LIFT_Z:
        return "release"
    if near:
        return "dest_approach"
    if cube[2] > LIFT_Z:
        return "carry"
    return "lift"


def predict(model, norm, feat):
    mu, sd, ymu, ysd = norm
    with torch.inference_mode():
        y = model(torch.from_numpy((feat - mu) / sd).float().unsqueeze(0)).squeeze(0).numpy()
    return y * ysd + ymu


def train(X, Y, seed, steps, lr=1e-3):
    seed_everything(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    ymu, ysd = Y.mean(0), Y.std(0) + 1e-6
    xt = torch.from_numpy((X - mu) / sd).float()
    yt = torch.from_numpy((Y - ymu) / ysd).float()
    model = MLP(d_in=FEAT_DIM)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    bs = min(256, len(xt))
    for _ in range(steps):
        idx = torch.randint(0, len(xt), (bs,))
        opt.zero_grad(); loss = lossf(model(xt[idx]), yt[idx]); loss.backward(); opt.step()
    return model, (mu, sd, ymu, ysd)


def reactive_demos(scenes):
    env = SO101PickPlaceTask()
    X, Y, stg = [], [], []
    demo_states = []
    for scene in scenes:
        command = int(scene["command"]); color = _color(command)
        positions = {c: np.asarray(v, float) for c, v in scene["positions"].items()}
        env.reset(command=command, positions=positions)
        prev = HOME_QPOS.astype(np.float32); dwell = 0
        for _ in range(EP_LEN):
            a = env.reactive_action(gain=0.25, max_dq=0.03).astype(np.float32)
            f = features(env, prev)
            X.append(f); Y.append(a); stg.append(stage_of(env, color)); demo_states.append(f[:22])
            env.step(a); prev = a
            dwell = dwell + 1 if env.success() else 0
            if dwell >= DWELL:
                break
    return np.asarray(X, np.float32), np.asarray(Y, np.float32), stg, np.asarray(demo_states, np.float32)


def learner_rollout(model, norm, scene, cap, radius, demo_states, collect):
    command = int(scene["command"]); color = _color(command)
    positions = {c: np.asarray(v, float) for c, v in scene["positions"].items()}
    env = SO101PickPlaceTask()
    lo, hi = env.ctrl_range[:, 0], env.ctrl_range[:, 1]
    env.reset(command=command, positions=positions)
    prev = HOME_QPOS.astype(np.float32)
    dmin, grasp_t, lifted, carried = float("inf"), -1, False, False
    first_fail_stage, visited = None, []
    dmu, dsd = demo_states.mean(0), demo_states.std(0) + 1e-6
    for t in range(cap):
        feat = features(env, prev)
        learner = np.clip(predict(model, norm, feat), lo, hi).astype(np.float32)
        if collect:
            expert = env.reactive_action(gain=0.25, max_dq=0.03).astype(np.float32)
            dist = float(np.sqrt((((feat[:22] - dmu) / dsd - (demo_states - dmu) / dsd) ** 2).sum(1)).min())
            visited.append({"snapshot": _snapshot(env), "feat": feat, "expert": expert,
                            "stage": stage_of(env, color),
                            "disagreement": float(np.linalg.norm(learner - expert)), "dist_to_demo": dist})
        dmin = min(dmin, float(np.linalg.norm(env.ee_pos() - env.cube_pos(color))))
        env.step(learner); prev = learner
        if grasp_t < 0 and env.grasped is not None:
            grasp_t = t
        if env.cube_pos(color)[2] > LIFT_Z:
            lifted = True
        if np.linalg.norm(env.cube_pos(color)[:2] - env._dest_xy(env.target_dest, color)) < 0.05:
            carried = True
    if not env.success():
        first_fail_stage = ("approach" if grasp_t < 0 else "carry_place" if lifted else "grasp")
    return {"success": bool(env.success()), "min_dist": round(dmin, 4), "grasp": grasp_t >= 0,
            "lift": lifted, "carry": carried, "first_fail_stage": first_fail_stage}, visited


def expert_takeover(snapshot, command, cap):
    env = SO101PickPlaceTask()
    env.reset(command=command, positions={c: np.asarray([0.2, 0.0]) for c in ("red", "blue")})
    _restore(env, snapshot)
    env.steps = COMMANDS[command]["steps"]; env.step_idx = 0
    for _ in range(cap):
        env.step(env.reactive_action(gain=0.25, max_dq=0.03))
        if env.success():
            return True
    return bool(env.success())


def eval_seeds(build_train, scenes, seeds, steps, cap, demo_states):
    out = []
    for seed in seeds:
        X, Y = build_train(seed)
        model, norm = train(X, Y, seed, steps)
        rolls = [learner_rollout(model, norm, s, cap, GRASP_RADIUS, demo_states, False)[0] for s in scenes]
        out.append(sum(r["success"] for r in rolls))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="artifacts/truth_harness/datasets/command0_4")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--cap", type=int, default=200)
    ap.add_argument("--stage-cap", type=int, default=600)
    ap.add_argument("--output", default="artifacts/truth_harness/controlled_dagger.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    scenes = json.loads((Path(args.root) / "scene_manifest.json").read_text())["scenes"]
    Xd, Yd, stg_d, demo_states = reactive_demos(scenes)

    # ---- expert-takeover recoverability from initial-model learner states, by stage
    seed_everything(0)
    m0, n0 = train(Xd, Yd, 0, args.steps)
    takeover = {s: {"n": 0, "recovered": 0} for s in STAGES}
    for scene in scenes:
        _, visited = learner_rollout(m0, n0, scene, args.cap, GRASP_RADIUS, demo_states, True)
        for v in visited[::5]:                              # subsample for speed
            st = v["stage"]
            takeover[st]["n"] += 1
            takeover[st]["recovered"] += int(expert_takeover(v["snapshot"], int(scene["command"]), args.cap))
    recover_rate = {s: (round(takeover[s]["recovered"] / takeover[s]["n"], 2) if takeover[s]["n"] else None)
                    for s in STAGES}

    # ---- DAgger per seed (independent aggregates), retrain from scratch each round
    dagger = {}
    final_aggr_sizes = []
    for seed in seeds:
        aggX, aggY, aggS = [Xd.copy()], [Yd.copy()], list(stg_d)
        curve = []
        for rnd in range(args.rounds):
            X = np.concatenate(aggX); Y = np.concatenate(aggY)
            # stage-balanced cap so stalled states don't dominate (keep all originals)
            keep = []
            counts = {s: 0 for s in set(aggS)}
            for i, s in enumerate(aggS):
                if i < len(Xd) or counts[s] < args.stage_cap:
                    keep.append(i); counts[s] += 1
            Xk, Yk = X[keep], Y[keep]
            model, norm = train(Xk, Yk, seed, args.steps)
            rolls, visited_all = [], []
            for s in scenes:
                r, vis = learner_rollout(model, norm, s, args.cap, GRASP_RADIUS, demo_states, True)
                rolls.append(r); visited_all += vis
            succ = sum(r["success"] for r in rolls)
            disagreement = float(np.mean([v["disagreement"] for v in visited_all])) if visited_all else 0.0
            stage_dist = {s: int(sum(1 for x in aggS if x == s)) for s in STAGES}
            curve.append({"round": rnd, "aggregate": int(len(Xk)), "new_states": len(visited_all),
                          "success": succ, "mean_disagreement": round(disagreement, 4),
                          "grasp": sum(r["grasp"] for r in rolls), "lift": sum(r["lift"] for r in rolls),
                          "carry": sum(r["carry"] for r in rolls),
                          "mean_min_dist": round(float(np.mean([r["min_dist"] for r in rolls])), 4),
                          "worst_min_dist": round(float(np.max([r["min_dist"] for r in rolls])), 4),
                          "first_fail_stages": [r["first_fail_stage"] for r in rolls],
                          "stage_dist": stage_dist})
            # aggregate on-policy states
            aggX.append(np.asarray([v["feat"] for v in visited_all], np.float32))
            aggY.append(np.asarray([v["expert"] for v in visited_all], np.float32))
            aggS += [v["stage"] for v in visited_all]
        dagger[str(seed)] = curve
        final_aggr_sizes.append(len(np.concatenate(aggX)))

    # ---- controls A/B/C at matched size & updates
    N = int(np.mean(final_aggr_sizes))
    # fixed random-perturbation recovery data for control B (off-policy noise, not on-policy)
    from scripts.perturbation_recovery import reactive_demos as pr_demos
    _, _, snaps, cmds = pr_demos(scenes)
    Xrec, Yrec, _ = recovery_set(snaps, cmds, MAGS, 3, 0)

    def build_A(seed):
        reps = max(1, N // len(Xd)); return np.tile(Xd, (reps, 1)), np.tile(Yd, (reps, 1))

    def build_B(seed):
        need = max(0, N - len(Xd)); idx = np.random.default_rng(seed).integers(0, len(Xrec), need)
        return np.concatenate([Xd, Xrec[idx]]), np.concatenate([Yd, Yrec[idx]])

    controls = {
        "A_demos_only_oversampled": eval_seeds(build_A, scenes, seeds, args.steps, args.cap, demo_states),
        "B_demos_plus_perturbation": eval_seeds(build_B, scenes, seeds, args.steps, args.cap, demo_states),
        "C_dagger_final": [dagger[str(s)][-1]["success"] for s in seeds],
    }

    out = {"seeds": seeds, "rounds": args.rounds, "matched_size_N": N,
           "expert_takeover_recoverability_by_stage": recover_rate,
           "expert_takeover_counts": takeover,
           "dagger_curves": dagger,
           "controls_success_out_of_4_per_seed": controls}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"expert_takeover": recover_rate,
                      "dagger_success_by_round": {s: [c["success"] for c in dagger[s]] for s in dagger},
                      "controls": controls}, indent=2))


if __name__ == "__main__":
    main()
