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
    """Reactive trajectories with per-state snapshots so images can be re-rendered."""
    env = SO101PickPlaceTask()
    X, Y, stg, snaps, cmds = [], [], [], [], []
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
            snaps.append(_snapshot(env)); cmds.append(command)
            env.step(a); prev = a
            dwell = dwell + 1 if env.success() else 0
            if dwell >= DWELL:
                break
    return (np.asarray(X, np.float32), np.asarray(Y, np.float32), stg,
            np.asarray(demo_states, np.float32), snaps, cmds)


def held_out_scenes(n, seed):
    """n deterministic unseen command-0 scenes (random cube layouts)."""
    env = SO101PickPlaceTask()
    scenes = []
    for i in range(n):
        env.rng = np.random.default_rng(10_000 + seed + i)
        env.reset(command=0)
        scenes.append({"episode": i, "command": 0, "instruction": env.instruction,
                       "positions": {c: env.cube_pos(c).tolist() for c in ("red", "blue")}})
    return scenes


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


def takeover_rate(visited, command, cap, subsample=8):
    """Expert-takeover recovery by stage from a set of visited states."""
    by = {s: {"n": 0, "rec": 0} for s in STAGES}
    fails = []
    for v in visited[::subsample]:
        st = v["stage"]; by[st]["n"] += 1
        ok = expert_takeover(v["snapshot"], command, cap)
        by[st]["rec"] += int(ok)
        if not ok:
            fails.append({"stage": st, "dist_to_demo": round(v["dist_to_demo"], 3)})
    rate = {s: (round(by[s]["rec"] / by[s]["n"], 2) if by[s]["n"] else None) for s in STAGES}
    return rate, by, fails


def export_aggregate(path, feats, labels, snaps, stages, sources, rounds, command, env):
    nq, nv = env.model.nq, env.model.nv
    n = len(feats)
    qpos = np.zeros((n, nq), np.float32); qvel = np.zeros((n, nv), np.float32)
    grasped = np.array([sn["grasped"] or "" for sn in snaps])
    offp = np.full((n, 3), np.nan, np.float32); offq = np.full((n, 4), np.nan, np.float32)
    for i, sn in enumerate(snaps):
        qpos[i] = sn["qpos"]; qvel[i] = sn["qvel"]
        if sn["off_pos"] is not None:
            offp[i] = sn["off_pos"]; offq[i] = sn["off_quat"]
    np.savez_compressed(path, sample_id=np.arange(n), feat=np.asarray(feats, np.float32),
                        label=np.asarray(labels, np.float32), qpos=qpos, qvel=qvel,
                        grasped=grasped, off_pos=offp, off_quat=offq,
                        command=np.full(n, command), stage=np.array(stages),
                        source=np.array(sources), source_round=np.asarray(rounds))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="artifacts/truth_harness/datasets/command0_4")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--cap", type=int, default=200)
    ap.add_argument("--stage-cap", type=int, default=600)
    ap.add_argument("--held-out", type=int, default=20)
    ap.add_argument("--export-dir", default="artifacts/truth_harness/dagger_dataset")
    ap.add_argument("--output", default="artifacts/truth_harness/controlled_dagger.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    scenes = json.loads((Path(args.root) / "scene_manifest.json").read_text())["scenes"]
    command0 = int(scenes[0]["command"])
    Xd, Yd, stg_d, demo_states, snap_d, cmd_d = reactive_demos(scenes)
    ho_scenes = held_out_scenes(args.held_out, 0)
    Path(args.export_dir).mkdir(parents=True, exist_ok=True)
    env_ref = SO101PickPlaceTask()

    dagger, ctrl_N, held_out, per_round_takeover, release_fail = {}, [], {}, {}, []
    for seed in seeds:
        # flat aggregate so snapshots/sources stay aligned with rows
        a_feat = [f for f in Xd]; a_lab = [y for y in Yd]; a_snap = list(snap_d)
        a_stage = list(stg_d); a_src = ["demo"] * len(Xd); a_round = [-1] * len(Xd)
        curve, last_keep = [], None
        for rnd in range(args.rounds):
            counts = {}; keep = []
            for i, s in enumerate(a_stage):
                counts.setdefault(s, 0)
                if a_src[i] == "demo" or counts[s] < args.stage_cap:
                    keep.append(i); counts[s] += 1
            Xk = np.asarray([a_feat[i] for i in keep], np.float32)
            Yk = np.asarray([a_lab[i] for i in keep], np.float32)
            model, norm = train(Xk, Yk, seed, args.steps)
            rolls, visited_all = [], []
            for s in scenes:
                r, vis = learner_rollout(model, norm, s, args.cap, GRASP_RADIUS, demo_states, True)
                rolls.append(r); visited_all += vis
            succ = sum(r["success"] for r in rolls)
            tk_rate, _, tk_fails = takeover_rate(visited_all, command0, args.cap)
            release_fail += [f for f in tk_fails if f["stage"] == "release"]
            curve.append({"round": rnd, "train_size": int(len(keep)), "new_states": len(visited_all),
                          "success": succ,
                          "mean_disagreement": round(float(np.mean([v["disagreement"] for v in visited_all])), 4),
                          "grasp": sum(r["grasp"] for r in rolls), "lift": sum(r["lift"] for r in rolls),
                          "carry": sum(r["carry"] for r in rolls),
                          "mean_min_dist": round(float(np.mean([r["min_dist"] for r in rolls])), 4),
                          "first_fail_stages": [r["first_fail_stage"] for r in rolls],
                          "takeover_rate": tk_rate,
                          # stage distribution of the CAPPED training set actually used
                          # this round (counts tracks kept rows), not the uncapped aggregate
                          "stage_dist": {s: counts.get(s, 0) for s in STAGES}})
            last_keep = keep
            # Only aggregate for a round that will actually be trained on. Appending
            # after the final round grows the aggregate with states no model ever sees
            # (last_keep/export/held-out all use this round's keep), so skip it.
            if rnd < args.rounds - 1:
                for v in visited_all:
                    a_feat.append(v["feat"]); a_lab.append(v["expert"]); a_snap.append(v["snapshot"])
                    a_stage.append(v["stage"]); a_src.append("dagger"); a_round.append(rnd)
        dagger[str(seed)] = curve
        ctrl_N.append(curve[-1]["train_size"])           # winning model's ACTUAL training size
        per_round_takeover[str(seed)] = [c["takeover_rate"] for c in curve]
        # export the exact winning (final-round) training set for the CNN transfer control
        export_aggregate(str(Path(args.export_dir) / f"seed{seed}.npz"),
                         [a_feat[i] for i in last_keep], [a_lab[i] for i in last_keep],
                         [a_snap[i] for i in last_keep], [a_stage[i] for i in last_keep],
                         [a_src[i] for i in last_keep], [a_round[i] for i in last_keep],
                         command0, env_ref)
        # held-out 20 with the final model
        fm, fn = train(np.asarray([a_feat[i] for i in last_keep], np.float32),
                       np.asarray([a_lab[i] for i in last_keep], np.float32), seed, args.steps)
        ho = [learner_rollout(fm, fn, s, args.cap, GRASP_RADIUS, demo_states, False)[0] for s in ho_scenes]
        held_out[str(seed)] = {"n": len(ho), "success": sum(r["success"] for r in ho),
                               "grasp": sum(r["grasp"] for r in ho), "lift": sum(r["lift"] for r in ho),
                               "carry": sum(r["carry"] for r in ho)}

    # ---- controls A/B/C matched to the WINNING model's training size (fixes off-by-one)
    N = int(np.mean(ctrl_N))
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

    out = {"seeds": seeds, "rounds": args.rounds, "matched_train_size_N": N,
           "expert_takeover_by_round": per_round_takeover,
           "release_stage_failures": release_fail,
           "dagger_curves": dagger, "held_out": held_out,
           "controls_success_out_of_4_per_seed": controls,
           "export_dir": args.export_dir}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"dagger_success_by_round": {s: [c["success"] for c in dagger[s]] for s in dagger},
                      "held_out_success_of_20": {s: held_out[s]["success"] for s in held_out},
                      "matched_train_size_N": N, "controls": controls,
                      "release_failures": len(release_fail)}, indent=2))


if __name__ == "__main__":
    main()
