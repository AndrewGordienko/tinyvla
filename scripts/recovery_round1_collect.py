"""Round-1 targeted recovery collection (scoped commands 1,3,4, front-only, Mac/GLFW).

Builds a ~50/50 pool of (a) scripted-expert NOMINAL demos and (b) DAgger RECOVERY
episodes: the CURRENT student is rolled out and every state it drifts into is
labelled with the reactive (stateless) expert. Then materialises a front-only
LeRobot dataset whose schema is asserted to match the student's EXACT deployed
observation/action schema (fail closed).

The reactive expert is used ONLY to label training data here. It never runs at
evaluation or deployment.

Run: .venv/bin/python scripts/recovery_round1_collect.py --device mps
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from tinyvla.dagger import (collect_expert_episodes, dagger_collect,
                            build_lerobot_dataset, pool_summary, pool_episodes, CAMERA)
from tinyvla.runtime import load_runtime
from tinyvla.paths import DATASETS_ROOT, CHECKPOINTS_ROOT

SCOPED = [1, 3, 4]


def frames_by_source(pool: Path):
    import numpy as np
    rec = nom = 0
    for p in pool_episodes(pool):
        d = np.load(p, allow_pickle=False)
        n = int(d["action"].shape[0])
        if str(d["source"]) == "dagger":
            rec += n
        else:
            nom += n
    return rec, nom


def assert_schema_matches_student(root: Path, student_cfg: dict):
    """Fail closed if the built dataset's schema differs from the student's deployed one."""
    meta = LeRobotDatasetMetadata("local/so101_recovery_r1", root=root)
    feats = meta.features
    problems = []
    # exactly the student's input image key(s): front only
    student_imgs = {k for k in student_cfg["input_features"] if k.startswith("observation.images.")}
    ds_imgs = {k for k in feats if k.startswith("observation.images.")}
    if student_imgs != ds_imgs:
        problems.append(f"camera keys differ: student={sorted(student_imgs)} dataset={sorted(ds_imgs)}")
    if ds_imgs != {f"observation.images.{CAMERA}"}:
        problems.append(f"dataset cameras are not front-only: {sorted(ds_imgs)}")
    # state / action dims
    if tuple(feats["observation.state"]["shape"]) != tuple(student_cfg["input_features"]["observation.state"]["shape"]):
        problems.append("observation.state shape mismatch")
    if tuple(feats["action"]["shape"]) != tuple(student_cfg["output_features"]["action"]["shape"]):
        problems.append("action shape mismatch")
    # absolute actions (student is absolute) -> no delta marker
    if (root / "delta_actions.json").exists():
        problems.append("dataset is delta-action but student is absolute")
    if problems:
        raise SystemExit("FAIL-CLOSED schema mismatch:\n  - " + "\n  - ".join(problems))
    print("schema OK: front-only, state/action dims match, absolute actions — matches deployed student")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default=str(CHECKPOINTS_ROOT /
                    "student291_recover_brain_v1" / "best_closed_loop"))
    ap.add_argument("--meta-root", default=str(DATASETS_ROOT / "so101_pickplace"))
    ap.add_argument("--meta-repo", default="local/so101_pickplace")
    ap.add_argument("--pool", default=str(DATASETS_ROOT / "recovery_pool_r1"))
    ap.add_argument("--out-root", default=str(DATASETS_ROOT / "so101_recovery_r1"))
    ap.add_argument("--recovery-per", type=int, default=12, help="student rollouts per scoped command")
    ap.add_argument("--nominal-batch", type=int, default=3, help="expert demos per command per balance step")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=4100)
    ap.add_argument("--fresh", action="store_true", help="wipe the pool first")
    args = ap.parse_args()

    pool = Path(args.pool)
    if args.fresh and pool.exists():
        import shutil
        shutil.rmtree(pool)

    device = torch.device(args.device)
    meta = LeRobotDatasetMetadata(args.meta_repo, root=args.meta_root)
    # deployed preprocessing exactly (stats_source=checkpoint) so we visit the SAME
    # drift distribution the deployed student visits
    rt = load_runtime(args.student, meta=meta, dataset_root=args.meta_root,
                      device=device, stats_source="checkpoint")
    policy = rt.policy.eval()
    student_cfg = json.load(open(Path(args.student) / "config.json"))
    assert rt.delta_actions is False, "student must be absolute-action for this pass"

    print(f"[1/4] RECOVERY: rolling out student on cmds {SCOPED}, {args.recovery_per}/cmd, labelling drift with reactive expert")
    n_rec = dagger_collect(pool, policy, rt.preprocessor, rt.postprocessor, SCOPED,
                           args.recovery_per, device=device, seed=args.seed, delta_actions=False)
    rec_f, _ = frames_by_source(pool)
    print(f"      recovery episodes={n_rec}  frames={rec_f}")

    print(f"[2/4] NOMINAL: scripted-expert demos on cmds {SCOPED} until ~50/50 by frames")
    seed = args.seed + 777
    while True:
        rec_f, nom_f = frames_by_source(pool)
        if nom_f >= rec_f or nom_f > 3 * rec_f:  # reached balance (or safety cap)
            break
        collect_expert_episodes(pool, SCOPED, args.nominal_batch, seed=seed)
        seed += 13
    rec_f, nom_f = frames_by_source(pool)
    print(f"      nominal frames={nom_f}  recovery frames={rec_f}  ratio nominal={nom_f/(nom_f+rec_f):.2f}")
    print("      pool:", json.dumps(pool_summary(pool)))

    print(f"[3/4] BUILD dataset -> {args.out_root}")
    root = build_lerobot_dataset(pool, "local/so101_recovery_r1", Path(args.out_root),
                                 delta_actions=False)

    # build_lerobot_dataset stores absolute actions but writes no semantics marker;
    # the trainer/loader requires one. The student is absolute, so mark it absolute.
    from tinyvla.runtime import write_action_semantics
    write_action_semantics(root, "absolute")

    print("[4/4] FAIL-CLOSED schema check vs deployed student")
    assert_schema_matches_student(root, student_cfg)
    print("DONE. dataset root:", root)


if __name__ == "__main__":
    main()
