"""Merge parallel DAgger shard pools -> balance nominal to ~50/50 -> build dataset.

Consumes the per-command shard pools produced by recovery_shard_collect.py (each an
independent 'arm'), merges them into one pool, tops up scripted-expert NOMINAL demos
(parallel, CPU-only) until frames are ~50/50 recovery/nominal, then materialises a
front-only LeRobot dataset whose schema is asserted to match the student's EXACT
deployed observation/action schema (fail closed). Local-only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from tinyvla.dagger import (_next_index, collect_expert_episodes_parallel,
                            build_lerobot_dataset, pool_summary, pool_episodes, CAMERA)
from tinyvla.runtime import write_action_semantics
from tinyvla.paths import DATASETS_ROOT, CHECKPOINTS_ROOT


def frames_by_source(pool: Path):
    rec = nom = 0
    for p in pool_episodes(pool):
        d = np.load(p, allow_pickle=False)
        n = int(d["action"].shape[0])
        if str(d["source"]) == "dagger":
            rec += n
        else:
            nom += n
    return rec, nom


def merge_shard(shard: Path, merged: Path) -> int:
    merged.mkdir(parents=True, exist_ok=True)
    moved = 0
    for f in sorted(Path(shard).glob("ep_*.npz")):
        f.rename(merged / f"ep_{_next_index(merged):06d}.npz")
        moved += 1
    return moved


def assert_schema_matches_student(root: Path, student_cfg: dict, repo_id: str):
    meta = LeRobotDatasetMetadata(repo_id, root=root)
    feats = meta.features
    problems = []
    student_imgs = {k for k in student_cfg["input_features"] if k.startswith("observation.images.")}
    ds_imgs = {k for k in feats if k.startswith("observation.images.")}
    if student_imgs != ds_imgs:
        problems.append(f"camera keys differ: student={sorted(student_imgs)} dataset={sorted(ds_imgs)}")
    if ds_imgs != {f"observation.images.{CAMERA}"}:
        problems.append(f"dataset cameras are not front-only: {sorted(ds_imgs)}")
    if tuple(feats["observation.state"]["shape"]) != tuple(student_cfg["input_features"]["observation.state"]["shape"]):
        problems.append("observation.state shape mismatch")
    if tuple(feats["action"]["shape"]) != tuple(student_cfg["output_features"]["action"]["shape"]):
        problems.append("action shape mismatch")
    if (root / "delta_actions.json").exists():
        problems.append("dataset is delta-action but student is absolute")
    if problems:
        raise SystemExit("FAIL-CLOSED schema mismatch:\n  - " + "\n  - ".join(problems))
    print("schema OK: front-only, state/action dims match, absolute actions — matches deployed student")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-root", default=str(DATASETS_ROOT / "recovery_pool_r2"),
                    help="dir containing dagger_cmd* shard pools")
    ap.add_argument("--pool", default=str(DATASETS_ROOT / "recovery_pool_r2" / "merged"))
    ap.add_argument("--out-root", default=str(DATASETS_ROOT / "so101_recovery_r2"))
    ap.add_argument("--repo-id", default="local/so101_recovery_r2")
    ap.add_argument("--commands", default="1,3,4")
    ap.add_argument("--student", default=str(CHECKPOINTS_ROOT / "student291_recovery_r1"))
    ap.add_argument("--nominal-batch", type=int, default=4, help="expert demos per command per balance step")
    ap.add_argument("--seed", type=int, default=4300)
    args = ap.parse_args()

    commands = [int(x) for x in args.commands.split(",") if x != ""]
    merged = Path(args.pool)
    if merged.exists():
        import shutil
        shutil.rmtree(merged)

    print("[1/4] MERGE shard pools")
    shard_root = Path(args.shard_root)
    total = 0
    for shard in sorted(shard_root.glob("dagger_cmd*")):
        m = merge_shard(shard, merged)
        total += m
        print(f"      merged {m:4d} episodes from {shard.name}")
    if total == 0:
        raise SystemExit(f"no shard episodes found under {shard_root}/dagger_cmd*")
    rec_f, nom_f = frames_by_source(merged)
    print(f"      merged recovery episodes={total} frames={rec_f}")

    print("[2/4] NOMINAL top-up (parallel) until ~50/50 by frames")
    seed = args.seed
    while True:
        rec_f, nom_f = frames_by_source(merged)
        if nom_f >= rec_f or nom_f > 3 * rec_f:
            break
        collect_expert_episodes_parallel(merged, commands, args.nominal_batch,
                                         workers=len(commands), seed=seed)
        seed += 13
    rec_f, nom_f = frames_by_source(merged)
    print(f"      nominal frames={nom_f}  recovery frames={rec_f}  "
          f"nominal ratio={nom_f/(nom_f+rec_f):.2f}")
    print("      pool:", json.dumps(pool_summary(merged)))

    print(f"[3/4] BUILD dataset -> {args.out_root}")
    root = build_lerobot_dataset(merged, args.repo_id, Path(args.out_root), delta_actions=False)
    write_action_semantics(root, "absolute")

    print("[4/4] FAIL-CLOSED schema check vs deployed student")
    student_cfg = json.load(open(Path(args.student) / "config.json"))
    assert_schema_matches_student(root, student_cfg, args.repo_id)
    print("DONE. dataset root:", root)


if __name__ == "__main__":
    main()
