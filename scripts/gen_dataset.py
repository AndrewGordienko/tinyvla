#!/usr/bin/env python3
"""Generate a large SO-101 dataset in parallel, then merge the shards.

Runs N independent `tinyvla.collect` processes (each a balanced round-robin over
all commands, different seed), then aggregates them into one LeRobot dataset.

Run (on a GPU box):
    MUJOCO_GL=egl python scripts/gen_dataset.py --shards 8 --eps-per-shard 300 \
        --out-repo local/so101_pickplace_v2 --out-root data/datasets/so101_pickplace_v2
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from lerobot.datasets.aggregate import aggregate_datasets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--eps-per-shard", type=int, default=300)
    ap.add_argument("--out-repo", default="local/so101_pickplace_v2")
    ap.add_argument("--out-root", default="data/datasets/so101_pickplace_v2")
    ap.add_argument("--shard-dir", default="data/datasets/_shards")
    args = ap.parse_args()

    os.makedirs(args.shard_dir, exist_ok=True)
    roots, repo_ids, procs, logs = [], [], [], []
    env = os.environ.copy()   # inherit MUJOCO_GL (set =egl on Linux GPU boxes)

    print(f"launching {args.shards} shards x {args.eps_per_shard} eps "
          f"= {args.shards * args.eps_per_shard} episodes", flush=True)
    for k in range(args.shards):
        root = os.path.join(args.shard_dir, f"shard_{k}")
        repo = f"local/_shard_{k}"
        log = open(os.path.join(args.shard_dir, f"shard_{k}.log"), "w")
        p = subprocess.Popen(
            [sys.executable, "-m", "tinyvla.collect",
             "--episodes", str(args.eps_per_shard), "--seed", str(1000 + k * 777),
             "--root", root, "--repo-id", repo],
            env=env, stdout=log, stderr=subprocess.STDOUT)
        procs.append(p); roots.append(root); repo_ids.append(repo); logs.append(log)

    # wait, reporting progress from the shard logs
    t0 = time.time()
    while any(p.poll() is None for p in procs):
        time.sleep(20)
        done = 0
        for k, root in enumerate(roots):
            lg = os.path.join(args.shard_dir, f"shard_{k}.log")
            try:
                last = [l for l in open(lg) if "episode " in l]
                done += int(last[-1].split("/")[0].split()[-1]) if last else 0
            except Exception:
                pass
        alive = sum(p.poll() is None for p in procs)
        print(f"  [{time.time()-t0:5.0f}s] ~{done} episodes done, {alive} shards running", flush=True)

    for lg in logs:
        lg.close()
    codes = [p.returncode for p in procs]
    print(f"shards finished, exit codes: {codes}", flush=True)
    if any(c != 0 for c in codes):
        print("WARNING: some shards failed; check shard logs", flush=True)

    print("aggregating shards ->", args.out_repo, flush=True)
    if os.path.exists(args.out_root):
        import shutil; shutil.rmtree(args.out_root)
    aggregate_datasets(repo_ids, args.out_repo, roots=roots, aggr_root=args.out_root)
    print(f"DONE -> {args.out_root}", flush=True)


if __name__ == "__main__":
    main()
