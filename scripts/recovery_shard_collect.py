"""One 'arm': single-command DAgger recovery shard collector (local-only).

Rolls out the CURRENT student on ONE scoped command and labels every drift state
with the reactive (stateless) expert -- classic DAgger -- writing episodes into
this shard's OWN pool dir. Run K of these as independent OS processes (one per
command / seed) to collect many recovery samples in parallel, then merge them with
recovery_merge_build.py. The reactive expert only labels training data; it never
runs at eval/deploy.

Run (one shard):
  MUJOCO_GL=glfw .venv/bin/python scripts/recovery_shard_collect.py \
    --command 1 --pool data/datasets/recovery_pool_r2/dagger_cmd1 --recovery-per 40 --device mps
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from tinyvla.dagger import dagger_collect, pool_summary
from tinyvla.runtime import load_runtime
from tinyvla.paths import DATASETS_ROOT, CHECKPOINTS_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--command", type=int, required=True)
    ap.add_argument("--pool", required=True, help="this shard's OWN pool dir")
    ap.add_argument("--student", default=str(CHECKPOINTS_ROOT / "student291_recovery_r1"))
    ap.add_argument("--meta-root", default=str(DATASETS_ROOT / "so101_pickplace"))
    ap.add_argument("--meta-repo", default="local/so101_pickplace")
    ap.add_argument("--recovery-per", type=int, default=40)
    ap.add_argument("--cap", type=int, default=200)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=4200)
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device(args.device)
    meta = LeRobotDatasetMetadata(args.meta_repo, root=args.meta_root)
    # deployed preprocessing exactly (stats_source=checkpoint) so we visit the SAME
    # drift distribution the deployed student visits.
    rt = load_runtime(args.student, meta=meta, dataset_root=args.meta_root,
                      device=device, stats_source="checkpoint")
    policy = rt.policy.eval()
    assert rt.delta_actions is False, "student must be absolute-action for this pass"

    pool = Path(args.pool)
    print(f"[shard cmd{args.command}] student loaded in {time.time()-t0:.1f}s; "
          f"rolling out {args.recovery_per} recovery episodes -> {pool}", flush=True)

    n = dagger_collect(pool, policy, rt.preprocessor, rt.postprocessor, [args.command],
                       args.recovery_per, device=device, seed=args.seed,
                       cap=args.cap, delta_actions=False)

    import json
    print(f"[shard cmd{args.command}] DONE {n} episodes in {time.time()-t0:.1f}s  "
          f"pool={json.dumps(pool_summary(pool))}", flush=True)


if __name__ == "__main__":
    main()
