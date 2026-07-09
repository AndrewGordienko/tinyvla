"""DAgger / targeted-data loop driver.

Each round:
  1. build a fresh LeRobot dataset from the episode pool
  2. train (subprocess -> tinyvla.train) from smolvla_base on it, with the levers
     (n_action_steps, delta actions) and closed-loop-in-the-loop eval
  3. load the trained checkpoint and score every command closed-loop
  4. find the worst commands (the knowledge gaps)
  5. top up the pool for those gaps two ways:
       - curriculum : more scripted-expert demos (fixes under-representation)
       - DAgger     : roll out THIS policy, label the states it drifts into with
                      the reactive expert (fixes compounding error)
  6. repeat -> the dataset grows where the policy is weakest

Training restarts from base each round (batch-DAgger); it's the *data* that
improves. Warm-starting from the previous checkpoint is a future optimisation
(noted in the H100 recipe).

Run (local smoke):
  python -m tinyvla.dagger_loop --rounds 1 --steps 3 --commands 0 \
      --curriculum-per 1 --dagger-per 1 --closed-loop-cap 40
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from .paths import CHECKPOINTS_ROOT, DATASETS_ROOT
from .dagger import (collect_expert_episodes_parallel, dagger_collect, build_lerobot_dataset,
                     pool_summary, pool_episodes)
from .eval_closedloop import evaluate_per_command, worst_commands, format_metrics


def _load_policy(ckpt: Path, device: str, meta):
    from .benchmark import load_policy, make_processors
    policy = load_policy(ckpt, device, meta).to(device)
    pre, post = make_processors(policy, ckpt, torch.device(device), meta)
    return policy, pre, post


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--commands", default="0,1,2,3,6,7",
                    help="Command indices in play (stacking 4,5 excluded by default: "
                         "the reactive labeler is weak there).")
    ap.add_argument("--seed-per", type=int, default=3, help="initial expert demos per command")
    ap.add_argument("--curriculum-per", type=int, default=2, help="expert top-ups per worst command / round")
    ap.add_argument("--dagger-per", type=int, default=2, help="DAgger rollouts per worst command / round")
    ap.add_argument("--worst-k", type=int, default=2, help="how many worst commands to target / round")
    ap.add_argument("--collect-workers", type=int, default=6,
                    help="Parallel processes for scripted-expert collection (seed + curriculum); "
                         "capped at the number of commands being collected. 1 = sequential.")
    ap.add_argument("--steps", type=int, default=600, help="train steps / round")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--n-action-steps", type=int, default=10)
    ap.add_argument("--trainable", default="checkpoint",
                    choices=("checkpoint", "expert", "brain", "brain_visual", "all"),
                    help="Trainability mode forwarded to tinyvla.train.")
    ap.add_argument("--backbone-lr-scale", type=float, default=1.0,
                    help="Forwarded to tinyvla.train (use ~0.1 with unfrozen modes).")
    ap.add_argument("--warmup-steps", type=int, default=0,
                    help="Forwarded to tinyvla.train (use ~500 with unfrozen modes).")
    ap.add_argument("--warm-start", action="store_true",
                    help="Rounds >1 initialise from the previous round's checkpoint instead of "
                         "smolvla_base, so --warm-steps can be much smaller than --steps. "
                         "Faster wall-clock; slight departure from pure batch-DAgger.")
    ap.add_argument("--warm-steps", type=int, default=0,
                    help="Train steps for warm-started rounds (default: same as --steps).")
    ap.add_argument("--delta-actions", action="store_true")
    ap.add_argument("--closed-loop-cap", type=int, default=180)
    ap.add_argument("--closed-loop-seed", type=int, default=100)
    ap.add_argument("--closed-loop-episodes", type=int, default=1,
                    help="Rollouts per command per eval; use >=3 so round-over-round "
                         "comparisons and worst-command ranking aren't coin flips.")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--pool", default=str(DATASETS_ROOT / "dagger_pool"))
    ap.add_argument("--work", default=str(CHECKPOINTS_ROOT / "dagger_run"))
    ap.add_argument("--repo-id", default="local/dagger")
    args = ap.parse_args()

    commands = [int(x) for x in args.commands.split(",") if x != ""]
    pool = Path(args.pool)
    work = Path(args.work); work.mkdir(parents=True, exist_ok=True)
    ds_root = Path(args.work) / "dataset"

    # ---- seed pool with initial expert demos (once) -----------------------
    if not pool_episodes(pool):
        print(f"[seed] collecting {args.seed_per} expert demos x {len(commands)} commands "
              f"({args.collect_workers} workers)", flush=True)
        collect_expert_episodes_parallel(pool, commands, args.seed_per, seed=1,
                                         workers=args.collect_workers)
    print(f"[pool] {pool_summary(pool)}", flush=True)

    history = []
    prev_ckpt: Path | None = None
    for rnd in range(1, args.rounds + 1):
        print(f"\n===== ROUND {rnd}/{args.rounds} =====", flush=True)

        # 1. build dataset from the whole pool
        build_lerobot_dataset(pool, args.repo_id, ds_root, delta_actions=args.delta_actions)

        # 2. train on it (subprocess), closed-loop eval during training; from base,
        #    or from the previous round's checkpoint when --warm-start
        warm = args.warm_start and prev_ckpt is not None
        steps = (args.warm_steps or args.steps) if warm else args.steps
        ckpt = work / f"round_{rnd:02d}"
        cmd = [
            sys.executable, "-m", "tinyvla.train",
            "--repo-id", args.repo_id, "--root", str(ds_root),
            "--steps", str(steps), "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers),
            "--n-action-steps", str(args.n_action_steps),
            "--trainable", args.trainable,
            "--backbone-lr-scale", str(args.backbone_lr_scale),
            "--warmup-steps", str(args.warmup_steps),
            "--device", args.device, "--output", str(ckpt),
            "--save-every", str(steps),
            "--closed-loop-every", str(steps),
            "--closed-loop-commands", args.commands,
            "--closed-loop-cap", str(args.closed_loop_cap),
            "--closed-loop-episodes", str(args.closed_loop_episodes),
            "--save-best-closed-loop",
        ]
        if warm:
            cmd += ["--init-from", str(prev_ckpt)]
        if args.delta_actions:
            cmd.append("--delta-actions")
        print(f"[train] {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)

        # 3. load checkpoint, score every command
        meta = LeRobotDatasetMetadata(args.repo_id, root=str(ds_root))
        eval_ckpt = ckpt / "best_closed_loop"
        if not eval_ckpt.exists():
            eval_ckpt = ckpt
        prev_ckpt = eval_ckpt   # warm-start source for the next round
        policy, pre, post = _load_policy(eval_ckpt, args.device, meta)
        per_cmd = evaluate_per_command(
            policy, pre, post, device=torch.device(args.device), commands=commands,
            cap=args.closed_loop_cap, seed=args.closed_loop_seed, delta_actions=args.delta_actions,
            episodes=args.closed_loop_episodes)
        for ci in commands:
            print(f"  cmd {ci}: {format_metrics(per_cmd[ci])}", flush=True)
        mean_succ = sum(m["success_rate"] for m in per_cmd.values()) / len(per_cmd)

        # 4. find gaps
        gaps = worst_commands(per_cmd, args.worst_k)
        print(f"[round {rnd}] mean success {mean_succ:.0%}  worst commands -> {gaps}", flush=True)

        # 5. top up the pool for the gaps: curriculum + DAgger
        if rnd < args.rounds:
            collect_expert_episodes_parallel(pool, gaps, args.curriculum_per, seed=100 + rnd,
                                             source="curriculum", workers=args.collect_workers)
            dagger_collect(pool, policy, pre, post, gaps, args.dagger_per,
                           device=torch.device(args.device), cap=args.closed_loop_cap,
                           seed=500 + rnd, delta_actions=args.delta_actions)
            print(f"[pool] after top-up: {pool_summary(pool)}", flush=True)

        history.append({"round": rnd, "checkpoint": str(eval_ckpt), "mean_success": mean_succ,
                        "per_command": {ci: per_cmd[ci] for ci in commands},
                        "gaps": gaps, "pool": pool_summary(pool)})
        del policy, pre, post
        if args.device == "mps":
            torch.mps.empty_cache()

    (work / "dagger_history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(f"\ndone -> {work / 'dagger_history.json'}", flush=True)
    print("mean success by round:", [f"{h['round']}:{h['mean_success']:.0%}" for h in history])


if __name__ == "__main__":
    main()
