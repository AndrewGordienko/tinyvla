"""Benchmark SmolVLA checkpoints on the local SO-101 pick/place task.

The fast default is an offline imitation benchmark over a LeRobot dataset:
it reports flow-matching loss and one-step action chunk MAE/RMSE against the
scripted expert actions. The optional closed-loop mode runs the policy in the
MuJoCo task and reports success rate.

Examples:
  python3 -m tinyvla.benchmark --model base=data/models/smolvla_base --batches 4
  python3 -m tinyvla.benchmark --model base=data/models/smolvla_base \
    --model pruned=data/models/smolvla_headless_vocab_so101 --batches 8
  python3 -m tinyvla.benchmark --teacher teacher=data/models/smolvla_base \
    --model pruned=data/models/smolvla_headless_vocab_so101 --batches 8
  python3 -m tinyvla.benchmark --model pruned=data/models/smolvla_headless_vocab_so101 --closed-loop --episodes 5
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

from .paths import ARTIFACTS_ROOT, DATASETS_ROOT

os.environ.setdefault("HF_DATASETS_CACHE", str((ARTIFACTS_ROOT / ".cache" / "huggingface" / "datasets").resolve()))

import numpy as np
import torch
from torch.utils.data import DataLoader

# Import datasets before policies to avoid LeRobot's policy/dataset import cycle.
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from .fast_dataset import FastChunkDataset
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.utils.constants import ACTION

from .collect import EP_LEN
from .determinism import preserve_rng_state
from .eval_closedloop import evaluate_closed_loop
from .runtime import (
    experiment_metadata,
    is_pruned_checkpoint,
    load_runtime,
    make_processors as _make_processors,
)


def parse_model_arg(value: str) -> tuple[str, Path]:
    if "=" in value:
        name, path = value.split("=", 1)
    else:
        path = value
        name = Path(path).name
    return name, Path(path)


def param_count(policy) -> int:
    return sum(param.numel() for param in policy.parameters())


def load_policy(path: Path, device: str, meta: LeRobotDatasetMetadata):
    """Compatibility wrapper; new code should retain the complete RuntimeBundle."""
    return load_runtime(
        path, meta=meta, dataset_root=getattr(meta, "root", None), device=device
    ).policy


def make_processors(policy, model_path: Path, device: torch.device, meta: LeRobotDatasetMetadata):
    """Compatibility wrapper around the canonical processor construction."""
    return _make_processors(
        policy, model_path, device, meta, stats_source="checkpoint"
    )


def _error_stats(abs_err_sum: float, sq_err_sum: float, max_abs: float, count: int) -> dict:
    if not count:
        return {}
    return {
        "mae": abs_err_sum / count,
        "rmse": (sq_err_sum / count) ** 0.5,
        "max_abs": max_abs,
    }


def _fixed_noise(policy, batch: dict, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    bsize = batch[ACTION].shape[0]
    return torch.randn(
        (bsize, policy.config.chunk_size, policy.config.max_action_dim),
        device=batch[ACTION].device,
        dtype=batch[ACTION].dtype,
    )


def offline_metrics(name: str, policy, preprocessor, postprocessor, dataset, args, teacher_bundle=None) -> dict:
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )

    policy.eval()
    losses: list[float] = []
    abs_err_sum = 0.0
    sq_err_sum = 0.0
    max_abs_err = 0.0
    err_count = 0
    teacher_abs_err_sum = 0.0
    teacher_sq_err_sum = 0.0
    teacher_max_abs_err = 0.0
    teacher_err_count = 0
    elapsed_start = time.time()

    with preserve_rng_state(), torch.inference_mode():
        for batch_index, raw_batch in enumerate(loader, start=1):
            if batch_index > args.batches:
                break
            batch = preprocessor(dict(raw_batch))

            torch.manual_seed(args.seed + batch_index)
            loss, _ = policy.forward(batch)
            losses.append(float(loss.item()))

            if args.action_metric:
                noise = _fixed_noise(policy, batch, args.seed + 100_000 + batch_index)
                policy.reset()
                pred = policy.predict_action_chunk(batch, noise=noise)
                pred = postprocessor(pred).cpu()
                target = postprocessor(batch[ACTION]).cpu()
                dims = target.shape[-1]
                pred = pred[:, :, :dims]
                err = pred - target
                if "action_is_pad" in batch:
                    valid = (~batch["action_is_pad"].bool()).cpu().unsqueeze(-1).expand_as(err)
                    err = err[valid]
                abs_err_sum += float(err.abs().sum())
                sq_err_sum += float((err * err).sum())
                max_abs_err = max(max_abs_err, float(err.abs().max()))
                err_count += int(err.numel())

            if teacher_bundle is not None:
                teacher_name, teacher_policy, teacher_preprocessor, teacher_postprocessor = teacher_bundle
                teacher_batch = teacher_preprocessor(dict(raw_batch))
                noise = _fixed_noise(policy, batch, args.seed + 200_000 + batch_index)
                teacher_policy.reset()
                policy.reset()
                teacher_pred = teacher_postprocessor(
                    teacher_policy.predict_action_chunk(teacher_batch, noise=noise)
                ).cpu()
                student_pred = postprocessor(
                    policy.predict_action_chunk(batch, noise=noise.clone())
                ).cpu()
                dims = min(teacher_pred.shape[-1], student_pred.shape[-1])
                err = student_pred[:, :, :dims] - teacher_pred[:, :, :dims]
                if "action_is_pad" in batch:
                    valid = (~batch["action_is_pad"].bool()).cpu().unsqueeze(-1).expand_as(err)
                    err = err[valid]
                teacher_abs_err_sum += float(err.abs().sum())
                teacher_sq_err_sum += float((err * err).sum())
                teacher_max_abs_err = max(teacher_max_abs_err, float(err.abs().max()))
                teacher_err_count += int(err.numel())

            print(f"    {name}: offline batch {batch_index}/{args.batches} loss={loss.item():.5f}")

    elapsed = time.time() - elapsed_start
    metrics = {
        "offline_batches": len(losses),
        "offline_loss_mean": float(np.mean(losses)) if losses else None,
        "offline_loss_std": float(np.std(losses)) if losses else None,
        "offline_seconds": elapsed,
    }
    if err_count:
        action_stats = _error_stats(abs_err_sum, sq_err_sum, max_abs_err, err_count)
        metrics["expert_action_mae"] = action_stats["mae"]
        metrics["expert_action_rmse"] = action_stats["rmse"]
        metrics["expert_action_max_abs"] = action_stats["max_abs"]
    if teacher_err_count:
        teacher_stats = _error_stats(
            teacher_abs_err_sum,
            teacher_sq_err_sum,
            teacher_max_abs_err,
            teacher_err_count,
        )
        metrics["teacher"] = teacher_name
        metrics["teacher_action_mae"] = teacher_stats["mae"]
        metrics["teacher_action_rmse"] = teacher_stats["rmse"]
        metrics["teacher_action_max_abs"] = teacher_stats["max_abs"]
        metrics["teacher_mae_tolerance"] = args.teacher_mae_tolerance
        metrics["teacher_max_abs_tolerance"] = args.teacher_max_abs_tolerance
        metrics["teacher_within_tolerance"] = (
            teacher_stats["mae"] <= args.teacher_mae_tolerance
            and teacher_stats["max_abs"] <= args.teacher_max_abs_tolerance
        )
    return metrics


def closed_loop_metrics(name: str, policy, preprocessor, postprocessor, args, *, delta_actions: bool) -> dict:
    elapsed_start = time.time()
    result = evaluate_closed_loop(
        policy,
        preprocessor,
        postprocessor,
        device=torch.device(args.device),
        commands=args.commands,
        cap=args.steps,
        seed=args.seed,
        delta_actions=delta_actions,
        episodes=args.episodes,
    )
    print(f"    {name}: closed-loop {result['successes']}/{result['n']}")
    return {"closed_loop": result, "closed_loop_seconds": time.time() - elapsed_start}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, help="NAME=PATH or PATH")
    parser.add_argument("--teacher", default=None, help="Optional NAME=PATH teacher model for vs-450M checks")
    parser.add_argument("--repo-id", default="local/so101_pickplace")
    parser.add_argument("--root", default=str(DATASETS_ROOT / "so101_pickplace"))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--no-action-metric", dest="action_metric", action="store_false")
    parser.add_argument("--teacher-mae-tolerance", type=float, default=1e-2)
    parser.add_argument("--teacher-max-abs-tolerance", type=float, default=1e-1)
    parser.add_argument("--closed-loop", action="store_true")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=EP_LEN)
    parser.add_argument("--commands", default="0,1,2,3",
                        help="Comma-separated command indices; long-horizon commands are reported separately.")
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--output", default="artifacts/benchmarks/latest.json")
    args = parser.parse_args()
    args.commands = [int(value) for value in args.commands.split(",") if value]

    device = torch.device(args.device)
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    delta_timestamps = {"action": [i / meta.fps for i in range(SmolVLAConfig().chunk_size)]}
    dataset = FastChunkDataset(args.repo_id, root=args.root, delta_timestamps=delta_timestamps)

    results = {
        "dataset": {"repo_id": args.repo_id, "root": args.root, "frames": meta.total_frames},
        "device": args.device,
        "models": {},
        "experiment": experiment_metadata(seed=args.seed),
    }

    teacher_bundle = None
    if args.teacher:
        teacher_name, teacher_path = parse_model_arg(args.teacher)
        print(f"\n== teacher {teacher_name} :: {teacher_path} ==")
        teacher_runtime = load_runtime(
            teacher_path, meta=meta, dataset_root=args.root, device=device, stats_source="checkpoint"
        )
        teacher_policy = teacher_runtime.policy.eval()
        teacher_bundle = (
            teacher_name, teacher_policy, teacher_runtime.preprocessor, teacher_runtime.postprocessor
        )
        results["teacher"] = {
            "name": teacher_name,
            "path": str(teacher_path),
            "parameters": param_count(teacher_policy),
            "mae_tolerance": args.teacher_mae_tolerance,
            "max_abs_tolerance": args.teacher_max_abs_tolerance,
        }

    for model_arg in args.model:
        name, path = parse_model_arg(model_arg)
        print(f"\n== {name} :: {path} ==")
        runtime = load_runtime(
            path, meta=meta, dataset_root=args.root, device=device, stats_source="checkpoint"
        )
        policy = runtime.policy.eval()
        preprocessor, postprocessor = runtime.preprocessor, runtime.postprocessor

        metrics = {
            "path": str(path),
            "pruned": is_pruned_checkpoint(path),
            "parameters": param_count(policy),
        }
        metrics["action_semantics"] = runtime.action_semantics
        metrics["load_report"] = runtime.load_report
        metrics.update(offline_metrics(
            name, policy, preprocessor, postprocessor, dataset, args, teacher_bundle
        ))
        if args.closed_loop:
            metrics.update(closed_loop_metrics(
                name, policy, preprocessor, postprocessor, args,
                delta_actions=runtime.delta_actions,
            ))

        results["models"][name] = metrics
        print(json.dumps(metrics, indent=2))

        del policy, preprocessor, postprocessor
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    if teacher_bundle is not None:
        _, teacher_policy, teacher_preprocessor, teacher_postprocessor = teacher_bundle
        del teacher_policy, teacher_preprocessor, teacher_postprocessor

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
