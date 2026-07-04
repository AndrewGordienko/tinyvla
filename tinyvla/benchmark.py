"""Benchmark SmolVLA checkpoints on the local SO-101 reach task.

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
import contextlib
import gc
import json
import os
import time
from pathlib import Path

from .paths import ARTIFACTS_ROOT, DATASETS_ROOT

os.environ.setdefault("HF_DATASETS_CACHE", str((ARTIFACTS_ROOT / ".cache" / "huggingface" / "datasets").resolve()))

import mujoco
import numpy as np
import torch
from torch.utils.data import DataLoader

# NOTE: import datasets before policies to avoid a circular import in lerobot 0.5.1
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import dataset_to_policy_features, make_policy, make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla import smolvlm_with_expert as smolvlm_module
from lerobot.utils.constants import ACTION

from .collect import EP_LEN, IMG
from .task import INSTRUCTION, SO101ReachTask
from tinyvla import load_pruned_smolvla


@contextlib.contextmanager
def local_transformers_only():
    """Force LeRobot's SmolVLA constructor to use cached VLM config/processor."""

    orig_auto_config = smolvlm_module.AutoConfig.from_pretrained
    orig_auto_processor = smolvlm_module.AutoProcessor.from_pretrained

    def local_config_from_pretrained(*args, **kwargs):
        kwargs.setdefault("local_files_only", True)
        return orig_auto_config(*args, **kwargs)

    def local_processor_from_pretrained(*args, **kwargs):
        kwargs.setdefault("local_files_only", True)
        return orig_auto_processor(*args, **kwargs)

    smolvlm_module.AutoConfig.from_pretrained = local_config_from_pretrained
    smolvlm_module.AutoProcessor.from_pretrained = local_processor_from_pretrained
    try:
        yield
    finally:
        smolvlm_module.AutoConfig.from_pretrained = orig_auto_config
        smolvlm_module.AutoProcessor.from_pretrained = orig_auto_processor


def parse_model_arg(value: str) -> tuple[str, Path]:
    if "=" in value:
        name, path = value.split("=", 1)
    else:
        path = value
        name = Path(path).name
    return name, Path(path)


def param_count(policy) -> int:
    return sum(param.numel() for param in policy.parameters())


def is_pruned_checkpoint(path: Path) -> bool:
    return (path / "pruning_meta.json").exists() or (path / "vocab_remap.json").exists()


def dataset_feature_overrides(meta: LeRobotDatasetMetadata) -> dict:
    features = dataset_to_policy_features(meta.features)
    output_features = {key: ft for key, ft in features.items() if ft.type.value == "ACTION"}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    return {"input_features": input_features, "output_features": output_features}


def load_policy(path: Path, device: str, meta: LeRobotDatasetMetadata):
    overrides = dataset_feature_overrides(meta)
    if is_pruned_checkpoint(path):
        return load_pruned_smolvla(path, device=device, config_overrides=overrides)

    cfg = SmolVLAConfig(pretrained_path=path, device=device, **overrides)
    with local_transformers_only():
        return make_policy(cfg=cfg, ds_meta=meta)


def make_processors(policy, model_path: Path, device: torch.device, meta: LeRobotDatasetMetadata):
    norm_feats = {**policy.config.input_features, **policy.config.output_features}
    return make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=model_path,
        preprocessor_overrides={
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": meta.stats,
                "features": norm_feats,
                "norm_map": policy.config.normalization_mapping,
            },
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "stats": meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        },
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


def offline_metrics(name: str, policy, preprocessor, dataset, args, teacher_bundle=None) -> dict:
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

    with torch.inference_mode():
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
                pred = pred.cpu()
                target = batch[ACTION].cpu()
                dims = target.shape[-1]
                pred = pred[:, :, :dims]
                err = pred - target
                abs_err_sum += float(err.abs().sum())
                sq_err_sum += float((err * err).sum())
                max_abs_err = max(max_abs_err, float(err.abs().max()))
                err_count += int(err.numel())

            if teacher_bundle is not None:
                teacher_name, teacher_policy, teacher_preprocessor = teacher_bundle
                teacher_batch = teacher_preprocessor(dict(raw_batch))
                noise = _fixed_noise(policy, batch, args.seed + 200_000 + batch_index)
                teacher_policy.reset()
                policy.reset()
                teacher_pred = teacher_policy.predict_action_chunk(teacher_batch, noise=noise).cpu()
                student_pred = policy.predict_action_chunk(batch, noise=noise.clone()).cpu()
                dims = min(teacher_pred.shape[-1], student_pred.shape[-1])
                err = student_pred[:, :, :dims] - teacher_pred[:, :, :dims]
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


def build_obs(env, renderer, device: torch.device):
    renderer.update_scene(env.data, camera="front")
    img = renderer.render()
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    state = torch.from_numpy(env.data.qpos[:6].copy().astype(np.float32))
    return {
        "observation.state": state.unsqueeze(0).to(device),
        "observation.images.front": img.unsqueeze(0).to(device),
        "task": [INSTRUCTION],
    }


def closed_loop_metrics(name: str, policy, preprocessor, postprocessor, args) -> dict:
    device = torch.device(args.device)
    env = SO101ReachTask(seed=args.seed)
    renderer = mujoco.Renderer(env.model, height=IMG, width=IMG)
    successes = 0
    distances: list[float] = []
    elapsed_start = time.time()

    for ep in range(args.episodes):
        env.reset()
        policy.reset()
        torch.manual_seed(args.seed + ep)
        for _ in range(args.steps):
            obs = preprocessor(build_obs(env, renderer, device))
            with torch.inference_mode():
                action = policy.select_action(obs)
            action = postprocessor(action).squeeze(0).cpu().numpy()
            env.step(action)
        ok = bool(env.success())
        dist = float(np.linalg.norm(env.ee_pos() - env.target_pos()))
        successes += int(ok)
        distances.append(dist)
        print(f"    {name}: closed-loop ep {ep + 1}/{args.episodes} {'reach' if ok else 'MISS'} dist={dist:.3f}")

    elapsed = time.time() - elapsed_start
    return {
        "closed_loop_episodes": args.episodes,
        "closed_loop_successes": successes,
        "closed_loop_success_rate": successes / args.episodes if args.episodes else 0.0,
        "closed_loop_final_dist_mean": float(np.mean(distances)) if distances else None,
        "closed_loop_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, help="NAME=PATH or PATH")
    parser.add_argument("--teacher", default=None, help="Optional NAME=PATH teacher model for vs-450M checks")
    parser.add_argument("--repo-id", default="local/so101_reach")
    parser.add_argument("--root", default=str(DATASETS_ROOT / "so101_reach"))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--no-action-metric", dest="action_metric", action="store_false")
    parser.add_argument("--teacher-mae-tolerance", type=float, default=1e-2)
    parser.add_argument("--teacher-max-abs-tolerance", type=float, default=1e-1)
    parser.add_argument("--closed-loop", action="store_true")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=EP_LEN)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--output", default="artifacts/benchmarks/latest.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    delta_timestamps = {"action": [i / meta.fps for i in range(SmolVLAConfig().chunk_size)]}
    dataset = LeRobotDataset(args.repo_id, root=args.root, delta_timestamps=delta_timestamps)

    results = {
        "dataset": {"repo_id": args.repo_id, "root": args.root, "frames": meta.total_frames},
        "device": args.device,
        "models": {},
    }

    teacher_bundle = None
    if args.teacher:
        teacher_name, teacher_path = parse_model_arg(args.teacher)
        print(f"\n== teacher {teacher_name} :: {teacher_path} ==")
        teacher_policy = load_policy(teacher_path, args.device, meta).to(device).eval()
        teacher_preprocessor, _ = make_processors(teacher_policy, teacher_path, device, meta)
        teacher_bundle = (teacher_name, teacher_policy, teacher_preprocessor)
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
        policy = load_policy(path, args.device, meta).to(device).eval()
        preprocessor, postprocessor = make_processors(policy, path, device, meta)

        metrics = {
            "path": str(path),
            "pruned": is_pruned_checkpoint(path),
            "parameters": param_count(policy),
        }
        metrics.update(offline_metrics(name, policy, preprocessor, dataset, args, teacher_bundle))
        if args.closed_loop:
            metrics.update(closed_loop_metrics(name, policy, preprocessor, postprocessor, args))

        results["models"][name] = metrics
        print(json.dumps(metrics, indent=2))

        del policy, preprocessor, postprocessor
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    if teacher_bundle is not None:
        _, teacher_policy, teacher_preprocessor = teacher_bundle
        del teacher_policy, teacher_preprocessor

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
