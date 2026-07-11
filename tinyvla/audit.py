"""Audit SmolVLA trainability and signal flow on a local LeRobot batch."""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from .fast_dataset import FastChunkDataset
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.utils.constants import ACTION

from .paths import DATASETS_ROOT
from .runtime import load_runtime
from .trainability import TRAINABLE_MODES, group_for_param, set_trainable


def tensor_batch_clone(batch: dict) -> dict:
    cloned = {}
    for key, value in batch.items():
        cloned[key] = value.detach().clone() if torch.is_tensor(value) else value
    return cloned


def param_summary(policy) -> dict:
    groups: dict[str, dict[str, float | int]] = {}
    for name, param in policy.named_parameters():
        group = group_for_param(name)
        row = groups.setdefault(group, {"params": 0, "trainable": 0, "grad_norm": 0.0, "grad_params": 0})
        row["params"] += param.numel()
        if param.requires_grad:
            row["trainable"] += param.numel()
        if param.grad is not None:
            grad_norm = float(param.grad.detach().float().norm().item())
            if math.isfinite(grad_norm):
                row["grad_norm"] += grad_norm
                row["grad_params"] += param.numel()
    return groups


def config_summary(policy) -> dict:
    keys = [
        "pretrained_path",
        "vlm_model_name",
        "load_vlm_weights",
        "freeze_vision_encoder",
        "train_expert_only",
        "train_state_proj",
        "num_vlm_layers",
        "num_expert_layers",
        "self_attn_every_n_layers",
        "expert_width_multiplier",
        "chunk_size",
        "num_steps",
        "max_state_dim",
        "max_action_dim",
        "image_features",
    ]
    return {key: str(getattr(policy.config, key, None)) for key in keys}


def fixed_noise_and_time(policy, batch: dict, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    noise = torch.randn(
        (batch[ACTION].shape[0], policy.config.chunk_size, policy.config.max_action_dim),
        device=batch[ACTION].device,
        dtype=batch[ACTION].dtype,
    )
    time = torch.rand(batch[ACTION].shape[0], device=batch[ACTION].device)
    time = time.to(dtype=torch.float32) * 0.999 + 0.001
    return noise, time


def image_keys(policy, batch: dict) -> list[str]:
    configured = list(getattr(policy.config, "image_features", {}) or {})
    return [key for key in configured if key in batch]


def audit_batch(policy, batch: dict, seed: int) -> dict:
    policy.train()
    policy.zero_grad(set_to_none=True)

    keys = image_keys(policy, batch)
    grad_batch = tensor_batch_clone(batch)
    for key in keys:
        grad_batch[key].requires_grad_(True)

    noise, time = fixed_noise_and_time(policy, grad_batch, seed)
    loss, loss_dict = policy.forward(grad_batch, noise=noise, time=time)
    loss.backward()

    pixel_grad_norms = {
        key: (
            float(grad_batch[key].grad.detach().float().norm().item())
            if grad_batch[key].grad is not None
            else 0.0
        )
        for key in keys
    }

    with torch.inference_mode():
        normal_batch = tensor_batch_clone(batch)
        black_batch = tensor_batch_clone(batch)
        for key in keys:
            black_batch[key].zero_()
        noise, time = fixed_noise_and_time(policy, normal_batch, seed)
        normal_loss, _ = policy.forward(normal_batch, noise=noise, time=time)
        black_loss, _ = policy.forward(black_batch, noise=noise, time=time)

    return {
        "loss": float(loss.item()),
        "loss_dict": loss_dict,
        "image_keys": keys,
        "pixel_grad_norms": pixel_grad_norms,
        "fixed_noise_loss_normal": float(normal_loss.item()),
        "fixed_noise_loss_black_images": float(black_loss.item()),
        "fixed_noise_loss_image_delta": float((black_loss - normal_loss).abs().item()),
        "parameter_groups": param_summary(policy),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to a SmolVLA checkpoint")
    parser.add_argument("--repo-id", default="local/so101_reach")
    parser.add_argument("--root", default=str(DATASETS_ROOT / "so101_reach"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--trainable", choices=TRAINABLE_MODES, default="checkpoint",
                        help="Apply a trainability mode before measuring gradients.")
    parser.add_argument("--base-checkpoint", action="store_true",
                        help="Treat --model as an immutable base checkpoint without an action marker.")
    args = parser.parse_args()

    device = torch.device(args.device)
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    delta_timestamps = {"action": [i / meta.fps for i in range(SmolVLAConfig().chunk_size)]}
    dataset = FastChunkDataset(args.repo_id, root=args.root, delta_timestamps=delta_timestamps)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)
    raw_batch = next(iter(loader))

    model_path = Path(args.model)
    with contextlib.redirect_stdout(sys.stderr):
        runtime = load_runtime(
            model_path, meta=meta, dataset_root=args.root, device=device, stats_source="dataset"
            , base_checkpoint=args.base_checkpoint
        )
        policy = runtime.policy
        trainable_params = set_trainable(policy, args.trainable)
        preprocessor = runtime.preprocessor
    batch = preprocessor(dict(raw_batch))

    result = {
        "model": str(model_path),
        "dataset": {"repo_id": args.repo_id, "root": args.root, "frames": meta.total_frames},
        "device": args.device,
        "trainable_mode": args.trainable,
        "trainable_params": trainable_params,
        "config": config_summary(policy),
        "audit": audit_batch(policy, batch, args.seed),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
