"""Recover layer-pruned SmolVLA candidates with short fine-tuning runs.

The default objective is the normal SmolVLA flow-matching loss on the local
SO-101 dataset. With ``--teacher`` enabled, the teacher first produces action
chunks and the student trains its flow objective toward those teacher chunks.
That is a cheaper approximation to full action-sampling distillation.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import shutil
import time
from pathlib import Path

from .paths import CHECKPOINTS_ROOT, DATASETS_ROOT

os.environ.setdefault("HF_DATASETS_CACHE", str((CHECKPOINTS_ROOT / ".cache" / "huggingface" / "datasets").resolve()))

import torch
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import cycle
from lerobot.policies.factory import dataset_to_policy_features, make_pre_post_processors
from lerobot.policies.smolvla import smolvlm_with_expert as smolvlm_module
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.utils.constants import ACTION

from tinyvla.benchmark import load_policy


@contextlib.contextmanager
def local_transformers_only():
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


def set_trainable(policy, mode: str) -> int:
    if mode == "all":
        for param in policy.parameters():
            param.requires_grad = True
    elif mode == "expert":
        for name, param in policy.named_parameters():
            param.requires_grad = (
                ".lm_expert." in name
                or ".action_" in name
                or ".state_proj" in name
            )
    else:
        raise ValueError(f"unknown trainable mode: {mode}")
    return sum(param.numel() for param in policy.parameters() if param.requires_grad)


def fixed_noise(policy, batch: dict, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    bsize = batch[ACTION].shape[0]
    return torch.randn(
        (bsize, policy.config.chunk_size, policy.config.max_action_dim),
        device=batch[ACTION].device,
        dtype=batch[ACTION].dtype,
    )


def copy_pruning_sidecars(source: Path, output: Path) -> None:
    for name in ("pruning_meta.json", "vocab_remap.json", "layer_pruning_meta.json"):
        src = source / name
        if src.exists():
            shutil.copy2(src, output / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--teacher", default=None)
    parser.add_argument("--repo-id", default="local/so101_reach")
    parser.add_argument("--root", default=str(DATASETS_ROOT / "so101_reach"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--trainable", choices=["expert", "all"], default="expert")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    student_path = Path(args.student)
    output = Path(args.output) if args.output else CHECKPOINTS_ROOT / f"{student_path.name}_recover"
    device = torch.device(args.device)
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    delta_timestamps = {"action": [i / meta.fps for i in range(SmolVLAConfig().chunk_size)]}
    dataset = LeRobotDataset(args.repo_id, root=args.root, delta_timestamps=delta_timestamps)

    student = load_policy(student_path, args.device, meta).to(device)
    student_preprocessor, student_postprocessor = make_processors(student, student_path, device, meta)
    trainable_params = set_trainable(student, args.trainable)

    teacher = None
    teacher_preprocessor = None
    if args.teacher:
        teacher_path = Path(args.teacher)
        teacher = load_policy(teacher_path, args.device, meta).to(device).eval()
        teacher_preprocessor, _ = make_processors(teacher, teacher_path, device, meta)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )
    optimizer = torch.optim.AdamW((p for p in student.parameters() if p.requires_grad), lr=args.lr)

    print(
        f"recovering {student_path} -> {output} on {device} | "
        f"steps={args.steps} batch={args.batch_size} lr={args.lr:g} "
        f"trainable={trainable_params/1e6:.1f}M teacher={'yes' if teacher else 'no'}"
    )

    output.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    losses: list[float] = []
    student.train()
    for step, raw_batch in zip(range(1, args.steps + 1), cycle(loader)):
        batch = student_preprocessor(dict(raw_batch))
        if teacher is not None and teacher_preprocessor is not None:
            with torch.inference_mode():
                teacher_batch = teacher_preprocessor(dict(raw_batch))
                noise = fixed_noise(teacher, teacher_batch, args.seed + step)
                teacher.reset()
                teacher_actions = teacher.predict_action_chunk(teacher_batch, noise=noise)
            batch[ACTION] = teacher_actions.detach()

        torch.manual_seed(args.seed + 10_000 + step)
        loss, _ = student.forward(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in student.parameters() if p.requires_grad), 10.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        losses.append(float(loss.item()))
        if step % args.log_every == 0 or step == 1 or step == args.steps:
            recent = losses[-args.log_every :]
            print(f"  step {step:5d}/{args.steps}  loss {sum(recent)/len(recent):.5f}")

        if args.save_every and step % args.save_every == 0:
            student.save_pretrained(output)
            student_preprocessor.save_pretrained(output)
            student_postprocessor.save_pretrained(output)
            copy_pruning_sidecars(student_path, output)

    student.save_pretrained(output)
    student_preprocessor.save_pretrained(output)
    student_postprocessor.save_pretrained(output)
    copy_pruning_sidecars(student_path, output)

    meta_out = {
        "student": str(student_path),
        "teacher": args.teacher,
        "repo_id": args.repo_id,
        "root": args.root,
        "device": args.device,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "trainable": args.trainable,
        "trainable_params": trainable_params,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_mean": sum(losses) / len(losses) if losses else None,
        "seconds": time.time() - t0,
    }
    (output / "recovery_meta.json").write_text(json.dumps(meta_out, indent=2) + "\n")
    print(json.dumps(meta_out, indent=2))
    print(f"saved recovered checkpoint to {output}")

    del student, student_preprocessor, student_postprocessor, teacher, teacher_preprocessor
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
