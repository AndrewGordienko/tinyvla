"""Recover layer-pruned SmolVLA candidates with short fine-tuning runs.

The default objective is the normal SmolVLA flow-matching loss on the local
SO-101 dataset. With ``--teacher`` enabled, the teacher first produces action
chunks and the student trains its flow objective toward those teacher chunks.
``--objective teacher_velocity`` instead matches the teacher's internal
flow-matching velocity field at the same noise/time sample.
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
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import cycle
from lerobot.policies.factory import dataset_to_policy_features, make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
from lerobot.policies.smolvla import smolvlm_with_expert as smolvlm_module
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from tinyvla.benchmark import load_policy
from tinyvla.eval_closedloop import evaluate_closed_loop, format_metrics


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


def fixed_time(batch: dict, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.rand(batch[ACTION].shape[0], device=batch[ACTION].device)


def flow_velocity(policy, batch: dict, noise: torch.Tensor, time_tensor: torch.Tensor) -> torch.Tensor:
    """Return SmolVLA's predicted flow velocity for a prepared batch."""

    if policy.config.adapt_to_pi_aloha:
        batch[OBS_LANGUAGE_TOKENS] = batch[OBS_LANGUAGE_TOKENS]
        batch = policy._prepare_batch(batch)

    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    actions = policy.prepare_action(batch)

    time_expanded = time_tensor[:, None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    model = policy.model
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state=state,
    )
    suffix_embs, suffix_pad_masks, suffix_att_masks = model.embed_suffix(x_t, time_tensor)

    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
    att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    (_, suffix_out), _ = model.vlm_with_expert.forward(
        attention_mask=att_2d_masks,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, suffix_embs],
        use_cache=False,
        fill_kv_cache=False,
    )
    suffix_out = suffix_out[:, -policy.config.chunk_size :].to(dtype=torch.float32)
    return model.action_out_proj(suffix_out)


def teacher_velocity_loss(
    student,
    student_batch: dict,
    teacher,
    teacher_batch: dict,
    seed: int,
) -> torch.Tensor:
    noise = fixed_noise(student, student_batch, seed)
    time_tensor = fixed_time(student_batch, seed + 500_000)
    with torch.no_grad():
        teacher_v = flow_velocity(teacher, teacher_batch, noise.detach(), time_tensor.detach()).detach().clone()
    student_v = flow_velocity(student, student_batch, noise, time_tensor)
    dims = min(student_v.shape[-1], teacher_v.shape[-1], student.config.action_feature.shape[0])
    return torch.nn.functional.mse_loss(student_v[:, :, :dims], teacher_v[:, :, :dims])


def copy_pruning_sidecars(source: Path, output: Path) -> None:
    for name in ("pruning_meta.json", "vocab_remap.json", "layer_pruning_meta.json"):
        src = source / name
        if src.exists():
            shutil.copy2(src, output / name)


def save_checkpoint(policy, preprocessor, postprocessor, source: Path, output: Path,
                    delta_actions: bool = False) -> None:
    output.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(output)
    preprocessor.save_pretrained(output)
    postprocessor.save_pretrained(output)
    copy_pruning_sidecars(source, output)
    if delta_actions:
        (output / "delta_actions.json").write_text('{"delta_actions": true}\n')


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
    parser.add_argument(
        "--objective",
        choices=["expert", "teacher_action", "teacher_velocity", "mixed_action_expert"],
        default=None,
        help="Defaults to teacher_action when --teacher is set, otherwise expert.",
    )
    parser.add_argument("--teacher-loss-weight", type=float, default=1.0)
    parser.add_argument("--expert-loss-weight", type=float, default=0.25)
    parser.add_argument("--save-step-subdirs", action="store_true")
    parser.add_argument("--step-offset", type=int, default=0)
    parser.add_argument("--trainable", choices=["expert", "all"], default="expert")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--closed-loop-every", type=int, default=0,
                        help="Run a closed-loop rollout eval every N steps (0=off). "
                             "Judge/select checkpoints by this, not offline loss.")
    parser.add_argument("--closed-loop-commands", default="0,1,2,3",
                        help="Comma-separated COMMANDS indices to roll out.")
    parser.add_argument("--closed-loop-cap", type=int, default=180)
    parser.add_argument("--closed-loop-seed", type=int, default=100)
    parser.add_argument("--save-best-closed-loop", action="store_true",
                        help="Keep the checkpoint with the best closed-loop success in <output>/best_closed_loop.")
    parser.add_argument("--n-action-steps", type=int, default=None,
                        help="Actions executed per replan (<=chunk_size). Try 10 for tighter closed-loop "
                             "(base default 50 causes open-loop drift).")
    parser.add_argument("--delta-actions", action="store_true",
                        help="Dataset stores joint deltas (action-state); closed-loop eval adds the live "
                             "pose back. Set this to match a delta-actions dataset.")
    args = parser.parse_args()
    cl_commands = [int(x) for x in args.closed_loop_commands.split(",") if x != ""]
    objective = args.objective or ("teacher_action" if args.teacher else "expert")
    if (objective.startswith("teacher") or objective.startswith("mixed")) and not args.teacher:
        raise SystemExit(f"--objective {objective} requires --teacher")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS requested, but torch.backends.mps.is_available() is false in this Python environment")

    student_path = Path(args.student)
    output = Path(args.output) if args.output else CHECKPOINTS_ROOT / f"{student_path.name}_recover"
    device = torch.device(args.device)
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    delta_timestamps = {"action": [i / meta.fps for i in range(SmolVLAConfig().chunk_size)]}
    dataset = LeRobotDataset(args.repo_id, root=args.root, delta_timestamps=delta_timestamps)

    student = load_policy(student_path, args.device, meta).to(device)
    if args.n_action_steps is not None:
        student.config.n_action_steps = args.n_action_steps
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
        f"trainable={trainable_params/1e6:.1f}M objective={objective} teacher={'yes' if teacher else 'no'}",
        flush=True,
    )

    output.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    losses: list[float] = []
    cl_history: list[dict] = []
    best_cl = -1.0
    student.train()
    for step, raw_batch in zip(range(1, args.steps + 1), cycle(loader)):
        batch = student_preprocessor(dict(raw_batch))
        expert_batch = None
        if objective == "mixed_action_expert":
            expert_batch = dict(batch)

        if objective in {"teacher_action", "mixed_action_expert"} and teacher is not None and teacher_preprocessor is not None:
            with torch.inference_mode():
                teacher_batch = teacher_preprocessor(dict(raw_batch))
                noise = fixed_noise(teacher, teacher_batch, args.seed + step)
                teacher.reset()
                teacher_actions = teacher.predict_action_chunk(teacher_batch, noise=noise)
            batch[ACTION] = teacher_actions.detach()

        torch.manual_seed(args.seed + 10_000 + step)
        if objective == "teacher_velocity" and teacher is not None and teacher_preprocessor is not None:
            teacher_batch = teacher_preprocessor(dict(raw_batch))
            loss = teacher_velocity_loss(student, batch, teacher, teacher_batch, args.seed + step)
        elif objective == "mixed_action_expert" and expert_batch is not None:
            teacher_loss, _ = student.forward(batch)
            expert_loss, _ = student.forward(expert_batch)
            loss = args.teacher_loss_weight * teacher_loss + args.expert_loss_weight * expert_loss
        else:
            loss, _ = student.forward(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in student.parameters() if p.requires_grad), 10.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        losses.append(float(loss.item()))
        if step % args.log_every == 0 or step == 1 or step == args.steps:
            recent = losses[-args.log_every :]
            print(f"  step {step:5d}/{args.steps}  loss {sum(recent)/len(recent):.5f}", flush=True)

        if args.save_every and step % args.save_every == 0:
            save_checkpoint(student, student_preprocessor, student_postprocessor, student_path, output,
                            delta_actions=args.delta_actions)
            if args.save_step_subdirs:
                save_checkpoint(
                    student,
                    student_preprocessor,
                    student_postprocessor,
                    student_path,
                    output / f"step_{args.step_offset + step:06d}",
                    delta_actions=args.delta_actions,
                )

        if args.closed_loop_every and (step % args.closed_loop_every == 0 or step == args.steps):
            cl = evaluate_closed_loop(
                student, student_preprocessor, student_postprocessor,
                device=device, commands=cl_commands,
                cap=args.closed_loop_cap, seed=args.closed_loop_seed,
                delta_actions=args.delta_actions,
            )
            print(f"  step {step:5d}/{args.steps}  closed-loop {format_metrics(cl)}", flush=True)
            cl_history.append({"step": args.step_offset + step, **cl})
            if args.save_best_closed_loop and cl["success_rate"] > best_cl:
                best_cl = cl["success_rate"]
                save_checkpoint(student, student_preprocessor, student_postprocessor,
                                student_path, output / "best_closed_loop",
                                delta_actions=args.delta_actions)
                print(f"    new best closed-loop success {cl['success_rate']:.0%} "
                      f"-> saved {output / 'best_closed_loop'}", flush=True)

    save_checkpoint(student, student_preprocessor, student_postprocessor, student_path, output,
                    delta_actions=args.delta_actions)

    meta_out = {
        "student": str(student_path),
        "teacher": args.teacher,
        "repo_id": args.repo_id,
        "root": args.root,
        "device": args.device,
        "steps": args.steps,
        "step_offset": args.step_offset,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "objective": objective,
        "teacher_loss_weight": args.teacher_loss_weight,
        "expert_loss_weight": args.expert_loss_weight,
        "trainable": args.trainable,
        "trainable_params": trainable_params,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_mean": sum(losses) / len(losses) if losses else None,
        "closed_loop_history": cl_history,
        "closed_loop_best_success": max((c["success_rate"] for c in cl_history), default=None),
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
