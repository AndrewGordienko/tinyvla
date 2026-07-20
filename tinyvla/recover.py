"""Recover layer-pruned SmolVLA candidates with short fine-tuning runs.

The default objective is the normal SmolVLA flow-matching loss on the local
SO-101 dataset. With ``--teacher`` enabled, the teacher first produces action
chunks and the student trains its flow objective toward those teacher chunks.
Normalized velocity matching is intentionally disabled unless teacher/student
normalization equivalence can be proven. Teacher actions cross the boundary in
physical units through the teacher postprocessor and student preprocessor.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

from .paths import CHECKPOINTS_ROOT, DATASETS_ROOT

os.environ.setdefault("HF_DATASETS_CACHE", str((CHECKPOINTS_ROOT / ".cache" / "huggingface" / "datasets").resolve()))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from tinyvla.fast_dataset import FastChunkDataset
from lerobot.datasets.utils import cycle
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from tinyvla.determinism import make_generator, seed_everything, seed_worker
from tinyvla.eval_closedloop import evaluate_closed_loop, format_metrics
from tinyvla.runtime import experiment_metadata, load_runtime, save_runtime
from tinyvla.trainability import RECOVERY_TRAINABLE_MODES, set_trainable


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


def compatible_noise(policy, batch: dict, noise: torch.Tensor | None) -> torch.Tensor | None:
    if noise is None:
        return None
    expected = (batch[ACTION].shape[0], policy.config.chunk_size, policy.config.max_action_dim)
    if tuple(noise.shape) != expected:
        return None
    return noise.to(device=batch[ACTION].device, dtype=batch[ACTION].dtype)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--teacher", default=None)
    parser.add_argument("--repo-id", default="local/so101_reach")
    parser.add_argument("--root", default=str(DATASETS_ROOT / "so101_reach"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--student-stats-source", choices=["dataset", "checkpoint"], default="dataset",
                        help="Where the student's normalizers come from. 'checkpoint' preserves the "
                             "student's EXACT deployed preprocessing (use for recovery/warm-start so the "
                             "observation/action schema does not drift); 'dataset' recomputes from --root.")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--objective",
        choices=["expert", "teacher_action", "teacher_velocity", "mixed_action_expert"],
        default=None,
        help="Defaults to teacher_action when --teacher is set. teacher_velocity is rejected "
             "until normalization equivalence is proven.",
    )
    parser.add_argument("--teacher-loss-weight", type=float, default=1.0)
    parser.add_argument("--expert-loss-weight", type=float, default=0.25)
    parser.add_argument("--save-step-subdirs", action="store_true")
    parser.add_argument("--step-offset", type=int, default=0)
    parser.add_argument("--trainable", choices=RECOVERY_TRAINABLE_MODES, default="expert")
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
    parser.add_argument("--closed-loop-episodes", type=int, default=1,
                        help="Rollouts per command per eval; use >=3 when comparing runs.")
    parser.add_argument("--save-best-closed-loop", action="store_true",
                        help="Keep the checkpoint with the best closed-loop success in <output>/best_closed_loop.")
    parser.add_argument("--n-action-steps", type=int, default=None,
                        help="Actions executed per replan (<=chunk_size). Try 10 for tighter closed-loop "
                             "(base default 50 causes open-loop drift).")
    parser.add_argument("--delta-actions", action="store_true", default=None,
                        help="Legacy assertion only; semantics are detected from markers.")
    args = parser.parse_args()
    seed_everything(args.seed)
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
    dataset = FastChunkDataset(args.repo_id, root=args.root, delta_timestamps=delta_timestamps)

    student_runtime = load_runtime(
        student_path, meta=meta, dataset_root=args.root, device=device,
        stats_source=args.student_stats_source
    )
    if args.delta_actions is not None and args.delta_actions != student_runtime.delta_actions:
        raise SystemExit(
            f"--delta-actions contradicts detected {student_runtime.action_semantics} semantics"
        )
    student = student_runtime.policy
    if args.n_action_steps is not None:
        student.config.n_action_steps = args.n_action_steps
        student.reset()
    student_preprocessor = student_runtime.preprocessor
    student_postprocessor = student_runtime.postprocessor
    trainable_params = set_trainable(student, args.trainable)

    teacher = None
    teacher_preprocessor = None
    teacher_postprocessor = None
    if args.teacher:
        teacher_path = Path(args.teacher)
        teacher_runtime = load_runtime(
            teacher_path, meta=meta, dataset_root=args.root, device=device, stats_source="checkpoint"
        )
        teacher = teacher_runtime.policy.eval()
        teacher_preprocessor = teacher_runtime.preprocessor
        teacher_postprocessor = teacher_runtime.postprocessor
        if objective == "teacher_velocity":
            raise SystemExit(
                "teacher_velocity is disabled: normalized teacher/student velocity fields are not "
                "proven to share normalization statistics. Use teacher_action physical conversion."
            )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=make_generator(args.seed),
    )
    trainable_parameters = [p for p in student.parameters() if p.requires_grad]
    if not trainable_parameters:
        raise SystemExit(f"no trainable parameters for --trainable {args.trainable}")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr)

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
        teacher_action_noise = None
        if objective == "mixed_action_expert":
            expert_batch = dict(batch)

        if (objective in {"teacher_action", "mixed_action_expert"} and teacher is not None
                and teacher_preprocessor is not None and teacher_postprocessor is not None):
            with torch.inference_mode():
                teacher_batch = teacher_preprocessor(dict(raw_batch))
                noise = fixed_noise(teacher, teacher_batch, args.seed + step)
                teacher.reset()
                teacher_actions = teacher.predict_action_chunk(teacher_batch, noise=noise)
                teacher_actions_physical = teacher_postprocessor(teacher_actions).detach()
            teacher_action_noise = compatible_noise(student, batch, noise)
            teacher_target_raw = dict(raw_batch)
            teacher_target_raw[ACTION] = teacher_actions_physical
            batch = student_preprocessor(teacher_target_raw)

        torch.manual_seed(args.seed + 10_000 + step)
        if objective == "teacher_velocity" and teacher is not None and teacher_preprocessor is not None:
            teacher_batch = teacher_preprocessor(dict(raw_batch))
            loss = teacher_velocity_loss(student, batch, teacher, teacher_batch, args.seed + step)
        elif objective == "mixed_action_expert" and expert_batch is not None:
            if teacher_action_noise is not None:
                teacher_loss, _ = student.forward(batch, noise=teacher_action_noise)
            else:
                teacher_loss, _ = student.forward(batch)
            expert_loss, _ = student.forward(expert_batch)
            loss = args.teacher_loss_weight * teacher_loss + args.expert_loss_weight * expert_loss
        else:
            if teacher_action_noise is not None:
                loss, _ = student.forward(batch, noise=teacher_action_noise)
            else:
                loss, _ = student.forward(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters, 10.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        losses.append(float(loss.item()))
        if step % args.log_every == 0 or step == 1 or step == args.steps:
            recent = losses[-args.log_every :]
            print(f"  step {step:5d}/{args.steps}  loss {sum(recent)/len(recent):.5f}", flush=True)

        if args.save_every and step % args.save_every == 0:
            save_runtime(student_runtime, output, seed=args.seed, extra_metadata={
                "repo_id": args.repo_id, "dataset_root": str(Path(args.root).resolve()), "step": step,
            })
            if args.save_step_subdirs:
                save_runtime(student_runtime, output / f"step_{args.step_offset + step:06d}",
                             seed=args.seed, extra_metadata={"step": args.step_offset + step})

        if args.closed_loop_every and (step % args.closed_loop_every == 0 or step == args.steps):
            cl = evaluate_closed_loop(
                student, student_preprocessor, student_postprocessor,
                device=device, commands=cl_commands,
                cap=args.closed_loop_cap, seed=args.closed_loop_seed,
                delta_actions=student_runtime.delta_actions, episodes=args.closed_loop_episodes,
            )
            print(f"  step {step:5d}/{args.steps}  closed-loop {format_metrics(cl)}", flush=True)
            cl_history.append({"step": args.step_offset + step, **cl})
            if args.save_best_closed_loop and cl["success_rate"] > best_cl:
                best_cl = cl["success_rate"]
                save_runtime(student_runtime, output / "best_closed_loop", seed=args.seed,
                             extra_metadata={"step": args.step_offset + step,
                                             "selection": "best_closed_loop"})
                print(f"    new best closed-loop success {cl['success_rate']:.0%} "
                      f"-> saved {output / 'best_closed_loop'}", flush=True)

    save_runtime(student_runtime, output, seed=args.seed, extra_metadata={
        "repo_id": args.repo_id, "dataset_root": str(Path(args.root).resolve()),
        "step": args.step_offset + args.steps,
    })

    meta_out = {
        "student": str(student_path),
        "teacher": args.teacher,
        "repo_id": args.repo_id,
        "root": args.root,
        "device": args.device,
        "seed": args.seed,
        "action_semantics": student_runtime.action_semantics,
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
        "experiment": experiment_metadata(seed=args.seed),
    }
    (output / "recovery_meta.json").write_text(json.dumps(meta_out, indent=2) + "\n")
    print(json.dumps(meta_out, indent=2))
    print(f"saved recovered checkpoint to {output}")

    del student, student_preprocessor, student_postprocessor, teacher, teacher_preprocessor, teacher_postprocessor
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
