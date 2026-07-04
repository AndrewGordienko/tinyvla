#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoProcessor

EMBED_KEY = "model.vlm_with_expert.vlm.model.text_model.embed_tokens.weight"
ID_MAP_KEY = "model.vlm_with_expert.vlm.model.text_model.embed_tokens.id_map"
LM_HEAD_KEY = "model.vlm_with_expert.vlm.lm_head.weight"


def load_tasks(dataset_root: Path | None, tasks_file: Path | None, task_args: list[str]) -> list[str]:
    tasks: list[str] = []
    if dataset_root is not None:
        task_parquet = dataset_root / "meta" / "tasks.parquet"
        if task_parquet.exists():
            df = pd.read_parquet(task_parquet)
            tasks.extend(str(task) for task in df.index.tolist())
    if tasks_file is not None:
        tasks.extend(line.strip() for line in tasks_file.read_text().splitlines() if line.strip())
    tasks.extend(task_args)
    return sorted(set(tasks))


def build_kept_token_ids(
    tokenizer_name: str,
    tasks: list[str],
    max_length: int,
    extra_token_ids: list[int],
) -> tuple[list[int], dict]:
    processor = AutoProcessor.from_pretrained(tokenizer_name, local_files_only=True)
    tokenizer = processor.tokenizer

    kept: set[int] = set(extra_token_ids)
    for attr in ("unk_token_id", "bos_token_id", "eos_token_id", "pad_token_id"):
        token_id = getattr(tokenizer, attr, None)
        if token_id is not None:
            kept.add(int(token_id))
    for attr in ("fake_image_token_id", "global_image_token_id"):
        token_id = getattr(tokenizer, attr, None)
        if token_id is not None:
            kept.add(int(token_id))

    encoded_tasks = {}
    for task in tasks:
        text = task if task.endswith("\n") else f"{task}\n"
        ids = tokenizer(text, padding="max_length", max_length=max_length, truncation=True).input_ids
        kept.update(int(token_id) for token_id in ids)
        encoded_tasks[task] = ids

    metadata = {
        "tokenizer_name": tokenizer_name,
        "original_vocab_size": len(tokenizer),
        "fallback_original_id": int(tokenizer.unk_token_id),
        "tasks": tasks,
        "encoded_tasks": encoded_tasks,
    }
    return sorted(kept), metadata


def copy_sidecars(source: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.name == "model.safetensors" or path.name == ".cache":
            continue
        if path.is_file():
            shutil.copy2(path, dest / path.name)


def count_params(tensors: dict[str, torch.Tensor]) -> int:
    return sum(t.numel() for t in tensors.values() if t.is_floating_point())


def prune_checkpoint(
    source: Path,
    dest: Path,
    *,
    drop_lm_head: bool,
    kept_token_ids: list[int] | None,
    vocab_metadata: dict | None,
) -> None:
    source_model = source / "model.safetensors"
    dest_model = dest / "model.safetensors"
    copy_sidecars(source, dest)

    tensors: dict[str, torch.Tensor] = {}
    removed_params = 0
    original_params = 0

    with safe_open(str(source_model), framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            if tensor.is_floating_point():
                original_params += tensor.numel()

            if drop_lm_head and key == LM_HEAD_KEY:
                removed_params += tensor.numel()
                continue

            if kept_token_ids is not None and key == EMBED_KEY:
                ids = torch.tensor(kept_token_ids, dtype=torch.long)
                compact_weight = tensor.index_select(0, ids).contiguous()
                id_map = torch.zeros(tensor.shape[0], dtype=torch.long)
                fallback_original_id = vocab_metadata["fallback_original_id"]
                fallback_row = kept_token_ids.index(fallback_original_id)
                id_map.fill_(fallback_row)
                for row, original_id in enumerate(kept_token_ids):
                    id_map[original_id] = row
                tensors[key] = compact_weight
                tensors[ID_MAP_KEY] = id_map
                removed_params += tensor.numel() - compact_weight.numel()
                continue

            tensors[key] = tensor

    save_file(tensors, str(dest_model))

    meta = {
        "source": str(source),
        "drop_lm_head": drop_lm_head,
        "compact_vocab": kept_token_ids is not None,
        "original_floating_params": original_params,
        "pruned_floating_params": count_params(tensors),
        "removed_floating_params": removed_params,
    }
    (dest / "pruning_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    if kept_token_ids is not None and vocab_metadata is not None:
        remap = {
            **vocab_metadata,
            "kept_token_ids": kept_token_ids,
            "compact_vocab_size": len(kept_token_ids),
        }
        (dest / "vocab_remap.json").write_text(json.dumps(remap, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune SmolVLA lm_head and optional vocab rows.")
    parser.add_argument("--source", default="data/models/smolvla_base")
    parser.add_argument("--dest", required=True)
    parser.add_argument("--keep-lm-head", action="store_true")
    parser.add_argument("--compact-vocab", action="store_true")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--tasks-file", default=None)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--tokenizer-name", default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--max-length", type=int, default=48)
    parser.add_argument("--extra-token-id", type=int, action="append", default=[])
    args = parser.parse_args()

    source = Path(args.source)
    dest = Path(args.dest)
    tasks = load_tasks(
        Path(args.dataset_root) if args.dataset_root else None,
        Path(args.tasks_file) if args.tasks_file else None,
        args.task,
    )

    kept_token_ids = None
    vocab_metadata = None
    if args.compact_vocab:
        if not tasks:
            raise SystemExit("--compact-vocab needs --dataset-root, --tasks-file, or --task")
        kept_token_ids, vocab_metadata = build_kept_token_ids(
            args.tokenizer_name,
            tasks,
            args.max_length,
            args.extra_token_id,
        )

    prune_checkpoint(
        source,
        dest,
        drop_lm_head=not args.keep_lm_head,
        kept_token_ids=kept_token_ids,
        vocab_metadata=vocab_metadata,
    )

    meta = json.loads((dest / "pruning_meta.json").read_text())
    print(f"wrote {dest}")
    print(f"params: {meta['original_floating_params']:,} -> {meta['pruned_floating_params']:,}")
    if kept_token_ids is not None:
        print(f"compact vocab rows: {len(kept_token_ids):,}")


if __name__ == "__main__":
    main()
