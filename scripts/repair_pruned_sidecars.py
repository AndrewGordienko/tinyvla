#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from transformers import AutoProcessor

EMBED_KEYS = (
    "model.vlm_with_expert.vlm.model.text_model.embed_tokens.weight",
    "model.vlm_with_expert.vlm.model.text_model.embed_tokens.embedding.weight",
)


def infer_layers(model_file: Path) -> tuple[list[int], list[int], int | None, dict]:
    text_layers: set[int] = set()
    expert_layers: set[int] = set()
    vision_layers: set[int] = set()
    compact_vocab_size = None
    text_intermediate_size = None
    vision_intermediate_size = None
    with safe_open(str(model_file), framework="pt", device="cpu") as tensors:
        for key in tensors.keys():
            if "text_model.layers." in key:
                part = key.split("text_model.layers.", 1)[1].split(".", 1)[0]
                if part.isdigit():
                    text_layers.add(int(part))
            if key.endswith("text_model.layers.0.mlp.gate_proj.weight"):
                text_intermediate_size = int(tensors.get_tensor(key).shape[0])
            if "lm_expert.layers." in key:
                part = key.split("lm_expert.layers.", 1)[1].split(".", 1)[0]
                if part.isdigit():
                    expert_layers.add(int(part))
            if "vision_model.encoder.layers." in key:
                part = key.split("vision_model.encoder.layers.", 1)[1].split(".", 1)[0]
                if part.isdigit():
                    vision_layers.add(int(part))
            if key.endswith("vision_model.encoder.layers.0.mlp.fc1.weight"):
                vision_intermediate_size = int(tensors.get_tensor(key).shape[0])
            if key in EMBED_KEYS:
                compact_vocab_size = int(tensors.get_tensor(key).shape[0])
    overrides = {
        "text_intermediate_size": text_intermediate_size,
        "vision_intermediate_size": vision_intermediate_size,
        "vision_num_hidden_layers": len(vision_layers) if vision_layers else None,
    }
    return sorted(text_layers), sorted(expert_layers), compact_vocab_size, overrides


def copy_sidecars(source: Path, dest: Path) -> None:
    for path in source.iterdir():
        if path.name == "model.safetensors" or path.name == ".cache":
            continue
        if path.is_file():
            shutil.copy2(path, dest / path.name)


def write_vocab_remap(
    dest: Path,
    kept_ids_path: Path,
    tokenizer_name: str,
    compact_vocab_size: int,
) -> None:
    kept_token_ids = json.loads(kept_ids_path.read_text())
    if len(kept_token_ids) != compact_vocab_size:
        raise SystemExit(
            f"{kept_ids_path} has {len(kept_token_ids)} ids but checkpoint embedding has "
            f"{compact_vocab_size} rows"
        )
    processor = AutoProcessor.from_pretrained(tokenizer_name, local_files_only=True)
    tokenizer = processor.tokenizer
    remap = {
        "tokenizer_name": tokenizer_name,
        "original_vocab_size": len(tokenizer),
        "fallback_original_id": int(tokenizer.unk_token_id),
        "tasks": [],
        "encoded_tasks": {},
        "kept_token_ids": kept_token_ids,
        "compact_vocab_size": compact_vocab_size,
    }
    (dest / "vocab_remap.json").write_text(json.dumps(remap, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recreate sidecars for a stripped pruned SmolVLA checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Directory containing model.safetensors")
    parser.add_argument("--source", required=True, help="Full/base SmolVLA checkpoint directory to copy sidecars from")
    parser.add_argument("--kept-ids", default="artifacts/kept_ids.json")
    parser.add_argument("--tokenizer-name", default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--n-action-steps", type=int, default=None)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    source = Path(args.source)
    model_file = checkpoint / "model.safetensors"
    if not model_file.exists():
        raise SystemExit(f"missing {model_file}")
    if not (source / "config.json").exists():
        raise SystemExit(f"missing {source / 'config.json'}")

    text_layers, expert_layers, compact_vocab_size, vlm_config_overrides = infer_layers(model_file)
    if not text_layers:
        raise SystemExit(f"could not infer text layers from {model_file}")
    expected = list(range(len(text_layers)))
    if text_layers != expected or expert_layers != expected:
        raise SystemExit(
            "checkpoint layers must already be remapped to contiguous 0..N-1 indices; "
            f"text={text_layers}, expert={expert_layers}"
        )

    copy_sidecars(source, checkpoint)

    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text())
    config["num_vlm_layers"] = len(text_layers)
    config["num_expert_layers"] = 0
    if args.n_action_steps is not None:
        config["n_action_steps"] = args.n_action_steps
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    if compact_vocab_size is not None:
        write_vocab_remap(checkpoint, Path(args.kept_ids), args.tokenizer_name, compact_vocab_size)

    meta = {
        "source": str(source),
        "repaired_sidecars": True,
        "kept_original_layers": text_layers,
        "new_layer_count": len(text_layers),
        "compact_vocab": compact_vocab_size is not None,
        "compact_vocab_size": compact_vocab_size,
        "vlm_config_overrides": vlm_config_overrides,
    }
    (checkpoint / "pruning_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"repaired {checkpoint}")
    print(f"layers: {len(text_layers)} compact_vocab: {compact_vocab_size}")


if __name__ == "__main__":
    main()
