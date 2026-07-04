#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

TEXT_LAYER_RE = re.compile(r"^(model\.vlm_with_expert\.vlm\.model\.text_model\.layers\.)(\d+)(\..+)$")
EXPERT_LAYER_RE = re.compile(r"^(model\.vlm_with_expert\.lm_expert\.layers\.)(\d+)(\..+)$")


def parse_layers(value: str) -> list[int]:
    layers = [int(part) for part in value.split(",") if part.strip()]
    if not layers:
        raise argparse.ArgumentTypeError("layer list cannot be empty")
    if len(set(layers)) != len(layers):
        raise argparse.ArgumentTypeError("layer list contains duplicates")
    if any(layer < 0 or layer > 15 for layer in layers):
        raise argparse.ArgumentTypeError("layers must be in [0, 15]")
    return layers


def first_n(n: int) -> list[int]:
    return list(range(n))


def even_spaced(n: int) -> list[int]:
    if n == 1:
        return [0]
    return sorted(round(i * 15 / (n - 1)) for i in range(n))


def paired_even_spaced(n: int) -> list[int]:
    if n % 2:
        raise SystemExit("--mode paired_even requires an even --layers-count")
    pair_count = n // 2
    if pair_count == 1:
        pair_indices = [0]
    else:
        pair_indices = sorted(round(i * 7 / (pair_count - 1)) for i in range(pair_count))
    layers: list[int] = []
    for pair_idx in pair_indices:
        layers.extend([pair_idx * 2, pair_idx * 2 + 1])
    return layers


def early_late(n: int) -> list[int]:
    if n <= 4:
        return sorted({0, 1, 14, 15} if n == 4 else set(even_spaced(n)))
    early = list(range(max(2, n // 2)))
    remaining = n - len(early)
    late = sorted(round((len(early) + i) * 15 / n) for i in range(remaining))
    layers = []
    for layer in early + late:
        if layer not in layers:
            layers.append(layer)
    candidate = 15
    while len(layers) < n:
        if candidate not in layers:
            layers.append(candidate)
        candidate -= 1
    return sorted(layers)


def layer_plan(mode: str, n: int | None, layers: list[int] | None) -> list[int]:
    if mode == "custom":
        if layers is None:
            raise SystemExit("--layers is required for --mode custom")
        return layers
    if n is None:
        raise SystemExit("--layers-count is required unless --mode custom")
    if mode == "first":
        return first_n(n)
    if mode == "even":
        return even_spaced(n)
    if mode == "paired_even":
        return paired_even_spaced(n)
    if mode == "early_late":
        return early_late(n)
    raise SystemExit(f"unknown mode: {mode}")


def validate_structural_mapping(keep_layers: list[int], self_attn_every_n_layers: int = 2) -> None:
    """Reject remaps that move expert layers into incompatible attention slots."""

    if self_attn_every_n_layers <= 0:
        return
    mismatches = [
        (old_idx, new_idx)
        for new_idx, old_idx in enumerate(keep_layers)
        if old_idx % self_attn_every_n_layers != new_idx % self_attn_every_n_layers
    ]
    if mismatches:
        formatted = ", ".join(f"{old}->{new}" for old, new in mismatches)
        raise SystemExit(
            "layer mapping changes expert attention slot type; use a parity-preserving "
            f"mapping or pass a compatible custom list. mismatches: {formatted}"
        )


def copy_sidecars(source: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.name == "model.safetensors" or path.name == ".cache":
            continue
        if path.is_file():
            shutil.copy2(path, dest / path.name)


def remap_key(key: str, old_to_new: dict[int, int]) -> str | None:
    for regex in (TEXT_LAYER_RE, EXPERT_LAYER_RE):
        match = regex.match(key)
        if not match:
            continue
        old_idx = int(match.group(2))
        if old_idx not in old_to_new:
            return None
        return f"{match.group(1)}{old_to_new[old_idx]}{match.group(3)}"
    return key


def floating_params(tensors: dict) -> int:
    return sum(t.numel() for t in tensors.values() if getattr(t, "is_floating_point")())


def prune_layers(source: Path, dest: Path, keep_layers: list[int], mode: str) -> None:
    validate_structural_mapping(keep_layers)
    copy_sidecars(source, dest)
    old_to_new = {old: new for new, old in enumerate(keep_layers)}

    tensors = {}
    original_params = 0
    with safe_open(str(source / "model.safetensors"), framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            if tensor.is_floating_point():
                original_params += tensor.numel()
            new_key = remap_key(key, old_to_new)
            if new_key is None:
                continue
            tensors[new_key] = tensor

    save_file(tensors, str(dest / "model.safetensors"))

    config_path = dest / "config.json"
    with config_path.open() as f:
        config = json.load(f)
    config["num_vlm_layers"] = len(keep_layers)
    config["num_expert_layers"] = 0
    with config_path.open("w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    meta = {
        "source": str(source),
        "mode": mode,
        "kept_original_layers": keep_layers,
        "new_layer_count": len(keep_layers),
        "original_floating_params": original_params,
        "pruned_floating_params": floating_params(tensors),
        "removed_floating_params": original_params - floating_params(tensors),
    }
    with (dest / "layer_pruning_meta.json").open("w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")

    # Keep the generic pruned-checkpoint marker for the custom loader.
    pruning_meta = dest / "pruning_meta.json"
    existing = {}
    if pruning_meta.exists():
        with pruning_meta.open() as f:
            existing = json.load(f)
    existing["layer_pruning"] = meta
    with pruning_meta.open("w") as f:
        json.dump(existing, f, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create layer-pruned SmolVLA candidates.")
    parser.add_argument("--source", default="data/models/smolvla_headless_vocab_so101")
    parser.add_argument("--dest", required=True)
    parser.add_argument("--mode", choices=["first", "even", "paired_even", "early_late", "custom"], required=True)
    parser.add_argument("--layers-count", type=int, default=None)
    parser.add_argument("--layers", type=parse_layers, default=None)
    args = parser.parse_args()

    keep = layer_plan(args.mode, args.layers_count, args.layers)
    prune_layers(Path(args.source), Path(args.dest), keep, args.mode)

    meta = json.loads((Path(args.dest) / "layer_pruning_meta.json").read_text())
    print(f"wrote {args.dest}")
    print(f"kept layers: {keep}")
    print(f"params: {meta['original_floating_params']:,} -> {meta['pruned_floating_params']:,}")


if __name__ == "__main__":
    main()
