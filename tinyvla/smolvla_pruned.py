from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_model
from torch import nn

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.smolvla import smolvlm_with_expert as smolvlm_module
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

EMBED_KEY = "model.vlm_with_expert.vlm.model.text_model.embed_tokens.weight"
EMBED_MODULE_KEY = "model.vlm_with_expert.vlm.model.text_model.embed_tokens.embedding.weight"
ID_MAP_KEY = "model.vlm_with_expert.vlm.model.text_model.embed_tokens.id_map"
LM_HEAD_KEY = "model.vlm_with_expert.vlm.lm_head.weight"


class CompactTokenEmbedding(nn.Module):
    """Embedding that accepts original tokenizer IDs and remaps them to compact rows."""

    def __init__(
        self,
        original_vocab_size: int,
        compact_vocab_size: int,
        embedding_dim: int,
        fallback_original_id: int,
    ):
        super().__init__()
        self.num_embeddings = original_vocab_size
        self.embedding_dim = embedding_dim
        self.fallback_original_id = fallback_original_id
        self.weight = nn.Parameter(torch.empty(compact_vocab_size, embedding_dim))
        self.register_buffer("id_map", torch.zeros(original_vocab_size, dtype=torch.long), persistent=True)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        compact_ids = self.id_map[input_ids]
        return F.embedding(compact_ids, self.weight)


class CompactTokenEmbeddingModule(nn.Module):
    """Compact embedding compatible with checkpoints saved as ``embedding.weight``."""

    def __init__(
        self,
        original_vocab_size: int,
        compact_vocab_size: int,
        embedding_dim: int,
        fallback_original_id: int,
    ):
        super().__init__()
        self.num_embeddings = original_vocab_size
        self.embedding_dim = embedding_dim
        self.fallback_original_id = fallback_original_id
        self.embedding = nn.Embedding(compact_vocab_size, embedding_dim)
        self.register_buffer("id_map", torch.zeros(original_vocab_size, dtype=torch.long), persistent=True)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        compact_ids = self.id_map[input_ids]
        return self.embedding(compact_ids)


def _replace_embedding(policy: SmolVLAPolicy, model_file: Path, remap_file: Path) -> None:
    with remap_file.open() as f:
        remap = json.load(f)

    with safe_open(str(model_file), framework="pt", device="cpu") as tensors:
        if EMBED_KEY in tensors.keys():
            key = EMBED_KEY
        elif EMBED_MODULE_KEY in tensors.keys():
            key = EMBED_MODULE_KEY
        else:
            raise KeyError(f"missing compact embedding tensor in {model_file}")
        compact_vocab_size, embedding_dim = tensors.get_tensor(key).shape

    embedding_cls = CompactTokenEmbedding if key == EMBED_KEY else CompactTokenEmbeddingModule
    compact = embedding_cls(
        original_vocab_size=remap["original_vocab_size"],
        compact_vocab_size=compact_vocab_size,
        embedding_dim=embedding_dim,
        fallback_original_id=remap["fallback_original_id"],
    )
    kept_token_ids = remap.get("kept_token_ids")
    if kept_token_ids is not None:
        fallback_row = kept_token_ids.index(remap["fallback_original_id"])
        id_map = torch.full((remap["original_vocab_size"],), fallback_row, dtype=torch.long)
        for row, original_id in enumerate(kept_token_ids):
            id_map[int(original_id)] = row
        compact.id_map.copy_(id_map)
    policy.model.vlm_with_expert.vlm.model.text_model.embed_tokens = compact


def _feature_from_json(value: dict) -> PolicyFeature:
    return PolicyFeature(type=FeatureType(value["type"]), shape=tuple(value["shape"]))


def _config_from_json(pretrained_path: Path) -> SmolVLAConfig:
    with (pretrained_path / "config.json").open() as f:
        data = json.load(f)

    data.pop("type", None)
    for feature_key in ("input_features", "output_features"):
        if data.get(feature_key) is not None:
            data[feature_key] = {
                name: _feature_from_json(feature) for name, feature in data[feature_key].items()
            }
    if data.get("normalization_mapping") is not None:
        data["normalization_mapping"] = {
            name: NormalizationMode(mode) for name, mode in data["normalization_mapping"].items()
        }
    return SmolVLAConfig(**data)


def _pruning_meta(pretrained_path: Path) -> dict:
    path = pretrained_path / "pruning_meta.json"
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def _apply_vlm_config_overrides(config, pruning_meta: dict) -> None:
    overrides = pruning_meta.get("vlm_config_overrides") or {}
    text_config = getattr(config, "text_config", None)
    vision_config = getattr(config, "vision_config", None)
    if text_config is not None and overrides.get("text_intermediate_size") is not None:
        text_config.intermediate_size = int(overrides["text_intermediate_size"])
    if vision_config is not None and overrides.get("vision_intermediate_size") is not None:
        vision_config.intermediate_size = int(overrides["vision_intermediate_size"])
    if vision_config is not None and overrides.get("vision_num_hidden_layers") is not None:
        vision_config.num_hidden_layers = int(overrides["vision_num_hidden_layers"])


def load_pruned_smolvla(
    pretrained_path: str | Path,
    *,
    device: str | torch.device | None = None,
    config_overrides: dict[str, Any] | None = None,
    strict: bool = False,
) -> SmolVLAPolicy:
    """Load a SmolVLA checkpoint pruned by scripts/prune_smolvla.py.

    Standard LeRobot loading reconstructs the original language head and full
    token embedding before loading weights. This loader installs the compact
    modules first, then loads the pruned safetensors file.
    """

    pretrained_path = Path(pretrained_path)
    model_file = pretrained_path / "model.safetensors"
    remap_file = pretrained_path / "vocab_remap.json"

    cfg = _config_from_json(pretrained_path)
    pruning_meta = _pruning_meta(pretrained_path)
    for key, value in (config_overrides or {}).items():
        setattr(cfg, key, value)
    if device is not None:
        cfg.device = str(device)

    # The pruned checkpoint contains the SmolVLA weights we need, so avoid
    # loading the full VLM checkpoint just to overwrite most of it.
    cfg.load_vlm_weights = False
    old_hf_offline = os.environ.get("HF_HUB_OFFLINE")
    old_transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE")
    orig_auto_config = smolvlm_module.AutoConfig.from_pretrained
    orig_auto_processor = smolvlm_module.AutoProcessor.from_pretrained

    def local_config_from_pretrained(*args, **kwargs):
        kwargs.setdefault("local_files_only", True)
        config = orig_auto_config(*args, **kwargs)
        _apply_vlm_config_overrides(config, pruning_meta)
        return config

    def local_processor_from_pretrained(*args, **kwargs):
        kwargs.setdefault("local_files_only", True)
        return orig_auto_processor(*args, **kwargs)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    smolvlm_module.AutoConfig.from_pretrained = local_config_from_pretrained
    smolvlm_module.AutoProcessor.from_pretrained = local_processor_from_pretrained
    try:
        policy = SmolVLAPolicy(cfg)
    finally:
        smolvlm_module.AutoConfig.from_pretrained = orig_auto_config
        smolvlm_module.AutoProcessor.from_pretrained = orig_auto_processor
        if old_hf_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = old_hf_offline
        if old_transformers_offline is None:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = old_transformers_offline

    with safe_open(str(model_file), framework="pt", device="cpu") as tensors:
        keys = set(tensors.keys())

    if remap_file.exists():
        _replace_embedding(policy, model_file, remap_file)

    if LM_HEAD_KEY not in keys:
        policy.model.vlm_with_expert.vlm.lm_head = None

    missing, unexpected = load_model(policy, str(model_file), strict=strict, device=str(cfg.device))
    if strict and (missing or unexpected):
        raise RuntimeError(f"strict load failed: missing={missing}, unexpected={unexpected}")

    policy.to(cfg.device)
    policy.eval()
    return policy
