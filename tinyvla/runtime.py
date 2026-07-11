"""One authoritative SmolVLA runtime for train, recovery, eval, benchmark, and demo."""
from __future__ import annotations

import contextlib
import json
import platform
import shutil
import subprocess
from dataclasses import dataclass
from fnmatch import fnmatch
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import torch
from safetensors import safe_open

# Import datasets before policies to avoid LeRobot's circular import.
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import dataset_to_policy_features, make_policy, make_pre_post_processors
from lerobot.policies.smolvla import smolvlm_with_expert as smolvlm_module
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

from .smolvla_loss import install_corrected_smolvla_loss
from .smolvla_pruned import load_pruned_smolvla

ActionSemantics = Literal["absolute", "delta"]
AUTHORITATIVE_VERSIONS = {
    "lerobot": "0.4.4",
    "torch": "2.10.0",
    "transformers": "4.57.6",
    "datasets": "4.8.5",
    "safetensors": "0.7.0",
    "mujoco": "3.9.0",
    "numpy": "2.2.6",
    "pillow": "12.1.1",
}


@dataclass
class RuntimeBundle:
    policy: Any
    preprocessor: Any
    postprocessor: Any
    action_semantics: ActionSemantics
    checkpoint_action_semantics: ActionSemantics | None
    dataset_action_semantics: ActionSemantics | None
    model_path: str
    load_report: dict[str, Any]

    @property
    def delta_actions(self) -> bool:
        return self.action_semantics == "delta"


class CompactVocabularyError(RuntimeError):
    def __init__(self, report: dict[str, Any]):
        self.report = report
        super().__init__(
            "compact vocabulary does not cover active instructions: "
            + json.dumps(report["missing_by_instruction"])
        )


def installed_versions() -> dict[str, str]:
    values: dict[str, str] = {}
    for package in AUTHORITATIVE_VERSIONS:
        try:
            values[package] = version(package)
        except PackageNotFoundError:
            values[package] = "missing"
    return values


def assert_authoritative_environment() -> dict[str, str]:
    values = installed_versions()
    mismatches = {
        name: {"expected": expected, "installed": values[name]}
        for name, expected in AUTHORITATIVE_VERSIONS.items()
        if values[name] != expected
    }
    if mismatches:
        raise RuntimeError(
            "non-authoritative environment; install this project with "
            "`.venv/bin/pip install -e '.[lerobot,test]'`: " + json.dumps(mismatches, sort_keys=True)
        )
    return values


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        )
        return bool(result.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def experiment_metadata(*, seed: int | None = None) -> dict[str, Any]:
    return {
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "versions": installed_versions(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "seed": seed,
    }


def write_action_semantics(path: str | Path, semantics: ActionSemantics) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "action_semantics.json").write_text(
        json.dumps({"representation": semantics, "schema_version": 1}, indent=2) + "\n"
    )
    legacy = path / "delta_actions.json"
    if semantics == "delta":
        legacy.write_text('{"delta_actions": true}\n')
    elif legacy.exists():
        legacy.unlink()


def detect_action_semantics(path: str | Path, *, legacy_default: ActionSemantics = "absolute") -> ActionSemantics:
    path = Path(path)
    explicit = path / "action_semantics.json"
    legacy = path / "delta_actions.json"
    if explicit.exists():
        data = json.loads(explicit.read_text())
        value = data.get("representation")
        if value not in {"absolute", "delta"}:
            raise ValueError(f"invalid action semantics marker {explicit}: {value!r}")
        if legacy.exists() and value != "delta":
            raise ValueError(f"contradictory action semantics markers in {path}")
        return value
    if legacy.exists():
        data = json.loads(legacy.read_text())
        if data.get("delta_actions") is not True:
            raise ValueError(f"invalid legacy delta marker {legacy}")
        return "delta"
    return legacy_default


def resolve_action_semantics(
    *,
    dataset: ActionSemantics | None,
    checkpoint: ActionSemantics | None,
    expected: ActionSemantics | None = None,
) -> ActionSemantics:
    semantics = expected or dataset or checkpoint or "absolute"
    if dataset is not None and dataset != semantics:
        raise RuntimeError(f"dataset action semantics are {dataset}, requested runtime is {semantics}")
    if checkpoint is not None and checkpoint != semantics:
        raise RuntimeError(
            f"checkpoint action semantics are {checkpoint}, dataset/runtime is {semantics}"
        )
    return semantics


def is_pruned_checkpoint(path: str | Path) -> bool:
    path = Path(path)
    return (path / "pruning_meta.json").exists() or (path / "vocab_remap.json").exists()


def instructions_from_metadata(meta: LeRobotDatasetMetadata) -> list[str]:
    tasks = getattr(meta, "tasks", None)
    if tasks is None:
        return []
    if hasattr(tasks, "index"):
        return [str(task) for task in tasks.index.tolist()]
    if isinstance(tasks, dict):
        return [str(task) for task in tasks]
    return [str(task) for task in tasks]


def dataset_feature_overrides(meta: LeRobotDatasetMetadata) -> dict[str, Any]:
    features = dataset_to_policy_features(meta.features)
    output_features = {key: ft for key, ft in features.items() if ft.type.value == "ACTION"}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    return {"input_features": input_features, "output_features": output_features}


def apply_saved_runtime_config(cfg: SmolVLAConfig, path: Path) -> SmolVLAConfig:
    config_path = path / "config.json"
    if config_path.exists():
        data = json.loads(config_path.read_text())
        for key in ("n_action_steps",):
            if data.get(key) is not None:
                setattr(cfg, key, data[key])
    return cfg


@contextlib.contextmanager
def local_transformers_only():
    orig_config = smolvlm_module.AutoConfig.from_pretrained
    orig_processor = smolvlm_module.AutoProcessor.from_pretrained

    def local_config(*args, **kwargs):
        kwargs.setdefault("local_files_only", True)
        return orig_config(*args, **kwargs)

    def local_processor(*args, **kwargs):
        kwargs.setdefault("local_files_only", True)
        return orig_processor(*args, **kwargs)

    smolvlm_module.AutoConfig.from_pretrained = local_config
    smolvlm_module.AutoProcessor.from_pretrained = local_processor
    try:
        yield
    finally:
        smolvlm_module.AutoConfig.from_pretrained = orig_config
        smolvlm_module.AutoProcessor.from_pretrained = orig_processor


def _matches_any(key: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch(key, pattern) for pattern in patterns)


def checkpoint_tensor_report(
    policy,
    path: str | Path,
    *,
    allowed_missing: tuple[str, ...] = (),
    allowed_unexpected: tuple[str, ...] = (),
) -> dict[str, Any]:
    path = Path(path)
    model_file = path / "model.safetensors"
    state = policy.state_dict()
    with safe_open(str(model_file), framework="pt", device="cpu") as tensors:
        saved_shapes = {key: tuple(tensors.get_slice(key).get_shape()) for key in tensors.keys()}
    state_shapes = {key: tuple(value.shape) for key, value in state.items()}
    missing = sorted(set(state_shapes) - set(saved_shapes))
    unexpected = sorted(set(saved_shapes) - set(state_shapes))
    shape_mismatches = {
        key: {"runtime": state_shapes[key], "checkpoint": saved_shapes[key]}
        for key in sorted(set(state_shapes) & set(saved_shapes))
        if state_shapes[key] != saved_shapes[key]
    }
    unexplained_missing = [key for key in missing if not _matches_any(key, allowed_missing)]
    unexplained_unexpected = [key for key in unexpected if not _matches_any(key, allowed_unexpected)]
    return {
        "checkpoint": str(path.resolve()),
        "checkpoint_tensor_count": len(saved_shapes),
        "runtime_tensor_count": len(state_shapes),
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatches": shape_mismatches,
        "allowed_missing": list(allowed_missing),
        "allowed_unexpected": list(allowed_unexpected),
        "unexplained_missing": unexplained_missing,
        "unexplained_unexpected": unexplained_unexpected,
        "ok": not unexplained_missing and not unexplained_unexpected and not shape_mismatches,
    }


def verify_compact_vocabulary(policy, path: str | Path, instructions: list[str]) -> dict[str, Any]:
    path = Path(path)
    remap_path = path / "vocab_remap.json"
    if not remap_path.exists():
        return {"compact": False, "instructions": len(instructions), "ok": True}
    remap = json.loads(remap_path.read_text())
    kept = {int(token_id) for token_id in remap.get("kept_token_ids", [])}
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    missing_by_instruction: dict[str, list[int]] = {}
    encoded: dict[str, list[int]] = {}
    max_length = int(policy.config.tokenizer_max_length)
    for instruction in instructions:
        text = instruction if instruction.endswith("\n") else instruction + "\n"
        token_ids = [
            int(token_id)
            for token_id in tokenizer(
                text, padding="max_length", max_length=max_length, truncation=True
            ).input_ids
        ]
        encoded[instruction] = token_ids
        missing = sorted(set(token_ids) - kept)
        if missing:
            missing_by_instruction[instruction] = missing
    report = {
        "compact": True,
        "instructions": len(instructions),
        "kept_token_count": len(kept),
        "missing_by_instruction": missing_by_instruction,
        "encoded_instructions": encoded,
        "ok": not missing_by_instruction,
    }
    if not report["ok"]:
        raise CompactVocabularyError(report)
    return report


def make_processors(
    policy,
    model_path: str | Path,
    device: torch.device,
    meta: LeRobotDatasetMetadata,
    *,
    stats_source: Literal["checkpoint", "dataset"] = "checkpoint",
):
    norm_feats = {**policy.config.input_features, **policy.config.output_features}
    pre_overrides: dict[str, Any] = {"device_processor": {"device": device.type}}
    post_overrides: dict[str, Any] = {}
    if stats_source == "dataset":
        pre_overrides["normalizer_processor"] = {
            "stats": meta.stats,
            "features": norm_feats,
            "norm_map": policy.config.normalization_mapping,
        }
        post_overrides["unnormalizer_processor"] = {
            "stats": meta.stats,
            "features": policy.config.output_features,
            "norm_map": policy.config.normalization_mapping,
        }
    return make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(model_path),
        preprocessor_overrides=pre_overrides,
        postprocessor_overrides=post_overrides,
    )


def load_runtime(
    model_path: str | Path,
    *,
    meta: LeRobotDatasetMetadata,
    dataset_root: str | Path | None,
    device: str | torch.device,
    stats_source: Literal["checkpoint", "dataset"] = "checkpoint",
    base_checkpoint: bool = False,
    expected_action_semantics: ActionSemantics | None = None,
    instructions: list[str] | None = None,
    allowed_missing: tuple[str, ...] = (),
    allowed_unexpected: tuple[str, ...] = (),
    enforce_versions: bool = True,
) -> RuntimeBundle:
    """Load policy, saved config, processors, action semantics, and audits together."""

    if enforce_versions:
        assert_authoritative_environment()
    device = torch.device(device)
    path = Path(model_path)
    local_checkpoint = path.is_dir()
    dataset_semantics = detect_action_semantics(dataset_root) if dataset_root is not None else None
    checkpoint_semantics = (
        None if base_checkpoint or not local_checkpoint else detect_action_semantics(path)
    )
    semantics = resolve_action_semantics(
        dataset=dataset_semantics,
        checkpoint=checkpoint_semantics,
        expected=expected_action_semantics,
    )

    overrides = dataset_feature_overrides(meta)
    if local_checkpoint and is_pruned_checkpoint(path):
        policy = load_pruned_smolvla(
            path,
            device=device,
            config_overrides=overrides,
            strict=True,
            allowed_missing=allowed_missing,
            allowed_unexpected=allowed_unexpected,
        )
        apply_saved_runtime_config(policy.config, path)
        load_report = policy._tinyvla_load_report
    else:
        cfg = SmolVLAConfig(pretrained_path=str(model_path), device=str(device), **overrides)
        if local_checkpoint:
            apply_saved_runtime_config(cfg, path)
        context = local_transformers_only() if local_checkpoint else contextlib.nullcontext()
        with context:
            policy = make_policy(cfg=cfg, ds_meta=meta)
        if local_checkpoint:
            load_report = checkpoint_tensor_report(
                policy, path, allowed_missing=allowed_missing, allowed_unexpected=allowed_unexpected
            )
            (path / "load_report.json").write_text(json.dumps(load_report, indent=2) + "\n")
            print(json.dumps({"smolvla_load_report": load_report}, indent=2))
            if not load_report["ok"]:
                raise RuntimeError(f"checkpoint tensor audit failed: {path / 'load_report.json'}")
        else:
            load_report = {"checkpoint": str(model_path), "remote": True, "ok": True}

    install_corrected_smolvla_loss(policy)
    policy.to(device)
    try:
        vocabulary = verify_compact_vocabulary(
            policy, path, instructions if instructions is not None else instructions_from_metadata(meta)
        ) if local_checkpoint else {"compact": False, "ok": True}
    except CompactVocabularyError as error:
        load_report = {**load_report, "vocabulary": error.report, "action_semantics": semantics, "ok": False}
        if local_checkpoint:
            (path / "load_report.json").write_text(json.dumps(load_report, indent=2) + "\n")
        print(json.dumps({"runtime_load_report": load_report}, indent=2))
        raise
    preprocessor, postprocessor = make_processors(
        policy, model_path, device, meta, stats_source=stats_source
    )
    load_report = {**load_report, "vocabulary": vocabulary, "action_semantics": semantics}
    if local_checkpoint:
        (path / "load_report.json").write_text(json.dumps(load_report, indent=2) + "\n")
    print(json.dumps({"runtime_load_report": load_report}, indent=2))
    return RuntimeBundle(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        action_semantics=semantics,
        checkpoint_action_semantics=checkpoint_semantics,
        dataset_action_semantics=dataset_semantics,
        model_path=str(model_path),
        load_report=load_report,
    )


def save_runtime(
    bundle: RuntimeBundle,
    output: str | Path,
    *,
    seed: int | None,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    bundle.policy.save_pretrained(output)
    bundle.preprocessor.save_pretrained(output)
    bundle.postprocessor.save_pretrained(output)
    source = Path(bundle.model_path)
    if source.is_dir():
        for name in ("pruning_meta.json", "vocab_remap.json", "layer_pruning_meta.json"):
            sidecar = source / name
            destination = output / name
            if sidecar.exists() and sidecar.resolve() != destination.resolve():
                shutil.copy2(sidecar, output / name)
    write_action_semantics(output, bundle.action_semantics)
    metadata = {
        **experiment_metadata(seed=seed),
        "action_semantics": bundle.action_semantics,
        "n_action_steps": bundle.policy.config.n_action_steps,
        **(extra_metadata or {}),
    }
    (output / "runtime_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
