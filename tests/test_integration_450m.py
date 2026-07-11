"""Heavyweight integration tests against the real 450M smolvla_base checkpoint.

These load the full model (~900MB) and are skipped automatically when it is not
present. Run explicitly on the M5 with:

    PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m pytest tests/test_integration_450m.py -v

They cover the Phase-1 claims that the fast unit tests (mocks / tiny layers)
cannot: canonical load -> inference -> save -> reload action equivalence,
read-only loading (checkpoint bytes unchanged, no report written into it), and
padded-suffix attention invariance on the actual transformer.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from tinyvla.paths import ARTIFACTS_ROOT, MODELS_ROOT
from tinyvla.runtime import load_runtime, save_runtime, sha256_tree

BASE = MODELS_ROOT / "smolvla_base"
DATASET_ROOT = ARTIFACTS_ROOT / "truth_harness" / "datasets" / "command0_4"
REPO_ID = "local/truth_gate_command0_4"
INSTRUCTION = "Pick up the red cube and place it in the box."

pytestmark = pytest.mark.skipif(
    not (BASE / "model.safetensors").exists() or not DATASET_ROOT.exists(),
    reason="requires local smolvla_base (450M) and the command0_4 dataset",
)


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _raw_obs(seed: int, device: torch.device) -> dict:
    """A deterministic single-frame observation matching the dataset features
    (front camera + 6-dof state). Synthetic pixels keep the test free of any GL
    dependency — the model weights, not scene realism, are under test."""
    gen = torch.Generator().manual_seed(seed)
    img = torch.rand(1, 3, 256, 256, generator=gen)
    state = torch.tensor([[0.0, -1.2, 0.6, 1.2, 0.0, 1.2]], dtype=torch.float32)
    return {
        "observation.state": state.to(device),
        "observation.images.front": img.to(device),
        "task": [INSTRUCTION],
    }


def _first_action(runtime, seed: int, device: torch.device) -> np.ndarray:
    policy = runtime.policy
    policy.reset()
    torch.manual_seed(seed)
    obs = runtime.preprocessor(_raw_obs(seed, device))
    with torch.inference_mode():
        action = policy.select_action(obs)
    return runtime.postprocessor(action).squeeze(0).float().cpu().numpy()


@pytest.fixture(scope="module")
def meta() -> LeRobotDatasetMetadata:
    return LeRobotDatasetMetadata(REPO_ID, root=DATASET_ROOT)


@pytest.fixture(scope="module")
def base_runtime(meta):
    return load_runtime(
        BASE, meta=meta, dataset_root=DATASET_ROOT, device=_device(),
        stats_source="dataset", base_checkpoint=True,
    )


def test_loading_base_is_read_only(meta):
    """Loading must not mutate the checkpoint: identical content hash before and
    after, and no load_report.json written into the checkpoint directory."""
    before = sha256_tree(BASE, patterns=("*.safetensors", "*.json"))
    load_runtime(
        BASE, meta=meta, dataset_root=DATASET_ROOT, device=_device(),
        stats_source="dataset", base_checkpoint=True,
    )
    after = sha256_tree(BASE, patterns=("*.safetensors", "*.json"))
    assert before == after
    assert not (BASE / "load_report.json").exists()


def test_load_infer_save_reload_action_equivalence(base_runtime, meta, tmp_path):
    """Canonical load -> inference -> save -> reload -> inference must reproduce
    the same action chunk, proving the round-trip preserves the policy exactly."""
    device = _device()
    seeds = [11, 12, 13]
    before = [_first_action(base_runtime, s, device) for s in seeds]

    out = tmp_path / "roundtrip_ckpt"
    save_runtime(base_runtime, out, seed=0, extra_metadata={"test": "integration"})

    # Reload as a local checkpoint (not base): strict tensor + semantics audit.
    reloaded = load_runtime(
        out, meta=meta, dataset_root=DATASET_ROOT, device=device,
        stats_source="checkpoint",
    )
    assert reloaded.action_semantics == base_runtime.action_semantics
    assert reloaded.load_report["ok"]
    assert not (out / "load_report.json").exists()  # reload is read-only too

    after = [_first_action(reloaded, s, device) for s in seeds]
    for s, a, b in zip(seeds, before, after):
        assert np.allclose(a, b, atol=2e-4), (
            f"seed {s}: action chunk changed across save/reload; max|d|={np.abs(a - b).max():.2e}"
        )


def test_padded_suffix_does_not_affect_valid_predictions(base_runtime):
    """End-to-end attention invariance: perturbing the action-expert inputs at
    padded (tail) timesteps must not change the model output at valid timesteps.
    Calls model.forward directly (bypassing the loss's target-zeroing) so the
    attention path itself is exercised."""
    device = _device()
    policy = base_runtime.policy
    chunk = policy.config.chunk_size
    valid = chunk - 8  # last 8 timesteps are tail padding

    raw = {
        "observation.state": torch.tensor([[0.0, -1.2, 0.6, 1.2, 0.0, 1.2]]).to(device),
        "observation.images.front": torch.rand(1, 3, 256, 256, generator=torch.Generator().manual_seed(7)).to(device),
        "action": torch.zeros(1, chunk, 6).to(device),
        "action_is_pad": torch.zeros(1, chunk, dtype=torch.bool).to(device),
        "task": [INSTRUCTION],
    }
    raw["action_is_pad"][:, valid:] = True
    batch = base_runtime.preprocessor(raw)

    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    actions = policy.prepare_action(batch)  # [1, chunk, max_action_dim]

    torch.manual_seed(0)
    noise = torch.randn_like(actions)
    time = torch.rand(actions.shape[0], device=actions.device)

    actions_perturbed = actions.clone()
    actions_perturbed[:, valid:] += 1000.0 * torch.randn_like(actions_perturbed[:, valid:])

    with torch.inference_mode():
        losses_ref = policy.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
        losses_pert = policy.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions_perturbed, noise, time)

    ref_valid = losses_ref[:, :valid].float().cpu()
    pert_valid = losses_pert[:, :valid].float().cpu()
    assert torch.allclose(ref_valid, pert_valid, atol=1e-4), (
        f"valid-timestep output changed when only padded tail inputs moved; "
        f"max|d|={(ref_valid - pert_valid).abs().max():.2e}"
    )
    # sanity: the perturbation really did change the padded positions
    ref_pad = losses_ref[:, valid:].float().cpu()
    pert_pad = losses_pert[:, valid:].float().cpu()
    assert not torch.allclose(ref_pad, pert_pad, atol=1e-4)
