from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)

from tinyvla.determinism import preserve_rng_state
from tinyvla.eval_closedloop import evaluate_closed_loop
from tinyvla.runtime import (
    AUTHORITATIVE_VERSIONS,
    apply_saved_runtime_config,
    assert_authoritative_environment,
    checkpoint_tensor_report,
    detect_action_semantics,
    resolve_action_semantics,
    verify_compact_vocabulary,
    write_action_semantics,
)
from tinyvla.smolvla_loss import install_corrected_smolvla_loss, reduce_valid_action_loss


def test_authoritative_dependency_versions_are_installed():
    assert assert_authoritative_environment() == AUTHORITATIVE_VERSIONS


def test_padding_values_and_extra_motor_dimensions_cannot_change_loss():
    base = torch.arange(2 * 4 * 32, dtype=torch.float32).reshape(2, 4, 32)
    padding = torch.tensor([[False, False, True, True], [False, True, True, True]])
    changed = base.clone()
    changed[padding] = torch.randn_like(changed[padding]) * 1_000_000
    changed[..., 6:] = torch.randn_like(changed[..., 6:]) * 1_000_000

    expected = reduce_valid_action_loss(base, padding, 6)
    actual = reduce_valid_action_loss(changed, padding, 6)
    assert actual == expected


def test_valid_scalar_normalization_is_not_diluted_by_padding():
    losses = torch.zeros(1, 3, 32)
    losses[:, 0, :6] = 2.0
    padding = torch.tensor([[False, True, True]])
    assert reduce_valid_action_loss(losses, padding, 6).item() == 2.0


class _PointwiseLossModel:
    def forward(self, images, image_masks, tokens, token_masks, state, actions, noise, time):
        del images, image_masks, tokens, token_masks, state, noise, time
        return actions.square()


class _LossPolicy:
    def __init__(self):
        self.config = SimpleNamespace(
            adapt_to_pi_aloha=False,
            action_feature=SimpleNamespace(shape=(6,)),
        )
        self.model = _PointwiseLossModel()

    def prepare_images(self, batch):
        del batch
        return [], []

    def prepare_state(self, batch):
        return batch["observation.state"]

    def prepare_action(self, batch):
        return torch.nn.functional.pad(batch[ACTION], (0, 26))


def test_installed_policy_loss_uses_action_is_pad_and_canonicalizes_padding():
    policy = install_corrected_smolvla_loss(_LossPolicy())
    batch = {
        ACTION: torch.ones(1, 3, 6),
        "observation.state": torch.zeros(1, 6),
        OBS_LANGUAGE_TOKENS: torch.zeros(1, 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 1, dtype=torch.bool),
        "action_is_pad": torch.tensor([[False, True, True]]),
    }
    first, report = policy.forward(batch)
    changed = dict(batch)
    changed[ACTION] = batch[ACTION].clone()
    changed[ACTION][:, 1:] = 100_000
    second, _ = policy.forward(changed)
    assert first == second == 1.0
    assert report["valid_action_scalars"] == 6

    legacy = dict(batch)
    legacy["actions_id_pad"] = legacy.pop("action_is_pad")
    with pytest.raises(KeyError, match="actions_id_pad"):
        policy.forward(legacy)


def test_action_semantics_markers_and_legacy_detection(tmp_path: Path):
    assert detect_action_semantics(tmp_path) == "absolute"
    write_action_semantics(tmp_path, "delta")
    assert detect_action_semantics(tmp_path) == "delta"
    write_action_semantics(tmp_path, "absolute")
    assert detect_action_semantics(tmp_path) == "absolute"
    assert not (tmp_path / "delta_actions.json").exists()


def test_saved_n_action_steps_is_restored(tmp_path: Path):
    (tmp_path / "config.json").write_text('{"n_action_steps": 5}\n')
    config = SimpleNamespace(n_action_steps=50)
    assert apply_saved_runtime_config(config, tmp_path).n_action_steps == 5


def test_contradictory_action_semantics_are_rejected(tmp_path: Path):
    (tmp_path / "action_semantics.json").write_text('{"representation": "absolute"}\n')
    (tmp_path / "delta_actions.json").write_text('{"delta_actions": true}\n')
    with pytest.raises(ValueError, match="contradictory"):
        detect_action_semantics(tmp_path)


def test_dataset_checkpoint_action_semantics_mismatch_is_rejected():
    with pytest.raises(RuntimeError, match="checkpoint action semantics"):
        resolve_action_semantics(dataset="delta", checkpoint="absolute")
    assert resolve_action_semantics(dataset="delta", checkpoint="delta") == "delta"


def test_checkpoint_tensor_coverage_reports_every_gap(tmp_path: Path):
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    saved = {key: value.detach().clone() for key, value in model.state_dict().items()}
    saved.pop("1.bias")
    saved["unexpected"] = torch.zeros(1)
    save_file(saved, str(tmp_path / "model.safetensors"))

    report = checkpoint_tensor_report(model, tmp_path)
    assert report["missing"] == ["1.bias"]
    assert report["unexpected"] == ["unexpected"]
    assert not report["ok"]

    allowed = checkpoint_tensor_report(
        model, tmp_path, allowed_missing=("1.bias",), allowed_unexpected=("unexpected",)
    )
    assert allowed["ok"]


class _Tokenizer:
    def __call__(self, text, **kwargs):
        del text, kwargs
        return SimpleNamespace(input_ids=[1, 2, 3])


def _fake_vocab_policy():
    return SimpleNamespace(
        config=SimpleNamespace(tokenizer_max_length=48),
        model=SimpleNamespace(
            vlm_with_expert=SimpleNamespace(processor=SimpleNamespace(tokenizer=_Tokenizer()))
        ),
    )


def test_compact_vocabulary_coverage_accepts_complete_set(tmp_path: Path):
    (tmp_path / "vocab_remap.json").write_text(json.dumps({"kept_token_ids": [1, 2, 3]}))
    report = verify_compact_vocabulary(_fake_vocab_policy(), tmp_path, ["instruction"])
    assert report["ok"]


def test_compact_vocabulary_coverage_rejects_missing_instruction_token(tmp_path: Path):
    (tmp_path / "vocab_remap.json").write_text(json.dumps({"kept_token_ids": [1, 2]}))
    with pytest.raises(RuntimeError, match="does not cover"):
        verify_compact_vocabulary(_fake_vocab_policy(), tmp_path, ["instruction"])


def test_rng_context_restores_python_numpy_and_torch():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    expected = (random.random(), np.random.rand(), torch.rand(()))
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    with preserve_rng_state():
        for _ in range(100):
            random.random()
            np.random.rand()
            torch.rand(())
    actual = (random.random(), np.random.rand(), torch.rand(()))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


class _ConstantPolicy:
    def __init__(self):
        self.training = True

    def eval(self):
        self.training = False
        return self

    def train(self):
        self.training = True
        return self

    def reset(self):
        pass

    def select_action(self, obs):
        del obs
        return torch.tensor([[0.0, -1.2, 0.6, 1.2, 0.0, 1.2]])


def test_repeated_closed_loop_evaluation_is_identical_and_rng_safe():
    policy = _ConstantPolicy()
    identity = lambda value: value
    torch.manual_seed(19)
    before = torch.get_rng_state().clone()
    first = evaluate_closed_loop(
        policy, identity, identity, device=torch.device("cpu"), commands=[0], cap=2, seed=22
    )
    middle = torch.get_rng_state().clone()
    second = evaluate_closed_loop(
        policy, identity, identity, device=torch.device("cpu"), commands=[0], cap=2, seed=22
    )
    after = torch.get_rng_state().clone()
    assert first == second
    assert torch.equal(before, middle)
    assert torch.equal(before, after)
    assert policy.training
    assert first["groups"]["single_step_0_3"]["n"] == 1
