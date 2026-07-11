import numpy as np
import torch

from scripts.deployable_controller import SharedEncoder, _masked_mse, expert_chunk_from_snapshot, restore, snapshot
from scripts.diagnose_supervised_gate import data_audit
from tinyvla.task import SO101PickPlaceTask


def test_expert_chunk_restores_learner_state():
    """DAgger chunk labels must not advance or mutate the learner simulator."""
    env = SO101PickPlaceTask(seed=0)
    env.reset(command=0)
    before = snapshot(env)
    labels = expert_chunk_from_snapshot(env, 4)
    after = snapshot(env)

    assert labels.shape == (4, 6)
    for field in ("qpos", "qvel", "ctrl", "act"):
        np.testing.assert_array_equal(after[field], before[field])
    assert after["time"] == before["time"]
    assert after["grasped"] == before["grasped"]


def test_restore_rewinds_a_step():
    env = SO101PickPlaceTask(seed=0)
    env.reset(command=0)
    before = snapshot(env)
    env.step(env.reactive_action())
    restore(env, before)
    np.testing.assert_array_equal(env.data.qpos, before["qpos"])
    np.testing.assert_array_equal(env.data.qvel, before["qvel"])


def test_padded_chunk_suffix_is_masked_from_loss():
    pred = torch.tensor([[[0.0] * 6, [10.0] * 6]])
    target = torch.zeros_like(pred)
    assert _masked_mse(pred, target, torch.tensor([[1.0, 0.0]])).item() == 0.0


def test_temporal_audit_rejects_no_invalid_chunk_suffix():
    data = {
        "imgs": np.zeros((2, 2, 1, 3, 2, 2), np.float32), "state": np.zeros((2, 18), np.float32),
        "label": np.zeros((2, 2, 6), np.float32), "mask": np.array([[1, 1], [1, 0]], np.float32),
        "frame_indices": np.array([[0, 1], [0, 0]], np.int32), "episode": np.array([0, 1], np.int32),
        "temporal_view_times": np.array([[[0.0], [0.04]], [[0.0], [0.0]]]),
    }
    audit = data_audit(data, chunk=2)
    assert audit["chunk_mask"]["suffix_mask_is_prefix_valid"]
    assert audit["alignment"]["same_timestep_all_views"]


def test_encoder_is_batch_size_independent_between_train_and_eval():
    encoder = SharedEncoder()
    x = torch.rand(1, 3, 32, 32)
    encoder.train(); train_out = encoder(x)
    encoder.eval(); eval_out = encoder(x)
    torch.testing.assert_close(train_out, eval_out)


def test_overfit_checkpoint_metadata_is_weights_only_loadable(tmp_path):
    path = tmp_path / "overfit.pt"
    torch.save({"state_dict": {"weight": torch.ones(1)}, "indices": [0, 3], "normalization": {"mean": [0.0]}}, path)
    loaded = torch.load(path, weights_only=True)
    assert loaded["indices"] == [0, 3]
