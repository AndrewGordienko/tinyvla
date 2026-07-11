import numpy as np

from scripts.deployable_controller import expert_chunk_from_snapshot, restore, snapshot
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
