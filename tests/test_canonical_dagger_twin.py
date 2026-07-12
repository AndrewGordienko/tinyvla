import numpy as np

from scripts.canonical_dagger_round1 import cached_observation_hash, snapshot, state_hash, verify_phase_clones
from tinyvla.task import SO101PickPlaceTask


def test_twin_clone_covers_all_grasp_phases_and_leaves_learner_unchanged():
    learner = SO101PickPlaceTask(seed=11)
    oracle = SO101PickPlaceTask(seed=12)
    result = verify_phase_clones(learner, oracle)
    assert set(result) == {"approach", "grasp", "transport", "release_before", "release"}


def test_snapshot_hash_includes_task_and_rng_state():
    env = SO101PickPlaceTask(seed=3)
    env.reset(command=0)
    before = snapshot(env)
    env.phase = 4
    assert state_hash(before) != state_hash(snapshot(env))
    env.phase = before["phase"]
    env.rng.random()
    assert state_hash(before) != state_hash(snapshot(env))


def test_cached_frame_hash_is_the_record_observation_hash():
    env = SO101PickPlaceTask(seed=7)
    env.reset(command=0)
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    frames = [frame, frame.copy()]
    record_hash = cached_observation_hash(env, frames)
    assert record_hash == cached_observation_hash(env, frames)
    frames[0][0, 0, 0] = 1
    assert record_hash != cached_observation_hash(env, frames)
