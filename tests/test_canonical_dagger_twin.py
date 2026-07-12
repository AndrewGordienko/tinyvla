import numpy as np

from scripts.canonical_dagger_round1 import (cached_observation_hash, new_event_tracker,
    physical_stage, snapshot, state_hash, update_event_tracker, verify_phase_clones)
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


def test_collection_cadence_runs_every_action_and_records_fixed_timesteps():
    env = SO101PickPlaceTask(seed=8); env.reset(command=0)
    cap, interval = 120, 5
    records = []; events = new_event_tracker()
    initial = state_hash(snapshot(env))
    for t in range(cap):
        if t % interval == 0:
            records.append(t)
        env.step(env.expert_action(gain=.5, max_dq=.06)); update_event_tracker(env, events)
    assert records == list(range(0, 120, 5))
    assert state_hash(snapshot(env)) != initial
    reference = SO101PickPlaceTask(seed=8); reference.reset(command=0)
    for _ in range(cap): reference.step(reference.expert_action(gain=.5, max_dq=.06))
    assert state_hash(snapshot(env)) == state_hash(snapshot(reference))


def test_physical_stage_tracker_never_relabels_release_as_approach():
    env = SO101PickPlaceTask(seed=9); env.reset(command=0)
    events = new_event_tracker(); seen = set()
    for _ in range(160):
        update_event_tracker(env, events); seen.add(physical_stage(events, env))
        env.step(env.expert_action(gain=.5, max_dq=.06))
    update_event_tracker(env, events); seen.add(physical_stage(events, env))
    assert {"approach", "grasp", "transport", "release"}.issubset(seen)
    assert physical_stage({"ever_grasped": True, "ever_lifted": True, "released_after_grasp": True}, env) == "release"
