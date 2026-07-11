from pathlib import Path

from tinyvla.interview_demo import DEFAULT_RESULT, DEFAULT_VIDEOS, load_payload


def test_interview_payload_is_deterministic_and_exposes_champion():
    payload = load_payload(Path(DEFAULT_RESULT), Path(DEFAULT_VIDEOS))
    assert payload["champion_round"] == 0
    assert payload["champion_result"] == {"success": 2, "n": 4}
    assert payload["checkpoint_sha256"] is None
    assert [row["video"] for row in payload["scenes"]].count(None) == 0
