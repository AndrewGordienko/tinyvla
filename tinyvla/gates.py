"""Run the local command-0 memorization and deterministic held-out gates."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from .eval_closedloop import evaluate_closed_loop
from .paths import ARTIFACTS_ROOT
from .runtime import experiment_metadata, load_runtime, sha256_file, sha256_tree


def _manifest_positions(root: Path) -> dict[tuple[int, int], dict[str, np.ndarray]]:
    path = root / "scene_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"four-scene gate requires {path}")
    data = json.loads(path.read_text())
    scenes = [scene for scene in data["scenes"] if int(scene["command"]) == 0]
    return {
        (0, index): {
            color: np.asarray(position, dtype=np.float64)
            for color, position in scene["positions"].items()
        }
        for index, scene in enumerate(scenes)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--repo-id", default="local/truth_gate_command0_4")
    parser.add_argument(
        "--root", default=str(ARTIFACTS_ROOT / "truth_harness" / "datasets" / "command0_4")
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--cap", type=int, default=140)
    parser.add_argument("--held-out", type=int, default=20)
    parser.add_argument(
        "--output", default=str(ARTIFACTS_ROOT / "truth_harness" / "latest_gates.json")
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="Always exit 0 and just write the report; by default a failed gate exits 1 "
             "so CI/shell pipelines cannot mistake a failing model for a passing one.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    device = torch.device(args.device)
    meta = LeRobotDatasetMetadata(args.repo_id, root=root)
    runtime = load_runtime(
        args.model, meta=meta, dataset_root=root, device=device, stats_source="checkpoint"
    )
    positions = _manifest_positions(root)
    overfit = evaluate_closed_loop(
        runtime.policy,
        runtime.preprocessor,
        runtime.postprocessor,
        device=device,
        commands=[0],
        cap=args.cap,
        seed=args.seed,
        delta_actions=runtime.delta_actions,
        episodes=len(positions),
        positions_by_rollout=positions,
    )
    held_out = evaluate_closed_loop(
        runtime.policy,
        runtime.preprocessor,
        runtime.postprocessor,
        device=device,
        commands=[0],
        cap=args.cap,
        seed=args.seed + 100_000,
        delta_actions=runtime.delta_actions,
        episodes=args.held_out,
    )
    result = {
        "model": str(Path(args.model).resolve()),
        "action_semantics": runtime.action_semantics,
        "load_report": runtime.load_report,
        "artifacts": {
            # content fingerprints so a result bundle is independently reproducible
            "checkpoint_sha256": sha256_tree(args.model, patterns=("*.safetensors", "*.json")),
            "dataset_manifest_sha256": sha256_file(root / "scene_manifest.json"),
            "repo_id": args.repo_id,
            "dataset_root": str(root.resolve()),
            "held_out_scenes": args.held_out,
            "cap": args.cap,
        },
        "four_scene_overfit": overfit,
        "held_out": held_out,
        "thresholds": {
            "four_scene_overfit": 0.95,
            "held_out": 0.80,
            "passed": overfit["success_rate"] >= 0.95 and held_out["success_rate"] >= 0.80,
        },
        "experiment": experiment_metadata(seed=args.seed),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

    passed = result["thresholds"]["passed"]
    if not passed and not args.report_only:
        print(
            f"GATE FAILED: four_scene_overfit={overfit['success_rate']:.0%} "
            f"(need >=95%), held_out={held_out['success_rate']:.0%} (need >=80%)",
            file=sys.stderr,
        )
        return 1
    print(f"GATE {'PASSED' if passed else 'FAILED (report-only)'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
