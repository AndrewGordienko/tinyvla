from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = ROOT / "artifacts"
CHECKPOINTS_ROOT = ARTIFACTS_ROOT / "checkpoints"
DATA_ROOT = ROOT / "data"
DATASETS_ROOT = DATA_ROOT / "datasets"
MODELS_ROOT = DATA_ROOT / "models"
SO_ARM_ROOT = ROOT / "SO-ARM100"
SO101_SIM_ROOT = SO_ARM_ROOT / "Simulation" / "SO101"
SO101_SCENE = SO101_SIM_ROOT / "scene.xml"
SO101_TASK = SO101_SIM_ROOT / "task.xml"
