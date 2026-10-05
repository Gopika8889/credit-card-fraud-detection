"""Shared constants for the fraud detection project."""

from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"