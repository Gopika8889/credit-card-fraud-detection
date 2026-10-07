"""Shared constants for the fraud detection project."""

from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

VAL_SIZE = 0.2               # share of a training set held out for early stopping
EARLY_STOPPING_ROUNDS = 50
MAX_ESTIMATORS = 2000        # upper bound; early stopping picks the real number
N_TRIALS = 30
REVIEW_COST = 5.0            # cost of manually reviewing one flagged transaction
MIN_RECALL = 0.90