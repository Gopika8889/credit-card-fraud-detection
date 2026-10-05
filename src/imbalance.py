"""Imbalance-handling strategies, applied inside training folds only."""

from typing import List, Tuple

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from src.config import RANDOM_STATE

STRATEGIES = ("none", "class_weight", "undersample", "smote", "smote_under")


def get_sampling_steps(
    strategy: str, random_state: int = RANDOM_STATE
) -> List[Tuple[str, object]]:
    """Return imblearn pipeline steps for a resampling strategy.

    Args:
        strategy: One of ``STRATEGIES``.
        random_state: Seed for the samplers.

    Returns:
        A list of ``(name, sampler)`` tuples (empty for strategies that do
        not resample).

    Raises:
        ValueError: If the strategy is unknown.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Use {STRATEGIES}.")
    if strategy == "undersample":
        return [("sampler", RandomUnderSampler(random_state=random_state))]
    if strategy == "smote":
        return [("sampler", SMOTE(random_state=random_state))]
    if strategy == "smote_under":
        return [
            ("smote", SMOTE(sampling_strategy=0.1, random_state=random_state)),
            (
                "under",
                RandomUnderSampler(
                    sampling_strategy=0.5, random_state=random_state
                ),
            ),
        ]
    return []


def uses_class_weight(strategy: str) -> bool:
    """Return True if the strategy relies on classifier class weights."""
    return strategy == "class_weight"


def compute_scale_pos_weight(y: pd.Series) -> float:
    """Compute XGBoost's ``scale_pos_weight`` = n_negative / n_positive.

    Used in Stage 3. Compute it on training labels only.
    """
    n_pos = int((y == 1).sum())
    return float((y == 0).sum() / n_pos)