"""Cleaning, feature engineering and scaling for the fraud dataset."""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler

from src.data_loader import PCA_COLUMNS

RAW_FEATURE_COLUMNS = ["Time", *PCA_COLUMNS, "Amount"]
SCALED_COLUMNS = ["log_amount", "Time"]
PASSTHROUGH_COLUMNS = [*PCA_COLUMNS, "hour_of_day"]


def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Drop exact duplicate rows and report how many were removed.

    Args:
        df: Raw transactions including the target column.

    Returns:
        Tuple of (deduplicated DataFrame with a fresh index, rows removed).
    """
    n_before = len(df)
    clean = df.drop_duplicates().reset_index(drop=True)
    n_removed = n_before - len(clean)
    print(
        f"Removed {n_removed:,} duplicate rows "
        f"({n_removed / n_before:.3%}); {len(clean):,} rows remain."
    )
    return clean, n_removed


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``hour_of_day`` and ``log_amount``. Stateless, so leak-free.

    ``Time`` is seconds since the first transaction, so ``hour_of_day`` is
    an hour *offset* (0-23), not a real clock hour.

    Args:
        df: DataFrame containing ``Time`` and ``Amount``.

    Returns:
        A copy of ``df`` with two extra columns.
    """
    out = df.copy()
    out["hour_of_day"] = ((out["Time"] // 3600) % 24).astype(int)
    out["log_amount"] = np.log1p(out["Amount"])
    return out


def build_preprocessor() -> Pipeline:
    """Build the reusable preprocessing pipeline (unfitted).

    Steps:
        1. ``features``: create ``hour_of_day`` and ``log_amount``.
        2. ``columns``: RobustScaler on ``log_amount`` and ``Time``;
           ``V1-V28`` and ``hour_of_day`` pass through; raw ``Amount`` is
           dropped (``log_amount`` replaces it).

    Returns:
        A scikit-learn Pipeline that outputs a 31-column DataFrame.
    """
    column_step = ColumnTransformer(
        transformers=[
            ("robust", RobustScaler(), SCALED_COLUMNS),
            ("keep", "passthrough", PASSTHROUGH_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    column_step.set_output(transform="pandas")
    return Pipeline(
        [
            ("features", FunctionTransformer(add_features)),
            ("columns", column_step),
        ]
    )