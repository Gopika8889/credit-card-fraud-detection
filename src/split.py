"""Train/test splitting utilities."""

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TEST_SIZE
from src.data_loader import TARGET_COLUMN

SplitResult = Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]


def stratified_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> SplitResult:
    """Stratified random split that preserves the fraud ratio.

    Args:
        df: Deduplicated DataFrame including the target.
        test_size: Fraction of rows held out for testing.
        random_state: Seed for reproducibility.

    Returns:
        ``X_train, X_test, y_train, y_test`` (raw feature columns only).
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def time_based_split(
    df: pd.DataFrame, test_size: float = TEST_SIZE
) -> SplitResult:
    """Chronological split: train on earlier rows, test on later ones.

    The cut is a ``Time`` threshold, so no timestamp is shared between
    train and test. The fraud ratio is NOT controlled here.

    Args:
        df: Deduplicated DataFrame including ``Time`` and the target.
        test_size: Approximate fraction of latest rows held out.

    Returns:
        ``X_train, X_test, y_train, y_test``.
    """
    ordered = df.sort_values("Time", kind="stable").reset_index(drop=True)
    cutoff = ordered["Time"].iloc[int(len(ordered) * (1 - test_size))]
    train = ordered[ordered["Time"] < cutoff]
    test = ordered[ordered["Time"] >= cutoff]
    return (
        train.drop(columns=[TARGET_COLUMN]),
        test.drop(columns=[TARGET_COLUMN]),
        train[TARGET_COLUMN],
        test[TARGET_COLUMN],
    )


def summarize_split(
    y_train: pd.Series, y_test: pd.Series, title: str = "Split"
) -> pd.DataFrame:
    """Print and return class counts per split to verify stratification.

    Args:
        y_train: Training labels.
        y_test: Test labels.
        title: Heading for the printout.

    Returns:
        DataFrame with rows ``train``/``test`` and columns ``n_rows``,
        ``n_fraud``, ``n_legit``, ``fraud_pct``.
    """
    rows = {
        name: {
            "n_rows": len(y),
            "n_fraud": int(y.sum()),
            "n_legit": int((y == 0).sum()),
            "fraud_pct": y.mean() * 100,
        }
        for name, y in (("train", y_train), ("test", y_test))
    }
    summary = pd.DataFrame.from_dict(rows, orient="index")
    print(f"\n{title}\n{summary.round(4)}")
    return summary