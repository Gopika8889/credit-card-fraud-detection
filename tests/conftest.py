"""Shared pytest fixtures."""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import EXPECTED_COLUMNS, PCA_COLUMNS


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Small synthetic dataset with the real schema (5% fraud)."""
    rng = np.random.default_rng(42)
    n = 1000
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=PCA_COLUMNS)
    df["Time"] = np.sort(rng.integers(0, 172_800, size=n)).astype(float)
    df["Amount"] = rng.exponential(scale=80, size=n).round(2)
    df["Class"] = 0
    df.loc[rng.choice(n, size=50, replace=False), "Class"] = 1
    return df[EXPECTED_COLUMNS]