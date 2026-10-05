"""Unit tests for src.preprocessing."""

import numpy as np
import pandas as pd

from src.preprocessing import (
    add_features, build_preprocessor, remove_duplicates)


def test_remove_duplicates(synthetic_df):
    doubled = pd.concat([synthetic_df, synthetic_df.iloc[:10]])
    clean, n_removed = remove_duplicates(doubled)
    assert n_removed == 10
    assert len(clean) == len(synthetic_df)


def test_add_features(synthetic_df):
    out = add_features(synthetic_df)
    assert out["hour_of_day"].between(0, 23).all()
    assert np.allclose(out["log_amount"], np.log1p(out["Amount"]))


def test_scaler_is_fit_on_train_only(synthetic_df):
    X = synthetic_df.drop(columns=["Class"])
    X_train, X_test = X.iloc[:800], X.iloc[800:]
    pre = build_preprocessor().fit(X_train)
    out = pre.transform(X_test)
    scaler = pre.named_steps["columns"].named_transformers_["robust"]
    assert np.isclose(
        scaler.center_[0], np.median(np.log1p(X_train["Amount"])))
    assert out.shape == (200, 31)