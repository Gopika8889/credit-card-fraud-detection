"""Unit tests for src.split."""

from src.split import stratified_split, time_based_split


def test_stratified_split_preserves_ratio(synthetic_df):
    X_train, X_test, y_train, y_test = stratified_split(synthetic_df)
    assert len(X_train) + len(X_test) == len(synthetic_df)
    assert abs(y_train.mean() - y_test.mean()) < 0.01
    assert set(X_train.index).isdisjoint(X_test.index)


def test_time_split_is_chronological(synthetic_df):
    X_train, X_test, _, _ = time_based_split(synthetic_df)
    assert X_train["Time"].max() < X_test["Time"].min()