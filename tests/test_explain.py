"""Unit tests for src.explain."""

from xgboost import XGBClassifier

from src.explain import explain_transaction
from src.preprocessing import build_preprocessor


def test_explain_transaction_returns_sorted_top_k(synthetic_df):
    X = synthetic_df.drop(columns=["Class"])
    y = synthetic_df["Class"]
    X_t = build_preprocessor().fit(X).transform(X)
    model = XGBClassifier(n_estimators=20, max_depth=3,
                          random_state=42).fit(X_t, y)
    out = explain_transaction(model, X_t.iloc[[0]], top_k=5)
    assert len(out) == 5
    assert {"feature", "value", "shap_value", "effect"} <= set(out[0])
    magnitudes = [abs(d["shap_value"]) for d in out]
    assert magnitudes == sorted(magnitudes, reverse=True)