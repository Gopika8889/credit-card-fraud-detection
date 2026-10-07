"""SHAP explanations for the final tree model."""

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import RANDOM_STATE


def build_explainer(model) -> shap.TreeExplainer:
    """Create a TreeExplainer for a fitted XGBoost/LightGBM model."""
    return shap.TreeExplainer(model)


def shap_values_for(explainer, X: pd.DataFrame) -> np.ndarray:
    """SHAP values for the fraud class as a 2-D array (rows x features).

    Handles the different output shapes SHAP returns across model types.
    """
    values = explainer.shap_values(X)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    return values


def base_value(explainer) -> float:
    """Expected model output (log-odds) for the fraud class."""
    return float(np.ravel(explainer.expected_value)[-1])


def sample_for_shap(
    X: pd.DataFrame,
    y: pd.Series,
    n_legit: int = 3000,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """All fraud rows plus a random sample of legit rows.

    Fraud-enriched on purpose so the summary plot shows fraud behaviour.
    Mean |SHAP| on this sample therefore measures fraud-vs-legit
    separation, not importance on the natural 0.17% mix.
    """
    labels = y.to_numpy()
    rng = np.random.default_rng(random_state)
    fraud_idx = np.flatnonzero(labels == 1)
    legit_idx = np.flatnonzero(labels == 0)
    legit_pick = rng.choice(
        legit_idx, size=min(n_legit, len(legit_idx)), replace=False)
    return X.iloc[np.sort(np.r_[fraud_idx, legit_pick])]


def plot_global_importance(explainer, X: pd.DataFrame) -> pd.Series:
    """Beeswarm summary plot and bar plot; returns mean |SHAP| per feature.

    Args:
        explainer: SHAP explainer.
        X: Preprocessed feature sample.

    Returns:
        Series of mean absolute SHAP values, sorted descending.
    """
    values = shap_values_for(explainer, X)
    plt.figure()
    shap.summary_plot(values, X, show=False, max_display=15)
    plt.tight_layout()
    plt.show()
    plt.figure()
    shap.summary_plot(values, X, plot_type="bar", show=False, max_display=15)
    plt.tight_layout()
    plt.show()
    return pd.Series(np.abs(values).mean(axis=0),
                     index=X.columns).sort_values(ascending=False)


def plot_waterfall(explainer, x_row: pd.DataFrame, title: str = "") -> None:
    """Waterfall plot for a single preprocessed transaction (one-row frame)."""
    row = x_row.iloc[[0]]
    explanation = shap.Explanation(
        values=shap_values_for(explainer, row)[0],
        base_values=base_value(explainer),
        data=row.iloc[0].to_numpy(),
        feature_names=list(row.columns),
    )
    shap.plots.waterfall(explanation, max_display=10, show=False)
    plt.gcf().suptitle(title, y=1.02)
    plt.show()


def explain_transaction(
    model,
    x_row,
    top_k: int = 5,
    explainer: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Top contributing features for one PREPROCESSED transaction.

    Intended for the Stage 4 API. Pass a prebuilt ``explainer`` to avoid
    rebuilding it on every request.

    Args:
        model: Fitted XGBoost/LightGBM classifier.
        x_row: One-row DataFrame (or Series) of preprocessed features, in
            the training column order.
        top_k: Number of features to return.
        explainer: Optional prebuilt SHAP explainer.

    Returns:
        List of dicts (feature, value, shap_value, effect), sorted by
        absolute SHAP value. ``value`` is the preprocessed (scaled) value
        for ``log_amount`` and ``Time``.
    """
    if explainer is None:
        explainer = build_explainer(model)
    if isinstance(x_row, pd.Series):
        x_row = x_row.to_frame().T.astype(float)
    row = x_row.iloc[[0]]
    values = shap_values_for(explainer, row)[0]
    order = np.argsort(-np.abs(values))[:top_k]
    return [
        {
            "feature": str(row.columns[i]),
            "value": float(row.iloc[0, i]),
            "shap_value": float(values[i]),
            "effect": ("increases fraud risk" if values[i] > 0
                       else "decreases fraud risk"),
        }
        for i in order
    ]