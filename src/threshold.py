"""Threshold selection from out-of-fold predictions (never the test set)."""

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import MIN_RECALL, REVIEW_COST
from src.evaluate import confusion_counts


def average_fraud_amount(X: pd.DataFrame, y: pd.Series) -> float:
    """Mean raw Amount of fraudulent transactions (training data)."""
    return float(X.loc[y == 1, "Amount"].mean())


def threshold_table(y_true, y_score) -> pd.DataFrame:
    """Confusion counts, precision and recall for every distinct threshold.

    A transaction is flagged when ``score >= threshold``.

    Args:
        y_true: True labels (0/1).
        y_score: Scores or probabilities (higher = more fraudulent).

    Returns:
        DataFrame with columns threshold, tp, fp, fn, precision, recall.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=float)
    order = np.argsort(-s, kind="stable")
    s_sorted, y_sorted = s[order], y[order]
    tp, fp = np.cumsum(y_sorted), np.cumsum(1 - y_sorted)
    # keep the last row of each tie so one row = one distinct threshold
    last = np.r_[np.where(np.diff(s_sorted) != 0)[0], len(s_sorted) - 1]
    n_pos = int(y.sum())
    table = pd.DataFrame(
        {"threshold": s_sorted[last], "tp": tp[last], "fp": fp[last]})
    table["fn"] = n_pos - table["tp"]
    table["precision"] = table["tp"] / (table["tp"] + table["fp"])
    table["recall"] = table["tp"] / n_pos
    return table


def add_cost(
    table: pd.DataFrame,
    missed_fraud_cost: float,
    false_alarm_cost: float = REVIEW_COST,
) -> pd.DataFrame:
    """Add ``cost = FN * missed_fraud_cost + FP * false_alarm_cost``."""
    out = table.copy()
    out["cost"] = (out["fn"] * missed_fraud_cost
                   + out["fp"] * false_alarm_cost)
    return out


def select_cost_threshold(
    table: pd.DataFrame,
    missed_fraud_cost: float,
    false_alarm_cost: float = REVIEW_COST,
) -> pd.Series:
    """Return the table row with the lowest total cost."""
    costed = add_cost(table, missed_fraud_cost, false_alarm_cost)
    return costed.loc[costed["cost"].idxmin()]


def select_recall_threshold(
    table: pd.DataFrame, min_recall: float = MIN_RECALL
) -> pd.Series:
    """Among thresholds with recall >= ``min_recall``, best precision.

    Raises:
        ValueError: If no threshold reaches the required recall.
    """
    eligible = table[table["recall"] >= min_recall]
    if eligible.empty:
        raise ValueError(f"No threshold reaches recall >= {min_recall}.")
    return eligible.loc[eligible["precision"].idxmax()]


def cost_at_threshold(
    y_true,
    y_score,
    threshold: float,
    missed_fraud_cost: float,
    false_alarm_cost: float = REVIEW_COST,
) -> Dict[str, float]:
    """Total cost at a fixed threshold, plus two do-nothing benchmarks.

    Returns:
        Dict with ``cost``, ``cost_no_model`` (flag nothing) and
        ``cost_flag_all`` (flag everything).
    """
    c = confusion_counts(y_true, y_score, threshold)
    n_pos, n_neg = c["tp"] + c["fn"], c["tn"] + c["fp"]
    return {
        "cost": c["fn"] * missed_fraud_cost + c["fp"] * false_alarm_cost,
        "cost_no_model": n_pos * missed_fraud_cost,
        "cost_flag_all": n_neg * false_alarm_cost,
    }


def plot_threshold_curves(
    table: pd.DataFrame,
    choices: Dict[str, float],
    missed_fraud_cost: float,
    false_alarm_cost: float = REVIEW_COST,
):
    """Plot precision/recall and total cost against the threshold.

    Args:
        table: Output of ``threshold_table`` (out-of-fold data).
        choices: Mapping of name -> chosen threshold (drawn as vlines).
        missed_fraud_cost: Cost X of a missed fraud.
        false_alarm_cost: Cost Y of a false alarm.

    Returns:
        The matplotlib figure.
    """
    t = add_cost(table, missed_fraud_cost, false_alarm_cost)
    t = t.sort_values("threshold")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    axes[0].plot(t["threshold"], t["precision"], label="precision")
    axes[0].plot(t["threshold"], t["recall"], label="recall")
    axes[1].plot(t["threshold"], t["cost"], color="crimson")
    for name, thr in choices.items():
        axes[0].axvline(thr, linestyle="--", color="grey")
        axes[0].annotate(f"{name}\n{thr:.3f}", (thr, 0.05), fontsize=8)
        axes[1].axvline(thr, linestyle="--", color="grey")
    axes[0].set(xlabel="Threshold", ylabel="Score", xlim=(0, 1),
                title="Precision and recall vs threshold (out-of-fold)")
    axes[0].legend()
    axes[1].set(xlabel="Threshold", ylabel="Total cost", xlim=(0, 1),
                title="Total cost vs threshold (out-of-fold)")
    plt.tight_layout()
    return fig