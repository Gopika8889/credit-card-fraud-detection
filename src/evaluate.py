"""Metrics and plots for imbalanced binary classification."""

from typing import Dict, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_NAMES = ("precision", "recall", "f1", "pr_auc", "roc_auc")


def compute_metrics(
    y_true, y_proba, threshold: float = 0.5
) -> Dict[str, float]:
    """Compute imbalance-appropriate metrics (no accuracy).

    Args:
        y_true: True labels (0/1).
        y_proba: Predicted fraud probabilities.
        threshold: Cut-off for converting probabilities to labels.

    Returns:
        Dict with precision, recall, f1, pr_auc (average precision), roc_auc.
    """
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_proba),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def confusion_counts(
    y_true, y_proba, threshold: float = 0.5
) -> Dict[str, int]:
    """Return TN/FP/FN/TP counts at a given threshold."""
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def plot_pr_curves(
    y_true, probas: Mapping[str, np.ndarray], ax: Optional[Axes] = None
) -> Axes:
    """Overlay Precision-Recall curves for several models.

    Args:
        y_true: True labels.
        probas: Mapping of label -> predicted probabilities.
        ax: Optional axes to draw on.

    Returns:
        The axes with the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    for label, proba in probas.items():
        precision, recall, _ = precision_recall_curve(y_true, proba)
        ap = average_precision_score(y_true, proba)
        ax.plot(recall, precision, label=f"{label} (AP={ap:.3f})")
    prevalence = float(np.mean(y_true))
    ax.axhline(
        prevalence, color="grey", linestyle="--",
        label=f"No-skill ({prevalence:.4f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves (out-of-fold, training set)")
    ax.legend(loc="upper right", fontsize=8)
    return ax


def plot_confusion_matrices(
    y_true, probas: Mapping[str, np.ndarray], threshold: float = 0.5
) -> None:
    """Plot one confusion matrix per model at a fixed threshold."""
    fig, axes = plt.subplots(
        1, len(probas), figsize=(4.5 * len(probas), 4), squeeze=False
    )
    for ax, (label, proba) in zip(axes[0], probas.items()):
        c = confusion_counts(y_true, proba, threshold)
        cm = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
            xticklabels=["Pred legit", "Pred fraud"],
            yticklabels=["True legit", "True fraud"],
        )
        ax.set_title(label, fontsize=10)
    plt.tight_layout()