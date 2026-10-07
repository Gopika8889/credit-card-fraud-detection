"""Boosted-tree models (XGBoost, LightGBM) and an anomaly-detection reference."""

import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline as SkPipeline
from xgboost import XGBClassifier

from src.config import RANDOM_STATE
from src.imbalance import (
    compute_scale_pos_weight, get_sampling_steps, uses_class_weight)
from src.preprocessing import build_preprocessor
from src.train_baseline import cross_validate_pipeline

BOOSTED_MODELS = ("xgb", "lgbm")
BOOST_STRATEGIES = ("none", "class_weight", "smote")


def get_boosted_classifier(
    model_name: str,
    params: Optional[Dict[str, Any]] = None,
    scale_pos_weight: float = 1.0,
    random_state: int = RANDOM_STATE,
):
    """Create an XGBoost or LightGBM classifier.

    Args:
        model_name: ``"xgb"`` or ``"lgbm"``.
        params: Overrides for the defaults (e.g. tuned hyperparameters).
        scale_pos_weight: Fraud-class weight; ``params`` may override it.
        random_state: Seed.

    Returns:
        An unfitted classifier.

    Raises:
        ValueError: If the model name is unknown.
    """
    params = dict(params or {})
    if model_name == "xgb":
        config = dict(
            n_estimators=300, learning_rate=0.1, max_depth=6,
            tree_method="hist", eval_metric="aucpr", n_jobs=-1,
            scale_pos_weight=scale_pos_weight, random_state=random_state,
        )
        config.update(params)
        return XGBClassifier(**config)
    if model_name == "lgbm":
        config = dict(
            n_estimators=300, learning_rate=0.1, num_leaves=31,
            metric="average_precision", n_jobs=-1, verbose=-1,
            deterministic=True, force_row_wise=True,
            scale_pos_weight=scale_pos_weight, random_state=random_state,
        )
        config.update(params)
        return LGBMClassifier(**config)
    raise ValueError(f"Unknown model '{model_name}'. Use {BOOSTED_MODELS}.")


def build_boosted_pipeline(
    model_name: str,
    strategy: str,
    params: Optional[Dict[str, Any]] = None,
    scale_pos_weight: float = 1.0,
) -> ImbPipeline:
    """Assemble preprocessing -> (resampler) -> boosted classifier.

    Steps are flattened because imblearn pipelines cannot be nested.

    Args:
        model_name: ``"xgb"`` or ``"lgbm"``.
        strategy: One of ``src.imbalance.STRATEGIES``.
        params: Optional classifier overrides.
        scale_pos_weight: Used only when ``strategy == "class_weight"``.

    Returns:
        An unfitted imblearn Pipeline.
    """
    steps = list(build_preprocessor().steps)
    steps += get_sampling_steps(strategy)
    weight = scale_pos_weight if uses_class_weight(strategy) else 1.0
    steps.append(
        ("clf", get_boosted_classifier(model_name, params, weight)))
    return ImbPipeline(steps)


def compare_imbalance_strategies(
    X: pd.DataFrame,
    y: pd.Series,
    model_names: Sequence[str] = BOOSTED_MODELS,
    strategies: Sequence[str] = BOOST_STRATEGIES,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Cross-validate boosted models under each imbalance strategy.

    Resampling happens only inside training folds (imblearn Pipeline).

    Args:
        X: Raw training features.
        y: Training labels.
        model_names: Models to compare.
        strategies: Imbalance strategies to compare.

    Returns:
        ``(results, oof_probas)``; ``results`` has one row per combination.
    """
    ratio = compute_scale_pos_weight(y)  # one scalar from training labels
    rows, oof_probas = {}, {}
    for model_name in model_names:
        for strategy in strategies:
            label = f"{model_name} + {strategy}"
            start = time.perf_counter()
            fold_df, oof = cross_validate_pipeline(
                build_boosted_pipeline(
                    model_name, strategy, scale_pos_weight=ratio), X, y)
            row = fold_df.mean().to_dict()
            row["pr_auc_std"] = fold_df["pr_auc"].std()
            row["cv_time_s"] = time.perf_counter() - start
            rows[label] = row
            oof_probas[label] = oof
            print(f"{label:<22} PR-AUC={row['pr_auc']:.3f} "
                  f"recall={row['recall']:.3f} ({row['cv_time_s']:.0f}s)")
    results = pd.DataFrame.from_dict(rows, orient="index")
    results.index.name = "model + strategy"
    return results, oof_probas


def fit_final_model(
    model_name: str,
    record: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
):
    """Fit preprocessor + tuned model on ALL training data.

    ``n_estimators`` is the median early-stopping iteration from CV, scaled
    by 1.1 because the final fit sees ~25% more data than a CV fold.

    Args:
        model_name: ``"xgb"`` or ``"lgbm"``.
        record: Tuning record from ``src.tuning.tune_model``.
        X_train: Raw training features.
        y_train: Training labels.

    Returns:
        ``(preprocessor, model, train_time_seconds)``.
    """
    start = time.perf_counter()
    preprocessor = build_preprocessor().fit(X_train)
    n_estimators = max(50, int(round(record["best_iteration"] * 1.1)))
    params = {**record["best_params"], "n_estimators": n_estimators}
    model = get_boosted_classifier(model_name, params)
    model.fit(preprocessor.transform(X_train), y_train)
    return preprocessor, model, time.perf_counter() - start


def fit_isolation_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> SkPipeline:
    """Fit preprocessing + Isolation Forest on legitimate training rows only.

    Args:
        X_train: Raw training features.
        y_train: Training labels (used only to select legit rows).
        random_state: Seed.

    Returns:
        A fitted pipeline.
    """
    steps = list(build_preprocessor().steps)
    steps.append(("iforest", IsolationForest(
        n_estimators=200, max_samples=256, random_state=random_state,
        n_jobs=-1)))
    pipeline = SkPipeline(steps)
    pipeline.fit(X_train.loc[y_train == 0])
    return pipeline


def anomaly_scores(pipeline: SkPipeline, X: pd.DataFrame) -> np.ndarray:
    """Return anomaly scores where HIGHER means more anomalous."""
    X_t = pipeline[:-1].transform(X)
    return -pipeline.named_steps["iforest"].score_samples(X_t)


def anomaly_threshold(
    pipeline: SkPipeline, X_train: pd.DataFrame, y_train: pd.Series
) -> float:
    """Score cut-off that flags the training fraud prevalence of rows."""
    scores = anomaly_scores(pipeline, X_train)
    return float(np.quantile(scores, 1 - y_train.mean()))