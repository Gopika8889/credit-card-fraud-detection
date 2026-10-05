"""Baseline model training with stratified cross-validation."""

import time
from pathlib import Path
from typing import Dict, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline as SkPipeline

from src.config import MODELS_DIR, N_SPLITS, RANDOM_STATE
from src.data_loader import load_data
from src.evaluate import compute_metrics, confusion_counts
from src.imbalance import STRATEGIES, get_sampling_steps, uses_class_weight
from src.preprocessing import build_preprocessor, remove_duplicates
from src.split import stratified_split, summarize_split

MODEL_NAMES = ("logreg", "rf")
PREPROCESS_STEP_NAMES = ("features", "columns")


def get_classifier(
    model_name: str,
    use_class_weight: bool = False,
    random_state: int = RANDOM_STATE,
):
    """Create a baseline classifier.

    Args:
        model_name: ``"logreg"`` or ``"rf"``.
        use_class_weight: Apply balanced class weights.
        random_state: Seed.

    Returns:
        An unfitted scikit-learn classifier.

    Raises:
        ValueError: If the model name is unknown.
    """
    if model_name == "logreg":
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced" if use_class_weight else None,
            random_state=random_state,
        )
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            n_jobs=-1,
            class_weight="balanced_subsample" if use_class_weight else None,
            random_state=random_state,
        )
    raise ValueError(f"Unknown model '{model_name}'. Use {MODEL_NAMES}.")


def build_pipeline(model_name: str, strategy: str) -> ImbPipeline:
    """Assemble preprocessing -> (resampler) -> classifier.

    imblearn's Pipeline cannot contain a nested Pipeline, so the
    preprocessor's steps ("features", "columns") are placed directly in the
    pipeline, followed by the optional sampler and the classifier.

    Args:
        model_name: ``"logreg"`` or ``"rf"``.
        strategy: One of ``src.imbalance.STRATEGIES``.

    Returns:
        An unfitted imblearn Pipeline.
    """
    steps = list(build_preprocessor().steps)
    steps += get_sampling_steps(strategy)
    steps.append(
        ("clf", get_classifier(model_name, uses_class_weight(strategy)))
    )
    return ImbPipeline(steps)


def cross_validate_pipeline(
    pipeline: ImbPipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Stratified K-fold CV returning per-fold metrics and OOF probabilities.

    Args:
        pipeline: Unfitted pipeline (cloned for every fold).
        X: Raw training features.
        y: Training labels.
        n_splits: Number of folds.
        random_state: Seed for fold shuffling.

    Returns:
        ``(fold_metrics, oof_proba)`` where ``fold_metrics`` has one row per
        fold and ``oof_proba`` aligns positionally with ``X``.
    """
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    oof_proba = np.zeros(len(y))
    fold_rows = []
    for train_idx, val_idx in skf.split(X, y):
        model = clone(pipeline)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = model.predict_proba(X.iloc[val_idx])[:, 1]
        oof_proba[val_idx] = proba
        fold_rows.append(compute_metrics(y.iloc[val_idx], proba))
    return pd.DataFrame(fold_rows), oof_proba


def run_experiments(
    X: pd.DataFrame,
    y: pd.Series,
    model_names: Sequence[str] = MODEL_NAMES,
    strategies: Sequence[str] = STRATEGIES,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Cross-validate every model x imbalance-strategy combination.

    Args:
        X: Raw training features.
        y: Training labels.
        model_names: Models to compare.
        strategies: Imbalance strategies to compare.

    Returns:
        ``(results, oof_probas)``. ``results`` has one row per combination
        (index ``"model + strategy"``) with mean fold metrics, PR-AUC std
        and pooled out-of-fold confusion counts at threshold 0.5.
    """
    rows, oof_probas = {}, {}
    for model_name in model_names:
        for strategy in strategies:
            label = f"{model_name} + {strategy}"
            start = time.perf_counter()
            fold_df, oof = cross_validate_pipeline(
                build_pipeline(model_name, strategy), X, y
            )
            row = fold_df.mean().to_dict()
            row["pr_auc_std"] = fold_df["pr_auc"].std()
            row.update(confusion_counts(y, oof))
            rows[label] = row
            oof_probas[label] = oof
            print(
                f"{label:<28} PR-AUC={row['pr_auc']:.3f} "
                f"recall={row['recall']:.3f} "
                f"({time.perf_counter() - start:.0f}s)"
            )
    results = pd.DataFrame.from_dict(rows, orient="index")
    results.index.name = "model + strategy"
    return results, oof_probas


def fit_and_save_best(
    best_label: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    models_dir=MODELS_DIR,
) -> ImbPipeline:
    """Fit the chosen pipeline on ALL training data and save artifacts.

    Saves ``best_baseline.joblib`` (full pipeline) and
    ``preprocessor.joblib`` (the already-fitted preprocessing steps, fit on
    training data only). Loading them requires ``src`` to be importable.

    Args:
        best_label: Label such as ``"rf + class_weight"``.
        X_train: Raw training features.
        y_train: Training labels.
        models_dir: Output directory.

    Returns:
        The fitted pipeline.
    """
    model_name, strategy = best_label.split(" + ")
    pipeline = build_pipeline(model_name, strategy)
    pipeline.fit(X_train, y_train)

    # Reuse the fitted step objects, so no refitting and no leakage.
    preprocessor = SkPipeline(
        [(name, pipeline.named_steps[name]) for name in PREPROCESS_STEP_NAMES]
    )

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, models_dir / "best_baseline.joblib")
    joblib.dump(preprocessor, models_dir / "preprocessor.joblib")
    return pipeline


def main() -> None:
    """Run the full baseline experiment end-to-end (test set untouched)."""
    df, _ = remove_duplicates(load_data(verbose=False))
    X_train, _, y_train, y_test = stratified_split(df)
    summarize_split(y_train, y_test, "Stratified 80/20")
    results, _ = run_experiments(X_train, y_train)
    print(results.round(3).to_string())
    fit_and_save_best(results["pr_auc"].idxmax(), X_train, y_train)


if __name__ == "__main__":
    main()