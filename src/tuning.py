"""Optuna tuning for XGBoost and LightGBM (training data only)."""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.config import (
    EARLY_STOPPING_ROUNDS, MAX_ESTIMATORS, MODELS_DIR, N_SPLITS, N_TRIALS,
    RANDOM_STATE, VAL_SIZE)
from src.preprocessing import build_preprocessor
from src.train_advanced import get_boosted_classifier


@dataclass
class Fold:
    """One CV fold with its early-stopping split (all preprocessed)."""

    val_idx: np.ndarray       # positions of the validation fold in X_train
    X_fit: pd.DataFrame
    y_fit: pd.Series
    X_es: pd.DataFrame        # early-stopping set carved from the fold's train
    y_es: pd.Series
    X_val: pd.DataFrame       # untouched validation fold (scoring only)
    y_val: pd.Series


def make_folds(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = N_SPLITS,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
) -> List[Fold]:
    """Build stratified folds with leak-free preprocessing.

    For each fold the preprocessor is fit on that fold's training rows only;
    the training rows are then split into fit/early-stopping parts.

    Args:
        X: Raw training features.
        y: Training labels.
        n_splits: Number of CV folds.
        val_size: Early-stopping share of each fold's training rows.
        random_state: Seed.

    Returns:
        A list of ``Fold`` objects.
    """
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
        pre = build_preprocessor().fit(X_tr)
        X_tr_t, X_va_t = pre.transform(X_tr), pre.transform(X_va)
        X_fit, X_es, y_fit, y_es = train_test_split(
            X_tr_t, y_tr, test_size=val_size, stratify=y_tr,
            random_state=random_state)
        folds.append(Fold(va_idx, X_fit, y_fit, X_es, y_es, X_va_t, y_va))
    return folds


def fit_with_early_stopping(
    model_name: str,
    params: Dict[str, Any],
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_es: pd.DataFrame,
    y_es: pd.Series,
) -> Tuple[Any, int]:
    """Fit a boosted model, stopping when the early-stopping PR-AUC stalls.

    Args:
        model_name: ``"xgb"`` or ``"lgbm"``.
        params: Model hyperparameters (not mutated).
        X_fit: Training features (preprocessed).
        y_fit: Training labels.
        X_es: Early-stopping features.
        y_es: Early-stopping labels.

    Returns:
        ``(fitted_model, best_number_of_trees)``.
    """
    params = {"n_estimators": MAX_ESTIMATORS, **params}
    if model_name == "xgb":
        params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
        model = get_boosted_classifier("xgb", params)
        model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)
        best = int(model.best_iteration) + 1      # 0-indexed -> count
    else:
        model = get_boosted_classifier("lgbm", params)
        model.fit(
            X_fit, y_fit, eval_set=[(X_es, y_es)],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS,
                                          verbose=False)])
        best = int(model.best_iteration_) or int(params["n_estimators"])
    return model, best


def suggest_params(
    trial: optuna.Trial, model_name: str, scale_ratio: float
) -> Dict[str, Any]:
    """Sample one hyperparameter set from the search space.

    Args:
        trial: Optuna trial.
        model_name: ``"xgb"`` or ``"lgbm"``.
        scale_ratio: n_negative / n_positive of the training labels.

    Returns:
        Keyword arguments ready for ``get_boosted_classifier``.
    """
    params: Dict[str, Any] = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3,
                                             log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "scale_pos_weight": scale_ratio * trial.suggest_float(
            "spw_fraction", 0.01, 1.0, log=True),
    }
    if model_name == "xgb":
        params.update(
            max_depth=trial.suggest_int("max_depth", 3, 10),
            min_child_weight=trial.suggest_float(
                "min_child_weight", 1.0, 20.0, log=True),
            gamma=trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        )
    else:
        params.update(
            num_leaves=trial.suggest_int("num_leaves", 8, 128, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            min_child_samples=trial.suggest_int(
                "min_child_samples", 5, 100, log=True),
            subsample_freq=1,  # LightGBM ignores subsample without this
        )
    return params


def make_objective(model_name: str, folds: List[Fold], scale_ratio: float):
    """Create the Optuna objective: mean PR-AUC over the CV folds."""

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, model_name, scale_ratio)
        scores, iterations = [], []
        for fold in folds:
            model, best = fit_with_early_stopping(
                model_name, params, fold.X_fit, fold.y_fit,
                fold.X_es, fold.y_es)
            proba = model.predict_proba(fold.X_val)[:, 1]
            scores.append(float(average_precision_score(fold.y_val, proba)))
            iterations.append(best)
        trial.set_user_attr("model_params", params)
        trial.set_user_attr("best_iteration", int(np.median(iterations)))
        trial.set_user_attr("fold_scores", scores)
        return float(np.mean(scores))

    return objective


def tune_model(
    model_name: str,
    folds: List[Fold],
    scale_ratio: float,
    n_trials: int = N_TRIALS,
    output_dir: Path = MODELS_DIR,
) -> Dict[str, Any]:
    """Run the Optuna search and log the best result to JSON.

    Args:
        model_name: ``"xgb"`` or ``"lgbm"``.
        folds: Output of ``make_folds`` (training data only).
        scale_ratio: n_negative / n_positive of the training labels.
        n_trials: Number of Optuna trials.
        output_dir: Where ``tuning_<model>.json`` is written.

    Returns:
        The tuning record (also saved to JSON).
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    start = time.perf_counter()
    study.optimize(make_objective(model_name, folds, scale_ratio),
                   n_trials=n_trials, show_progress_bar=True)
    best = study.best_trial
    fold_scores = best.user_attrs["fold_scores"]
    record = {
        "model": model_name,
        "metric": "average_precision, mean over stratified folds",
        "n_trials": n_trials,
        "best_cv_pr_auc": float(best.value),
        "cv_pr_auc_std": float(np.std(fold_scores, ddof=1)),
        "fold_scores": fold_scores,
        "best_iteration": best.user_attrs["best_iteration"],
        "best_params": best.user_attrs["model_params"],
        "random_state": RANDOM_STATE,
        "search_time_s": round(time.perf_counter() - start, 1),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"tuning_{model_name}.json").write_text(
        json.dumps(record, indent=2))
    return record


def load_record(model_name: str, models_dir: Path = MODELS_DIR) -> Dict:
    """Load a saved tuning record (lets you skip re-tuning)."""
    path = Path(models_dir) / f"tuning_{model_name}.json"
    return json.loads(path.read_text())


def cv_oof_predictions(
    model_name: str,
    params: Dict[str, Any],
    folds: List[Fold],
    n_samples: int,
) -> np.ndarray:
    """Out-of-fold probabilities for the tuned parameters.

    Used for threshold selection, so no test data is involved.

    Args:
        model_name: ``"xgb"`` or ``"lgbm"``.
        params: Tuned hyperparameters.
        folds: Output of ``make_folds``.
        n_samples: Length of the training set.

    Returns:
        Array aligned positionally with the training rows.
    """
    oof = np.zeros(n_samples)
    for fold in folds:
        model, _ = fit_with_early_stopping(
            model_name, params, fold.X_fit, fold.y_fit, fold.X_es, fold.y_es)
        oof[fold.val_idx] = model.predict_proba(fold.X_val)[:, 1]
    return oof