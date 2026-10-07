"""Save/load model artifacts and score raw transactions (used by the API)."""

import json
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

from src.config import MODELS_DIR, RANDOM_STATE
from src.explain import build_explainer, explain_transaction
from src.preprocessing import RAW_FEATURE_COLUMNS

LIBRARIES = ("numpy", "pandas", "scikit-learn", "imbalanced-learn",
             "xgboost", "lightgbm", "optuna", "shap")


def library_versions() -> Dict[str, str]:
    """Installed versions of the key libraries."""
    versions = {}
    for name in LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def save_artifacts(
    model,
    preprocessor,
    threshold_info: Dict[str, float],
    explainer,
    model_name: str,
    metrics: Dict[str, float],
    params: Dict[str, Any],
    models_dir: Path = MODELS_DIR,
) -> None:
    """Persist everything the API needs to ``models_dir``.

    Args:
        model: Fitted classifier.
        preprocessor: Fitted preprocessing pipeline (takes RAW columns).
        threshold_info: Dict containing at least ``"chosen"``.
        explainer: SHAP explainer (rebuilt from the model if unpicklable).
        model_name: ``"xgb"`` or ``"lgbm"``.
        metrics: Test metrics at the chosen threshold.
        params: Final hyperparameters.
        models_dir: Output directory.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / "final_model.joblib")
    joblib.dump(preprocessor, models_dir / "preprocessor.joblib")
    joblib.dump(threshold_info, models_dir / "threshold.joblib")
    joblib.dump(explainer, models_dir / "shap_explainer.joblib")
    meta = {
        "model_name": model_name,
        "training_date_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "random_state": RANDOM_STATE,
        "library_versions": library_versions(),
        "threshold": threshold_info,
        "metrics_test_at_chosen_threshold": metrics,
        "hyperparameters": params,
        "input_columns": RAW_FEATURE_COLUMNS,
    }
    (models_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, default=float))


def load_artifacts(models_dir: Path = MODELS_DIR) -> Dict[str, Any]:
    """Load model, preprocessor, threshold, explainer and metadata.

    ``src`` must be importable (the preprocessor references
    ``src.preprocessing.add_features``). If the saved explainer cannot be
    loaded, it is rebuilt from the model.
    """
    models_dir = Path(models_dir)
    model = joblib.load(models_dir / "final_model.joblib")
    try:
        explainer = joblib.load(models_dir / "shap_explainer.joblib")
    except Exception:  # version mismatch etc.: rebuild is cheap
        explainer = build_explainer(model)
    return {
        "model": model,
        "preprocessor": joblib.load(models_dir / "preprocessor.joblib"),
        "threshold": joblib.load(models_dir / "threshold.joblib"),
        "explainer": explainer,
        "metadata": json.loads((models_dir / "metadata.json").read_text()),
    }


def score_transactions(
    artifacts: Dict[str, Any], raw: pd.DataFrame
) -> pd.DataFrame:
    """Score RAW transactions (Time, V1-V28, Amount).

    Args:
        artifacts: Output of ``load_artifacts``.
        raw: DataFrame with the raw feature columns.

    Returns:
        DataFrame with ``fraud_probability`` and ``is_fraud``.

    Raises:
        ValueError: If raw feature columns are missing.
    """
    missing = [c for c in RAW_FEATURE_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing input columns: {missing}")
    features = artifacts["preprocessor"].transform(raw[RAW_FEATURE_COLUMNS])
    proba = artifacts["model"].predict_proba(features)[:, 1]
    chosen = artifacts["threshold"]["chosen"]
    return pd.DataFrame(
        {"fraud_probability": proba, "is_fraud": proba >= chosen},
        index=raw.index)


def score_and_explain(
    artifacts: Dict[str, Any], raw_row: pd.DataFrame, top_k: int = 5
) -> Dict[str, Any]:
    """Score one raw transaction (one-row DataFrame) and explain it."""
    result = score_transactions(artifacts, raw_row).iloc[0]
    features = artifacts["preprocessor"].transform(
        raw_row[RAW_FEATURE_COLUMNS])
    return {
        "fraud_probability": float(result["fraud_probability"]),
        "is_fraud": bool(result["is_fraud"]),
        "threshold": float(artifacts["threshold"]["chosen"]),
        "top_factors": explain_transaction(
            artifacts["model"], features, top_k, artifacts["explainer"]),
    }