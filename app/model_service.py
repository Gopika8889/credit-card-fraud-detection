"""Model service: loads artifacts once and scores raw transactions."""

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import Request

from src.explain import explain_transaction, shap_values_for
from src.inference import library_versions, load_artifacts
from src.preprocessing import RAW_FEATURE_COLUMNS

logger = logging.getLogger(__name__)

RISK_LEVELS = ("Low", "Medium", "High")
VERSION_SENSITIVE_LIBS = ("scikit-learn", "xgboost", "lightgbm", "shap")


class ModelService:
    """Holds the fitted model, preprocessor, threshold and SHAP explainer."""

    def __init__(
        self, artifacts: Dict[str, Any], high_risk_fraction: float = 0.5
    ) -> None:
        """Wrap artifacts returned by ``src.inference.load_artifacts``.

        Args:
            artifacts: Loaded model, preprocessor, threshold, explainer,
                metadata.
            high_risk_fraction: Where "High" starts, as a fraction of the
                distance between the threshold and 1.0.
        """
        self.model = artifacts["model"]
        self.preprocessor = artifacts["preprocessor"]
        self.explainer = artifacts["explainer"]
        self.metadata: Dict[str, Any] = artifacts["metadata"]
        self.threshold = float(artifacts["threshold"]["chosen"])
        self.high_cutoff = self.threshold + high_risk_fraction * (
            1.0 - self.threshold
        )
        self._lock = threading.Lock()
        self._check_metadata()

    @classmethod
    def load(
        cls, models_dir: Path, high_risk_fraction: float = 0.5
    ) -> "ModelService":
        """Load all artifacts from ``models_dir`` (call once at startup)."""
        return cls(load_artifacts(Path(models_dir)), high_risk_fraction)

    def _check_metadata(self) -> None:
        """Fail on a schema mismatch; warn on library version drift."""
        saved_columns = self.metadata.get("input_columns")
        if saved_columns != RAW_FEATURE_COLUMNS:
            raise RuntimeError(
                "metadata.json input_columns do not match "
                "src.preprocessing.RAW_FEATURE_COLUMNS; retrain or fix."
            )
        saved = self.metadata.get("library_versions", {})
        current = library_versions()
        for lib in VERSION_SENSITIVE_LIBS:
            if saved.get(lib) != current.get(lib):
                logger.warning(
                    "Library version drift for %s: trained with %s, "
                    "serving with %s", lib, saved.get(lib), current.get(lib))

    @property
    def model_version(self) -> str:
        """Identifier such as ``xgb-2026-10-07`` (name + training date)."""
        date = str(self.metadata.get("training_date_utc", "unknown"))[:10]
        return f"{self.metadata.get('model_name', 'model')}-{date}"

    def risk_levels(self, proba: np.ndarray) -> np.ndarray:
        """Vectorized risk level: Low < threshold <= Medium < High cut-off."""
        return np.select(
            [proba >= self.high_cutoff, proba >= self.threshold],
            ["High", "Medium"],
            default="Low",
        )

    def risk_level(self, probability: float) -> str:
        """Risk level for a single probability."""
        return str(self.risk_levels(np.array([probability]))[0])

    def _score(self, raw: pd.DataFrame):
        """Preprocess RAW rows (fixed column order) and predict.

        Returns:
            ``(features, probabilities)`` where ``features`` is the
            preprocessed DataFrame the model actually sees.
        """
        missing = [c for c in RAW_FEATURE_COLUMNS if c not in raw.columns]
        if missing:
            raise ValueError(f"Missing input columns: {missing}")
        ordered = raw[RAW_FEATURE_COLUMNS].astype(float)
        with self._lock:
            features = self.preprocessor.transform(ordered)
            proba = self.model.predict_proba(features)[:, 1]
        return features, proba

    @staticmethod
    def _top_factors(
        row: pd.Series, shap_row: np.ndarray, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Top-k features by |SHAP|; same format as ``explain_transaction``."""
        order = np.argsort(-np.abs(shap_row))[:top_k]
        return [
            {
                "feature": str(row.index[i]),
                "value": float(row.iloc[i]),
                "shap_value": float(shap_row[i]),
                "effect": ("increases fraud risk" if shap_row[i] > 0
                           else "decreases fraud risk"),
            }
            for i in order
        ]

    def predict_one(
        self, transaction: Dict[str, float], top_k: int = 5
    ) -> Dict[str, Any]:
        """Score and explain one raw transaction.

        Args:
            transaction: Mapping with Time, V1-V28, Amount.
            top_k: Number of SHAP factors to return.

        Returns:
            Dict with fraud_probability, is_fraud, risk_level, top_factors.
        """
        features, proba = self._score(pd.DataFrame([transaction]))
        p = float(proba[0])
        with self._lock:
            factors = explain_transaction(
                self.model, features.iloc[[0]], top_k, self.explainer)
        return {
            "fraud_probability": p,
            "is_fraud": bool(p >= self.threshold),
            "risk_level": self.risk_level(p),
            "top_factors": factors,
        }

    def predict_batch(
        self,
        raw: pd.DataFrame,
        explain: bool = True,
        max_explain: Optional[int] = None,
    ) -> pd.DataFrame:
        """Score many raw transactions (vectorized).

        SHAP is computed only for flagged rows, highest probability first,
        capped at ``max_explain`` rows.

        Args:
            raw: DataFrame containing the raw feature columns.
            explain: Whether to compute SHAP factors at all.
            max_explain: Cap on explained rows (None = all flagged).

        Returns:
            DataFrame aligned with ``raw`` (same index) and columns
            fraud_probability, is_fraud, risk_level, top_factors
            (list of dicts, or None when not explained).
        """
        features, proba = self._score(raw)
        flagged = proba >= self.threshold
        factors: List[Optional[List[Dict[str, Any]]]] = [None] * len(raw)
        if explain and flagged.any():
            idx = np.flatnonzero(flagged)
            idx = idx[np.argsort(-proba[idx])][:max_explain]
            with self._lock:
                values = shap_values_for(self.explainer, features.iloc[idx])
            for pos, shap_row in zip(idx, values):
                factors[pos] = self._top_factors(features.iloc[pos], shap_row)
        return pd.DataFrame(
            {
                "fraud_probability": proba,
                "is_fraud": flagged,
                "risk_level": self.risk_levels(proba),
                "top_factors": pd.Series(factors, index=raw.index,
                                         dtype=object),
            },
            index=raw.index,
        )


def get_model_service(request: Request) -> ModelService:
    """FastAPI dependency returning the service loaded at startup."""
    return request.app.state.model_service