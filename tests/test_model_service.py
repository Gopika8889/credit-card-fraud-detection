"""Model service tests: serving must match training exactly."""

import numpy as np
import pandas as pd
import pytest

from app.config import PROJECT_ROOT
from app.model_service import ModelService
from src.preprocessing import RAW_FEATURE_COLUMNS
from tests.conftest import TEST_THRESHOLD


@pytest.fixture
def service(model_dir) -> ModelService:
    return ModelService.load(model_dir)


def _model_features(model):
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        names = getattr(model, "feature_name_", None)
    return None if names is None else list(names)


def test_matches_manual_pipeline_and_ignores_column_order(
        service, sample_rows):
    raw = pd.DataFrame([sample_rows["fraud"], sample_rows["legit"]])
    shuffled = raw[list(reversed(raw.columns))]
    out = service.predict_batch(shuffled, explain=False)
    expected = service.model.predict_proba(
        service.preprocessor.transform(raw[RAW_FEATURE_COLUMNS]))[:, 1]
    assert np.allclose(out["fraud_probability"], expected)


def test_preprocessed_columns_match_model_features(service, sample_rows):
    raw = pd.DataFrame([sample_rows["legit"]])[RAW_FEATURE_COLUMNS]
    features = service.preprocessor.transform(raw)
    assert list(features.columns) == _model_features(service.model)


def test_risk_levels_relative_to_threshold(service):
    high = TEST_THRESHOLD + 0.5 * (1 - TEST_THRESHOLD)  # 0.65
    assert service.risk_level(TEST_THRESHOLD - 0.01) == "Low"
    assert service.risk_level(TEST_THRESHOLD) == "Medium"
    assert service.risk_level(high - 0.01) == "Medium"
    assert service.risk_level(high) == "High"


def test_batch_explanations_match_single(service, sample_rows):
    single = service.predict_one(sample_rows["fraud"])
    batch = service.predict_batch(pd.DataFrame([sample_rows["fraud"]]))
    factors = batch.iloc[0]["top_factors"]
    assert [f["feature"] for f in factors] == [
        f["feature"] for f in single["top_factors"]]
    assert np.allclose([f["shap_value"] for f in factors],
                       [f["shap_value"] for f in single["top_factors"]])


def test_known_samples_are_stable(service, sample_rows):
    first = service.predict_one(sample_rows["fraud"])
    second = service.predict_one(sample_rows["fraud"])
    assert first["fraud_probability"] == second["fraud_probability"]
    assert first["is_fraud"] is True
    assert service.predict_one(sample_rows["legit"])["is_fraud"] is False


def test_metadata_input_columns_match_training(service):
    assert service.metadata["input_columns"] == RAW_FEATURE_COLUMNS


@pytest.mark.skipif(
    not (PROJECT_ROOT / "models" / "final_model.joblib").exists(),
    reason="Stage 3 artifacts not present")
def test_real_artifacts_are_consistent():
    real = ModelService.load(PROJECT_ROOT / "models")
    row = pd.DataFrame([{c: 0.0 for c in RAW_FEATURE_COLUMNS}])
    features = real.preprocessor.transform(row)
    expected = _model_features(real.model)
    if expected is not None:
        assert list(features.columns) == expected
    assert 0.0 < real.threshold < 1.0