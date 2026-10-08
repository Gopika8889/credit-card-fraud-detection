"""Shared pytest fixtures."""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import EXPECTED_COLUMNS, PCA_COLUMNS


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Small synthetic dataset with the real schema (5% fraud)."""
    rng = np.random.default_rng(42)
    n = 1000
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=PCA_COLUMNS)
    df["Time"] = np.sort(rng.integers(0, 172_800, size=n)).astype(float)
    df["Amount"] = rng.exponential(scale=80, size=n).round(2)
    df["Class"] = 0
    df.loc[rng.choice(n, size=50, replace=False), "Class"] = 1
    return df[EXPECTED_COLUMNS]
"""Shared pytest fixtures (Stages 1-4)."""

import os
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest

# --- Must run BEFORE any `app` import (engine reads these at import) ----
_TMP = Path(tempfile.mkdtemp(prefix="fraud_api_tests_"))
os.environ["DATABASE_URL"] = f"sqlite:///{(_TMP / 'test.db').as_posix()}"
os.environ["SECRET_KEY"] = "test-secret-" + "k" * 40
os.environ["MODELS_DIR"] = str(_TMP / "models")
os.environ["BCRYPT_ROUNDS"] = "4"
# -------------------------------------------------------------------------

from src.data_loader import EXPECTED_COLUMNS, PCA_COLUMNS  # noqa: E402
from src.preprocessing import RAW_FEATURE_COLUMNS  # noqa: E402

TEST_THRESHOLD = 0.3
USER = ("alice", "user-password-1", "user")
ANALYST = ("bob", "analyst-password-1", "analyst")


def make_synthetic_df() -> pd.DataFrame:
    """Small synthetic dataset with the real schema (5% fraud)."""
    rng = np.random.default_rng(42)
    n = 1000
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=PCA_COLUMNS)
    df["Time"] = np.sort(rng.integers(0, 172_800, size=n)).astype(float)
    df["Amount"] = rng.exponential(scale=80, size=n).round(2)
    df["Class"] = 0
    df.loc[rng.choice(n, size=50, replace=False), "Class"] = 1
    return df[EXPECTED_COLUMNS]


def _learnable_df() -> pd.DataFrame:
    """Synthetic data where fraud is separable (V1 shifted by +4)."""
    df = make_synthetic_df()
    df.loc[df["Class"] == 1, "V1"] += 4.0
    return df


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Fresh synthetic dataset for Stage 1-3 tests."""
    return make_synthetic_df()


@pytest.fixture(scope="session")
def model_dir() -> Path:
    """Train a tiny model and save artifacts exactly like Stage 3 does."""
    from xgboost import XGBClassifier

    from src.explain import build_explainer
    from src.inference import save_artifacts
    from src.preprocessing import build_preprocessor

    df = _learnable_df()
    X, y = df.drop(columns=["Class"]), df["Class"]
    pre = build_preprocessor().fit(X)
    model = XGBClassifier(n_estimators=40, max_depth=3, random_state=42)
    model.fit(pre.transform(X), y)
    path = Path(os.environ["MODELS_DIR"])
    save_artifacts(
        model=model, preprocessor=pre,
        threshold_info={"chosen": TEST_THRESHOLD},
        explainer=build_explainer(model), model_name="xgb",
        metrics={"pr_auc": 0.9}, params={"max_depth": 3}, models_dir=path)
    return path


@pytest.fixture(scope="session")
def sample_rows() -> Dict[str, Dict[str, float]]:
    """A clear fraud row and a clear legit row (raw columns)."""
    df = _learnable_df()
    X = df[RAW_FEATURE_COLUMNS]
    fraud = X[df["Class"] == 1].nlargest(1, "V1").iloc[0]
    legit = X[df["Class"] == 0].nsmallest(1, "V1").iloc[0]
    return {"fraud": {k: float(v) for k, v in fraud.items()},
            "legit": {k: float(v) for k, v in legit.items()}}


@pytest.fixture
def client(model_dir):
    """TestClient with a clean database and two seeded users."""
    from fastapi.testclient import TestClient

    from app.database import Base, SessionLocal, engine
    from app.main import app
    from app.seed_users import create_or_update_user

    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        create_or_update_user(db, *USER)
        create_or_update_user(db, *ANALYST)
    with TestClient(app) as test_client:
        yield test_client


def _login(client, credentials) -> Dict[str, str]:
    username, password, _ = credentials
    resp = client.post("/auth/login",
                       data={"username": username, "password": password})
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


@pytest.fixture
def user_headers(client) -> Dict[str, str]:
    """Bearer header for the 'user' role."""
    return _login(client, USER)


@pytest.fixture
def analyst_headers(client) -> Dict[str, str]:
    """Bearer header for the 'analyst' role."""
    return _login(client, ANALYST)