"""Model metadata, PR-curve image and demo samples (for the dashboard)."""

from typing import Any, Dict, Literal

import pandas as pd
from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from app.config import get_settings
from app.errors import ApiError
from app.model_service import ModelService, get_model_service
from app.models import User
from app.security import get_current_user
from src.preprocessing import RAW_FEATURE_COLUMNS

router = APIRouter(tags=["model"])


@router.get("/model/info")
def model_info(
    service: ModelService = Depends(get_model_service),
    user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return metadata.json (metrics, versions, threshold, parameters)."""
    return service.metadata


@router.get("/model/pr-curve")
def pr_curve(user: User = Depends(get_current_user)) -> FileResponse:
    """Serve the saved precision-recall curve image."""
    path = get_settings().models_dir / "pr_curve.png"
    if not path.exists():
        raise ApiError(404, "asset_missing",
                       "Run: python -m app.make_demo_assets")
    return FileResponse(path, media_type="image/png")


@router.get("/demo/sample")
def demo_sample(
    kind: Literal["fraud", "legit"] = "fraud",
    user: User = Depends(get_current_user),
) -> Dict[str, float]:
    """Return a random held-out raw transaction for demos."""
    path = get_settings().models_dir / "demo_samples.csv"
    if not path.exists():
        raise ApiError(404, "asset_missing",
                       "Run: python -m app.make_demo_assets")
    frame = pd.read_csv(path)
    subset = frame[frame["Class"] == (1 if kind == "fraud" else 0)]
    row = subset.sample(1).iloc[0]
    return {c: float(row[c]) for c in RAW_FEATURE_COLUMNS}