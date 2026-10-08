"""FastAPI application: startup, routers, /health."""

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from app.config import get_settings, validate_settings
from app.database import init_db
from app.errors import register_handlers
from app.model_service import ModelService, get_model_service
from app.routers import (
    alerts, auth, batch, model_info, predict, stats, transactions)
from app.schemas import HealthOut

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Validate config, create tables and load the model ONCE."""
    settings = get_settings()
    validate_settings(settings)
    init_db()
    app.state.model_service = ModelService.load(
        settings.models_dir, settings.high_risk_fraction)
    service = app.state.model_service
    logger.info("Loaded model %s (threshold %.4f)",
                service.model_version, service.threshold)
    yield


app = FastAPI(title="Fraud Detection API", version="1.0.0", lifespan=lifespan)
register_handlers(app)

for module in (auth, predict, batch, transactions, alerts, stats, model_info):
    app.include_router(module.router)


@app.get("/health", response_model=HealthOut, tags=["system"])
def health(service: ModelService = Depends(get_model_service)) -> HealthOut:
    """Public liveness check with model version and threshold."""
    return HealthOut(
        status="ok",
        model_name=str(service.metadata.get("model_name", "unknown")),
        model_version=service.model_version,
        threshold=service.threshold,
        high_risk_cutoff=service.high_cutoff)