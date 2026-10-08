"""Analyst alert review (analyst role required)."""

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session, contains_eager

from app.database import get_db
from app.errors import ApiError
from app.models import Alert, Transaction, User, utcnow
from app.schemas import AlertOut, AlertPage, AlertUpdate
from app.security import require_analyst

router = APIRouter(prefix="/alerts", tags=["alerts"])


def alert_to_out(alert: Alert) -> AlertOut:
    """Join an Alert with its Transaction for the API."""
    tx = alert.transaction
    return AlertOut(
        id=alert.id, transaction_id=tx.id, status=alert.status,
        created_at=alert.created_at, reviewed_by=alert.reviewed_by,
        reviewed_at=alert.reviewed_at, notes=alert.notes,
        fraud_probability=tx.probability, risk_level=tx.risk_level,
        amount=tx.features.get("Amount"), transaction_time=tx.timestamp,
        top_factors=tx.explanation or [])


@router.get("", response_model=AlertPage)
def list_alerts(
    status: Literal["open", "confirmed_fraud", "false_positive", "all"] = "open",
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    analyst: User = Depends(require_analyst),
) -> AlertPage:
    """List alerts, highest fraud probability first."""
    conditions = [] if status == "all" else [Alert.status == status]
    total = db.scalar(select(func.count()).select_from(Alert)
                      .where(*conditions)) or 0
    rows = db.scalars(
        select(Alert).join(Alert.transaction)
        .options(contains_eager(Alert.transaction))
        .where(*conditions)
        .order_by(Transaction.probability.desc(), Alert.id)
        .offset((page - 1) * page_size).limit(page_size)).all()
    return AlertPage(items=[alert_to_out(a) for a in rows], total=total,
                     page=page, page_size=page_size)


@router.patch("/{alert_id}", response_model=AlertOut)
def review_alert(
    alert_id: int,
    body: AlertUpdate,
    db: Session = Depends(get_db),
    analyst: User = Depends(require_analyst),
) -> AlertOut:
    """Mark an alert as confirmed fraud or a false positive."""
    alert = db.get(Alert, alert_id)
    if alert is None:
        raise ApiError(404, "alert_not_found", "Unknown alert.")
    alert.status = body.status
    alert.notes = body.notes
    alert.reviewed_by = analyst.username
    alert.reviewed_at = utcnow()
    db.commit()
    return alert_to_out(alert)