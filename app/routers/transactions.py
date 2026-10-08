"""Transaction history with filters and pagination."""

from datetime import date, datetime, time, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database import get_db
from app.errors import ApiError
from app.models import Alert, Transaction, User
from app.schemas import (
    RiskLevel, TransactionDetail, TransactionPage, TransactionSummary)
from app.security import get_current_user

router = APIRouter(tags=["transactions"])


def to_summary(tx: Transaction) -> TransactionSummary:
    """Map a Transaction row to its API summary."""
    return TransactionSummary(
        transaction_id=tx.id, timestamp=tx.timestamp,
        fraud_probability=tx.probability, is_fraud=tx.label,
        risk_level=tx.risk_level, source=tx.source,
        model_version=tx.model_version)


@router.get("/transactions", response_model=TransactionPage)
def list_transactions(
    is_fraud: Optional[bool] = None,
    risk_level: Optional[RiskLevel] = None,
    source: Optional[str] = Query(None, pattern="^(single|batch)$"),
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> TransactionPage:
    """List transactions, newest first. ``date_to`` is inclusive (UTC)."""
    conditions = []
    if is_fraud is not None:
        conditions.append(Transaction.label.is_(is_fraud))
    if risk_level:
        conditions.append(Transaction.risk_level == risk_level)
    if source:
        conditions.append(Transaction.source == source)
    if date_from:
        conditions.append(
            Transaction.timestamp >= datetime.combine(date_from, time.min))
    if date_to:
        conditions.append(Transaction.timestamp < datetime.combine(
            date_to + timedelta(days=1), time.min))
    total = db.scalar(select(func.count()).select_from(Transaction)
                      .where(*conditions)) or 0
    rows = db.scalars(
        select(Transaction).where(*conditions)
        .order_by(Transaction.timestamp.desc())
        .offset((page - 1) * page_size).limit(page_size)).all()
    return TransactionPage(items=[to_summary(t) for t in rows], total=total,
                           page=page, page_size=page_size)


@router.get("/transactions/{transaction_id}", response_model=TransactionDetail)
def get_transaction(
    transaction_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> TransactionDetail:
    """Return one transaction with its inputs and SHAP factors."""
    tx = db.get(Transaction, transaction_id)
    if tx is None:
        raise ApiError(404, "transaction_not_found", "Unknown transaction.")
    alert = db.scalar(select(Alert).where(Alert.transaction_id == tx.id))
    return TransactionDetail(
        **to_summary(tx).model_dump(),
        features=tx.features, top_factors=tx.explanation or [],
        alert_id=alert.id if alert else None,
        alert_status=alert.status if alert else None)