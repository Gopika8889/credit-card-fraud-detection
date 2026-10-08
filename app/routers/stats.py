"""GET /stats: fraud rate, risk mix, daily counts, analyst precision."""

from datetime import timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from app.database import get_db
from app.model_service import RISK_LEVELS
from app.models import Alert, Transaction, User, utcnow
from app.schemas import AlertCounts, DailyStat, StatsOut
from app.security import get_current_user

router = APIRouter(tags=["stats"])


@router.get("/stats", response_model=StatsOut)
def stats(
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> StatsOut:
    """Aggregate statistics over the last ``days`` days (UTC)."""
    in_window = Transaction.timestamp >= utcnow() - timedelta(days=days)
    flagged_col = Transaction.label.is_(True)

    total = db.scalar(select(func.count()).select_from(Transaction)
                      .where(in_window)) or 0
    flagged = db.scalar(select(func.count()).select_from(Transaction)
                        .where(in_window, flagged_col)) or 0

    risk_rows = db.execute(
        select(Transaction.risk_level, func.count())
        .where(in_window).group_by(Transaction.risk_level)).all()
    risk_counts = {level: 0 for level in RISK_LEVELS}
    risk_counts.update({level: int(n) for level, n in risk_rows})

    day = func.date(Transaction.timestamp).label("day")
    daily_rows = db.execute(
        select(day, func.count(),
               func.sum(case((flagged_col, 1), else_=0)))
        .where(in_window).group_by(day).order_by(day)).all()
    daily = [DailyStat(date=str(d), total=int(t), flagged=int(f or 0))
             for d, t, f in daily_rows]

    status_rows = db.execute(
        select(Alert.status, func.count()).group_by(Alert.status)).all()
    by_status = {s: int(n) for s, n in status_rows}
    counts = AlertCounts(
        open=by_status.get("open", 0),
        confirmed_fraud=by_status.get("confirmed_fraud", 0),
        false_positive=by_status.get("false_positive", 0))
    reviewed = counts.confirmed_fraud + counts.false_positive

    return StatsOut(
        window_days=days, total_transactions=total,
        flagged_transactions=flagged,
        fraud_rate=flagged / total if total else 0.0,
        risk_counts=risk_counts, daily=daily, alerts=counts,
        reviewed_alerts=reviewed,
        analyst_precision=(counts.confirmed_fraud / reviewed
                           if reviewed else None))