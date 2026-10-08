"""POST /predict/batch (CSV upload) and the flagged-rows download."""

import io
from typing import List, Tuple

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.errors import ApiError
from app.model_service import RISK_LEVELS, ModelService, get_model_service
from app.models import Alert, Transaction, User, new_id, utcnow
from app.schemas import (
    AMOUNT_MAX, TIME_MAX, V_LIMIT, BatchSummary, RowError)
from app.security import get_current_user
from src.data_loader import PCA_COLUMNS
from src.preprocessing import RAW_FEATURE_COLUMNS

router = APIRouter(prefix="/predict", tags=["batch"])
MAX_REPORTED_ERRORS = 50


def split_valid_rows(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[RowError], int]:
    """Separate valid rows from rejected ones (vectorized checks).

    Args:
        df: Uploaded frame containing the raw feature columns.

    Returns:
        ``(valid_numeric_rows, first_errors, total_rejected)``. ``row`` in
        each error is the line number in the CSV (header = line 1).
    """
    num = df[RAW_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    finite = pd.Series(
        np.isfinite(num.to_numpy(dtype=float)).all(axis=1), index=num.index)
    checks = {
        "missing, non-numeric or infinite value": ~finite,
        f"Amount outside 0..{AMOUNT_MAX:,.0f}":
            ~num["Amount"].between(0, AMOUNT_MAX),
        f"Time outside 0..{TIME_MAX:,.0f}":
            ~num["Time"].between(0, TIME_MAX),
        f"V1-V28 outside +/-{V_LIMIT:,.0f}":
            (num[PCA_COLUMNS].abs() > V_LIMIT).any(axis=1),
    }
    bad = pd.concat(list(checks.values()), axis=1).any(axis=1)
    errors = []
    for idx in bad[bad].index[:MAX_REPORTED_ERRORS]:
        reason = next(msg for msg, mask in checks.items() if mask[idx])
        errors.append(RowError(row=int(idx) + 2, reason=reason))
    return num[~bad], errors, int(bad.sum())


@router.post("/batch", response_model=BatchSummary)
def predict_batch(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    service: ModelService = Depends(get_model_service),
    user: User = Depends(get_current_user),
) -> BatchSummary:
    """Score a CSV of raw transactions and log every scored row."""
    settings = get_settings()
    if not (file.filename or "").lower().endswith(".csv"):
        raise ApiError(415, "unsupported_file_type", "Upload a .csv file.")
    content = file.file.read(settings.max_upload_bytes + 1)
    if len(content) > settings.max_upload_bytes:
        raise ApiError(413, "file_too_large",
                       f"File exceeds {settings.max_upload_mb} MB.")
    try:
        df = pd.read_csv(io.BytesIO(content))
    except (pd.errors.EmptyDataError, pd.errors.ParserError,
            UnicodeDecodeError) as exc:
        raise ApiError(400, "invalid_csv",
                       f"Could not parse the CSV: {exc}") from exc

    df.columns = df.columns.astype(str).str.strip()
    missing = [c for c in RAW_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ApiError(
            422, "missing_columns",
            f"CSV is missing {len(missing)} required column(s): {missing}",
            details={"missing": missing, "required": RAW_FEATURE_COLUMNS})
    if len(df) > settings.max_batch_rows:
        raise ApiError(413, "too_many_rows",
                       f"At most {settings.max_batch_rows} rows per upload.")

    valid, errors, n_rejected = split_valid_rows(df)
    if valid.empty:
        raise ApiError(422, "no_valid_rows", "No valid rows to score.",
                       details=[e.model_dump() for e in errors])

    result = service.predict_batch(
        valid, explain=True, max_explain=settings.max_explain_rows)

    batch_id, now = new_id(), utcnow()
    transactions, alerts = [], []
    for features, p, flagged, risk, factors in zip(
        valid.to_dict("records"),
        result["fraud_probability"].to_numpy(),
        result["is_fraud"].to_numpy(),
        result["risk_level"].to_numpy(),
        result["top_factors"],
    ):
        tx = Transaction(
            id=new_id(), timestamp=now, features=features,
            probability=float(p), label=bool(flagged), risk_level=str(risk),
            source="batch", batch_id=batch_id,
            model_version=service.model_version, created_by=user.username,
            explanation=factors if isinstance(factors, list) else None)
        transactions.append(tx)
        if tx.label:
            alerts.append(Alert(transaction_id=tx.id))
    db.add_all(transactions)
    db.add_all(alerts)
    db.commit()

    counts = result["risk_level"].value_counts()
    return BatchSummary(
        batch_id=batch_id,
        n_rows=len(df),
        n_scored=len(valid),
        n_rejected=n_rejected,
        n_flagged=len(alerts),
        flagged_rate=len(alerts) / len(valid),
        risk_counts={lvl: int(counts.get(lvl, 0)) for lvl in RISK_LEVELS},
        explained_rows=int(sum(isinstance(f, list)
                               for f in result["top_factors"])),
        ignored_columns=[c for c in df.columns
                         if c not in RAW_FEATURE_COLUMNS],
        rejected_rows=errors,
        download_url=f"/predict/batch/{batch_id}/flagged.csv",
    )


@router.get("/batch/{batch_id}/flagged.csv")
def download_flagged(
    batch_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Response:
    """Download the flagged rows of a batch as CSV (highest risk first)."""
    exists = db.scalar(select(func.count()).select_from(Transaction)
                       .where(Transaction.batch_id == batch_id))
    if not exists:
        raise ApiError(404, "batch_not_found", "Unknown batch id.")
    rows = db.scalars(
        select(Transaction)
        .where(Transaction.batch_id == batch_id, Transaction.label.is_(True))
        .order_by(Transaction.probability.desc())).all()
    frame = pd.DataFrame(
        [{
            "transaction_id": t.id,
            "fraud_probability": round(t.probability, 6),
            "risk_level": t.risk_level,
            "Time": t.features["Time"],
            "Amount": t.features["Amount"],
            "top_factors": "; ".join(
                f"{f['feature']}({f['shap_value']:+.2f})"
                for f in (t.explanation or [])),
        } for t in rows],
        columns=["transaction_id", "fraud_probability", "risk_level",
                 "Time", "Amount", "top_factors"])
    return Response(
        content=frame.to_csv(index=False), media_type="text/csv",
        headers={"Content-Disposition":
                 f'attachment; filename="flagged_{batch_id}.csv"'})