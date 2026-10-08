"""POST /predict: score, explain and log a single transaction."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.model_service import ModelService, get_model_service
from app.models import Alert, Transaction, User, new_id
from app.schemas import PredictionOut, TransactionIn
from app.security import get_current_user

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictionOut)
def predict(
    transaction: TransactionIn,  # type: ignore[valid-type]
    db: Session = Depends(get_db),
    service: ModelService = Depends(get_model_service),
    user: User = Depends(get_current_user),
) -> PredictionOut:
    """Score one transaction, log it, and open an alert if flagged."""
    features = transaction.model_dump()
    result = service.predict_one(features)
    record = Transaction(
        id=new_id(),
        features=features,
        probability=result["fraud_probability"],
        label=result["is_fraud"],
        risk_level=result["risk_level"],
        source="single",
        model_version=service.model_version,
        created_by=user.username,
        explanation=result["top_factors"],
    )
    db.add(record)
    alert = None
    if record.label:
        alert = Alert(transaction_id=record.id)
        db.add(alert)
    db.commit()
    return PredictionOut(
        transaction_id=record.id,
        fraud_probability=record.probability,
        is_fraud=record.label,
        risk_level=record.risk_level,
        threshold=service.threshold,
        model_version=service.model_version,
        top_factors=result["top_factors"],
        alert_id=alert.id if alert else None,
    )