"""Pydantic request/response models."""

from datetime import date, datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, create_model

from src.data_loader import PCA_COLUMNS

# Sanity bounds (the real dataset: Amount <= ~26k, |V| <= ~120).
V_LIMIT = 1_000.0
AMOUNT_MAX = 1_000_000.0
TIME_MAX = 1_000_000_000.0

RiskLevel = Literal["Low", "Medium", "High"]
AlertStatus = Literal["open", "confirmed_fraud", "false_positive"]


class _TransactionBase(BaseModel):
    """Time and Amount, with strict ranges."""

    model_config = ConfigDict(extra="forbid")

    Time: float = Field(
        ge=0, le=TIME_MAX, allow_inf_nan=False,
        description="Seconds since the first transaction in the dataset.")
    Amount: float = Field(
        ge=0, le=AMOUNT_MAX, allow_inf_nan=False,
        description="Transaction amount (>= 0).")


TransactionIn = create_model(  # one required float per PCA column
    "TransactionIn",
    __base__=_TransactionBase,
    **{c: (float, Field(..., ge=-V_LIMIT, le=V_LIMIT, allow_inf_nan=False))
       for c in PCA_COLUMNS},
)


class ShapFactor(BaseModel):
    """One SHAP contribution (log-odds)."""

    feature: str
    value: float
    shap_value: float
    effect: str


class PredictionOut(BaseModel):
    """Response of POST /predict."""

    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    risk_level: RiskLevel
    threshold: float
    model_version: str
    top_factors: List[ShapFactor]
    alert_id: Optional[int] = None


class RowError(BaseModel):
    """A rejected CSV row (``row`` is the line number in the file)."""

    row: int
    reason: str


class BatchSummary(BaseModel):
    """Response of POST /predict/batch."""

    batch_id: str
    n_rows: int
    n_scored: int
    n_rejected: int
    n_flagged: int
    flagged_rate: float
    risk_counts: Dict[str, int]
    explained_rows: int
    ignored_columns: List[str]
    rejected_rows: List[RowError]
    download_url: str


class TransactionSummary(BaseModel):
    """One row of the transaction history."""

    transaction_id: str
    timestamp: datetime
    fraud_probability: float
    is_fraud: bool
    risk_level: RiskLevel
    source: str
    model_version: str


class TransactionDetail(TransactionSummary):
    """Full record including inputs and explanation."""

    features: Dict[str, float]
    top_factors: List[ShapFactor]
    alert_id: Optional[int] = None
    alert_status: Optional[AlertStatus] = None


class TransactionPage(BaseModel):
    """Paginated transactions."""

    items: List[TransactionSummary]
    total: int
    page: int
    page_size: int


class AlertUpdate(BaseModel):
    """Body of PATCH /alerts/{id}."""

    status: Literal["confirmed_fraud", "false_positive"]
    notes: Optional[str] = Field(default=None, max_length=500)


class AlertOut(BaseModel):
    """An alert joined with its transaction."""

    id: int
    transaction_id: str
    status: AlertStatus
    created_at: datetime
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    notes: Optional[str] = None
    fraud_probability: float
    risk_level: RiskLevel
    amount: Optional[float] = None
    transaction_time: datetime
    top_factors: List[ShapFactor]


class AlertPage(BaseModel):
    """Paginated alerts."""

    items: List[AlertOut]
    total: int
    page: int
    page_size: int


class DailyStat(BaseModel):
    """Counts for one UTC day."""

    date: str
    total: int
    flagged: int


class AlertCounts(BaseModel):
    """Alerts by review status."""

    open: int
    confirmed_fraud: int
    false_positive: int


class StatsOut(BaseModel):
    """Response of GET /stats."""

    window_days: int
    total_transactions: int
    flagged_transactions: int
    fraud_rate: float
    risk_counts: Dict[str, int]
    daily: List[DailyStat]
    alerts: AlertCounts
    reviewed_alerts: int
    analyst_precision: Optional[float] = None


class HealthOut(BaseModel):
    """Response of GET /health."""

    status: str
    model_name: str
    model_version: str
    threshold: float
    high_risk_cutoff: float


class Token(BaseModel):
    """Response of POST /auth/login."""

    access_token: str
    token_type: str = "bearer"
    role: str
    expires_in: int


class UserOut(BaseModel):
    """Current user."""

    username: str
    role: str