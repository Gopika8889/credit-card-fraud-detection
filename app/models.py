"""Database tables."""

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def utcnow() -> datetime:
    """Current UTC time as a naive datetime."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def new_id() -> str:
    """Random 32-char hex id (generated in Python so it is known early)."""
    return uuid.uuid4().hex


class User(Base):
    """An API user with a role ("user" or "analyst")."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(128))
    role: Mapped[str] = mapped_column(String(16), default="user")


class Transaction(Base):
    """Every scored transaction (single or batch)."""

    __tablename__ = "transactions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, index=True)
    features: Mapped[dict[str, Any]] = mapped_column(JSON)
    probability: Mapped[float] = mapped_column(Float)
    label: Mapped[bool] = mapped_column(Boolean, index=True)
    risk_level: Mapped[str] = mapped_column(String(10), index=True)
    source: Mapped[str] = mapped_column(String(10))
    batch_id: Mapped[Optional[str]] = mapped_column(
        String(32), index=True, nullable=True)
    model_version: Mapped[str] = mapped_column(String(64))
    created_by: Mapped[Optional[str]] = mapped_column(String(64))
    explanation: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(
        JSON, nullable=True)


class Alert(Base):
    """Review workflow record; one per flagged transaction."""

    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    transaction_id: Mapped[str] = mapped_column(
        ForeignKey("transactions.id"), unique=True, index=True)
    status: Mapped[str] = mapped_column(
        String(20), default="open", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    reviewed_by: Mapped[Optional[str]] = mapped_column(String(64))
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    notes: Mapped[Optional[str]] = mapped_column(Text)

    transaction: Mapped["Transaction"] = relationship()