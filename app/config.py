"""Application settings, read from environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIN_SECRET_LENGTH = 32


@dataclass(frozen=True)
class Settings:
    """Runtime configuration (all values come from the environment)."""

    models_dir: Path
    database_url: str
    secret_key: str
    access_token_expire_minutes: int
    bcrypt_rounds: int
    max_upload_mb: int
    max_batch_rows: int
    max_explain_rows: int
    high_risk_fraction: float

    @property
    def max_upload_bytes(self) -> int:
        """Upload size cap in bytes."""
        return self.max_upload_mb * 1024 * 1024


def get_settings() -> Settings:
    """Read settings from the environment (cheap, so tests can override)."""
    env = os.environ.get
    default_db = f"sqlite:///{(PROJECT_ROOT / 'data' / 'app.db').as_posix()}"
    return Settings(
        models_dir=Path(env("MODELS_DIR", str(PROJECT_ROOT / "models"))),
        database_url=env("DATABASE_URL", default_db),
        secret_key=env("SECRET_KEY", ""),
        access_token_expire_minutes=int(env("ACCESS_TOKEN_EXPIRE_MINUTES", "60")),
        bcrypt_rounds=int(env("BCRYPT_ROUNDS", "12")),
        max_upload_mb=int(env("MAX_UPLOAD_MB", "5")),
        max_batch_rows=int(env("MAX_BATCH_ROWS", "10000")),
        max_explain_rows=int(env("MAX_EXPLAIN_ROWS", "200")),
        high_risk_fraction=float(env("HIGH_RISK_FRACTION", "0.5")),
    )


def validate_settings(settings: Settings) -> None:
    """Fail fast at startup if the configuration is unsafe.

    Raises:
        RuntimeError: If SECRET_KEY is missing or too short.
    """
    if len(settings.secret_key) < MIN_SECRET_LENGTH:
        raise RuntimeError(
            f"SECRET_KEY must be set and at least {MIN_SECRET_LENGTH} "
            "characters. Generate one with: python -c "
            "\"import secrets; print(secrets.token_urlsafe(48))\""
        )