"""Password hashing, JWT tokens and authorization dependencies."""

from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Optional

import jwt
from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.errors import ApiError
from app.models import User

ALGORITHM = "HS256"
ROLES = ("user", "analyst")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
_BEARER = {"WWW-Authenticate": "Bearer"}


@lru_cache(maxsize=4)
def _context(rounds: int) -> CryptContext:
    return CryptContext(schemes=["bcrypt"], bcrypt__rounds=rounds)


def hash_password(password: str) -> str:
    """Hash a password with bcrypt."""
    return _context(get_settings().bcrypt_rounds).hash(password)


def verify_password(password: str, hashed: str) -> bool:
    """Check a password against its bcrypt hash."""
    return _context(get_settings().bcrypt_rounds).verify(password, hashed)


def authenticate(db: Session, username: str, password: str) -> Optional[User]:
    """Return the user if the credentials are valid, else None."""
    user = db.scalar(select(User).where(User.username == username))
    if user is None or not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(user: User) -> str:
    """Create a signed JWT (secret from the environment)."""
    settings = get_settings()
    expires = datetime.now(timezone.utc) + timedelta(
        minutes=settings.access_token_expire_minutes)
    return jwt.encode(
        {"sub": user.username, "role": user.role, "exp": expires},
        settings.secret_key, algorithm=ALGORITHM)


def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
) -> User:
    """Resolve the bearer token to a user (role is read from the DB)."""
    try:
        payload = jwt.decode(
            token, get_settings().secret_key, algorithms=[ALGORITHM])
        username = payload.get("sub")
    except jwt.PyJWTError as exc:
        raise ApiError(401, "invalid_token", "Invalid or expired token.",
                       headers=_BEARER) from exc
    user = db.scalar(select(User).where(User.username == username))
    if user is None:
        raise ApiError(401, "invalid_token", "Unknown user.", headers=_BEARER)
    return user


def require_analyst(user: User = Depends(get_current_user)) -> User:
    """Allow only users with the analyst role."""
    if user.role != "analyst":
        raise ApiError(403, "forbidden", "Analyst role required.")
    return user