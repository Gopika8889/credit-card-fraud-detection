"""Login and current-user endpoints."""

from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.errors import ApiError
from app.models import User
from app.schemas import Token, UserOut
from app.security import authenticate, create_access_token, get_current_user

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=Token)
def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
) -> Token:
    """Exchange username/password (form fields) for a JWT."""
    user = authenticate(db, form.username, form.password)
    if user is None:
        raise ApiError(
            401, "invalid_credentials", "Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"})
    minutes = get_settings().access_token_expire_minutes
    return Token(access_token=create_access_token(user), role=user.role,
                 expires_in=minutes * 60)


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)) -> UserOut:
    """Return the authenticated user."""
    return UserOut(username=user.username, role=user.role)