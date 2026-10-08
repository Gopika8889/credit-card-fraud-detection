"""Create or update a user.

Usage (run from the project root):
    python -m app.seed_users --username analyst1 --role analyst
    python -m app.seed_users --username demo_user --role user

The password is read from the SEED_PASSWORD environment variable or, if
unset, prompted for (never passed on the command line).
"""

import argparse
import getpass
import os

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import SessionLocal, init_db
from app.models import User
from app.security import ROLES, hash_password


def create_or_update_user(
    db: Session, username: str, password: str, role: str
) -> User:
    """Insert a user, or update the password and role if it exists.

    Raises:
        ValueError: For an unknown role or an unsuitable password length
            (bcrypt only uses the first 72 bytes).
    """
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    if not 8 <= len(password.encode()) <= 72:
        raise ValueError("password must be 8-72 bytes long")
    user = db.scalar(select(User).where(User.username == username))
    if user is None:
        user = User(username=username, role=role,
                    hashed_password=hash_password(password))
        db.add(user)
    else:
        user.role = role
        user.hashed_password = hash_password(password)
    db.commit()
    return user


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Create/update a user.")
    parser.add_argument("--username", required=True)
    parser.add_argument("--role", choices=ROLES, default="analyst")
    args = parser.parse_args()
    password = os.environ.get("SEED_PASSWORD") or getpass.getpass("Password: ")
    init_db()
    with SessionLocal() as db:
        user = create_or_update_user(db, args.username, password, args.role)
    print(f"Saved user '{user.username}' with role '{user.role}'.")


if __name__ == "__main__":
    main()