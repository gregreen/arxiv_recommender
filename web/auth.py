"""
Auth helpers: password hashing and JWT creation/verification.

No FastAPI imports here — this module is pure Python so it can be used
by management scripts (e.g. activate_user.py) without starting the app.
"""

from datetime import datetime, timedelta, timezone

import jwt
from passlib.context import CryptContext

from arxiv_lib.config import JWT_ALGORITHM, JWT_EXPIRE_HOURS, SECRET_KEY

_crypt = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    return _crypt.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return _crypt.verify(plain, hashed)


def create_access_token(user_id: int) -> str:
    payload = {
        "sub": str(user_id),
        "exp": datetime.now(tz=timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> int:
    """
    Decode and validate the token.  Returns the user_id (int).
    Raises jwt.InvalidTokenError (or subclass) on any problem.
    """
    payload = jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])
    return int(payload["sub"])
