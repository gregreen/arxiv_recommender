"""
Auth helpers: password hashing and JWT creation/verification.

No FastAPI imports here — this module is pure Python so it can be used
by management scripts (e.g. activate_user.py) without starting the app.
"""

from datetime import datetime, timedelta, timezone

import hashlib

import bcrypt
import jwt

from arxiv_lib.config import JWT_ALGORITHM, JWT_EXPIRE_HOURS, SECRET_KEY


def _prehash(plain: str) -> bytes:
    """SHA-256 digest of the password, hex-encoded as bytes.

    Passing a fixed-length digest to bcrypt sidesteps bcrypt's 72-byte input
    limit while preserving the full entropy of arbitrarily long passwords.
    """
    return hashlib.sha256(plain.encode()).hexdigest().encode()


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(_prehash(plain), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(_prehash(plain), hashed.encode())


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
