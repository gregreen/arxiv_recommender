"""
FastAPI dependency functions: database connection and current-user resolution.
"""

import sqlite3

import jwt
from fastapi import Cookie, Depends, HTTPException, status

from arxiv_lib.appdb import get_connection
from web.auth import decode_access_token


def get_db():
    """Yield an open app.db connection; close it when the request finishes."""
    con = get_connection()
    try:
        yield con
    finally:
        con.close()


def get_current_user(
    access_token: str | None = Cookie(default=None),
    db: sqlite3.Connection = Depends(get_db),
) -> sqlite3.Row:
    """
    Resolve the JWT cookie to an active user row.

    Raises 401 if the cookie is missing, invalid, or expired.
    Raises 403 if the account exists but is not yet active.
    """
    if not access_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated.",
        )
    try:
        user_id = decode_access_token(access_token)
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
        )

    row = db.execute(
        "SELECT id, email, is_active FROM users WHERE id = ?", (user_id,)
    ).fetchone()

    if row is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found.")
    if not row["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account pending review. Contact the administrator.",
        )
    return row
