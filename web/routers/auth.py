"""
Auth endpoints: register, login, logout, email verification.

POST /api/auth/register              — create account
POST /api/auth/login                 — verify credentials and set JWT cookie
POST /api/auth/logout                — clear JWT cookie
GET  /api/auth/verify-email          — consume a verification token
POST /api/auth/resend-verification   — resend the verification email (with exponential cooldown)
GET  /api/auth/me                    — return the current user's profile
"""

import logging
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, EmailStr
from slowapi import Limiter
from slowapi.util import get_remote_address

from arxiv_lib.config import EMAIL_VERIFICATION_ENABLED
from web.auth import create_access_token, hash_password, verify_password
from web.dependencies import get_current_user, get_db
from web.email import send_verification_email

log = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])
limiter = Limiter(key_func=get_remote_address)

# Maximum cooldown between resend attempts (capped at 24 hours)
_MAX_RESEND_COOLDOWN_MINUTES = 1440


def _generate_verify_token() -> str:
    return secrets.token_urlsafe(32)


def _token_expiry() -> str:
    """Return an ISO 8601 UTC string 24 hours from now."""
    return (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()


def _next_resend_at(resend_count: int) -> str:
    """Exponential cooldown: 2^resend_count minutes, capped at 24 hours."""
    minutes = min(2 ** resend_count, _MAX_RESEND_COOLDOWN_MINUTES)
    return (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat()


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class ResendVerificationRequest(BaseModel):
    email: EmailStr


@router.post("/register", status_code=status.HTTP_202_ACCEPTED)
def register(body: RegisterRequest, db: sqlite3.Connection = Depends(get_db)):
    existing = db.execute(
        "SELECT id FROM users WHERE email = ?", (body.email,)
    ).fetchone()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with that email already exists.",
        )
    if len(body.password) < 8:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Password must be at least 8 characters.",
        )

    if EMAIL_VERIFICATION_ENABLED:
        token = _generate_verify_token()
        expiry = _token_expiry()
        next_resend = _next_resend_at(0)
        db.execute(
            """INSERT INTO users
               (email, password_hash, is_active, email_verified,
                email_verify_token, email_verify_token_expires_at,
                email_verify_resend_count, email_verify_next_resend_at)
               VALUES (?, ?, 0, 0, ?, ?, 0, ?)""",
            (body.email, hash_password(body.password), token, expiry, next_resend),
        )
        db.commit()
        try:
            send_verification_email(body.email, token)
        except RuntimeError as exc:
            log.error("register: email send failed for %s: %s", body.email, exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Account created but verification email could not be sent. Contact the administrator.",
            )
        return {"message": "Registration received. Please check your email to verify your account."}
    else:
        db.execute(
            "INSERT INTO users (email, password_hash, is_active) VALUES (?, ?, 0)",
            (body.email, hash_password(body.password)),
        )
        db.commit()
        return {"message": "Registration received. Your account is pending review by the administrator."}


@router.post("/login")
@limiter.limit("5/minute")
def login(request: Request, body: LoginRequest, response: Response, db: sqlite3.Connection = Depends(get_db)):
    row = db.execute(
        "SELECT id, password_hash, is_active, email_verified FROM users WHERE email = ?",
        (body.email,),
    ).fetchone()

    if row is None or not verify_password(body.password, row["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )

    # email_verified = 0 means verification was requested but not yet completed.
    # (Tokens are retained after verification, so token presence is not a reliable check.)
    if not row["email_verified"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="verify_email_pending",
        )

    if not row["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account pending review. Contact the administrator.",
        )

    token = create_access_token(row["id"])
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax",
        secure=True,    # requires HTTPS in production; browsers exempt localhost
        max_age=86400,  # 24 hours
    )
    return {"user_id": row["id"], "email": body.email}


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie("access_token")
    return {"message": "Logged out."}


@router.get("/verify-email")
def verify_email(token: str, db: sqlite3.Connection = Depends(get_db)):
    row = db.execute(
        """SELECT id, email_verified, email_verify_token_expires_at
           FROM users WHERE email_verify_token = ?""",
        (token,),
    ).fetchone()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="invalid_token",
        )

    if row["email_verified"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="already_verified",
        )

    expires_at = datetime.fromisoformat(row["email_verify_token_expires_at"])
    if datetime.now(timezone.utc) > expires_at:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Verification link has expired. Please request a new one.",
        )

    # Retain token for 7 days so repeat clicks return "already_verified" instead
    # of "invalid_token". cleanup_tokens.py will NULL it after that window.
    retention_expiry = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
    db.execute(
        """UPDATE users
           SET email_verified = 1,
               is_active = 1,
               email_verify_token_expires_at = ?,
               email_verify_resend_count = 0,
               email_verify_next_resend_at = NULL
           WHERE id = ?""",
        (retention_expiry, row["id"]),
    )
    db.commit()
    return {"message": "Email verified. You can now sign in."}


@router.post("/resend-verification", status_code=status.HTTP_200_OK)
def resend_verification(body: ResendVerificationRequest, db: sqlite3.Connection = Depends(get_db)):
    # Always return 200 with the same message to prevent email enumeration.
    _generic_ok = {"message": "If that address is registered and awaiting verification, a new email has been sent."}

    if not EMAIL_VERIFICATION_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Email verification is not enabled.",
        )

    row = db.execute(
        """SELECT id, email_verified, email_verify_token,
                  email_verify_resend_count, email_verify_next_resend_at
           FROM users WHERE email = ?""",
        (body.email,),
    ).fetchone()

    if row is None or row["email_verified"] or row["email_verify_token"] is None:
        return _generic_ok

    # Enforce exponential cooldown
    if row["email_verify_next_resend_at"]:
        next_allowed = datetime.fromisoformat(row["email_verify_next_resend_at"])
        if datetime.now(timezone.utc) < next_allowed:
            wait_seconds = int((next_allowed - datetime.now(timezone.utc)).total_seconds())
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Please wait {wait_seconds} seconds before requesting another verification email.",
            )

    new_count = row["email_verify_resend_count"] + 1
    new_token = _generate_verify_token()
    new_expiry = _token_expiry()
    new_next_resend = _next_resend_at(new_count)

    db.execute(
        """UPDATE users
           SET email_verify_token = ?,
               email_verify_token_expires_at = ?,
               email_verify_resend_count = ?,
               email_verify_next_resend_at = ?
           WHERE id = ?""",
        (new_token, new_expiry, new_count, new_next_resend, row["id"]),
    )
    db.commit()

    try:
        send_verification_email(body.email, new_token)
    except RuntimeError as exc:
        log.error("resend_verification: email send failed for %s: %s", body.email, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not send verification email. Please try again later.",
        )

    return _generic_ok


@router.get("/me")
def me(user=Depends(get_current_user)):
    return {
        "user_id": user["id"],
        "email": user["email"],
        "is_admin": bool(user["is_admin"]),
        "email_verified": bool(user["email_verified"]),
    }

