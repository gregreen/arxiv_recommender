"""
Auth endpoints: register, login, logout.

POST /api/auth/register  — create account (inactive by default; admin must activate)
POST /api/auth/login     — verify credentials and set JWT cookie
POST /api/auth/logout    — clear JWT cookie
"""

import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, EmailStr

from web.auth import create_access_token, hash_password, verify_password
from web.dependencies import get_current_user, get_db

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


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
    db.execute(
        "INSERT INTO users (email, password_hash, is_active) VALUES (?, ?, 0)",
        (body.email, hash_password(body.password)),
    )
    db.commit()
    return {"message": "Registration received. Your account is pending review by the administrator."}


@router.post("/login")
def login(body: LoginRequest, response: Response, db: sqlite3.Connection = Depends(get_db)):
    row = db.execute(
        "SELECT id, password_hash, is_active FROM users WHERE email = ?", (body.email,)
    ).fetchone()

    if row is None or not verify_password(body.password, row["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
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
        secure=False,   # set to True in production (HTTPS)
        max_age=86400,  # 24 hours
    )
    return {"user_id": row["id"], "email": body.email}


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie("access_token")
    return {"message": "Logged out."}


@router.get("/me")
def me(user=Depends(get_current_user)):
    return {"user_id": user["id"], "email": user["email"]}
