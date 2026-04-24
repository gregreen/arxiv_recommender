"""
Tests for /api/auth/* endpoints:

  POST /api/auth/register
  POST /api/auth/login
  POST /api/auth/logout
  GET  /api/auth/verify-email
  GET  /api/auth/me
  POST /api/auth/reset-password
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from web.auth import hash_password
from web.routers.auth import limiter as _auth_limiter

_EMAIL = "user@example.com"
_PASSWORD = "correctpassword"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _insert_user(
    db,
    email=_EMAIL,
    password=_PASSWORD,
    is_active=1,
    email_verified=1,
    verify_token=None,
    verify_token_expires=None,
    reset_token=None,
    reset_token_expires=None,
):
    db.execute(
        """INSERT INTO users
           (email, password_hash, is_active, email_verified,
            email_verify_token, email_verify_token_expires_at,
            password_reset_token, password_reset_token_expires_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            email,
            hash_password(password),
            int(is_active),
            int(email_verified),
            verify_token,
            verify_token_expires,
            reset_token,
            reset_token_expires,
        ),
    )
    db.commit()


def _future(hours=24):
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()


def _past(hours=1):
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


# ---------------------------------------------------------------------------
# POST /api/auth/register
# ---------------------------------------------------------------------------

class TestRegister:
    def test_register_without_verification(self, raw_client, web_db):
        """When email verification is disabled registration should return 202
        and create a new user row with is_active=0 and no verify token."""
        with patch("web.routers.auth.EMAIL_VERIFICATION_ENABLED", False):
            r = raw_client.post(
                "/api/auth/register", json={"email": _EMAIL, "password": _PASSWORD}
            )
        assert r.status_code == 202
        row = web_db.execute("SELECT * FROM users WHERE email = ?", (_EMAIL,)).fetchone()
        assert row is not None
        assert row["is_active"] == 0
        assert row["email_verify_token"] is None

    def test_register_with_verification_sends_email(self, raw_client, web_db):
        """When email verification is enabled a token should be stored in the DB
        and the verification email helper should be called exactly once."""
        with (
            patch("web.routers.auth.EMAIL_VERIFICATION_ENABLED", True),
            patch("web.routers.auth.send_verification_email") as mock_send,
        ):
            r = raw_client.post(
                "/api/auth/register", json={"email": _EMAIL, "password": _PASSWORD}
            )
        assert r.status_code == 202
        row = web_db.execute("SELECT * FROM users WHERE email = ?", (_EMAIL,)).fetchone()
        assert row["email_verify_token"] is not None
        mock_send.assert_called_once_with(_EMAIL, row["email_verify_token"])

    def test_register_duplicate_email_returns_202(self, raw_client, web_db):
        """Re-registering an existing email must return 202 without revealing
        whether the address is already registered (enumeration prevention)."""
        _insert_user(web_db)
        with patch("web.routers.auth.EMAIL_VERIFICATION_ENABLED", False):
            r = raw_client.post(
                "/api/auth/register", json={"email": _EMAIL, "password": _PASSWORD}
            )
        assert r.status_code == 202
        count = web_db.execute(
            "SELECT COUNT(*) FROM users WHERE email = ?", (_EMAIL,)
        ).fetchone()[0]
        assert count == 1

    def test_register_short_password_422(self, raw_client):
        """Passwords shorter than 8 characters should be rejected with 422."""
        with patch("web.routers.auth.EMAIL_VERIFICATION_ENABLED", False):
            r = raw_client.post(
                "/api/auth/register", json={"email": _EMAIL, "password": "short"}
            )
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# POST /api/auth/login
# ---------------------------------------------------------------------------

class TestLogin:
    def test_login_success_sets_cookie(self, raw_client, web_db):
        """A valid email/password for an active, verified account should return
        200 and set the httponly access_token cookie."""
        _insert_user(web_db)
        with patch("web.routers.auth.EMAIL_VERIFICATION_ENABLED", False):
            r = raw_client.post(
                "/api/auth/login", json={"email": _EMAIL, "password": _PASSWORD}
            )
        assert r.status_code == 200
        assert "access_token" in r.cookies

    def test_login_wrong_password_401(self, raw_client, web_db):
        """An incorrect password should return 401 Unauthorized."""
        _insert_user(web_db)
        with patch("web.routers.auth.EMAIL_VERIFICATION_ENABLED", False):
            r = raw_client.post(
                "/api/auth/login", json={"email": _EMAIL, "password": "wrongpassword"}
            )
        assert r.status_code == 401

    def test_login_inactive_account_403(self, raw_client, web_db):
        """A correct password for an account pending admin approval (is_active=0)
        should return 403 Forbidden."""
        _insert_user(web_db, is_active=0, email_verified=1)
        with patch("web.routers.auth.EMAIL_VERIFICATION_ENABLED", False):
            r = raw_client.post(
                "/api/auth/login", json={"email": _EMAIL, "password": _PASSWORD}
            )
        assert r.status_code == 403


# ---------------------------------------------------------------------------
# GET /api/auth/verify-email
# ---------------------------------------------------------------------------

class TestVerifyEmail:
    def test_verify_email_activates_account(self, raw_client, web_db):
        """Consuming a valid, unexpired token should set email_verified=1,
        is_active=1 on the user row and return 200."""
        token = "validtoken123"
        _insert_user(
            web_db,
            is_active=0,
            email_verified=0,
            verify_token=token,
            verify_token_expires=_future(hours=24),
        )
        r = raw_client.get(f"/api/auth/verify-email?token={token}")
        assert r.status_code == 200
        row = web_db.execute("SELECT * FROM users WHERE email = ?", (_EMAIL,)).fetchone()
        assert row["email_verified"] == 1
        assert row["is_active"] == 1

    def test_verify_email_expired_token_400(self, raw_client, web_db):
        """An expired verification token should return 400 Bad Request."""
        token = "expiredtoken456"
        _insert_user(
            web_db,
            is_active=0,
            email_verified=0,
            verify_token=token,
            verify_token_expires=_past(hours=1),
        )
        r = raw_client.get(f"/api/auth/verify-email?token={token}")
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# POST /api/auth/reset-password
# ---------------------------------------------------------------------------

class TestResetPassword:
    def test_reset_password_success(self, raw_client, web_db):
        """A valid, unexpired reset token should allow setting a new password.
        After success the reset token should be nulled in the DB."""
        reset_token = "resettoken789"
        _insert_user(
            web_db,
            reset_token=reset_token,
            reset_token_expires=_future(hours=1),
        )
        r = raw_client.post(
            "/api/auth/reset-password",
            json={"token": reset_token, "password": "newpassword123"},
        )
        assert r.status_code == 200
        row = web_db.execute("SELECT * FROM users WHERE email = ?", (_EMAIL,)).fetchone()
        assert row["password_reset_token"] is None

    def test_reset_password_invalid_token_400(self, raw_client):
        """An unknown reset token should return 400 Bad Request."""
        r = raw_client.post(
            "/api/auth/reset-password",
            json={"token": "unknowntoken", "password": "newpassword123"},
        )
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# GET /api/auth/me
# ---------------------------------------------------------------------------

class TestMe:
    def test_me_returns_user_profile(self, client):
        """An authenticated request to /me should return the current user's
        id, email, is_admin, and email_verified fields."""
        r = client.get("/api/auth/me")
        assert r.status_code == 200
        data = r.json()
        assert data["user_id"] == 1
        assert data["email"] == "test@example.com"
        assert data["is_admin"] is False
        assert data["email_verified"] is True


# ---------------------------------------------------------------------------
# POST /api/auth/login  — per-account rate limit
# ---------------------------------------------------------------------------

class TestLoginAccountRateLimit:
    """Tests for the per-account 20/hour rate limit on POST /api/auth/login.

    The limiter's default key_func (IP-based, 5/minute) is replaced with a
    rotating-IP function so it never triggers, allowing us to accumulate 21
    attempts against the per-account limit and verify the 21st is rejected.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Reset limiter storage and rotate the per-IP key so the 5/minute
        IP limit never triggers while we accumulate account-based attempts."""
        _auth_limiter._storage.reset()

        login_limits = _auth_limiter._route_limits.get("web.routers.auth.login", [])
        ip_limit = next(
            (lim for lim in login_limits if "1 minute" in str(lim.limit)), None
        )
        original_key_func = ip_limit.key_func if ip_limit else None
        counter = [0]

        def rotating_ip(request):
            counter[0] += 1
            n = counter[0]
            return f"10.0.{(n >> 8) & 0xFF}.{n & 0xFF}"

        if ip_limit:
            ip_limit.key_func = rotating_ip
        yield
        if ip_limit and original_key_func is not None:
            ip_limit.key_func = original_key_func
        _auth_limiter._storage.reset()

    def test_account_rate_limit_triggers_after_20_attempts(self, raw_client, web_db):
        """The 21st login attempt for the same email within an hour should
        return 429 regardless of the source IP."""
        email = "rl_account@example.com"
        _insert_user(web_db, email=email)

        with patch("web.routers.auth.EMAIL_VERIFICATION_ENABLED", False):
            for _ in range(20):
                r = raw_client.post(
                    "/api/auth/login", json={"email": email, "password": "wrong"}
                )
                assert r.status_code == 401, "Requests 1-20 should reach auth logic"

            r = raw_client.post(
                "/api/auth/login", json={"email": email, "password": "wrong"}
            )
        assert r.status_code == 429

    def test_account_rate_limit_is_per_email(self, raw_client, web_db):
        """Exhausting the rate limit for one email address does not affect
        requests for a different email address."""
        email_a = "rl_account_a@example.com"
        email_b = "rl_account_b@example.com"
        _insert_user(web_db, email=email_a)
        _insert_user(web_db, email=email_b)

        with patch("web.routers.auth.EMAIL_VERIFICATION_ENABLED", False):
            for _ in range(20):
                raw_client.post(
                    "/api/auth/login", json={"email": email_a, "password": "wrong"}
                )

            r_a = raw_client.post(
                "/api/auth/login", json={"email": email_a, "password": "wrong"}
            )
            assert r_a.status_code == 429, "email_a should be rate limited"

            r_b = raw_client.post(
                "/api/auth/login", json={"email": email_b, "password": "wrong"}
            )
        assert r_b.status_code == 401, "email_b should not be rate limited"
