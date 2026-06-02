"""
Tests for POST /api/users/me/export  (GDPR data export).

Happy path:
  - Returns 200 with Content-Disposition attachment header
  - JSON body contains all expected top-level keys
  - last_export_at is updated in the database

Error paths:
  - Wrong password → 400
  - Second request within cooldown window → 429
"""

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from web.app import create_app
from web.auth import hash_password
from web.dependencies import get_current_user, get_db
from web.limiter import limiter as _limiter

_USER_ID = 1
_PASSWORD = "export-test-password"
_PASSWORD_HASH = hash_password(_PASSWORD)


def _export(client, password):
    return client.post("/api/users/me/export", json={"password": password})


def _make_client(db):
    app = create_app()
    app.dependency_overrides[get_db] = lambda: db
    user_row = db.execute(
        "SELECT id, email, is_active, is_admin, email_verified, tutorial_shown "
        "FROM users WHERE id = ?",
        (_USER_ID,),
    ).fetchone()
    app.dependency_overrides[get_current_user] = lambda: user_row
    return app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _disable_limiter():
    _limiter.enabled = False
    yield
    _limiter.enabled = True


@pytest.fixture()
def export_db(web_db):
    """web_db with a known password hash for the test user."""
    web_db.execute(
        "UPDATE users SET password_hash = ? WHERE id = ?",
        (_PASSWORD_HASH, _USER_ID),
    )
    web_db.commit()
    yield web_db


@pytest.fixture()
def export_client(export_db):
    app = _make_client(export_db)
    with patch("web.app.init_app_db"), patch("web.app.SECRET_KEY", "a" * 32):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_export_returns_json_attachment(export_client, export_db):
    resp = _export(export_client, _PASSWORD)
    assert resp.status_code == 200
    assert "attachment" in resp.headers.get("content-disposition", "")
    data = resp.json()
    for key in ("exported_at", "account", "library", "categories", "search_terms", "groups"):
        assert key in data, f"missing key: {key}"


def test_export_updates_last_export_at(export_client, export_db):
    _export(export_client, _PASSWORD)
    row = export_db.execute(
        "SELECT last_export_at FROM users WHERE id = ?", (_USER_ID,)
    ).fetchone()
    assert row["last_export_at"] is not None


def test_export_wrong_password_returns_400(export_client):
    resp = _export(export_client, "wrong-password")
    assert resp.status_code == 400


def test_export_cooldown_returns_429(export_client, export_db):
    # Set last_export_at to 1 day ago (within the 7-day window)
    recent = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    export_db.execute(
        "UPDATE users SET last_export_at = ? WHERE id = ?", (recent, _USER_ID)
    )
    export_db.commit()
    resp = _export(export_client, _PASSWORD)
    assert resp.status_code == 429


def test_export_after_cooldown_succeeds(export_client, export_db):
    # Set last_export_at to 8 days ago (outside the 7-day window)
    old = (datetime.now(timezone.utc) - timedelta(days=8)).strftime("%Y-%m-%dT%H:%M:%SZ")
    export_db.execute(
        "UPDATE users SET last_export_at = ? WHERE id = ?", (old, _USER_ID)
    )
    export_db.commit()
    resp = _export(export_client, _PASSWORD)
    assert resp.status_code == 200
