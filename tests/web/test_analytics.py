"""
Tests for the analytics system:
  POST /api/analytics/event — visit counter and dedup table
  GET  /api/admin/analytics  — DAU/WAU/MAU summary, daily breakdown, page breakdown
"""

import pytest
from unittest.mock import patch

from fastapi.testclient import TestClient

from web.app import create_app
from web.dependencies import get_admin_user, get_current_user, get_db
from web.limiter import limiter as _limiter

_USER_ID = 1
_USER_ID_2 = 2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _disable_limiter():
    _limiter.enabled = False
    yield
    _limiter.enabled = True


@pytest.fixture()
def analytics_db(web_db):
    """web_db with a second user for multi-user tests."""
    web_db.execute(
        "INSERT INTO users (id, email, password_hash, is_active, is_admin, email_verified) "
        "VALUES (?, 'user2@example.com', 'x', 1, 0, 1)",
        (_USER_ID_2,),
    )
    web_db.commit()
    yield web_db


def _make_event_client(db, user_id: int):
    app = create_app()
    app.dependency_overrides[get_db] = lambda: db
    user_row = db.execute(
        "SELECT id, email, is_active, is_admin, email_verified, tutorial_shown FROM users WHERE id = ?",
        (user_id,),
    ).fetchone()
    app.dependency_overrides[get_current_user] = lambda: user_row
    return app


def _make_admin_client(db):
    db.execute("UPDATE users SET is_admin = 1 WHERE id = ?", (_USER_ID,))
    db.commit()
    app = create_app()
    app.dependency_overrides[get_db] = lambda: db
    admin_row = db.execute(
        "SELECT id, email, is_active, is_admin, email_verified, tutorial_shown FROM users WHERE id = ?",
        (_USER_ID,),
    ).fetchone()
    app.dependency_overrides[get_current_user] = lambda: admin_row
    app.dependency_overrides[get_admin_user] = lambda: admin_row
    return app


@pytest.fixture()
def event_client(analytics_db):
    app = _make_event_client(analytics_db, _USER_ID)
    with patch("web.app.init_app_db"), patch("web.app.SECRET_KEY", "a" * 32):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


@pytest.fixture()
def event_client_2(analytics_db):
    app = _make_event_client(analytics_db, _USER_ID_2)
    with patch("web.app.init_app_db"), patch("web.app.SECRET_KEY", "a" * 32):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


@pytest.fixture()
def analytics_admin_client(analytics_db):
    app = _make_admin_client(analytics_db)
    with patch("web.app.init_app_db"), patch("web.app.SECRET_KEY", "a" * 32):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


# ---------------------------------------------------------------------------
# Event recording tests
# ---------------------------------------------------------------------------

def test_record_event_returns_204(event_client):
    resp = event_client.post("/api/analytics/event", json={"page": "/recommendations"})
    assert resp.status_code == 204


def test_visit_count_increments(event_client, analytics_db):
    event_client.post("/api/analytics/event", json={"page": "/recommendations"})
    event_client.post("/api/analytics/event", json={"page": "/recommendations"})
    row = analytics_db.execute(
        "SELECT visits FROM page_stats_daily WHERE page = '/recommendations'"
    ).fetchone()
    assert row is not None
    assert row["visits"] == 2


def test_same_user_deduped_in_daily_users(event_client, analytics_db):
    """Same user visiting the same page twice should only appear once in the dedup table."""
    event_client.post("/api/analytics/event", json={"page": "/recommendations"})
    event_client.post("/api/analytics/event", json={"page": "/recommendations"})
    count = analytics_db.execute(
        "SELECT COUNT(*) FROM page_stats_daily_users WHERE page = '/recommendations'"
    ).fetchone()[0]
    assert count == 1


def test_second_user_added_to_dedup_table(event_client, event_client_2, analytics_db):
    event_client.post("/api/analytics/event", json={"page": "/recommendations"})
    event_client_2.post("/api/analytics/event", json={"page": "/recommendations"})
    count = analytics_db.execute(
        "SELECT COUNT(*) FROM page_stats_daily_users WHERE page = '/recommendations'"
    ).fetchone()[0]
    assert count == 2


def test_last_active_at_set_on_event(event_client, analytics_db):
    event_client.post("/api/analytics/event", json={"page": "/recommendations"})
    row = analytics_db.execute(
        "SELECT last_active_at FROM users WHERE id = ?", (_USER_ID,)
    ).fetchone()
    assert row["last_active_at"] is not None


# ---------------------------------------------------------------------------
# Admin analytics endpoint tests
# ---------------------------------------------------------------------------

def test_admin_analytics_response_shape(analytics_admin_client):
    resp = analytics_admin_client.get("/api/admin/analytics?days=30")
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert "daily" in data
    assert "pages" in data
    summary = data["summary"]
    assert "dau" in summary and "wau" in summary and "mau" in summary


def test_admin_analytics_daily_rows_have_users_field(analytics_admin_client, event_client):
    event_client.post("/api/analytics/event", json={"page": "/recommendations"})
    resp = analytics_admin_client.get("/api/admin/analytics?days=30")
    daily = resp.json()["daily"]
    assert len(daily) > 0
    assert "users" in daily[0]
    assert "visits" in daily[0]


def test_admin_analytics_page_rows_have_users_field(analytics_admin_client, event_client):
    event_client.post("/api/analytics/event", json={"page": "/recommendations"})
    resp = analytics_admin_client.get("/api/admin/analytics?days=30")
    pages = resp.json()["pages"]
    assert len(pages) > 0
    assert "users" in pages[0]
    assert "visits" in pages[0]


def test_admin_analytics_dau_counts_active_user(analytics_admin_client, analytics_db):
    analytics_db.execute(
        "UPDATE users SET last_active_at = date('now') WHERE id = ?", (_USER_ID,)
    )
    analytics_db.commit()
    resp = analytics_admin_client.get("/api/admin/analytics?days=30")
    assert resp.json()["summary"]["dau"] >= 1


def test_admin_analytics_dau_excludes_inactive_user(analytics_admin_client, analytics_db):
    analytics_db.execute(
        "UPDATE users SET last_active_at = date('now', '-5 days') WHERE id = ?", (_USER_ID,)
    )
    analytics_db.commit()
    resp = analytics_admin_client.get("/api/admin/analytics?days=30")
    assert resp.json()["summary"]["dau"] == 0


def test_admin_analytics_page_distinct_users(
    analytics_admin_client, event_client, event_client_2
):
    event_client.post("/api/analytics/event", json={"page": "/recommendations"})
    event_client_2.post("/api/analytics/event", json={"page": "/recommendations"})
    resp = analytics_admin_client.get("/api/admin/analytics?days=30")
    pages = resp.json()["pages"]
    rec = next(p for p in pages if p["page"] == "/recommendations")
    assert rec["users"] == 2
