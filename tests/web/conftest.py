"""
Shared fixtures for web router integration tests.

Strategy
--------
- FastAPI dependency overrides replace ``get_db`` (yields the test connection)
  and ``get_current_user`` (returns a real sqlite3.Row for user id=1).
- ``web.app.init_app_db`` is patched to a no-op so the lifespan startup does
  not touch the real app.db on disk.
- ``raw_client`` omits the ``get_current_user`` override so the real auth path
  runs — used for 401 tests where no cookie is sent.
- All path functions (SUMMARY_CACHE_DIR, EMBEDDING_CACHE_DB, …) read
  ``config.DATA_DIR`` at call time, which the inherited ``data_dir`` fixture
  has already redirected to a fresh tmp directory.
"""

import pytest
from unittest.mock import patch

from fastapi.testclient import TestClient  # noqa: E402

import arxiv_lib.ingest as ingest  # noqa: E402
from web.app import create_app  # noqa: E402
from web.dependencies import get_admin_user, get_current_user, get_db  # noqa: E402

_USER_ID = 1
_USER_EMAIL = "test@example.com"


@pytest.fixture()
def web_db(app_db_con):
    """app_db_con with embedding tables and a test user pre-inserted."""
    ingest._init_embedding_db()
    app_db_con.execute(
        "INSERT INTO users (id, email, password_hash, is_active, is_admin, email_verified) "
        "VALUES (?, ?, 'x', 1, 0, 1)",
        (_USER_ID, _USER_EMAIL),
    )
    app_db_con.commit()
    yield app_db_con


def _build_app(web_db, override_auth: bool):
    """Create a fresh FastAPI app with dependency overrides applied."""
    app = create_app()
    app.dependency_overrides[get_db] = lambda: web_db
    if override_auth:
        user_row = web_db.execute(
            "SELECT id, email, is_active, is_admin, email_verified FROM users WHERE id = ?",
            (_USER_ID,),
        ).fetchone()
        app.dependency_overrides[get_current_user] = lambda: user_row
    return app


@pytest.fixture()
def client(web_db):
    """Authenticated TestClient — get_db and get_current_user both overridden."""
    app = _build_app(web_db, override_auth=True)
    with patch("web.app.init_app_db"), patch("web.app.SECRET_KEY", "a" * 32):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


@pytest.fixture()
def raw_client(web_db):
    """Unauthenticated TestClient — only get_db overridden, real auth path runs."""
    app = _build_app(web_db, override_auth=False)
    with patch("web.app.init_app_db"), patch("web.app.SECRET_KEY", "a" * 32):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


@pytest.fixture()
def admin_client(web_db):
    """Authenticated TestClient where the current user is an admin.

    Overrides both ``get_current_user`` and ``get_admin_user`` so that every
    admin-protected route sees an active admin row without needing a real JWT.
    """
    web_db.execute("UPDATE users SET is_admin = 1 WHERE id = ?", (_USER_ID,))
    web_db.commit()
    app = create_app()
    app.dependency_overrides[get_db] = lambda: web_db
    admin_row = web_db.execute(
        "SELECT id, email, is_active, is_admin, email_verified FROM users WHERE id = ?",
        (_USER_ID,),
    ).fetchone()
    app.dependency_overrides[get_current_user] = lambda: admin_row
    app.dependency_overrides[get_admin_user] = lambda: admin_row
    with patch("web.app.init_app_db"), patch("web.app.SECRET_KEY", "a" * 32):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c
