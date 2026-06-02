"""
Tests for DELETE /api/users/me  (account self-deletion).

Happy path:
  - Returns 204 with cookie cleared
  - All associated user data is removed
  - On re-fetch, user no longer exists

Error paths:
  - Wrong password → 400
  - Admin user → 403
  - Sole group admin → 409
"""

import pytest
from unittest.mock import patch

from fastapi.testclient import TestClient

from web.app import create_app
from web.auth import hash_password
from web.dependencies import get_current_user, get_db
from web.limiter import limiter as _limiter

_USER_ID = 1
_PASSWORD = "correct-horse-battery"
_PASSWORD_HASH = hash_password(_PASSWORD)


def _delete(client, password):
    return client.request(
        "DELETE",
        "/api/users/me",
        json={"password": password},
    )


def _make_client(db, override_auth: bool = True):
    """Create a TestClient with rate limiting disabled."""
    app = create_app()
    app.dependency_overrides[get_db] = lambda: db
    if override_auth:
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
    """Disable the rate limiter for every test in this module."""
    _limiter.enabled = False
    yield
    _limiter.enabled = True

@pytest.fixture()
def deletion_db(web_db):
    """web_db fixture with the test user's password_hash set to a known value."""
    web_db.execute(
        "UPDATE users SET password_hash = ? WHERE id = ?",
        (_PASSWORD_HASH, _USER_ID),
    )
    web_db.commit()
    yield web_db


@pytest.fixture()
def del_client(deletion_db):
    """Authenticated TestClient for account-deletion tests, rate limiting disabled."""
    app = _make_client(deletion_db)
    with patch("web.app.init_app_db"), patch("web.app.SECRET_KEY", "a" * 32):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestDeleteAccountHappy:
    def test_returns_204(self, del_client):
        r = _delete(del_client, _PASSWORD)
        assert r.status_code == 204

    def test_user_row_removed(self, del_client, deletion_db):
        _delete(del_client, _PASSWORD)
        row = deletion_db.execute(
            "SELECT id FROM users WHERE id = ?", (_USER_ID,)
        ).fetchone()
        assert row is None

    def test_user_papers_removed(self, del_client, deletion_db):
        deletion_db.execute(
            "INSERT OR IGNORE INTO papers (arxiv_id, title, abstract, authors, published_date, categories) "
            "VALUES ('2309.06676', 'T', 'A', '[]', '2023-01-01', '[]')"
        )
        deletion_db.execute(
            "INSERT INTO user_papers (user_id, arxiv_id, liked) VALUES (?, '2309.06676', 1)",
            (_USER_ID,),
        )
        deletion_db.commit()

        _delete(del_client, _PASSWORD)

        count = deletion_db.execute(
            "SELECT COUNT(*) FROM user_papers WHERE user_id = ?", (_USER_ID,)
        ).fetchone()[0]
        assert count == 0

    def test_group_membership_removed(self, del_client, deletion_db):
        """group_members rows should be gone after deletion."""
        deletion_db.execute("INSERT INTO groups (name) VALUES ('TestGroup')")
        group_id = deletion_db.execute("SELECT last_insert_rowid()").fetchone()[0]
        # Another admin so user is NOT sole admin
        deletion_db.execute(
            "INSERT INTO users (id, email, password_hash, is_active, is_admin, email_verified) "
            "VALUES (99, 'other@example.com', 'x', 1, 0, 1)"
        )
        deletion_db.execute(
            "INSERT INTO group_members (group_id, user_id, is_admin) VALUES (?, 99, 1)",
            (group_id,),
        )
        deletion_db.execute(
            "INSERT INTO group_members (group_id, user_id, is_admin) VALUES (?, ?, 0)",
            (group_id, _USER_ID),
        )
        deletion_db.commit()

        _delete(del_client, _PASSWORD)

        count = deletion_db.execute(
            "SELECT COUNT(*) FROM group_members WHERE user_id = ?", (_USER_ID,)
        ).fetchone()[0]
        assert count == 0


# ---------------------------------------------------------------------------
# Wrong password
# ---------------------------------------------------------------------------

class TestDeleteAccountWrongPassword:
    def test_wrong_password_returns_400(self, del_client):
        r = _delete(del_client, "wrong-password")
        assert r.status_code == 400

    def test_wrong_password_user_not_deleted(self, del_client, deletion_db):
        _delete(del_client, "wrong-password")
        row = deletion_db.execute(
            "SELECT id FROM users WHERE id = ?", (_USER_ID,)
        ).fetchone()
        assert row is not None


# ---------------------------------------------------------------------------
# Admin user blocked
# ---------------------------------------------------------------------------

class TestDeleteAccountAdmin:
    def test_admin_returns_403(self, deletion_db):
        """Admin users must not be able to delete themselves via the API."""
        deletion_db.execute(
            "UPDATE users SET is_admin = 1 WHERE id = ?", (_USER_ID,)
        )
        deletion_db.commit()

        app = _make_client(deletion_db)
        with patch("web.app.init_app_db"), patch("web.app.SECRET_KEY", "a" * 32):
            with TestClient(app, raise_server_exceptions=False) as c:
                r = _delete(c, _PASSWORD)

        assert r.status_code == 403


# ---------------------------------------------------------------------------
# Sole group admin blocked
# ---------------------------------------------------------------------------

class TestDeleteAccountSoleGroupAdmin:
    def test_sole_group_admin_returns_409(self, del_client, deletion_db):
        """Deleting an account that is the sole admin of a group should return 409."""
        deletion_db.execute("INSERT INTO groups (name) VALUES ('MyGroup')")
        group_id = deletion_db.execute("SELECT last_insert_rowid()").fetchone()[0]
        deletion_db.execute(
            "INSERT INTO group_members (group_id, user_id, is_admin) VALUES (?, ?, 1)",
            (group_id, _USER_ID),
        )
        deletion_db.commit()

        r = _delete(del_client, _PASSWORD)
        assert r.status_code == 409
        assert "MyGroup" in r.json()["detail"]

    def test_not_sole_group_admin_succeeds(self, del_client, deletion_db):
        """Deleting is allowed when there is another admin in the group."""
        deletion_db.execute("INSERT INTO groups (name) VALUES ('SharedGroup')")
        group_id = deletion_db.execute("SELECT last_insert_rowid()").fetchone()[0]
        deletion_db.execute(
            "INSERT INTO users (id, email, password_hash, is_active, is_admin, email_verified) "
            "VALUES (98, 'admin2@example.com', 'x', 1, 0, 1)"
        )
        deletion_db.execute(
            "INSERT INTO group_members (group_id, user_id, is_admin) VALUES (?, ?, 1)",
            (group_id, _USER_ID),
        )
        deletion_db.execute(
            "INSERT INTO group_members (group_id, user_id, is_admin) VALUES (?, 98, 1)",
            (group_id,),
        )
        deletion_db.commit()

        r = _delete(del_client, _PASSWORD)
        assert r.status_code == 204


