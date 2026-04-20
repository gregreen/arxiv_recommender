"""
Tests for /api/groups/* endpoints:

  POST   /api/groups
  GET    /api/groups
  GET    /api/groups/{group_id}
  POST   /api/groups/{group_id}/invites
  GET    /api/groups/{group_id}/invites
  DELETE /api/groups/{group_id}/invites/{invite_id}
  GET    /api/groups/join-info
  POST   /api/groups/join
  DELETE /api/groups/{group_id}/members/{user_id}
  PATCH  /api/groups/{group_id}/members/{user_id}
  DELETE /api/groups/{group_id}
"""

from unittest.mock import patch

import pytest

from fastapi.testclient import TestClient

from web.app import create_app
from web.dependencies import get_current_user, get_db

_USER_ID = 1
_USER2_ID = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _insert_second_user(db):
    db.execute(
        "INSERT INTO users (id, email, password_hash, is_active, email_verified) "
        "VALUES (?, 'user2@example.com', 'x', 1, 1)",
        (_USER2_ID,),
    )
    db.commit()


def _create_group(client, name="Test Group"):
    r = client.post("/api/groups", json={"name": name})
    assert r.status_code == 201
    return r.json()["id"]


def _create_invite(client, group_id, max_uses=1):
    r = client.post(f"/api/groups/{group_id}/invites", json={"max_uses": max_uses})
    assert r.status_code == 201
    return r.json()


# ---------------------------------------------------------------------------
# second_client fixture — same DB, but authenticated as user 2
# ---------------------------------------------------------------------------

@pytest.fixture()
def second_client(web_db):
    """Authenticated TestClient for a second, non-admin user (id=2)."""
    _insert_second_user(web_db)
    app = create_app()
    app.dependency_overrides[get_db] = lambda: web_db
    user2_row = web_db.execute(
        "SELECT id, email, is_active, is_admin, email_verified FROM users WHERE id = ?",
        (_USER2_ID,),
    ).fetchone()
    app.dependency_overrides[get_current_user] = lambda: user2_row
    with patch("web.app.init_app_db"), patch("web.app.SECRET_KEY", "a" * 32):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


# ---------------------------------------------------------------------------
# POST /api/groups  &  GET /api/groups
# ---------------------------------------------------------------------------

class TestCreateGroup:
    def test_create_group(self, client):
        """Creating a group should return 201 with the group name and a members
        list where the creator is listed as the admin."""
        r = client.post("/api/groups", json={"name": "My Group"})
        assert r.status_code == 201
        data = r.json()
        assert data["name"] == "My Group"
        assert len(data["members"]) == 1
        assert data["members"][0]["user_id"] == _USER_ID
        assert data["members"][0]["is_admin"] is True

    def test_create_group_blank_name_422(self, client):
        """A group name of empty string should be rejected with 422."""
        r = client.post("/api/groups", json={"name": ""})
        assert r.status_code == 422


class TestListGroups:
    def test_list_groups_empty(self, client):
        """A user with no group memberships should receive an empty list."""
        r = client.get("/api/groups")
        assert r.status_code == 200
        assert r.json() == []

    def test_list_groups_shows_membership(self, client):
        """After creating a group the user should see it in their membership
        list with the correct name and is_admin=True."""
        _create_group(client, "Listed Group")
        r = client.get("/api/groups")
        assert r.status_code == 200
        groups = r.json()
        assert len(groups) == 1
        assert groups[0]["name"] == "Listed Group"
        assert groups[0]["is_admin"] is True


# ---------------------------------------------------------------------------
# GET /api/groups/{group_id}
# ---------------------------------------------------------------------------

class TestGetGroup:
    def test_get_group_as_member(self, client):
        """A member fetching their group should receive the group info and
        the full members list."""
        group_id = _create_group(client)
        r = client.get(f"/api/groups/{group_id}")
        assert r.status_code == 200
        data = r.json()
        assert data["id"] == group_id
        assert len(data["members"]) == 1

    def test_get_group_non_member_403(self, client, second_client):
        """A user who is not a member of the group should receive 403 Forbidden."""
        group_id = _create_group(client)
        r = second_client.get(f"/api/groups/{group_id}")
        assert r.status_code == 403


# ---------------------------------------------------------------------------
# Invites
# ---------------------------------------------------------------------------

class TestInvites:
    def test_create_invite(self, client):
        """The group admin should be able to create an invite token that is
        returned with remaining_uses and expires_at fields."""
        group_id = _create_group(client)
        invite = _create_invite(client, group_id)
        assert "token" in invite
        assert invite["remaining_uses"] == 1

    def test_list_invites(self, client):
        """After creating an invite it should appear in the invites list for
        the group."""
        group_id = _create_group(client)
        _create_invite(client, group_id)
        r = client.get(f"/api/groups/{group_id}/invites")
        assert r.status_code == 200
        assert len(r.json()) == 1

    def test_revoke_invite(self, client):
        """Revoking an invite should return 204 and remove it from the invites
        list."""
        group_id = _create_group(client)
        invite = _create_invite(client, group_id)
        r = client.delete(f"/api/groups/{group_id}/invites/{invite['id']}")
        assert r.status_code == 204
        remaining = client.get(f"/api/groups/{group_id}/invites").json()
        assert remaining == []

    def test_non_admin_cannot_create_invite_403(self, client, second_client, web_db):
        """A group member without admin rights should receive 403 when trying
        to create an invite."""
        group_id = _create_group(client)
        # second_client fixture already inserted user 2; just add them as a non-admin member
        web_db.execute(
            "INSERT INTO group_members (group_id, user_id, is_admin) VALUES (?, ?, 0)",
            (group_id, _USER2_ID),
        )
        web_db.commit()
        r = second_client.post(f"/api/groups/{group_id}/invites", json={"max_uses": 1})
        assert r.status_code == 403


# ---------------------------------------------------------------------------
# Join
# ---------------------------------------------------------------------------

class TestJoinGroup:
    def test_join_via_valid_token(self, client, second_client):
        """Consuming a valid invite token should add the user to the group and
        return the updated group info with the new member included."""
        group_id = _create_group(client)
        invite = _create_invite(client, group_id)
        r = second_client.post(f"/api/groups/join?token={invite['token']}")
        assert r.status_code == 200
        data = r.json()
        member_ids = [m["user_id"] for m in data["members"]]
        assert _USER2_ID in member_ids

    def test_join_invalid_token_404(self, second_client):
        """An unknown or already-used invite token should return 404."""
        r = second_client.post("/api/groups/join?token=doesnotexist")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Member management
# ---------------------------------------------------------------------------

class TestMemberManagement:
    def test_remove_member_as_admin(self, client, second_client, web_db):
        """The group admin should be able to remove another member, which
        returns 204 and deletes the group_members row."""
        group_id = _create_group(client)
        invite = _create_invite(client, group_id)
        second_client.post(f"/api/groups/join?token={invite['token']}")
        r = client.delete(f"/api/groups/{group_id}/members/{_USER2_ID}")
        assert r.status_code == 204
        row = web_db.execute(
            "SELECT 1 FROM group_members WHERE group_id = ? AND user_id = ?",
            (group_id, _USER2_ID),
        ).fetchone()
        assert row is None

    def test_remove_last_admin_409(self, client, web_db):
        """The sole admin of a group should not be able to remove themselves
        because there would be no admin left; the endpoint must return 409."""
        # Insert group directly to avoid exhausting the rate-limited POST /api/groups
        web_db.execute("INSERT INTO groups (name) VALUES ('Test Group')")
        group_id = web_db.execute("SELECT last_insert_rowid()").fetchone()[0]
        web_db.execute(
            "INSERT INTO group_members (group_id, user_id, is_admin) VALUES (?, ?, 1)",
            (group_id, _USER_ID),
        )
        web_db.commit()
        r = client.delete(f"/api/groups/{group_id}/members/{_USER_ID}")
        assert r.status_code == 409
