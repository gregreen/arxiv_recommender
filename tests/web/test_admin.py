"""
Tests for /api/admin/* endpoints:

  GET    /api/admin/users
  PATCH  /api/admin/users/{user_id}
  GET    /api/admin/tasks
  POST   /api/admin/tasks/{task_id}/reset
  GET    /api/admin/papers
  GET    /api/admin/groups
  DELETE /api/admin/groups/{group_id}
"""

import json

import pytest

_USER_ID = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _insert_paper(db, arxiv_id, title="Test Paper"):
    db.execute(
        "INSERT INTO papers (arxiv_id, title, abstract, authors, published_date, categories) "
        "VALUES (?, ?, 'Abstract.', '[]', '2023-09-12', '[]')",
        (arxiv_id, title),
    )
    db.commit()


def _insert_task(db, task_type="fetch_meta", status="pending", arxiv_id="2309.06676"):
    db.execute(
        "INSERT INTO task_queue (type, payload, status) VALUES (?, ?, ?)",
        (task_type, json.dumps({"arxiv_id": arxiv_id}), status),
    )
    db.commit()
    return db.execute("SELECT last_insert_rowid()").fetchone()[0]


def _insert_group(db, name="Test Group"):
    db.execute("INSERT INTO groups (name) VALUES (?)", (name,))
    db.commit()
    group_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
    db.execute(
        "INSERT INTO group_members (group_id, user_id, is_admin) VALUES (?, ?, 1)",
        (group_id, _USER_ID),
    )
    db.commit()
    return group_id


# ---------------------------------------------------------------------------
# GET /api/admin/users
# ---------------------------------------------------------------------------

class TestAdminUsers:
    def test_list_users_returns_all(self, admin_client, web_db):
        """Admin user list should return all registered users enriched with
        paper_count and import_count aggregates."""
        # Insert a second user
        web_db.execute(
            "INSERT INTO users (email, password_hash, is_active) VALUES ('other@example.com', 'x', 1)"
        )
        web_db.commit()
        r = admin_client.get("/api/admin/users")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 2
        assert all("paper_count" in u for u in data)
        assert all("import_count" in u for u in data)

    def test_patch_user_toggles_active(self, admin_client, web_db):
        """Patching is_active on a user should update the DB and create an
        admin_audit_log row recording the action."""
        r = admin_client.patch(f"/api/admin/users/{_USER_ID}", json={"is_active": False})
        assert r.status_code == 200
        assert r.json()["is_active"] is False
        row = web_db.execute(
            "SELECT is_active FROM users WHERE id = ?", (_USER_ID,)
        ).fetchone()
        assert row["is_active"] == 0
        audit = web_db.execute(
            "SELECT action FROM admin_audit_log WHERE target_id = ?", (_USER_ID,)
        ).fetchone()
        assert audit is not None
        assert audit["action"] == "patch_user"

    def test_non_admin_forbidden_403(self, client):
        """A regular (non-admin) authenticated user should receive 403 Forbidden
        when accessing any admin endpoint."""
        r = client.get("/api/admin/users")
        assert r.status_code == 403


# ---------------------------------------------------------------------------
# GET /api/admin/tasks  &  POST /api/admin/tasks/{id}/reset
# ---------------------------------------------------------------------------

class TestAdminTasks:
    def test_list_tasks_empty(self, admin_client):
        """With no tasks in the queue the response should have total=0 and
        an empty items list."""
        r = admin_client.get("/api/admin/tasks")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 0
        assert data["items"] == []

    def test_list_tasks_with_type_filter(self, admin_client, web_db):
        """The ?type= filter should restrict results to tasks of that type only."""
        _insert_task(web_db, task_type="fetch_meta", arxiv_id="2309.00001")
        _insert_task(web_db, task_type="embed", arxiv_id="2309.00002")
        r = admin_client.get("/api/admin/tasks?type=embed")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 1
        assert data["items"][0]["type"] == "embed"

    def test_reset_task(self, admin_client, web_db):
        """Resetting a failed task should set its status back to 'pending' and
        clear its attempt count and error field."""
        task_id = _insert_task(web_db, status="failed")
        web_db.execute(
            "UPDATE task_queue SET attempts = 3, error = 'boom' WHERE id = ?", (task_id,)
        )
        web_db.commit()
        r = admin_client.post(f"/api/admin/tasks/{task_id}/reset")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "pending"
        assert data["attempts"] == 0
        assert data["error"] is None


# ---------------------------------------------------------------------------
# GET /api/admin/papers
# ---------------------------------------------------------------------------

class TestAdminPapers:
    def test_list_papers(self, admin_client, web_db):
        """Admin paper listing should return all ingested papers with the
        correct total count."""
        _insert_paper(web_db, "2309.00001", "Paper One")
        _insert_paper(web_db, "2309.00002", "Paper Two")
        r = admin_client.get("/api/admin/papers")
        assert r.status_code == 200
        assert r.json()["total"] == 2

    def test_list_papers_with_query(self, admin_client, web_db):
        """The ?q= search filter should match by title substring and exclude
        papers that do not match."""
        _insert_paper(web_db, "2309.00001", "Unique Title Here")
        _insert_paper(web_db, "2309.00002", "Something Else")
        r = admin_client.get("/api/admin/papers?q=Unique")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 1
        assert data["items"][0]["arxiv_id"] == "2309.00001"


# ---------------------------------------------------------------------------
# GET /api/admin/groups  &  DELETE /api/admin/groups/{group_id}
# ---------------------------------------------------------------------------

class TestAdminGroups:
    def test_list_groups_empty(self, admin_client):
        """When there are no groups the admin groups list should be empty."""
        r = admin_client.get("/api/admin/groups")
        assert r.status_code == 200
        assert r.json() == []

    def test_delete_group(self, admin_client, web_db):
        """Deleting a group via the admin endpoint should return 204 and create
        an admin_audit_log row recording the action."""
        group_id = _insert_group(web_db)
        r = admin_client.delete(f"/api/admin/groups/{group_id}")
        assert r.status_code == 204
        audit = web_db.execute(
            "SELECT action FROM admin_audit_log WHERE target_id = ?", (group_id,)
        ).fetchone()
        assert audit is not None
        assert audit["action"] == "delete_group"
