"""
Admin API endpoints.

All routes require an authenticated user with is_admin=1 (enforced by
get_admin_user dependency).  Admin status can only be granted via the CLI
script scripts/activate_user.py --make-admin.

GET  /api/admin/users                  — list all users with enrichment
PATCH /api/admin/users/{user_id}       — toggle is_active (not is_admin)
GET  /api/admin/tasks                  — task_queue entries, filterable
GET  /api/admin/papers                 — ingested papers, searchable
GET  /api/admin/groups                 — list all groups with aggregate stats
GET  /api/admin/groups/{group_id}      — group detail with members + pending invites
DELETE /api/admin/groups/{group_id}    — delete a group (cascades to members/invites)
"""

import json
import sqlite3
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from web.dependencies import get_admin_user, get_db

router = APIRouter(prefix="/admin", tags=["admin"])


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

@router.get("/users")
def list_users(
    db: sqlite3.Connection = Depends(get_db),
    _admin=Depends(get_admin_user),
):
    rows = db.execute("""
        SELECT
            u.id,
            u.email,
            u.is_active,
            u.is_admin,
            u.email_verified,
            u.created_at,
            COUNT(DISTINCT up.arxiv_id) AS paper_count,
            um.trained_at     AS model_trained_at,
            COUNT(DISTINCT il.id) AS import_count
        FROM users u
        LEFT JOIN user_papers  up ON up.user_id = u.id AND up.liked != 0
        LEFT JOIN user_models  um ON um.user_id = u.id
        LEFT JOIN user_import_log il ON il.user_id = u.id
        GROUP BY u.id
        ORDER BY u.created_at DESC
    """).fetchall()
    return [
        {
            "id":              r["id"],
            "email":           r["email"],
            "is_active":       bool(r["is_active"]),
            "is_admin":        bool(r["is_admin"]),
            "email_verified":  bool(r["email_verified"]),
            "created_at":      r["created_at"],
            "paper_count":     r["paper_count"],
            "model_trained_at": r["model_trained_at"],
            "import_count":    r["import_count"],
        }
        for r in rows
    ]


class PatchUserBody(BaseModel):
    is_active: bool


@router.patch("/users/{user_id}")
def patch_user(
    user_id: int,
    body: PatchUserBody,
    db: sqlite3.Connection = Depends(get_db),
    _admin=Depends(get_admin_user),
):
    row = db.execute("SELECT id FROM users WHERE id = ?", (user_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    db.execute("UPDATE users SET is_active = ? WHERE id = ?", (int(body.is_active), user_id))
    db.execute(
        "INSERT INTO admin_audit_log (admin_id, action, target_id, detail) VALUES (?, ?, ?, ?)",
        (_admin["id"], "patch_user", user_id, json.dumps({"is_active": body.is_active})),
    )
    db.commit()
    return {"user_id": user_id, "is_active": body.is_active}


@router.delete("/users/{user_id}/import-log", status_code=status.HTTP_204_NO_CONTENT)
def reset_user_import_log(
    user_id: int,
    db: sqlite3.Connection = Depends(get_db),
    _admin=Depends(get_admin_user),
):
    row = db.execute("SELECT id FROM users WHERE id = ?", (user_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    db.execute("DELETE FROM user_import_log WHERE user_id = ?", (user_id,))
    db.execute(
        "INSERT INTO admin_audit_log (admin_id, action, target_id) VALUES (?, ?, ?)",
        (_admin["id"], "reset_import_log", user_id),
    )
    db.commit()


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

TaskType   = Literal["fetch_meta", "embed", "recommend"]
TaskStatus = Literal["pending", "running", "done", "failed"]


@router.get("/tasks")
def list_tasks(
    type:   str | None = Query(default=None),
    status_: str | None = Query(default=None, alias="status"),
    q:      str | None = Query(default=None, description="Filter by payload substring"),
    limit:  int = Query(default=50, ge=1, le=8192),
    offset: int = Query(default=0, ge=0),
    db: sqlite3.Connection = Depends(get_db),
    _admin=Depends(get_admin_user),
):
    conditions = []
    params: list = []
    if type:
        conditions.append("type = ?")
        params.append(type)
    if status_:
        conditions.append("status = ?")
        params.append(status_)
    if q:
        conditions.append("payload LIKE ?")
        params.append(f"%{q}%")
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    rows = db.execute(
        f"""
        SELECT id, type, payload, status, priority, attempts,
               created_at, started_at, completed_at, error
        FROM task_queue
        {where}
        ORDER BY id DESC
        LIMIT ? OFFSET ?
        """,
        params + [limit, offset],
    ).fetchall()
    total = db.execute(
        f"SELECT COUNT(*) FROM task_queue {where}", params
    ).fetchone()[0]
    return {
        "total":  total,
        "offset": offset,
        "limit":  limit,
        "items": [
            {
                "id":           r["id"],
                "type":         r["type"],
                "payload":      r["payload"],
                "status":       r["status"],
                "priority":     r["priority"],
                "attempts":     r["attempts"],
                "created_at":   r["created_at"],
                "started_at":   r["started_at"],
                "completed_at": r["completed_at"],
                "error":        r["error"],
            }
            for r in rows
        ],
    }


@router.post("/tasks/{task_id}/reset")
def reset_task(
    task_id: int,
    db: sqlite3.Connection = Depends(get_db),
    _admin=Depends(get_admin_user),
):
    row = db.execute("SELECT * FROM task_queue WHERE id = ?", (task_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found.")
    db.execute(
        """
        UPDATE task_queue
           SET status = 'pending', attempts = 0, error = NULL,
               started_at = NULL, completed_at = NULL
         WHERE id = ?
        """,
        (task_id,),
    )
    db.execute(
        "INSERT INTO admin_audit_log (admin_id, action, target_id) VALUES (?, ?, ?)",
        (_admin["id"], "reset_task", task_id),
    )
    db.commit()
    row = db.execute("SELECT * FROM task_queue WHERE id = ?", (task_id,)).fetchone()
    return {
        "id":           row["id"],
        "type":         row["type"],
        "payload":      row["payload"],
        "status":       row["status"],
        "priority":     row["priority"],
        "attempts":     row["attempts"],
        "created_at":   row["created_at"],
        "started_at":   row["started_at"],
        "completed_at": row["completed_at"],
        "error":        row["error"],
    }


@router.delete("/tasks/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_task(
    task_id: int,
    db: sqlite3.Connection = Depends(get_db),
    _admin=Depends(get_admin_user),
):
    cur = db.execute("DELETE FROM task_queue WHERE id = ?", (task_id,))
    if cur.rowcount == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found.")
    db.execute(
        "INSERT INTO admin_audit_log (admin_id, action, target_id) VALUES (?, ?, ?)",
        (_admin["id"], "delete_task", task_id),
    )
    db.commit()


# ---------------------------------------------------------------------------
# Papers
# ---------------------------------------------------------------------------

@router.get("/papers")
def list_papers(
    q:      str | None = Query(default=None, description="Search arxiv_id or title"),
    limit:  int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: sqlite3.Connection = Depends(get_db),
    _admin=Depends(get_admin_user),
):
    if q:
        pattern = f"%{q}%"
        rows = db.execute(
            """
            SELECT arxiv_id, title, authors, published_date, categories, embedded_at
            FROM papers
            WHERE arxiv_id LIKE ? OR title LIKE ?
            ORDER BY embedded_at DESC
            LIMIT ? OFFSET ?
            """,
            (pattern, pattern, limit, offset),
        ).fetchall()
        total = db.execute(
            "SELECT COUNT(*) FROM papers WHERE arxiv_id LIKE ? OR title LIKE ?",
            (pattern, pattern),
        ).fetchone()[0]
    else:
        rows = db.execute(
            """
            SELECT arxiv_id, title, authors, published_date, categories, embedded_at
            FROM papers
            ORDER BY embedded_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
        total = db.execute("SELECT COUNT(*) FROM papers").fetchone()[0]

    return {
        "total":  total,
        "offset": offset,
        "limit":  limit,
        "items": [
            {
                "arxiv_id":       r["arxiv_id"],
                "title":          r["title"],
                "authors":        r["authors"],
                "published_date": r["published_date"],
                "categories":     r["categories"],
                "embedded_at":    r["embedded_at"],
            }
            for r in rows
        ],
    }


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------

@router.get("/groups")
def list_groups(
    db: sqlite3.Connection = Depends(get_db),
    _admin=Depends(get_admin_user),
):
    rows = db.execute("""
        SELECT
            g.id,
            g.name,
            g.created_at,
            COUNT(DISTINCT gm.user_id)                                   AS member_count,
            MAX(gm.joined_at)                                            AS last_joined_at,
            SUM(CASE WHEN gm.is_admin = 1 THEN 1 ELSE 0 END)            AS admin_count,
            COUNT(DISTINCT CASE
                WHEN gi.remaining_uses > 0 AND gi.expires_at > datetime('now')
                THEN gi.id END)                                          AS pending_invite_count
        FROM groups g
        LEFT JOIN group_members gm ON gm.group_id = g.id
        LEFT JOIN group_invites gi ON gi.group_id = g.id
        GROUP BY g.id
        ORDER BY g.created_at DESC
    """).fetchall()

    # Fetch admin emails for each group separately (avoids comma_group concat portability issues)
    result = []
    for r in rows:
        admin_rows = db.execute("""
            SELECT u.email
            FROM group_members gm
            JOIN users u ON u.id = gm.user_id
            WHERE gm.group_id = ? AND gm.is_admin = 1
            ORDER BY gm.joined_at ASC
        """, (r["id"],)).fetchall()
        result.append({
            "id":                   r["id"],
            "name":                 r["name"],
            "created_at":           r["created_at"],
            "member_count":         r["member_count"],
            "last_joined_at":       r["last_joined_at"],
            "admin_emails":         [a["email"] for a in admin_rows],
            "pending_invite_count": r["pending_invite_count"] or 0,
        })
    return result


@router.get("/groups/{group_id}")
def get_group(
    group_id: int,
    db: sqlite3.Connection = Depends(get_db),
    _admin=Depends(get_admin_user),
):
    group = db.execute(
        "SELECT id, name, created_at FROM groups WHERE id = ?", (group_id,)
    ).fetchone()
    if group is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Group not found.")

    members = db.execute("""
        SELECT u.id AS user_id, u.email, gm.is_admin, gm.joined_at
        FROM group_members gm
        JOIN users u ON u.id = gm.user_id
        WHERE gm.group_id = ?
        ORDER BY gm.joined_at ASC
    """, (group_id,)).fetchall()

    invites = db.execute("""
        SELECT gi.id, gi.token, u.email AS created_by_email, gi.created_at, gi.expires_at, gi.remaining_uses
        FROM group_invites gi
        JOIN users u ON u.id = gi.created_by
        WHERE gi.group_id = ?
          AND gi.remaining_uses > 0
          AND gi.expires_at > datetime('now')
        ORDER BY gi.created_at DESC
    """, (group_id,)).fetchall()

    return {
        "id":         group["id"],
        "name":       group["name"],
        "created_at": group["created_at"],
        "members": [
            {
                "user_id":   m["user_id"],
                "email":     m["email"],
                "is_admin":  bool(m["is_admin"]),
                "joined_at": m["joined_at"],
            }
            for m in members
        ],
        "pending_invites": [
            {
                "id":               i["id"],
                "token":            i["token"],
                "created_by_email": i["created_by_email"],
                "created_at":       i["created_at"],
                "expires_at":       i["expires_at"],
                "remaining_uses":   i["remaining_uses"],
            }
            for i in invites
        ],
    }


@router.delete("/groups/{group_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_group(
    group_id: int,
    db: sqlite3.Connection = Depends(get_db),
    _admin=Depends(get_admin_user),
):
    cur = db.execute("DELETE FROM groups WHERE id = ?", (group_id,))
    if cur.rowcount == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Group not found.")
    db.execute(
        "INSERT INTO admin_audit_log (admin_id, action, target_id) VALUES (?, ?, ?)",
        (_admin["id"], "delete_group", group_id),
    )
    db.commit()

