"""
Admin API endpoints.

All routes require an authenticated user with is_admin=1 (enforced by
get_admin_user dependency).  Admin status can only be granted via the CLI
script scripts/activate_user.py --make-admin.

GET  /api/admin/users                  — list all users with enrichment
PATCH /api/admin/users/{user_id}       — toggle is_active (not is_admin)
GET  /api/admin/tasks                  — task_queue entries, filterable
GET  /api/admin/papers                 — ingested papers, searchable
"""

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
            u.created_at,
            COUNT(up.arxiv_id) AS paper_count,
            um.trained_at     AS model_trained_at
        FROM users u
        LEFT JOIN user_papers  up ON up.user_id = u.id AND up.liked != 0
        LEFT JOIN user_models  um ON um.user_id = u.id
        GROUP BY u.id
        ORDER BY u.created_at DESC
    """).fetchall()
    return [
        {
            "id":              r["id"],
            "email":           r["email"],
            "is_active":       bool(r["is_active"]),
            "is_admin":        bool(r["is_admin"]),
            "created_at":      r["created_at"],
            "paper_count":     r["paper_count"],
            "model_trained_at": r["model_trained_at"],
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
    db.commit()
    return {"user_id": user_id, "is_active": body.is_active}


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

TaskType   = Literal["fetch_meta", "embed", "recommend"]
TaskStatus = Literal["pending", "running", "done", "failed"]


@router.get("/tasks")
def list_tasks(
    type:   str | None = Query(default=None),
    status_: str | None = Query(default=None, alias="status"),
    limit:  int = Query(default=50, ge=1, le=500),
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
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    rows = db.execute(
        f"""
        SELECT id, type, payload, status, attempts,
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
                "attempts":     r["attempts"],
                "created_at":   r["created_at"],
                "started_at":   r["started_at"],
                "completed_at": r["completed_at"],
                "error":        r["error"],
            }
            for r in rows
        ],
    }


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
