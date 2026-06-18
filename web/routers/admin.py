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
GET  /api/admin/analytics              — page-visit telemetry summary
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
            u.last_active_at,
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
            "last_active_at":  r["last_active_at"],
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


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

@router.get("/analytics")
def get_analytics(
    days: int = Query(default=30, ge=1, le=365),
    db: sqlite3.Connection = Depends(get_db),
    _admin=Depends(get_admin_user),
):
    """Return page-visit telemetry for the last *days* days.

    Response shape::

        {
            "summary": {"dau": int, "wau": int, "mau": int},
            "first_date": str | null,
            "daily":   [{"date": str, "visits": int, "users": int}, ...],
            "pages":   [{"page": str, "visits": int, "users": int}, ...]
        }

    *dau* = distinct users active today; *wau* = last 7 days; *mau* = last 30 days.
    *first_date* is the earliest date in page_stats_daily (or null), used to avoid
    zero-filling days before data collection began.
    Daily rows show total visit and distinct-user counts. Page rows include distinct-user counts
    from the dedup table (accurate within the 90-day retention window).
    """
    # Summary counts — always fixed windows regardless of the ?days parameter
    summary_row = db.execute("""
        SELECT
            COUNT(*) FILTER (WHERE last_active_at >= date('now'))              AS dau,
            COUNT(*) FILTER (WHERE last_active_at >= date('now', '-6 days'))   AS wau,
            COUNT(*) FILTER (WHERE last_active_at >= date('now', '-29 days'))  AS mau
        FROM users
    """).fetchone()

    # Earliest date with any data (so we don't zero-fill before collection began)
    first_date_row = db.execute(
        "SELECT MIN(date) AS first_date FROM page_stats_daily"
    ).fetchone()
    first_date = first_date_row["first_date"] if first_date_row else None

    # Daily visit counts for the requested window
    daily_rows = db.execute("""
        SELECT
            p.date,
            SUM(p.visits)               AS visits,
            COUNT(DISTINCT u.user_id)   AS users
        FROM page_stats_daily p
        LEFT JOIN page_stats_daily_users u USING (date, page)
        WHERE p.date >= date('now', ? || ' days')
        GROUP BY p.date
        ORDER BY p.date ASC
    """, (f"-{days}",)).fetchall()

    # Page breakdown: visits from counter table, distinct users from dedup table
    page_rows = db.execute("""
        SELECT
            p.page,
            SUM(p.visits)           AS visits,
            COUNT(DISTINCT u.user_id) AS users
        FROM page_stats_daily p
        LEFT JOIN page_stats_daily_users u USING (date, page)
        WHERE p.date >= date('now', ? || ' days')
        GROUP BY p.page
        ORDER BY visits DESC
    """, (f"-{days}",)).fetchall()

    return {
        "summary": {
            "dau": summary_row["dau"] or 0,
            "wau": summary_row["wau"] or 0,
            "mau": summary_row["mau"] or 0,
        },
        "first_date": first_date,
        "daily": [
            {"date": r["date"], "visits": r["visits"], "users": r["users"]}
            for r in daily_rows
        ],
        "pages": [
            {"page": r["page"], "visits": r["visits"], "users": r["users"]}
            for r in page_rows
        ],
    }


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

_DAEMON_TYPES = ("embed", "fetch_meta")


def _dt_to_ts(dt_str: str) -> int:
    """Convert 'YYYY-MM-DD HH:MM:SS' (SQLite UTC) → Unix timestamp (seconds)."""
    from datetime import datetime, timezone
    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _ts_to_dt(ts: int) -> str:
    """Convert Unix timestamp → 'YYYY-MM-DDTHH:MM:SS' ISO string."""
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def _daemon_health(db: sqlite3.Connection, task_type: str) -> dict:
    """Return health metrics for a single daemon task type."""

    # Queue size
    queue_size = db.execute(
        "SELECT COUNT(*) FROM task_queue WHERE type=? AND status='pending'",
        (task_type,),
    ).fetchone()[0]

    # Avg completion time (last 8 done tasks)
    avg_row = db.execute("""
        SELECT AVG(julianday(completed_at) - julianday(started_at)) * 86400.0
        FROM task_queue
        WHERE type=? AND status='done' AND started_at IS NOT NULL AND completed_at IS NOT NULL
        ORDER BY completed_at DESC LIMIT 8
    """, (task_type,)).fetchone()
    avg_completion_ms = round(avg_row[0] * 1000) if avg_row and avg_row[0] is not None else None

    # Recent failure rate (last 24h)
    rate_row = db.execute("""
        SELECT
            COUNT(*) FILTER (WHERE status='failed') * 1.0 / NULLIF(COUNT(*), 0)
        FROM task_queue
        WHERE type=? AND status IN ('done','failed')
          AND created_at >= datetime('now', '-1 day')
    """, (task_type,)).fetchone()
    recent_failure_rate = round(rate_row[0], 4) if rate_row and rate_row[0] is not None else None

    # Permanently failed (all-time)
    perm_failed = db.execute(
        "SELECT COUNT(*) FROM task_queue WHERE type=? AND status='failed'",
        (task_type,),
    ).fetchone()[0]

    # Completion times for the last 3 days (from the most recent completed task)
    ct_rows = db.execute("""
        SELECT completed_at,
               (julianday(completed_at) - julianday(started_at)) * 86400.0 AS duration_sec
        FROM task_queue
        WHERE type=? AND status='done' AND completed_at IS NOT NULL AND started_at IS NOT NULL
          AND completed_at >= COALESCE(
            (SELECT DATE(MAX(completed_at), '-3 days') FROM task_queue WHERE type=? AND status='done'),
            '1970-01-01'
          )
        ORDER BY completed_at ASC
    """, (task_type, task_type)).fetchall()
    completion_times = [
        {"completed_at": r["completed_at"], "duration_ms": round(r["duration_sec"] * 1000)}
        for r in ct_rows
    ]

    # Queue size history (sweep-line over task lifetimes)
    q_rows = db.execute("""
        SELECT created_at,
               COALESCE(completed_at, datetime('now')) AS end_at
        FROM task_queue
        WHERE type = ?
          AND created_at < datetime('now')
          AND COALESCE(completed_at, datetime('now')) >= COALESCE(
            (SELECT DATE(MAX(completed_at), '-3 days') FROM task_queue WHERE type=? AND status='done'),
            datetime('now', '-3 days')
          )
        ORDER BY created_at
    """, (task_type, task_type)).fetchall()

    queue_times: list[dict] = []
    if q_rows:
        # Build (ts, delta) events; +1 at created_at, -1 at end_at
        events: list[tuple[int, int]] = []
        for r in q_rows:
            c_ts = _dt_to_ts(r["created_at"])
            e_ts = _dt_to_ts(r["end_at"])
            events.append((c_ts, +1))
            events.append((e_ts, -1))
        events.sort(key=lambda x: x[0])

        # Sweep: cumulative sum
        cur = 0
        t0 = events[0][0]
        i = 0
        # Sample at ~288 points across the full range (~15-minute intervals over 3 days)
        t_end = max(e[0] for e in events)
        interval = max(1, (t_end - t0) // 288)
        for t in range(t0, t_end + 1, interval):
            while i < len(events) and events[i][0] <= t:
                cur += events[i][1]
                i += 1
            queue_times.append({
                "ts":   _ts_to_dt(t),
                "size": max(0, cur),
            })

    result = {
        "queue_size":         queue_size,
        "avg_completion_ms":  avg_completion_ms,
        "recent_failure_rate": recent_failure_rate,
        "permanently_failed": perm_failed,
        "completion_times":   completion_times,
        "queue_times":        queue_times,
    }

    # Meta daemon gets an extra stale-pending metric
    if task_type == "fetch_meta":
        stale = db.execute(
            "SELECT COUNT(*) FROM task_queue "
            "WHERE type=? AND status='pending' AND created_at < datetime('now', '-1 hour')",
            (task_type,),
        ).fetchone()[0]
        result["stale_pending"] = stale

    return result


@router.get("/health")
def get_health(
    db: sqlite3.Connection = Depends(get_db),
    _admin=Depends(get_admin_user),
):
    """Return per-daemon health metrics for embed and fetch_meta task pipelines."""
    return {
        "embed": _daemon_health(db, "embed"),
        "meta":  _daemon_health(db, "fetch_meta"),
    }

