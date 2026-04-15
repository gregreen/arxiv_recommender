"""
Group endpoints.

Groups allow multiple users to view aggregated recommendations computed from
their combined per-user scores.

POST   /api/groups                           – Create a new group
GET    /api/groups                           – List groups the current user belongs to
GET    /api/groups/{group_id}                – Group info + member list (members only)
GET    /api/groups/{group_id}/recommendations – Aggregated group recommendations
POST   /api/groups/{group_id}/invites        – Generate a new invite token (admin only)
GET    /api/groups/{group_id}/invites        – List pending invites (admin only)
DELETE /api/groups/{group_id}/invites/{invite_id} – Revoke an invite (admin only)
GET    /api/groups/join-info                 – Peek at a group name by invite token
POST   /api/groups/join                      – Consume an invite token and join its group
DELETE /api/groups/{group_id}/members/{user_id} – Remove a member (admin or self)
PATCH  /api/groups/{group_id}/members/{user_id} – Transfer admin rights (admin only)
DELETE /api/groups/{group_id}                – Delete a group (admin only)
"""

import secrets
import sqlite3
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from arxiv_lib.config import RECOMMEND_TIME_WINDOWS
from arxiv_lib.recommend import (
    NotEnoughDataError,
    aggregate_group_scores,
    get_group_recommendations,
)
from web.dependencies import get_current_user, get_db
from web.limiter import limiter

router = APIRouter(prefix="/groups", tags=["groups"])

_INVITE_EXPIRY_DAYS = 7


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class CreateGroupRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_member(db: sqlite3.Connection, group_id: int, user_id: int) -> None:
    """Raise 404 if the group doesn't exist, 403 if the user is not a member."""
    group = db.execute("SELECT id FROM groups WHERE id = ?", (group_id,)).fetchone()
    if group is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Group not found.")
    member = db.execute(
        "SELECT 1 FROM group_members WHERE group_id = ? AND user_id = ?",
        (group_id, user_id),
    ).fetchone()
    if member is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not a member of this group.")


def _require_admin(db: sqlite3.Connection, group_id: int, user_id: int) -> None:
    """Raise 403 if the user is not the group admin (also checks group existence)."""
    _require_member(db, group_id, user_id)
    row = db.execute(
        "SELECT is_admin FROM group_members WHERE group_id = ? AND user_id = ?",
        (group_id, user_id),
    ).fetchone()
    if not row["is_admin"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Group admin access required.")


def _group_response(db: sqlite3.Connection, group_id: int, include_members: bool = False) -> dict:
    group = db.execute("SELECT id, name, created_at FROM groups WHERE id = ?", (group_id,)).fetchone()
    result = {"id": group["id"], "name": group["name"], "created_at": group["created_at"]}
    if include_members:
        member_rows = db.execute(
            """
            SELECT u.id, u.email, gm.is_admin, gm.joined_at
              FROM group_members gm
              JOIN users u ON u.id = gm.user_id
             WHERE gm.group_id = ?
             ORDER BY gm.joined_at ASC
            """,
            (group_id,),
        ).fetchall()
        result["members"] = [
            {
                "user_id":   r["id"],
                "email":     r["email"],
                "is_admin":  bool(r["is_admin"]),
                "joined_at": r["joined_at"],
            }
            for r in member_rows
        ]
    return result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("", status_code=status.HTTP_201_CREATED)
@limiter.limit("10/hour")
def create_group(
    request: Request,
    body: CreateGroupRequest,
    db: sqlite3.Connection = Depends(get_db),
    user=Depends(get_current_user),
):
    """Create a new group. The creator becomes the group admin."""
    now = datetime.now(timezone.utc).isoformat()
    cursor = db.execute(
        "INSERT INTO groups (name, created_at) VALUES (?, ?)",
        (body.name.strip(), now),
    )
    group_id = cursor.lastrowid
    db.execute(
        "INSERT INTO group_members (group_id, user_id, is_admin, joined_at) VALUES (?, ?, 1, ?)",
        (group_id, user["id"], now),
    )
    db.commit()
    return _group_response(db, group_id, include_members=True)


@router.get("")
def list_groups(
    db: sqlite3.Connection = Depends(get_db),
    user=Depends(get_current_user),
):
    """List all groups the current user belongs to."""
    rows = db.execute(
        """
        SELECT g.id, g.name, g.created_at, gm.is_admin
          FROM groups g
          JOIN group_members gm ON gm.group_id = g.id
         WHERE gm.user_id = ?
         ORDER BY g.name ASC
        """,
        (user["id"],),
    ).fetchall()
    return [
        {
            "id":         r["id"],
            "name":       r["name"],
            "created_at": r["created_at"],
            "is_admin":   bool(r["is_admin"]),
        }
        for r in rows
    ]


@router.get("/join-info")
def join_info(
    token: str = Query(...),
    db: sqlite3.Connection = Depends(get_db),
    user=Depends(get_current_user),
):
    """
    Return the group name for an invite token without consuming it.
    Used by JoinGroupPage to confirm group identity before joining.
    """
    now = datetime.now(timezone.utc).isoformat()
    row = db.execute(
        """
        SELECT g.id, g.name
          FROM group_invites gi
          JOIN groups g ON g.id = gi.group_id
         WHERE gi.token = ? AND gi.used_at IS NULL AND gi.expires_at > ?
        """,
        (token, now),
    ).fetchone()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invite not found, already used, or expired.",
        )
    return {"group_id": row["id"], "group_name": row["name"]}


@router.get("/{group_id}")
def get_group(
    group_id: int,
    db: sqlite3.Connection = Depends(get_db),
    user=Depends(get_current_user),
):
    """Return group info and member list (members only)."""
    _require_member(db, group_id, user["id"])
    return _group_response(db, group_id, include_members=True)


@router.get("/{group_id}/recommendations")
def group_recommendations(
    group_id: int,
    window: str = Query(default="week"),
    method: str = Query(default="softmax_sum"),
    db: sqlite3.Connection = Depends(get_db),
    user=Depends(get_current_user),
):
    """Return aggregated group recommendations for the given time window."""
    _require_member(db, group_id, user["id"])

    if window not in RECOMMEND_TIME_WINDOWS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid window {window!r}. Must be one of: {RECOMMEND_TIME_WINDOWS}",
        )

    supported_methods = ["softmax_sum"]
    if method not in supported_methods:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid method {method!r}. Must be one of: {supported_methods}",
        )

    results, group_member_count, active_member_count = get_group_recommendations(
        db, group_id, window, user["id"], method=method
    )

    group = db.execute("SELECT name FROM groups WHERE id = ?", (group_id,)).fetchone()
    return {
        "group_id":           group_id,
        "group_name":         group["name"],
        "window":             window,
        "method":             method,
        "count":              len(results),
        "group_member_count": group_member_count,
        "active_member_count": active_member_count,
        "results":            results,
    }


@router.post("/{group_id}/invites", status_code=status.HTTP_201_CREATED)
@limiter.limit("20/hour")
def create_invite(
    request: Request,
    group_id: int,
    db: sqlite3.Connection = Depends(get_db),
    user=Depends(get_current_user),
):
    """Generate a new single-use invite token for the group (admin only)."""
    _require_admin(db, group_id, user["id"])
    now = datetime.now(timezone.utc)
    token = secrets.token_urlsafe(32)
    expires_at = (now + timedelta(days=_INVITE_EXPIRY_DAYS)).isoformat()
    cursor = db.execute(
        """
        INSERT INTO group_invites (group_id, token, created_by, created_at, expires_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (group_id, token, user["id"], now.isoformat(), expires_at),
    )
    db.commit()
    return {
        "id":         cursor.lastrowid,
        "token":      token,
        "expires_at": expires_at,
    }


@router.get("/{group_id}/invites")
def list_invites(
    group_id: int,
    db: sqlite3.Connection = Depends(get_db),
    user=Depends(get_current_user),
):
    """List pending (unused and unexpired) invites for the group (admin only)."""
    _require_admin(db, group_id, user["id"])
    now = datetime.now(timezone.utc).isoformat()
    rows = db.execute(
        """
        SELECT id, token, created_at, expires_at
          FROM group_invites
         WHERE group_id = ? AND used_at IS NULL AND expires_at > ?
         ORDER BY created_at DESC
        """,
        (group_id, now),
    ).fetchall()
    return [
        {
            "id":         r["id"],
            "token":      r["token"],
            "created_at": r["created_at"],
            "expires_at": r["expires_at"],
        }
        for r in rows
    ]


@router.delete("/{group_id}/invites/{invite_id}", status_code=status.HTTP_204_NO_CONTENT)
def revoke_invite(
    group_id: int,
    invite_id: int,
    db: sqlite3.Connection = Depends(get_db),
    user=Depends(get_current_user),
):
    """Revoke (delete) a pending invite (admin only)."""
    _require_admin(db, group_id, user["id"])
    result = db.execute(
        "DELETE FROM group_invites WHERE id = ? AND group_id = ? AND used_at IS NULL",
        (invite_id, group_id),
    )
    db.commit()
    if result.rowcount == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invite not found or already used.",
        )


@router.post("/join", status_code=status.HTTP_200_OK)
@limiter.limit("10/hour")
def join_group(
    request: Request,
    token: str = Query(...),
    db: sqlite3.Connection = Depends(get_db),
    user=Depends(get_current_user),
):
    """Consume an invite token and join its group."""
    now = datetime.now(timezone.utc).isoformat()
    invite = db.execute(
        """
        SELECT id, group_id
          FROM group_invites
         WHERE token = ? AND used_at IS NULL AND expires_at > ?
        """,
        (token, now),
    ).fetchone()
    if invite is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invite not found, already used, or expired.",
        )

    group_id = invite["group_id"]

    # Idempotent: already a member is fine
    existing = db.execute(
        "SELECT 1 FROM group_members WHERE group_id = ? AND user_id = ?",
        (group_id, user["id"]),
    ).fetchone()
    if existing:
        return _group_response(db, group_id, include_members=True)

    db.execute(
        "INSERT INTO group_members (group_id, user_id, is_admin, joined_at) VALUES (?, ?, 0, ?)",
        (group_id, user["id"], now),
    )
    db.execute(
        "UPDATE group_invites SET used_at = ?, used_by = ? WHERE id = ?",
        (now, user["id"], invite["id"]),
    )
    db.commit()
    return _group_response(db, group_id, include_members=True)


@router.delete("/{group_id}/members/{target_user_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_member(
    group_id: int,
    target_user_id: int,
    db: sqlite3.Connection = Depends(get_db),
    user=Depends(get_current_user),
):
    """
    Remove a member from a group.
    Admins can remove anyone; non-admins can only remove themselves (leave).
    The last admin cannot leave without first transferring admin rights.
    """
    _require_member(db, group_id, user["id"])

    is_self = target_user_id == user["id"]
    requester_row = db.execute(
        "SELECT is_admin FROM group_members WHERE group_id = ? AND user_id = ?",
        (group_id, user["id"]),
    ).fetchone()
    is_requester_admin = bool(requester_row["is_admin"])

    if not is_self and not is_requester_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Group admin access required.")

    # Prevent the last admin from leaving
    target_row = db.execute(
        "SELECT is_admin FROM group_members WHERE group_id = ? AND user_id = ?",
        (group_id, target_user_id),
    ).fetchone()
    if target_row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Member not found.")

    if target_row["is_admin"]:
        admin_count = db.execute(
            "SELECT COUNT(*) FROM group_members WHERE group_id = ? AND is_admin = 1",
            (group_id,),
        ).fetchone()[0]
        if admin_count <= 1:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Cannot remove the last admin. Transfer admin rights first.",
            )

    db.execute(
        "DELETE FROM group_members WHERE group_id = ? AND user_id = ?",
        (group_id, target_user_id),
    )
    db.commit()


@router.patch("/{group_id}/members/{target_user_id}", status_code=status.HTTP_200_OK)
def transfer_admin(
    group_id: int,
    target_user_id: int,
    db: sqlite3.Connection = Depends(get_db),
    user=Depends(get_current_user),
):
    """
    Grant admin rights to another member (admin only).
    The existing admin retains their admin status — use remove_member to step down.
    """
    _require_admin(db, group_id, user["id"])
    target_row = db.execute(
        "SELECT user_id FROM group_members WHERE group_id = ? AND user_id = ?",
        (group_id, target_user_id),
    ).fetchone()
    if target_row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Member not found.")
    db.execute(
        "UPDATE group_members SET is_admin = 1 WHERE group_id = ? AND user_id = ?",
        (group_id, target_user_id),
    )
    db.commit()
    return _group_response(db, group_id, include_members=True)


@router.delete("/{group_id}", status_code=status.HTTP_204_NO_CONTENT)
@limiter.limit("5/hour")
def delete_group(
    request: Request,
    group_id: int,
    db: sqlite3.Connection = Depends(get_db),
    user=Depends(get_current_user),
):
    """Delete the group entirely (admin only). Cascades to members and invites."""
    _require_admin(db, group_id, user["id"])
    db.execute("DELETE FROM groups WHERE id = ?", (group_id,))
    db.commit()
