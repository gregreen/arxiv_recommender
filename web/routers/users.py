"""
User paper management endpoints.

GET    /api/users/me/papers                  — list liked/disliked papers
POST   /api/users/me/papers                  — add a paper to liked set
POST   /api/users/me/papers/import/ads       — bulk import from NASA ADS export text
PATCH  /api/users/me/papers/{arxiv_id}       — update liked flag
DELETE /api/users/me/papers/{arxiv_id}       — remove from set
GET    /api/users/me/categories              — list subscribed categories
PUT    /api/users/me/categories              — replace subscribed categories
"""

import json
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from web.limiter import limiter

from arxiv_lib.appdb import enqueue_task
from arxiv_lib.config import (
    ARXIV_CATEGORIES,
    EMBEDDING_CACHE_DB,
    IMPORT_TIER_THRESHOLD,
    IMPORT_DAILY_LIMIT_TIER_A,
    IMPORT_DAILY_LIMIT_TIER_B,
)
from web.auth import verify_password
from web.dependencies import get_current_user, get_db

router = APIRouter(prefix="/users/me", tags=["users"])


class DeleteAccountRequest(BaseModel):
    password: str


class ExportRequest(BaseModel):
    password: str


_EXPORT_COOLDOWN_DAYS = 7
_ADS_INPUT_CAP = 64


def _validate_arxiv_id(arxiv_id: str) -> str:
    """Semantically validate and canonicalise an arXiv ID. Raises 422 on failure."""
    from arxiv_lib.arxiv_id import validate_arxiv_id

    try:
        return validate_arxiv_id(arxiv_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc


def _get_daily_limit(db: sqlite3.Connection, user_id: int) -> int:
    """Return the daily import limit based on the user's lifetime import count."""
    lifetime = db.execute(
        "SELECT COUNT(*) FROM user_import_log WHERE user_id = ?", (user_id,)
    ).fetchone()[0]
    return IMPORT_DAILY_LIMIT_TIER_A if lifetime < IMPORT_TIER_THRESHOLD else IMPORT_DAILY_LIMIT_TIER_B


def _count_recent_imports(db: sqlite3.Connection, user_id: int) -> int:
    """Count this user's imports in the rolling 24-hour window."""
    return db.execute(
        """
        SELECT COUNT(*) FROM user_import_log
         WHERE user_id = ? AND imported_at >= datetime('now', '-24 hours')
        """,
        (user_id,),
    ).fetchone()[0]


def _task_pending_or_running(db: sqlite3.Connection, task_type: str, arxiv_id: str) -> bool:
    """Return True if there is already a pending or running task of task_type for arxiv_id."""
    return db.execute(
        """
        SELECT 1 FROM task_queue
         WHERE type = ? AND json_extract(payload, '$.arxiv_id') = ?
           AND status IN ('pending', 'running')
        """,
        (task_type, arxiv_id),
    ).fetchone() is not None


def _ensure_ingest_enqueued(db: sqlite3.Connection, arxiv_id: str) -> None:
    """
    Ensure the full ingest pipeline (metadata + embeddings) will run for arxiv_id.

    - Paper not in papers table          → enqueue fetch_meta (meta daemon writes
                                           metadata then enqueues embed).
    - Paper in papers table but missing  → enqueue embed directly (skips the
      one or both embeddings               meta step, embeddings are the only
                                           thing outstanding).
    - Paper fully ingested               → nothing to do.

    Deduplicates: won't add a second task if one is already pending/running.
    """
    has_meta = db.execute(
        "SELECT 1 FROM papers WHERE arxiv_id = ?", (arxiv_id,)
    ).fetchone() is not None

    if not has_meta:
        if not _task_pending_or_running(db, "fetch_meta", arxiv_id):
            enqueue_task(db, "fetch_meta", {"arxiv_id": arxiv_id})
        return

    # Metadata is present — check whether embeddings are missing.
    try:
        with sqlite3.connect(EMBEDDING_CACHE_DB()) as emb_con:
            has_search = emb_con.execute(
                "SELECT 1 FROM search_embeddings WHERE arxiv_id = ?", (arxiv_id,)
            ).fetchone() is not None
            has_rec = emb_con.execute(
                "SELECT 1 FROM recommendation_embeddings WHERE arxiv_id = ?", (arxiv_id,)
            ).fetchone() is not None
    except Exception:
        # If we can't open the embedding DB, be conservative and enqueue.
        has_search = has_rec = False

    if not (has_search and has_rec):
        if not _task_pending_or_running(db, "embed", arxiv_id):
            enqueue_task(db, "embed", {"arxiv_id": arxiv_id})


# ---------------------------------------------------------------------------
# Account deletion
# ---------------------------------------------------------------------------

@router.delete("", status_code=204)
@limiter.limit("3/hour")
def delete_account(
    request: Request,
    body: DeleteAccountRequest,
    response: Response,
    db: sqlite3.Connection = Depends(get_db),
    user=Depends(get_current_user),
):
    """Permanently delete the authenticated user's account and all their data."""
    if user["is_admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin accounts cannot be self-deleted. Ask another admin to remove your account.",
        )

    # Block if sole admin of any group.
    sole_admin_groups = db.execute(
        """
        SELECT g.name
          FROM group_members gm
          JOIN groups g ON g.id = gm.group_id
         WHERE gm.user_id = ? AND gm.is_admin = 1
           AND (SELECT COUNT(*) FROM group_members gm2
                 WHERE gm2.group_id = gm.group_id AND gm2.is_admin = 1) = 1
        """,
        (user["id"],),
    ).fetchall()
    if sole_admin_groups:
        names = ", ".join(f'"{r["name"]}"' for r in sole_admin_groups)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"You are the sole admin of: {names}. "
                "Transfer admin rights or delete those groups before deleting your account."
            ),
        )

    row = db.execute(
        "SELECT password_hash FROM users WHERE id = ?", (user["id"],)
    ).fetchone()
    if row is None or not verify_password(body.password, row["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password is incorrect.",
        )

    uid = user["id"]
    db.execute("UPDATE group_invites SET used_by = NULL WHERE used_by = ?", (uid,))
    db.execute("DELETE FROM group_invites WHERE created_by = ?", (uid,))
    db.execute("DELETE FROM recommendations WHERE user_id = ?", (uid,))
    db.execute("DELETE FROM user_models WHERE user_id = ?", (uid,))
    db.execute("DELETE FROM user_import_log WHERE user_id = ?", (uid,))
    db.execute("DELETE FROM user_papers WHERE user_id = ?", (uid,))
    db.execute("DELETE FROM user_categories WHERE user_id = ?", (uid,))
    db.execute("DELETE FROM user_search_terms WHERE user_id = ?", (uid,))
    db.execute(
        "DELETE FROM task_queue WHERE type = 'recommend' AND json_extract(payload, '$.user_id') = ?",
        (uid,),
    )
    db.execute("DELETE FROM users WHERE id = ?", (uid,))
    db.commit()

    response.delete_cookie("access_token")


# ---------------------------------------------------------------------------
# Data export (GDPR)
# ---------------------------------------------------------------------------

@router.post("/export", status_code=200)
@limiter.limit("3/day")
def export_user_data(
    request: Request,
    body: ExportRequest,
    db: sqlite3.Connection = Depends(get_db),
    user=Depends(get_current_user),
):
    """Return a JSON export of all personal data held for the authenticated user."""
    row = db.execute(
        "SELECT password_hash, last_export_at FROM users WHERE id = ?", (user["id"],)
    ).fetchone()
    if row is None or not verify_password(body.password, row["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password is incorrect.",
        )

    # Enforce per-user cooldown independent of IP-based limiter.
    if row["last_export_at"]:
        last = datetime.fromisoformat(row["last_export_at"].rstrip("Z")).replace(tzinfo=timezone.utc)
        next_allowed = last + timedelta(days=_EXPORT_COOLDOWN_DAYS)
        if datetime.now(timezone.utc) < next_allowed:
            wait_seconds = int((next_allowed - datetime.now(timezone.utc)).total_seconds())
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Data export is limited to once every {_EXPORT_COOLDOWN_DAYS} days. "
                       f"Please wait {wait_seconds} seconds before requesting another export.",
            )

    db.execute(
        "UPDATE users SET last_export_at = datetime('now') WHERE id = ?", (user["id"],)
    )
    db.commit()

    uid = user["id"]

    # Account
    account_row = db.execute(
        "SELECT email, created_at FROM users WHERE id = ?", (uid,)
    ).fetchone()
    account = {"email": account_row["email"], "created_at": account_row["created_at"]}

    # Library
    library = [
        {"arxiv_id": r["arxiv_id"], "liked": r["liked"], "added_at": r["added_at"]}
        for r in db.execute(
            "SELECT arxiv_id, liked, added_at FROM user_papers WHERE user_id = ? ORDER BY added_at DESC",
            (uid,),
        ).fetchall()
    ]

    # Categories
    categories = [
        r["category"]
        for r in db.execute(
            "SELECT category FROM user_categories WHERE user_id = ? ORDER BY category",
            (uid,),
        ).fetchall()
    ]

    # Search terms
    search_terms = [
        {"query": r["query"], "last_searched_at": r["last_searched_at"]}
        for r in db.execute(
            "SELECT query, last_searched_at FROM user_search_terms WHERE user_id = ? ORDER BY last_searched_at DESC",
            (uid,),
        ).fetchall()
    ]

    # Groups
    groups = [
        {"name": r["name"], "is_admin": bool(r["is_admin"]), "joined_at": r["joined_at"]}
        for r in db.execute(
            """
            SELECT g.name, gm.is_admin, gm.joined_at
              FROM group_members gm
              JOIN groups g ON g.id = gm.group_id
             WHERE gm.user_id = ?
             ORDER BY gm.joined_at
            """,
            (uid,),
        ).fetchall()
    ]

    export = {
        "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "account": account,
        "library": library,
        "categories": categories,
        "search_terms": search_terms,
        "groups": groups,
    }

    return JSONResponse(
        content=export,
        headers={"Content-Disposition": 'attachment; filename="arxiv-recommender-data.json"'},
    )


# ---------------------------------------------------------------------------
# Paper list
# ---------------------------------------------------------------------------

@router.get("/papers")
def list_papers(
    db: sqlite3.Connection = Depends(get_db),
    user=Depends(get_current_user),
):
    rows = db.execute(
        """
        SELECT up.arxiv_id, p.title, p.authors, p.published_date,
               up.liked, up.added_at
          FROM user_papers up
          LEFT JOIN papers p ON p.arxiv_id = up.arxiv_id
         WHERE up.user_id = ? AND up.liked != 0
         ORDER BY up.added_at DESC
        """,
        (user["id"],),
    ).fetchall()
    return [
        {
            "arxiv_id":       r["arxiv_id"],
            "title":          r["title"],
            "authors":        json.loads(r["authors"]) if r["authors"] else [],
            "published_date": r["published_date"],
            "liked":          r["liked"],
            "added_at":       r["added_at"],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Add / import papers
# ---------------------------------------------------------------------------

class AddPaperRequest(BaseModel):
    arxiv_id: str
    liked: int = 1

    @field_validator("liked")
    @classmethod
    def liked_must_be_valid(cls, v):
        if v not in (-1, 0, 1):
            raise ValueError("liked must be -1, 0, or 1")
        return v


@router.post("/papers", status_code=status.HTTP_201_CREATED)
def add_paper(
    body: AddPaperRequest,
    db: sqlite3.Connection = Depends(get_db),
    user=Depends(get_current_user),
):
    arxiv_id = _validate_arxiv_id(body.arxiv_id)
    already_exists = db.execute(
        "SELECT 1 FROM user_papers WHERE user_id = ? AND arxiv_id = ?",
        (user["id"], arxiv_id),
    ).fetchone()
    if not already_exists:
        daily_limit = _get_daily_limit(db, user["id"])
        recent = _count_recent_imports(db, user["id"])
        if recent >= daily_limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Daily import limit of {daily_limit} reached. Try again later.",
            )
    db.execute(
        "INSERT OR IGNORE INTO user_papers (user_id, arxiv_id, liked) VALUES (?, ?, ?)",
        (user["id"], arxiv_id, body.liked),
    )
    if not already_exists:
        db.execute(
            "INSERT INTO user_import_log (user_id, arxiv_id) VALUES (?, ?)",
            (user["id"], arxiv_id),
        )
    _ensure_ingest_enqueued(db, arxiv_id)
    db.commit()
    return {"arxiv_id": arxiv_id, "liked": body.liked}


class AdsImportRequest(BaseModel):
    text: str


@router.post("/papers/import/ads")
def import_ads(
    body: AdsImportRequest,
    db: sqlite3.Connection = Depends(get_db),
    user=Depends(get_current_user),
):
    # Accept IDs with or without the "arXiv:" prefix (case-insensitive).
    found_ids = re.findall(r"(?i:arxiv:)?((?:\d{4}\.\d{4,5}|[a-z][a-z-]*(?:\.[A-Z]{2})?/\d{7}))(?:v\d+)?", body.text)
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_ids = [aid for aid in found_ids if not (aid in seen or seen.add(aid))]  # type: ignore[func-returns-value]
    unique_ids = unique_ids[:_ADS_INPUT_CAP]

    # Semantically validate each extracted ID; silently drop impossible ones.
    from arxiv_lib.arxiv_id import validate_arxiv_id
    valid_ids: list[str] = []
    invalid = 0
    for raw_id in unique_ids:
        try:
            valid_ids.append(validate_arxiv_id(raw_id))
        except ValueError:
            invalid += 1

    existing = {
        r[0] for r in db.execute(
            "SELECT arxiv_id FROM user_papers WHERE user_id = ?", (user["id"],)
        ).fetchall()
    }

    imported = 0
    rate_limited = 0
    daily_limit = _get_daily_limit(db, user["id"])
    recent = _count_recent_imports(db, user["id"])

    for arxiv_id in valid_ids:
        if arxiv_id in existing:
            continue
        if recent >= daily_limit:
            rate_limited += 1
            continue
        db.execute(
            "INSERT OR IGNORE INTO user_papers (user_id, arxiv_id, liked) VALUES (?, ?, 1)",
            (user["id"], arxiv_id),
        )
        db.execute(
            "INSERT INTO user_import_log (user_id, arxiv_id) VALUES (?, ?)",
            (user["id"], arxiv_id),
        )
        _ensure_ingest_enqueued(db, arxiv_id)
        imported += 1
        recent += 1

    db.commit()
    return {"imported": imported, "skipped": len(valid_ids) - imported - rate_limited, "rate_limited": rate_limited, "invalid": invalid}


# ---------------------------------------------------------------------------
# Update / delete a paper
# ---------------------------------------------------------------------------

class PatchPaperRequest(BaseModel):
    liked: Literal[-1, 0, 1]


@router.patch("/papers/{arxiv_id:path}")
def update_paper(
    arxiv_id: str,
    body: PatchPaperRequest,
    db: sqlite3.Connection = Depends(get_db),
    user=Depends(get_current_user),
):
    arxiv_id = _validate_arxiv_id(arxiv_id)
    db.execute(
        """
        INSERT INTO user_papers (user_id, arxiv_id, liked)
        VALUES (?, ?, ?)
        ON CONFLICT (user_id, arxiv_id) DO UPDATE SET liked = excluded.liked
        """,
        (user["id"], arxiv_id, body.liked),
    )
    _ensure_ingest_enqueued(db, arxiv_id)
    db.commit()
    return {"arxiv_id": arxiv_id, "liked": body.liked}


@router.delete("/papers/{arxiv_id:path}", status_code=status.HTTP_204_NO_CONTENT)
def delete_paper(
    arxiv_id: str,
    db: sqlite3.Connection = Depends(get_db),
    user=Depends(get_current_user),
):
    cur = db.execute(
        "DELETE FROM user_papers WHERE user_id = ? AND arxiv_id = ?",
        (user["id"], arxiv_id),
    )
    if cur.rowcount == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Paper not in your library.")
    db.commit()


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------

@router.get("/categories")
def get_categories(
    db: sqlite3.Connection = Depends(get_db),
    user=Depends(get_current_user),
):
    rows = db.execute(
        "SELECT category FROM user_categories WHERE user_id = ? ORDER BY category",
        (user["id"],),
    ).fetchall()
    return {"categories": [r["category"] for r in rows]}


class SetCategoriesRequest(BaseModel):
    categories: list[str]


@router.put("/categories")
@limiter.limit("15/minute")
@limiter.limit("200/day")
def set_categories(
    request: Request,
    body: SetCategoriesRequest,
    db: sqlite3.Connection = Depends(get_db),
    user=Depends(get_current_user),
):
    invalid = [c for c in body.categories if c not in ARXIV_CATEGORIES]
    if invalid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Unknown arXiv categories: {invalid}. "
                   f"Supported: {sorted(ARXIV_CATEGORIES)}",
        )
    db.execute("DELETE FROM user_categories WHERE user_id = ?", (user["id"],))
    db.executemany(
        "INSERT INTO user_categories (user_id, category) VALUES (?, ?)",
        [(user["id"], c) for c in body.categories],
    )
    db.commit()
    return {"categories": sorted(body.categories)}
