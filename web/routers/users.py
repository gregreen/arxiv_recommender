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
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
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
from web.dependencies import get_current_user, get_db

router = APIRouter(prefix="/users/me", tags=["users"])

_ADS_IMPORT_LIMIT = 64


def _validate_arxiv_id(arxiv_id: str) -> str:
    """Semantically validate and canonicalise an arXiv ID. Raises 422 on failure."""
    from arxiv_lib.arxiv_id import validate_arxiv_id

    try:
        return validate_arxiv_id(arxiv_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
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

    if len(unique_ids) > _ADS_IMPORT_LIMIT:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Too many IDs ({len(unique_ids)}). Maximum is {_ADS_IMPORT_LIMIT} per import.",
        )

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
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
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
