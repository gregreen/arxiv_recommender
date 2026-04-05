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

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, field_validator

from arxiv_lib.appdb import enqueue_task
from arxiv_lib.config import ARXIV_CATEGORIES
from web.dependencies import get_current_user, get_db

router = APIRouter(prefix="/users/me", tags=["users"])

_NEW_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
_OLD_ID_RE = re.compile(r"^[a-z][a-z-]*(\.[A-Z]{2})?/\d{7}(v\d+)?$")
_ADS_IMPORT_LIMIT = 64


def _validate_arxiv_id(arxiv_id: str) -> str:
    """Strip version suffix and validate format. Returns clean ID or raises 422."""
    stripped = arxiv_id.strip()
    clean = re.sub(r"v\d+$", "", stripped)
    if not (_NEW_ID_RE.match(stripped) or _OLD_ID_RE.match(stripped)):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid arXiv ID format: {arxiv_id!r}",
        )
    return clean


def _ensure_fetch_meta_enqueued(db: sqlite3.Connection, arxiv_id: str) -> None:
    """Enqueue a fetch_meta task if the paper is not yet in the papers table."""
    exists = db.execute(
        "SELECT 1 FROM papers WHERE arxiv_id = ?", (arxiv_id,)
    ).fetchone()
    if not exists:
        enqueue_task(db, "fetch_meta", {"arxiv_id": arxiv_id})


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
    db.execute(
        "INSERT OR IGNORE INTO user_papers (user_id, arxiv_id, liked) VALUES (?, ?, ?)",
        (user["id"], arxiv_id, body.liked),
    )
    _ensure_fetch_meta_enqueued(db, arxiv_id)
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
    found_ids = re.findall(r"arXiv:((?:\d{4}\.\d{4,5}|[a-z][a-z-]*(?:\.[A-Z]{2})?/\d{7}))(?:v\d+)?", body.text)
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_ids = [aid for aid in found_ids if not (aid in seen or seen.add(aid))]  # type: ignore[func-returns-value]

    if len(unique_ids) > _ADS_IMPORT_LIMIT:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Too many IDs ({len(unique_ids)}). Maximum is {_ADS_IMPORT_LIMIT} per import.",
        )

    existing = {
        r[0] for r in db.execute(
            "SELECT arxiv_id FROM user_papers WHERE user_id = ?", (user["id"],)
        ).fetchall()
    }

    imported = 0
    for arxiv_id in unique_ids:
        if arxiv_id in existing:
            continue
        db.execute(
            "INSERT OR IGNORE INTO user_papers (user_id, arxiv_id, liked) VALUES (?, ?, 1)",
            (user["id"], arxiv_id),
        )
        _ensure_fetch_meta_enqueued(db, arxiv_id)
        imported += 1

    db.commit()
    return {"imported": imported, "skipped": len(unique_ids) - imported}


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
    _ensure_fetch_meta_enqueued(db, arxiv_id)
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
def set_categories(
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
