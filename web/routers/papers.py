"""
Paper detail endpoint.

GET /api/papers/{arxiv_id}  — metadata + LLM summary for a single paper
"""

import json
import os
import sqlite3

from fastapi import APIRouter, Depends, HTTPException, status

from arxiv_lib.config import SUMMARY_CACHE_DIR
from web.dependencies import get_current_user, get_db

router = APIRouter(prefix="/papers", tags=["papers"])


@router.get("/{arxiv_id:path}")
def get_paper(
    arxiv_id: str,
    db: sqlite3.Connection = Depends(get_db),
    _user=Depends(get_current_user),
):
    row = db.execute(
        "SELECT arxiv_id, title, abstract, authors, published_date, categories"
        "  FROM papers WHERE arxiv_id = ?",
        (arxiv_id,),
    ).fetchone()

    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Paper not found.")

    summary_cache_id = arxiv_id.replace("/", "_")
    summary_path = os.path.join(SUMMARY_CACHE_DIR(), f"{summary_cache_id}.txt")
    summary = None
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = f.read()

    return {
        "arxiv_id":       row["arxiv_id"],
        "title":          row["title"],
        "abstract":       row["abstract"],
        "authors":        json.loads(row["authors"]) if row["authors"] else [],
        "published_date": row["published_date"],
        "categories":     json.loads(row["categories"]) if row["categories"] else [],
        "summary":        summary,
        "url":            f"https://arxiv.org/abs/{arxiv_id}",
    }
