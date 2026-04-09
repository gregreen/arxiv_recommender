"""
Search endpoint.

POST /api/search
"""

import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from arxiv_lib.config import RECOMMEND_TIME_WINDOWS
from arxiv_lib.search import SearchEmbeddingError, search_papers
from web.dependencies import get_current_user, get_db
from web.limiter import limiter

router = APIRouter(prefix="/search", tags=["search"])


class SearchRequest(BaseModel):
    query: str


@router.post("")
@limiter.limit("100/hour")
@limiter.limit("10/minute")
def search(
    request: Request,
    body: SearchRequest,
    db: sqlite3.Connection = Depends(get_db),
    user=Depends(get_current_user),
):
    try:
        results = search_papers(db, user["id"], body.query)
    except SearchEmbeddingError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Embedding service unavailable: {exc}",
        ) from exc

    return results
