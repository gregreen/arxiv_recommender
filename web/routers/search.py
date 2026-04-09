"""
Search endpoint.

POST /api/search
"""

import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from arxiv_lib.config import ONBOARDING_BROWSE_LIMIT, RECOMMEND_TIME_WINDOWS
from arxiv_lib.search import SearchEmbeddingError, search_papers
from web.dependencies import get_current_user, get_db
from web.limiter import limiter

router = APIRouter(prefix="/search", tags=["search"])


class SearchRequest(BaseModel):
    query: str
    window: str = "week"


@router.post("")
@limiter.limit("100/hour")
@limiter.limit("10/minute")
def search(
    request: Request,
    body: SearchRequest,
    db: sqlite3.Connection = Depends(get_db),
    user=Depends(get_current_user),
):
    if body.window not in RECOMMEND_TIME_WINDOWS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid window {body.window!r}. Must be one of: {RECOMMEND_TIME_WINDOWS}",
        )

    try:
        results = search_papers(
            db, user["id"], body.query, body.window, ONBOARDING_BROWSE_LIMIT
        )
    except SearchEmbeddingError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Embedding service unavailable: {exc}",
        ) from exc

    return {"count": len(results), "results": results}
