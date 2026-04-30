"""
Search endpoint.

POST /api/search
"""

import logging
import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from arxiv_lib.arxiv_id import validate_arxiv_id
from arxiv_lib.config import RECOMMEND_TIME_WINDOWS
from arxiv_lib.search import SearchEmbeddingError, lookup_paper_by_id, search_papers
from web.dependencies import get_current_user, get_db
from web.limiter import limiter

router = APIRouter(prefix="/search", tags=["search"])
log = logging.getLogger(__name__)


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
    # If the query looks like an arXiv ID, do a direct DB lookup instead of
    # a semantic search.
    try:
        canonical_id = validate_arxiv_id(body.query.strip())
        paper = lookup_paper_by_id(db, user["id"], canonical_id)
        return {"kind": "id_lookup", "arxiv_id": canonical_id, "paper": paper}
    except ValueError:
        pass  # not an arXiv ID — fall through to semantic search

    try:
        results = search_papers(db, user["id"], body.query)
    except SearchEmbeddingError as exc:
        log.error("search: embedding service error for user %s: %s", user["id"], exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding service temporarily unavailable.",
        ) from exc

    return {"kind": "semantic", **results}
