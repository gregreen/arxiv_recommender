"""
Recommendations endpoint.

GET /api/recommendations?window=day|week|month
"""

import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Query, status

from arxiv_lib.config import RECOMMEND_TIME_WINDOWS
from arxiv_lib.recommend import NotEnoughDataError, get_recommendations
from web.dependencies import get_current_user, get_db

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.get("")
def recommendations(
    window: str = Query(default="week", description="Time window: day, week, or month"),
    db: sqlite3.Connection = Depends(get_db),
    user=Depends(get_current_user),
):
    if window not in RECOMMEND_TIME_WINDOWS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid window {window!r}. Must be one of: {RECOMMEND_TIME_WINDOWS}",
        )

    try:
        results = get_recommendations(db, user["id"], window)
    except NotEnoughDataError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )

    generated_at = results[0]["generated_at"] if results else None
    return {
        "window":       window,
        "count":        len(results),
        "generated_at": generated_at,
        "results":      results,
    }
