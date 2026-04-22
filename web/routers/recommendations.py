"""
Recommendations endpoint.

GET /api/recommendations?window=day|week|month
"""

import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from arxiv_lib.config import ONBOARDING_BROWSE_LIMIT, RECOMMEND_TIME_WINDOWS
from arxiv_lib.recommend import NotEnoughDataError, get_onboarding_papers, get_recommendations
from web.dependencies import get_current_user, get_db
from web.limiter import limiter

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.get("")
@limiter.limit("30/minute")
def recommendations(
    request: Request,
    window: str = Query(default="week", description="Time window: day, week, or month"),
    db: sqlite3.Connection = Depends(get_db),
    user=Depends(get_current_user),
):
    if window not in RECOMMEND_TIME_WINDOWS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid window {window!r}. Must be one of: {RECOMMEND_TIME_WINDOWS}",
        )

    onboarding = False
    message = None
    try:
        results = get_recommendations(db, user["id"], window)
    except NotEnoughDataError:
        results = get_onboarding_papers(db, window, ONBOARDING_BROWSE_LIMIT, seed=user["id"])
        onboarding = True
        message = "Not enough data to generate recommendations yet. Mark papers as relevant or add papers to your Library."

    generated_at = next((r["generated_at"] for r in results if r["generated_at"]), None)
    return {
        "window":       window,
        "count":        len(results),
        "generated_at": generated_at,
        "onboarding":   onboarding,
        "message":      message,
        "results":      results,
    }
