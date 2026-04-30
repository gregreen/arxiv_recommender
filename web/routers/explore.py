"""
Paper explorer endpoint.

GET /api/explore?window=day|week|month

Returns the UMAP coordinates for papers in the given time window, plus a
separate liked-overlay array (the user's 128 most recently liked papers,
regardless of window).

If the UMAP data is missing, umap_available is False and both arrays are empty.
If the UMAP model is older than UMAP_STALE_DAYS days, a background thread is
started to recompute it (the response is still returned immediately).

On each request:
 1. Candidate IDs are collected (window papers + liked overlay).
 2. ensure_umap_coords() projects any IDs not yet in paper_umap (or stale vs.
    the model file).
 3. score_papers_for_explore() fills in recommendation scores for unscored
    papers (no-op if the user has too few liked papers).
 4. The main query JOINs paper_umap and LEFT JOINs recommendations for scores.
"""

import logging
import os
import sqlite3
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, Query, Request

from arxiv_lib.config import RECOMMEND_TIME_WINDOWS, UMAP_MODEL_PATH
from arxiv_lib.umap_utils import ensure_umap_coords
from arxiv_lib.recommend import NotEnoughDataError, score_papers_for_explore
from web.dependencies import get_current_user, get_db
from web.limiter import limiter

router = APIRouter(prefix="/explore", tags=["explore"])
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UMAP_STALE_DAYS = 7
LIKED_OVERLAY_LIMIT = 128
WINDOW_PAPERS_LIMIT = 5000

_WINDOW_DELTAS = {
    "day":   timedelta(days=1),
    "week":  timedelta(weeks=1),
    "month": timedelta(days=30),
}

_SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"

# ---------------------------------------------------------------------------
# Background recompute
# ---------------------------------------------------------------------------

_recompute_lock = threading.Lock()


def _trigger_recompute() -> None:
    """Run compute_umap.py in a background thread (non-blocking, non-concurrent)."""
    if not _recompute_lock.acquire(blocking=False):
        log.debug("explore: UMAP recompute already in progress, skipping.")
        return
    try:
        script = _SCRIPTS_DIR / "compute_umap.py"
        log.info("explore: starting background UMAP recompute via %s", script)
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            log.info("explore: UMAP recompute finished successfully.")
        else:
            log.error(
                "explore: UMAP recompute failed (exit %d):\n%s",
                result.returncode, result.stderr,
            )
    finally:
        _recompute_lock.release()


def _model_mtime_iso() -> str | None:
    """Return the ISO timestamp of the UMAP model file, or None if it doesn't exist."""
    path = UMAP_MODEL_PATH()
    try:
        mtime = os.path.getmtime(path)
        return datetime.utcfromtimestamp(mtime).strftime("%Y-%m-%dT%H:%M:%S")
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.get("")
@limiter.limit("60/minute")
def explore(
    request: Request,
    window: str = Query(default="week", description="Time window: day, week, or month"),
    db: sqlite3.Connection = Depends(get_db),
    user=Depends(get_current_user),
):
    if window not in RECOMMEND_TIME_WINDOWS:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid window {window!r}. Must be one of: {list(RECOMMEND_TIME_WINDOWS)}",
        )

    # Check whether the UMAP model file exists and how fresh it is.
    # Fall back to MAX(computed_at) from paper_umap if no model file is present
    # (supports legacy mode when compute_umap.py hasn't been run yet).
    umap_model_ts = _model_mtime_iso()
    if umap_model_ts is None:
        db_row = db.execute("SELECT MAX(computed_at) FROM paper_umap").fetchone()
        umap_model_ts = db_row[0] if db_row else None
    umap_available = umap_model_ts is not None

    # Trigger background recompute if stale (but don't block the response).
    if umap_available:
        try:
            model_dt = datetime.fromisoformat(umap_model_ts)  # type: ignore[arg-type]
            if model_dt.tzinfo is None:
                model_dt = model_dt.replace(tzinfo=timezone.utc)
            age = datetime.now(timezone.utc) - model_dt
            if age > timedelta(days=UMAP_STALE_DAYS):
                log.info(
                    "explore: UMAP model is %.1f days old (threshold %d), triggering background recompute.",
                    age.total_seconds() / 86400, UMAP_STALE_DAYS,
                )
                t = threading.Thread(target=_trigger_recompute, daemon=True)
                t.start()
        except (ValueError, TypeError):
            log.warning("explore: could not parse umap_model_ts=%r", umap_model_ts)

    if not umap_available:
        # Nothing to show yet; kick off first-time computation.
        t = threading.Thread(target=_trigger_recompute, daemon=True)
        t.start()
        return {
            "papers":          [],
            "liked_overlay":   [],
            "umap_computed_at": None,
            "umap_available":  False,
        }

    # ------------------------------------------------------------------
    # Pass 1: collect candidate IDs for coord projection + scoring
    # ------------------------------------------------------------------
    cutoff = (datetime.now(timezone.utc) - _WINDOW_DELTAS[window]).strftime("%Y-%m-%d")

    window_id_rows = db.execute(
        "SELECT arxiv_id FROM papers WHERE published_date >= ? ORDER BY published_date DESC LIMIT ?",
        (cutoff, WINDOW_PAPERS_LIMIT),
    ).fetchall()
    window_ids = [r[0] for r in window_id_rows]

    liked_id_rows = db.execute(
        "SELECT arxiv_id FROM user_papers WHERE user_id = ? AND liked = 1 ORDER BY added_at DESC LIMIT ?",
        (user["id"], LIKED_OVERLAY_LIMIT),
    ).fetchall()
    liked_ids = [r[0] for r in liked_id_rows]

    # Ensure UMAP coords are current for all relevant papers (blocking, fast).
    all_candidate_ids = list(dict.fromkeys(window_ids + liked_ids))  # deduplicated
    ensure_umap_coords(db, all_candidate_ids)

    # Fill in any missing recommendation scores (no-op if not enough liked papers).
    try:
        score_papers_for_explore(db, user["id"], window)
    except NotEnoughDataError:
        pass  # No model yet — scores will be NULL in the response

    # ------------------------------------------------------------------
    # Pass 2: main query with UMAP coords + scores
    # ------------------------------------------------------------------
    papers_rows = db.execute(
        """
        SELECT p.arxiv_id, p.title, p.published_date,
               u.x, u.y,
               COALESCE(ul.liked, 0) AS liked,
               r.score
          FROM papers p
          JOIN paper_umap u   ON u.arxiv_id = p.arxiv_id
          LEFT JOIN user_papers ul
                 ON ul.arxiv_id = p.arxiv_id AND ul.user_id = ?
          LEFT JOIN recommendations r
                 ON r.arxiv_id = p.arxiv_id AND r.user_id = ? AND r.time_window = ?
         WHERE p.published_date >= ?
         ORDER BY p.published_date DESC
         LIMIT ?
        """,
        (user["id"], user["id"], window, cutoff, WINDOW_PAPERS_LIMIT),
    ).fetchall()

    papers = [
        {
            "arxiv_id":       r["arxiv_id"],
            "title":          r["title"] or "",
            "published_date": r["published_date"],
            "x":              r["x"],
            "y":              r["y"],
            "liked":          r["liked"],
            "score":          r["score"],
        }
        for r in papers_rows
    ]

    # Liked overlay: 128 most recently liked papers (regardless of window).
    overlay_rows = db.execute(
        """
        SELECT ul.arxiv_id, p.title,
               u.x, u.y
          FROM user_papers ul
          JOIN papers p       ON p.arxiv_id = ul.arxiv_id
          JOIN paper_umap u   ON u.arxiv_id = ul.arxiv_id
         WHERE ul.user_id = ? AND ul.liked = 1
         ORDER BY ul.added_at DESC
         LIMIT ?
        """,
        (user["id"], LIKED_OVERLAY_LIMIT),
    ).fetchall()

    liked_overlay = [
        {
            "arxiv_id": r["arxiv_id"],
            "title":    r["title"] or "",
            "x":        r["x"],
            "y":        r["y"],
        }
        for r in overlay_rows
    ]

    return {
        "papers":           papers,
        "liked_overlay":    liked_overlay,
        "umap_computed_at": umap_model_ts,
        "umap_available":   True,
    }
