"""
Paper explorer endpoint.

GET /api/explore?window=day|week|month

Returns low-resolution projection coordinates for papers in the given time window, plus a
separate liked-overlay array (the user's 128 most recently liked papers,
regardless of window).

If projection data is missing, lowres_proj_available is False and both arrays are empty.
If the projection model is older than LOWRES_PROJ_STALE_DAYS days, a background thread is
started to recompute it (the response is still returned immediately).

On each request:
 1. Candidate IDs are collected (window papers + liked overlay).
 2. ensure_lowres_proj_coords() projects any IDs not yet in paper_lowres_proj (or stale vs.
    the model file).
 3. score_papers_for_explore() fills in recommendation scores for unscored
    papers (no-op if the user has too few liked papers).
 4. The main query JOINs paper_lowres_proj and LEFT JOINs recommendations for scores.
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

from arxiv_lib.config import LOWRES_PROJ_MODEL_PATH, RECOMMEND_TIME_WINDOWS
from arxiv_lib.lowres_proj_utils import ensure_lowres_proj_coords
from arxiv_lib.recommend import NotEnoughDataError, _window_cutoff, score_papers_for_explore
from web.dependencies import get_current_user, get_db
from web.limiter import limiter

router = APIRouter(prefix="/explore", tags=["explore"])
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOWRES_PROJ_STALE_DAYS = 7
LIKED_OVERLAY_LIMIT = 128
WINDOW_PAPERS_LIMIT = 5000

_SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"

# ---------------------------------------------------------------------------
# Background recompute
# ---------------------------------------------------------------------------

_recompute_lock = threading.Lock()


def _trigger_recompute() -> None:
    """Run compute_lowres_proj.py in a background thread (non-blocking, non-concurrent)."""
    if not _recompute_lock.acquire(blocking=False):
        log.debug("explore: lowres projection recompute already in progress, skipping.")
        return
    try:
        script = _SCRIPTS_DIR / "compute_lowres_proj.py"
        log.info("explore: starting background lowres projection recompute via %s", script)
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            log.info("explore: lowres projection recompute finished successfully.")
        else:
            log.error(
                "explore: lowres projection recompute failed (exit %d):\n%s",
                result.returncode, result.stderr,
            )
    finally:
        _recompute_lock.release()


def _model_mtime_iso() -> str | None:
    """Return the ISO timestamp of the projection model file, or None if absent."""
    path = LOWRES_PROJ_MODEL_PATH()
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

    # Check whether the projection model file exists and how fresh it is.
    # Fall back to MAX(computed_at) from paper_lowres_proj if no model file is present.
    lowres_proj_model_ts = _model_mtime_iso()
    if lowres_proj_model_ts is None:
        db_row = db.execute("SELECT MAX(computed_at) FROM paper_lowres_proj").fetchone()
        lowres_proj_model_ts = db_row[0] if db_row else None
    lowres_proj_available = lowres_proj_model_ts is not None

    # Trigger background recompute if stale (but don't block the response).
    if lowres_proj_available:
        try:
            model_dt = datetime.fromisoformat(lowres_proj_model_ts)  # type: ignore[arg-type]
            if model_dt.tzinfo is None:
                model_dt = model_dt.replace(tzinfo=timezone.utc)
            age = datetime.now(timezone.utc) - model_dt
            if age > timedelta(days=LOWRES_PROJ_STALE_DAYS):
                log.info(
                    "explore: lowres projection model is %.1f days old (threshold %d), triggering background recompute.",
                    age.total_seconds() / 86400, LOWRES_PROJ_STALE_DAYS,
                )
                t = threading.Thread(target=_trigger_recompute, daemon=True)
                t.start()
        except (ValueError, TypeError):
            log.warning("explore: could not parse lowres_proj_model_ts=%r", lowres_proj_model_ts)

    if not lowres_proj_available:
        # Nothing to show yet; kick off first-time computation.
        t = threading.Thread(target=_trigger_recompute, daemon=True)
        t.start()
        return {
            "papers":          [],
            "liked_overlay":   [],
            "lowres_proj_computed_at": None,
            "lowres_proj_available":  False,
        }

    # ------------------------------------------------------------------
    # Pass 1: collect candidate IDs for coord projection + scoring
    # ------------------------------------------------------------------
    # Use the same anchor logic as the recommendations endpoint.
    cutoff = _window_cutoff(window, db)

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

    # Ensure lowres coords are current for all relevant papers (blocking, fast).
    all_candidate_ids = list(dict.fromkeys(window_ids + liked_ids))  # deduplicated
    log.info(
        "explore: window=%s cutoff=%s window_ids=%d liked_ids=%d total_candidates=%d",
        window, cutoff, len(window_ids), len(liked_ids), len(all_candidate_ids),
    )
    ensure_lowres_proj_coords(db, all_candidate_ids)
    if window_ids:
        placeholders = ",".join("?" * len(window_ids))
        n_coords = db.execute(
            f"SELECT COUNT(*) FROM paper_lowres_proj WHERE arxiv_id IN ({placeholders})",
            window_ids,
        ).fetchone()[0]
        log.info("explore: after ensure_lowres_proj_coords: %d/%d window_ids have coords.", n_coords, len(window_ids))

    # Fill in any missing recommendation scores (no-op if not enough liked papers).
    # Pass liked_ids as extra_ids so papers older than the window cutoff also get scored.
    try:
        score_papers_for_explore(db, user["id"], window, extra_ids=liked_ids)
    except NotEnoughDataError:
        pass  # No model yet — scores will be NULL in the response

    # ------------------------------------------------------------------
    # Pass 2: main query with lowres coords + scores
    # ------------------------------------------------------------------
    papers_rows = db.execute(
        """
        SELECT p.arxiv_id, p.title, p.published_date,
               u.x, u.y,
               COALESCE(ul.liked, 0) AS liked,
               r.score
          FROM papers p
          JOIN paper_lowres_proj u   ON u.arxiv_id = p.arxiv_id
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
    log.info("explore: main query returned %d papers for window=%s.", len(papers), window)

    # Liked overlay: 128 most recently liked papers (regardless of window).
    overlay_rows = db.execute(
        """
        SELECT ul.arxiv_id, p.title,
               u.x, u.y,
               r.score
          FROM user_papers ul
          JOIN papers p       ON p.arxiv_id = ul.arxiv_id
          JOIN paper_lowres_proj u   ON u.arxiv_id = ul.arxiv_id
          LEFT JOIN recommendations r
                 ON r.arxiv_id = ul.arxiv_id AND r.user_id = ? AND r.time_window = ?
         WHERE ul.user_id = ? AND ul.liked = 1
         ORDER BY ul.added_at DESC
         LIMIT ?
        """,
        (user["id"], window, user["id"], LIKED_OVERLAY_LIMIT),
    ).fetchall()

    liked_overlay = [
        {
            "arxiv_id": r["arxiv_id"],
            "title":    r["title"] or "",
            "x":        r["x"],
            "y":        r["y"],
            "score":    r["score"],
        }
        for r in overlay_rows
    ]

    return {
        "papers":           papers,
        "liked_overlay":    liked_overlay,
        "lowres_proj_computed_at": lowres_proj_model_ts,
        "lowres_proj_available":   True,
    }
