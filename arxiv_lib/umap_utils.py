"""
Utilities for on-the-fly UMAP coordinate projection.

The fitted UMAP model (reducer + normalisation params) is loaded from the
joblib bundle written by scripts/compute_umap.py.  The bundle is cached
in-process and reloaded only when the file's mtime changes.

Public API
----------
ensure_umap_coords(con, arxiv_ids)
    Project any IDs that are missing from paper_umap or whose coords are older
    than the current model, and write the results to paper_umap.
"""

import logging
import os
import sqlite3
from datetime import datetime, timezone

import numpy as np

from arxiv_lib.config import EMBEDDING_CACHE_DB, UMAP_MODEL_PATH

logger = logging.getLogger(__name__)

_BATCH = 256

# Module-level cache: avoids re-loading the joblib file on every request.
_cache: dict = {"bundle": None, "mtime": 0.0}


def _load_bundle() -> dict | None:
    """Return the cached bundle, reloading from disk only if mtime changed."""
    path = UMAP_MODEL_PATH()
    if not os.path.exists(path):
        return None
    mtime = os.path.getmtime(path)
    if _cache["bundle"] is None or _cache["mtime"] != mtime:
        try:
            import joblib
            _cache["bundle"] = joblib.load(path)
            _cache["mtime"] = mtime
            logger.debug("Loaded UMAP model bundle from %s (mtime=%s)", path, mtime)
        except Exception:
            logger.exception("Failed to load UMAP model bundle from %s", path)
            return None
    return _cache["bundle"]


def _load_embeddings_batch(
    batch_ids: list[str], emb_db_path: str, dim: int
) -> tuple[list[str], np.ndarray]:
    """
    Load embeddings for a list of IDs, returning only the IDs that have
    embeddings (and the corresponding matrix rows).
    """
    with sqlite3.connect(emb_db_path) as con:
        placeholders = ",".join("?" * len(batch_ids))
        rows = con.execute(
            f"SELECT arxiv_id, vector FROM recommendation_embeddings"
            f" WHERE arxiv_id IN ({placeholders})",
            batch_ids,
        ).fetchall()

    found_ids: list[str] = []
    vectors: list[np.ndarray] = []
    for arxiv_id, blob in rows:
        vec = np.frombuffer(blob, dtype=np.float32)[:dim]
        if len(vec) < dim:
            vec = np.pad(vec, (0, dim - len(vec)))
        found_ids.append(arxiv_id)
        vectors.append(vec)

    if not found_ids:
        return [], np.zeros((0, dim), dtype=np.float32)
    return found_ids, np.stack(vectors)


def ensure_umap_coords(con: sqlite3.Connection, arxiv_ids: list[str]) -> None:
    """
    Project any IDs in *arxiv_ids* that are missing from paper_umap or whose
    stored coords pre-date the current UMAP model, and write the results to
    paper_umap.

    This is a no-op if no UMAP model file exists.

    Parameters
    ----------
    con : sqlite3.Connection
        Open app.db connection (read/write).
    arxiv_ids : list[str]
        The arXiv IDs to ensure have up-to-date coords.
    """
    if not arxiv_ids:
        return
    bundle = _load_bundle()
    if bundle is None:
        return

    # Derive a timestamp string for the model file so we can compare it against
    # computed_at stored in paper_umap.
    model_mtime = _cache["mtime"]
    model_ts = datetime.utcfromtimestamp(model_mtime).strftime("%Y-%m-%dT%H:%M:%S")

    # Find which IDs need (re-)projecting: missing or computed before the model.
    placeholders = ",".join("?" * len(arxiv_ids))
    up_to_date = {
        r[0]
        for r in con.execute(
            f"SELECT arxiv_id FROM paper_umap"
            f" WHERE arxiv_id IN ({placeholders}) AND computed_at >= ?",
            (*arxiv_ids, model_ts),
        ).fetchall()
    }
    todo = [aid for aid in arxiv_ids if aid not in up_to_date]
    if not todo:
        return

    logger.debug("ensure_umap_coords: projecting %d IDs", len(todo))

    reducer = bundle["reducer"]
    x_min: float = bundle["x_min"]
    x_max: float = bundle["x_max"]
    y_min: float = bundle["y_min"]
    y_max: float = bundle["y_max"]
    dim: int = bundle.get("dim", 64)
    emb_db = EMBEDDING_CACHE_DB()

    rx = (x_max - x_min) if x_max != x_min else 1.0
    ry = (y_max - y_min) if y_max != y_min else 1.0

    computed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    rows_to_write: list[tuple[str, float, float, str]] = []

    for i in range(0, len(todo), _BATCH):
        batch = todo[i : i + _BATCH]
        found_ids, matrix = _load_embeddings_batch(batch, emb_db, dim)
        if not found_ids:
            continue
        try:
            raw_coords = reducer.transform(matrix)  # (n, 2)
        except Exception:
            logger.exception("UMAP transform failed for batch starting at index %d", i)
            continue
        for j, aid in enumerate(found_ids):
            nx = float(np.clip((raw_coords[j, 0] - x_min) / rx, 0.0, 1.0))
            ny = float(np.clip((raw_coords[j, 1] - y_min) / ry, 0.0, 1.0))
            rows_to_write.append((aid, nx, ny, computed_at))

    if rows_to_write:
        con.executemany(
            "INSERT OR REPLACE INTO paper_umap (arxiv_id, x, y, computed_at) VALUES (?, ?, ?, ?)",
            rows_to_write,
        )
        logger.debug("ensure_umap_coords: wrote %d rows", len(rows_to_write))
