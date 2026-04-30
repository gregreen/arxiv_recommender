#!/usr/bin/env python3
"""
Generate a low-resolution 2D projection plot for:
- papers in a selected time window (day/week/month), plus
- liked papers (liked=1) for one user.

The projection method is intentionally left unimplemented via project_to_2d().
Implement that function to try different projection methods.

Usage
-----
python scripts/project_recent_and_liked.py --user-id 1 --window week --output /tmp/proj.png
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np

from arxiv_lib.config import APP_DB_PATH, EMBEDDING_CACHE_DB, RECOMMEND_TIME_WINDOWS, RECOMMENDATION_EMBEDDING_DIM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a low-resolution 2D projection of recent papers + one user's liked papers."
    )
    parser.add_argument(
        "--user-id",
        type=int,
        required=True,
        help="User id whose liked papers (liked=1) should be overlaid.",
    )
    parser.add_argument(
        "--window",
        type=str,
        default="week",
        choices=RECOMMEND_TIME_WINDOWS,
        help="Time window for recent papers (default: week).",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=RECOMMENDATION_EMBEDDING_DIM,
        help=f"Embedding dimension used for projection input (default: {RECOMMENDATION_EMBEDDING_DIM}).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=3000,
        help="Maximum points to plot in low-resolution mode (default: 3000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling (default: 42).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="pacmap",
        choices=["umap", "pacmap", "svd"],
        help="Projection method to use (default: pacmap).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="projection.png",
        help="Output image path (default: projection.png).",
    )
    return parser.parse_args()


def window_cutoff(con: sqlite3.Connection, window: str) -> str:
    """Return ISO UTC cutoff anchored to latest published paper in DB."""
    deltas = {
        "day": timedelta(days=1),
        "week": timedelta(weeks=1),
        "month": timedelta(days=30),
    }
    if window not in deltas:
        raise ValueError(f"Unknown time window: {window!r}. Expected one of {RECOMMEND_TIME_WINDOWS}")

    row = con.execute(
        "SELECT MAX(published_date) FROM papers WHERE published_date IS NOT NULL"
    ).fetchone()

    if row and row[0]:
        raw = row[0].rstrip("Z")
        try:
            anchor = datetime.fromisoformat(raw)
        except ValueError:
            anchor = datetime.now(tz=timezone.utc)
        if anchor.tzinfo is None:
            anchor = anchor.replace(tzinfo=timezone.utc)
    else:
        anchor = datetime.now(tz=timezone.utc)

    cutoff = anchor - deltas[window] + timedelta(seconds=1)
    return cutoff.strftime("%Y-%m-%dT%H:%M:%S")


def load_window_ids(app_db_path: str, emb_db_path: str, cutoff: str) -> list[str]:
    """Load recent paper ids in the given window that have recommendation embeddings."""
    with sqlite3.connect(app_db_path) as con:
        con.execute("ATTACH DATABASE ? AS emb", (emb_db_path,))
        rows = con.execute(
            """
            SELECT p.arxiv_id
              FROM papers p
              JOIN emb.recommendation_embeddings e ON e.arxiv_id = p.arxiv_id
             WHERE p.published_date IS NOT NULL
               AND p.published_date >= ?
             ORDER BY p.published_date DESC, p.arxiv_id
            """,
            (cutoff,),
        ).fetchall()
    return [r[0] for r in rows]


def load_liked_ids(app_db_path: str, emb_db_path: str, user_id: int) -> list[str]:
    """Load liked paper ids for one user that have recommendation embeddings."""
    with sqlite3.connect(app_db_path) as con:
        con.execute("ATTACH DATABASE ? AS emb", (emb_db_path,))
        rows = con.execute(
            """
            SELECT up.arxiv_id
              FROM user_papers up
              JOIN emb.recommendation_embeddings e ON e.arxiv_id = up.arxiv_id
             WHERE up.user_id = ? AND up.liked = 1
             ORDER BY up.added_at DESC, up.arxiv_id
            """,
            (user_id,),
        ).fetchall()
    return [r[0] for r in rows]


def merge_ids(window_ids: list[str], liked_ids: list[str]) -> list[str]:
    """Merge ids with deterministic order: window first, then liked-only additions."""
    merged: list[str] = []
    seen: set[str] = set()
    for aid in window_ids + liked_ids:
        if aid in seen:
            continue
        merged.append(aid)
        seen.add(aid)
    return merged


def load_vectors(emb_db_path: str, arxiv_ids: list[str], dim: int) -> tuple[list[str], np.ndarray]:
    """Load vectors in requested id order; truncate/pad to dim."""
    if not arxiv_ids:
        return [], np.zeros((0, dim), dtype=np.float32)

    id_to_idx = {aid: i for i, aid in enumerate(arxiv_ids)}
    matrix = np.zeros((len(arxiv_ids), dim), dtype=np.float32)
    present = np.zeros((len(arxiv_ids),), dtype=bool)

    with sqlite3.connect(emb_db_path) as con:
        placeholders = ",".join("?" * len(arxiv_ids))
        rows = con.execute(
            f"SELECT arxiv_id, vector FROM recommendation_embeddings WHERE arxiv_id IN ({placeholders})",
            arxiv_ids,
        ).fetchall()

    for aid, blob in rows:
        vec = np.frombuffer(blob, dtype=np.float32)[:dim]
        if len(vec) < dim:
            vec = np.pad(vec, (0, dim - len(vec)))
        idx = id_to_idx[aid]
        matrix[idx] = vec
        present[idx] = True

    kept_ids = [aid for i, aid in enumerate(arxiv_ids) if present[i]]
    kept_matrix = matrix[present]
    return kept_ids, kept_matrix


def downsample_ids(ids: list[str], liked_set: set[str], max_points: int, seed: int) -> list[str]:
    """Keep all liked ids where possible; sample remaining ids deterministically."""
    if len(ids) <= max_points:
        return ids

    liked_ids = [aid for aid in ids if aid in liked_set]
    non_liked_ids = [aid for aid in ids if aid not in liked_set]

    if len(liked_ids) >= max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(liked_ids), size=max_points, replace=False)
        chosen = sorted(idx.tolist())
        return [liked_ids[i] for i in chosen]

    remaining = max_points - len(liked_ids)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(non_liked_ids), size=remaining, replace=False)
    chosen = sorted(idx.tolist())
    sampled_non_liked = [non_liked_ids[i] for i in chosen]
    return liked_ids + sampled_non_liked


def project_to_2d(vectors: np.ndarray, method: str = "pacmap") -> np.ndarray:
    """Project an (N, D) matrix to (N, 2)."""

    if method == "umap":
        return project_to_2d_umap(vectors)
    elif method == "pacmap":
        return project_to_2d_pacmap(vectors)
    elif method == "svd":
        return project_to_2d_svd(vectors)
    else:
        raise ValueError(f"Unknown projection method: {method}")


def project_to_2d_umap(vectors: np.ndarray) -> np.ndarray:
    """
    Project an (N, D) matrix to (N, 2) using UMAP.

    Note: UMAP is not deterministic and can produce different results across runs.
    """
    try:
        import umap as _umap
    except ImportError as exc:
        raise RuntimeError("umap-learn is required for the 'umap' projection method.") from exc

    reducer = _umap.UMAP(n_components=2, random_state=42)
    return reducer.fit_transform(vectors)


def project_to_2d_pacmap(vectors: np.ndarray) -> np.ndarray:
    """
    Project an (N, D) matrix to (N, 2) using PaCMAP.

    Note: PaCMAP is not deterministic and can produce different results across runs.
    """
    try:
        import pacmap as _pacmap
    except ImportError as exc:
        raise RuntimeError("pacmap is required for the 'pacmap' projection method.") from exc

    reducer = _pacmap.PaCMAP(n_components=2, apply_pca=True, random_state=42, num_iters=64, n_neighbors=10)
    return reducer.fit_transform(vectors)


def project_to_2d_svd(vectors: np.ndarray) -> np.ndarray:
    """
    Project an (N, D) matrix to (N, 2).

    This is a deterministic linear projection that captures the top 2 principal components.
    """
    if vectors.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {vectors.shape}.")

    n_samples, _ = vectors.shape
    if n_samples == 0:
        return np.zeros((0, 2), dtype=np.float32)

    # Center before SVD so the first components capture variance around the mean.
    x = vectors.astype(np.float64, copy=False)
    x_centered = x - x.mean(axis=0, keepdims=True)

    # Economy SVD: X = U S V^T. PCA scores in 2D are U[:, :2] * S[:2].
    u, s, _ = np.linalg.svd(x_centered, full_matrices=False)
    k = min(2, s.shape[0])

    coords = np.zeros((n_samples, 2), dtype=np.float64)
    if k > 0:
        coords[:, :k] = u[:, :k] * s[:k]

    return coords.astype(np.float32)


def normalize_01(coords: np.ndarray) -> np.ndarray:
    """Normalize (N,2) coords to [0,1] range per axis."""
    mn = coords.min(axis=0)
    mx = coords.max(axis=0)
    w = mx - mn
    return (coords - mn) / (w + 1e-12)


def plot_points(coords_01: np.ndarray, ids: list[str], liked_set: set[str], output_path: str, title: str) -> None:
    """Plot and save a low-resolution scatter image."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to generate the scatter plot.") from exc

    base_mask = np.array([aid not in liked_set for aid in ids], dtype=bool)
    liked_mask = ~base_mask

    fig, ax = plt.subplots(figsize=(9, 6), dpi=120)
    if np.any(base_mask):
        ax.scatter(
            coords_01[base_mask, 0],
            coords_01[base_mask, 1],
            s=8,
            c="#94a3b8",
            alpha=0.6,
            linewidths=0,
            label="Window papers",
        )
    if np.any(liked_mask):
        ax.scatter(
            coords_01[liked_mask, 0],
            coords_01[liked_mask, 1],
            s=12,
            c="#22c55e",
            alpha=0.9,
            linewidths=0,
            label="Liked papers",
        )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> int:
    args = parse_args()

    if args.max_points <= 0:
        print("Error: --max-points must be > 0.", file=sys.stderr)
        return 2
    if args.dim <= 1:
        print("Error: --dim must be > 1.", file=sys.stderr)
        return 2

    app_db = APP_DB_PATH()
    emb_db = EMBEDDING_CACHE_DB()

    with sqlite3.connect(app_db) as con:
        cutoff = window_cutoff(con, args.window)

    window_ids = load_window_ids(app_db, emb_db, cutoff)
    liked_ids = load_liked_ids(app_db, emb_db, args.user_id)

    merged_ids = merge_ids(window_ids, liked_ids)
    if not merged_ids:
        print("No candidate papers found for the selected window/user.", file=sys.stderr)
        return 1

    liked_set = set(liked_ids)

    sampled_ids = downsample_ids(merged_ids, liked_set, args.max_points, args.seed)

    kept_ids, vectors = load_vectors(emb_db, sampled_ids, args.dim)
    if len(kept_ids) == 0:
        print("No embeddings found for selected papers.", file=sys.stderr)
        return 1

    try:
        coords_raw = project_to_2d(vectors, method=args.method)
    except NotImplementedError as exc:
        print(str(exc), file=sys.stderr)
        return 3

    if not isinstance(coords_raw, np.ndarray):
        print("Error: project_to_2d must return a numpy.ndarray.", file=sys.stderr)
        return 2
    if coords_raw.shape != (len(kept_ids), 2):
        print(
            f"Error: project_to_2d returned shape {coords_raw.shape}, expected ({len(kept_ids)}, 2).",
            file=sys.stderr,
        )
        return 2

    coords = normalize_01(coords_raw)

    liked_in_plot = sum(1 for aid in kept_ids if aid in liked_set)
    title = (
        f"Low-res projection | user={args.user_id} | window={args.window} | "
        f"points={len(kept_ids)} | liked={liked_in_plot}"
    )

    plot_points(coords, kept_ids, liked_set, args.output, title)

    dropped = len(sampled_ids) - len(kept_ids)
    print("Projection plot generated.")
    print(f"  cutoff: {cutoff}")
    print(f"  window ids: {len(window_ids)}")
    print(f"  liked ids: {len(liked_ids)}")
    print(f"  merged ids: {len(merged_ids)}")
    print(f"  sampled ids: {len(sampled_ids)}")
    print(f"  embedded ids: {len(kept_ids)}")
    print(f"  dropped (missing embedding): {dropped}")
    print(f"  liked in plot: {liked_in_plot}")
    print(f"  output: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
