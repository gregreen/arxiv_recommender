#!/usr/bin/env python3
"""
Compute a 2D UMAP projection of all embedded papers and store the results in
app.db's paper_umap table.

Usage
-----
    python scripts/compute_umap.py
    python scripts/compute_umap.py --max-papers 2048 --dim 64 --seed 42
    python scripts/compute_umap.py --n-neighbors 15 --min-dist 0.1

The script samples up to --max-papers papers from all papers that have an
embedding, using 1/rank harmonic weighting (rank 1 = most recently published).
This gives a recency-biased sample with a long tail towards older papers.

The fitted UMAP reducer plus normalisation parameters are saved to
umap_model.joblib so that new papers can be projected on the fly without
re-running UMAP from scratch.

All papers in the last 30 days (and any user-liked papers) that are not part
of the training sample are projected using the saved reducer and stored in
paper_umap as well.

x and y coordinates are normalised to [0, 1] before storage.
The result is a complete replacement: all previous paper_umap rows are deleted
and the new ones inserted in a single transaction.

Exit codes
----------
    0  success
    1  not enough papers (fewer than --min-papers)
    2  umap-learn not installed
"""

import argparse
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np

try:
    import umap
except ImportError:
    print("Error: umap-learn is not installed.  Run: pip install umap-learn", file=sys.stderr)
    sys.exit(2)

try:
    import joblib
except ImportError:
    print("Error: joblib is not installed.  Run: pip install joblib", file=sys.stderr)
    sys.exit(2)

from arxiv_lib.config import APP_DB_PATH, EMBEDDING_CACHE_DB, RECOMMENDATION_EMBEDDING_DIM, UMAP_MODEL_PATH

_BATCH = 256


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute 2D UMAP projection of paper embeddings.")
    p.add_argument("--max-papers", type=int, default=2048,
                   help="Maximum number of papers to include in the UMAP training set (default: 2048).")
    p.add_argument("--min-papers", type=int, default=50,
                   help="Minimum papers needed; exit 1 if fewer are available (default: 50).")
    p.add_argument("--dim", type=int, default=64,
                   help="Truncate embeddings to this many dimensions before UMAP (default: 64).")
    p.add_argument("--n-neighbors", type=int, default=15,
                   help="UMAP n_neighbors parameter (default: 15).")
    p.add_argument("--min-dist", type=float, default=0.1,
                   help="UMAP min_dist parameter (default: 0.1).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility (default: 42).")
    return p.parse_args()


def load_candidate_ids(app_db_path: str, emb_db_path: str) -> list[str]:
    """
    Return arXiv IDs for all papers that have both metadata in app.db and an
    embedding in embeddings_cache.db, ordered by published_date DESC (newest
    first).  Papers with a NULL published_date are sorted last.
    """
    with sqlite3.connect(app_db_path) as app_con:
        app_con.execute("ATTACH DATABASE ? AS emb", (emb_db_path,))
        rows = app_con.execute(
            """
            SELECT p.arxiv_id
              FROM papers p
              JOIN emb.recommendation_embeddings e ON e.arxiv_id = p.arxiv_id
             ORDER BY p.published_date DESC NULLS LAST
            """
        ).fetchall()
    return [r[0] for r in rows]


def load_relevant_ids(app_db_path: str, emb_db_path: str, train_id_set: set[str]) -> list[str]:
    """
    Return arXiv IDs for papers that should be projected but are NOT in the
    training set: papers published in the last 30 days OR liked by any user,
    that have embeddings.
    """
    month_cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
    with sqlite3.connect(app_db_path) as app_con:
        app_con.execute("ATTACH DATABASE ? AS emb", (emb_db_path,))
        rows = app_con.execute(
            """
            SELECT DISTINCT p.arxiv_id
              FROM papers p
              JOIN emb.recommendation_embeddings e ON e.arxiv_id = p.arxiv_id
             WHERE p.published_date >= ?
                OR p.arxiv_id IN (
                       SELECT arxiv_id FROM user_papers WHERE liked = 1
                   )
            """,
            (month_cutoff,),
        ).fetchall()
    return [r[0] for r in rows if r[0] not in train_id_set]


def sample_ids(all_ids: list[str], n: int, rng: np.random.Generator) -> list[str]:
    """
    Sample up to n IDs using 1/rank harmonic weighting (rank 1 = most recent).
    """
    N = len(all_ids)
    n = min(n, N)
    ranks = np.arange(1, N + 1, dtype=float)
    weights = 1.0 / ranks
    weights /= weights.sum()
    chosen = rng.choice(N, size=n, replace=False, p=weights)
    return [all_ids[i] for i in chosen]


def load_embeddings(arxiv_ids: list[str], emb_db_path: str, dim: int) -> np.ndarray:
    """
    Load embedding vectors for the given IDs and truncate to `dim` dimensions.
    Returns a (len(arxiv_ids), dim) float32 matrix.
    """
    id_to_idx = {aid: i for i, aid in enumerate(arxiv_ids)}
    matrix = np.zeros((len(arxiv_ids), dim), dtype=np.float32)

    with sqlite3.connect(emb_db_path) as con:
        placeholders = ",".join("?" * len(arxiv_ids))
        rows = con.execute(
            f"SELECT arxiv_id, vector FROM recommendation_embeddings WHERE arxiv_id IN ({placeholders})",
            arxiv_ids,
        ).fetchall()

    for arxiv_id, blob in rows:
        vec = np.frombuffer(blob, dtype=np.float32)[:dim]
        if len(vec) < dim:
            vec = np.pad(vec, (0, dim - len(vec)))
        idx = id_to_idx[arxiv_id]
        matrix[idx] = vec

    return matrix


def load_embeddings_batch(batch_ids: list[str], emb_db_path: str, dim: int) -> tuple[list[str], np.ndarray]:
    """
    Load embeddings for a batch, returning only IDs that actually have embeddings.
    Returns (found_ids, matrix) where matrix has shape (len(found_ids), dim).
    """
    with sqlite3.connect(emb_db_path) as con:
        placeholders = ",".join("?" * len(batch_ids))
        rows = con.execute(
            f"SELECT arxiv_id, vector FROM recommendation_embeddings WHERE arxiv_id IN ({placeholders})",
            batch_ids,
        ).fetchall()

    found_ids = []
    vectors = []
    for arxiv_id, blob in rows:
        vec = np.frombuffer(blob, dtype=np.float32)[:dim]
        if len(vec) < dim:
            vec = np.pad(vec, (0, dim - len(vec)))
        found_ids.append(arxiv_id)
        vectors.append(vec)

    if not found_ids:
        return [], np.zeros((0, dim), dtype=np.float32)
    return found_ids, np.stack(vectors)


def write_results(arxiv_ids: list[str], coords: np.ndarray, app_db_path: str) -> None:
    """Replace all paper_umap rows with the new results in a single transaction."""
    computed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    rows = [(aid, float(coords[i, 0]), float(coords[i, 1]), computed_at)
            for i, aid in enumerate(arxiv_ids)]

    with sqlite3.connect(app_db_path) as con:
        con.execute("DELETE FROM paper_umap")
        con.executemany(
            "INSERT INTO paper_umap (arxiv_id, x, y, computed_at) VALUES (?, ?, ?, ?)",
            rows,
        )
        con.commit()

    print(f"Wrote {len(rows)} UMAP coordinates to paper_umap (computed_at={computed_at}).")


def main() -> None:
    args = parse_args()

    app_db = APP_DB_PATH()
    emb_db = EMBEDDING_CACHE_DB()
    model_path = UMAP_MODEL_PATH()

    print(f"Loading candidate IDs from {app_db} + {emb_db} …")
    all_ids = load_candidate_ids(app_db, emb_db)
    print(f"  {len(all_ids)} papers with embeddings found.")

    if len(all_ids) < args.min_papers:
        print(
            f"Error: only {len(all_ids)} papers available, need at least {args.min_papers}.",
            file=sys.stderr,
        )
        sys.exit(1)

    rng = np.random.default_rng(seed=args.seed)
    train_ids = sample_ids(all_ids, args.max_papers, rng)
    print(f"  Sampled {len(train_ids)} papers for UMAP training (max={args.max_papers}, seed={args.seed}).")

    print(f"Loading training embeddings (dim={args.dim}) …")
    train_matrix = load_embeddings(train_ids, emb_db, args.dim)

    print(f"Fitting UMAP (n_neighbors={args.n_neighbors}, min_dist={args.min_dist}) …")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.seed,
        low_memory=True,
    )
    reducer.fit(train_matrix)
    train_coords_raw = reducer.embedding_
    print(f"  UMAP fit complete.")

    # Find additional papers to project (last 30 days + any liked, not in training set)
    train_id_set = set(train_ids)
    relevant_ids = load_relevant_ids(app_db, emb_db, train_id_set)
    print(f"  {len(relevant_ids)} additional relevant papers to project.")

    # Transform relevant papers in batches of _BATCH
    relevant_coords_list: list[tuple[str, np.ndarray]] = []
    for i in range(0, len(relevant_ids), _BATCH):
        batch = relevant_ids[i:i + _BATCH]
        found_ids, batch_matrix = load_embeddings_batch(batch, emb_db, args.dim)
        if not found_ids:
            continue
        batch_coords = reducer.transform(batch_matrix)
        for j, aid in enumerate(found_ids):
            relevant_coords_list.append((aid, batch_coords[j]))
        print(f"  Projected batch {i // _BATCH + 1}/{(len(relevant_ids) + _BATCH - 1) // _BATCH} …")

    # Combine training + relevant coords to compute global normalisation bounds
    all_raw_coords_list = list(train_coords_raw)
    for _, c in relevant_coords_list:
        all_raw_coords_list.append(c)
    all_raw = np.stack(all_raw_coords_list) if all_raw_coords_list else train_coords_raw

    x_min = float(all_raw[:, 0].min())
    x_max = float(all_raw[:, 0].max())
    y_min = float(all_raw[:, 1].min())
    y_max = float(all_raw[:, 1].max())

    def _norm(raw: np.ndarray) -> np.ndarray:
        result = np.empty_like(raw)
        rx = x_max - x_min if x_max != x_min else 1.0
        ry = y_max - y_min if y_max != y_min else 1.0
        result[:, 0] = np.clip((raw[:, 0] - x_min) / rx, 0.0, 1.0)
        result[:, 1] = np.clip((raw[:, 1] - y_min) / ry, 0.0, 1.0)
        return result

    train_coords = _norm(train_coords_raw)

    # Save fitted reducer + normalisation params
    bundle = {
        "reducer": reducer,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "dim": args.dim,
    }
    joblib.dump(bundle, model_path)
    print(f"Saved UMAP model bundle to {model_path}.")

    # Build combined ids + coords arrays
    all_ids_out = list(train_ids)
    all_coords_out = list(train_coords)

    for aid, raw_c in relevant_coords_list:
        raw_2d = raw_c.reshape(1, 2)
        norm_c = _norm(raw_2d)[0]
        all_ids_out.append(aid)
        all_coords_out.append(norm_c)

    all_coords_array = np.stack(all_coords_out)
    write_results(all_ids_out, all_coords_array, app_db)


if __name__ == "__main__":
    main()
