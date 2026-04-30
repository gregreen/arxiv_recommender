#!/usr/bin/env python3
"""
Compute a 2D low-resolution projection of paper embeddings and store the
results in app.db's paper_lowres_proj table.

Current method: PaCMAP.

Usage
-----
    python scripts/compute_lowres_proj.py
    python scripts/compute_lowres_proj.py --max-papers 2048 --dim 64 --seed 42

The script samples up to --max-papers papers from all papers that have an
embedding, using 1/rank harmonic weighting (rank 1 = most recently published).
This gives a recency-biased sample with a long tail towards older papers.

The fitted projection model plus normalisation parameters are saved to
lowres_proj_model.joblib so that new papers can be projected on the fly.

All papers in the last 30 days (and any user-liked papers) that are not part
of the training sample are projected using the saved model and stored in
paper_lowres_proj as well.

x and y coordinates are normalised to [0, 1] before storage.
The result is a complete replacement: all previous paper_lowres_proj rows are
deleted and the new ones inserted in a single transaction.
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
    import pacmap
except ImportError:
    print("Error: pacmap is not installed. Run: pip install pacmap", file=sys.stderr)
    sys.exit(2)

try:
    import joblib
except ImportError:
    print("Error: joblib is not installed. Run: pip install joblib", file=sys.stderr)
    sys.exit(2)

from arxiv_lib.config import APP_DB_PATH, EMBEDDING_CACHE_DB, LOWRES_PROJ_MODEL_PATH, RECOMMENDATION_EMBEDDING_DIM

_BATCH = 256


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute 2D lowres projection of paper embeddings (PaCMAP).")
    p.add_argument("--max-papers", type=int, default=2048,
                   help="Maximum number of papers to include in training set (default: 2048).")
    p.add_argument("--min-papers", type=int, default=50,
                   help="Minimum papers needed; exit 1 if fewer are available (default: 50).")
    p.add_argument("--dim", type=int, default=64,
                   help="Truncate embeddings to this many dimensions before projection (default: 64).")
    p.add_argument("--n-neighbors", type=int, default=10,
                   help="PaCMAP n_neighbors parameter (default: 10).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility (default: 42).")
    return p.parse_args()


def load_candidate_ids(app_db_path: str, emb_db_path: str) -> list[str]:
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
    N = len(all_ids)
    n = min(n, N)
    ranks = np.arange(1, N + 1, dtype=float)
    weights = 1.0 / ranks
    weights /= weights.sum()
    chosen = rng.choice(N, size=n, replace=False, p=weights)
    return [all_ids[i] for i in chosen]


def load_embeddings(arxiv_ids: list[str], emb_db_path: str, dim: int) -> np.ndarray:
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
    with sqlite3.connect(emb_db_path) as con:
        placeholders = ",".join("?" * len(batch_ids))
        rows = con.execute(
            f"SELECT arxiv_id, vector FROM recommendation_embeddings WHERE arxiv_id IN ({placeholders})",
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


def write_results(arxiv_ids: list[str], coords: np.ndarray, app_db_path: str) -> None:
    computed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    rows = [(aid, float(coords[i, 0]), float(coords[i, 1]), computed_at)
            for i, aid in enumerate(arxiv_ids)]

    with sqlite3.connect(app_db_path) as con:
        con.execute("DELETE FROM paper_lowres_proj")
        con.executemany(
            "INSERT INTO paper_lowres_proj (arxiv_id, x, y, computed_at) VALUES (?, ?, ?, ?)",
            rows,
        )
        con.commit()

    print(f"Wrote {len(rows)} lowres coordinates to paper_lowres_proj (computed_at={computed_at}).")


def main() -> None:
    args = parse_args()

    app_db = APP_DB_PATH()
    emb_db = EMBEDDING_CACHE_DB()
    model_path = LOWRES_PROJ_MODEL_PATH()

    print(f"Loading candidate IDs from {app_db} + {emb_db} ...")
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
    print(f"  Sampled {len(train_ids)} papers for training (max={args.max_papers}, seed={args.seed}).")

    print(f"Loading training embeddings (dim={args.dim}) ...")
    train_matrix = load_embeddings(train_ids, emb_db, args.dim)

    print(f"Fitting PaCMAP (n_neighbors={args.n_neighbors}) ...")
    projector = pacmap.PaCMAP(n_components=2, n_neighbors=args.n_neighbors, random_state=args.seed, save_tree=True)
    train_coords_raw = projector.fit_transform(train_matrix)
    print("  PaCMAP fit complete.")

    train_id_set = set(train_ids)
    relevant_ids = load_relevant_ids(app_db, emb_db, train_id_set)
    print(f"  {len(relevant_ids)} additional relevant papers to project.")

    relevant_coords_list: list[tuple[str, np.ndarray]] = []
    for i in range(0, len(relevant_ids), _BATCH):
        batch = relevant_ids[i:i + _BATCH]
        found_ids, batch_matrix = load_embeddings_batch(batch, emb_db, args.dim)
        if not found_ids:
            continue
        if hasattr(projector, "transform"):
            batch_coords = projector.transform(batch_matrix)
            for j, aid in enumerate(found_ids):
                relevant_coords_list.append((aid, batch_coords[j]))
        print(f"  Processed batch {i // _BATCH + 1}/{(len(relevant_ids) + _BATCH - 1) // _BATCH} ...")

    all_raw_coords_list = [np.array(c) for c in train_coords_raw]
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

    bundle = {
        "projector": projector,
        "method": "pacmap",
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "dim": args.dim,
    }
    joblib.dump(bundle, model_path)
    print(f"Saved lowres projection bundle to {model_path}.")

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
