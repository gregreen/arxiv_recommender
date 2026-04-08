#!/usr/bin/env python3
"""
Plot a 2D UMAP of all embeddings in the database, coloured by arXiv category.

Requires:
    pip install umap-learn matplotlib

Usage:
    python experiments/umap_plot.py
    python experiments/umap_plot.py --dim 64          # truncate to 64 dims before UMAP (default: full)
    python experiments/umap_plot.py --out my_plot.png # save to a specific path
    python experiments/umap_plot.py --n-neighbors 20 --min-dist 0.1
"""

import argparse
import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import sqlite3

import numpy as np

try:
    import umap
except ImportError:
    print("Error: umap-learn is not installed.  Run: pip install umap-learn", file=sys.stderr)
    sys.exit(1)

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; works without a display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from arxiv_lib.config import EMBEDDING_CACHE_DB, APP_DB_PATH

_EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))


def load_embeddings() -> tuple[list[str], np.ndarray]:
    """Return (arxiv_ids, matrix) for all embeddings in the DB."""
    with sqlite3.connect(EMBEDDING_CACHE_DB) as con:
        rows = con.execute("SELECT arxiv_id, vector FROM embeddings ORDER BY arxiv_id").fetchall()
    if not rows:
        print("No embeddings found in the database.", file=sys.stderr)
        sys.exit(1)
    arxiv_ids = [r[0] for r in rows]
    matrix = np.stack(
        [np.frombuffer(r[1], dtype=np.float32) for r in rows]
    )
    return arxiv_ids, matrix


def load_primary_categories(arxiv_ids: list[str]) -> list[str]:
    """
    Return the primary category (first entry in the JSON categories array) for
    each arXiv ID.  Falls back to 'unknown' if the paper is not in papers table
    or has no categories stored.
    """
    id_to_cat: dict[str, str] = {}
    with sqlite3.connect(APP_DB_PATH) as con:
        placeholders = ",".join("?" * len(arxiv_ids))
        rows = con.execute(
            f"SELECT arxiv_id, categories FROM papers WHERE arxiv_id IN ({placeholders})",
            arxiv_ids,
        ).fetchall()
    import json
    for arxiv_id, cats_json in rows:
        if cats_json:
            try:
                cats = json.loads(cats_json)
                if cats:
                    # Use the archive prefix only (e.g. "astro-ph.GA" → "astro-ph")
                    id_to_cat[arxiv_id] = cats[0].split(".")[0]
                    continue
            except (ValueError, IndexError):
                pass
        id_to_cat[arxiv_id] = "unknown"
    return [id_to_cat.get(aid, "unknown") for aid in arxiv_ids]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a 2D UMAP of all embeddings in the database."
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=None,
        metavar="N",
        help="Truncate embedding vectors to N dimensions before UMAP (default: use full vectors).",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        metavar="N",
        help="UMAP n_neighbors parameter (default: 15).",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        metavar="F",
        help="UMAP min_dist parameter (default: 0.1).",
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        default=None,
        help="Output file path (default: umap_plot.png in experiments/).",
    )
    args = parser.parse_args()

    out_path = os.path.abspath(args.out) if args.out else os.path.join(_EXPERIMENTS_DIR, "umap_plot.svg")

    # ------------------------------------------------------------------
    # Load embeddings
    # ------------------------------------------------------------------
    print("Loading embeddings from database...")
    arxiv_ids, matrix = load_embeddings()
    print(f"  {len(arxiv_ids)} embeddings loaded, shape {matrix.shape}.")

    if args.dim is not None:
        if args.dim > matrix.shape[1]:
            parser.error(f"--dim {args.dim} exceeds stored vector length {matrix.shape[1]}.")
        matrix = matrix[:, : args.dim]
        print(f"  Truncated to {args.dim} dimensions.")

    # ------------------------------------------------------------------
    # Load categories for colouring
    # ------------------------------------------------------------------
    print("Loading paper categories...")
    categories = load_primary_categories(arxiv_ids)
    unique_cats = sorted(set(categories))
    cat_to_idx  = {c: i for i, c in enumerate(unique_cats)}
    color_idxs  = np.array([cat_to_idx[c] for c in categories])
    n_cats      = len(unique_cats)
    print(f"  {n_cats} unique archive prefix(es): {', '.join(unique_cats)}")

    # ------------------------------------------------------------------
    # Load titles for a random sample of papers (for annotation)
    # ------------------------------------------------------------------
    print("Loading paper titles for annotation sample...")
    rng = np.random.default_rng(seed=42)
    sample_indices = rng.choice(len(arxiv_ids), size=min(64, len(arxiv_ids)), replace=False)
    sample_ids = [arxiv_ids[i] for i in sample_indices]
    id_to_title: dict[str, str] = {}
    with sqlite3.connect(APP_DB_PATH) as con:
        placeholders = ",".join("?" * len(sample_ids))
        title_rows = con.execute(
            f"SELECT arxiv_id, title FROM papers WHERE arxiv_id IN ({placeholders})",
            sample_ids,
        ).fetchall()
    for aid, title in title_rows:
        id_to_title[aid] = title or aid

    # ------------------------------------------------------------------
    # Run UMAP
    # ------------------------------------------------------------------
    print(f"\nRunning UMAP (n_neighbors={args.n_neighbors}, min_dist={args.min_dist})...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric="cosine",
        random_state=42,
        verbose=True,
    )
    embedding_2d = reducer.fit_transform(matrix)
    print("UMAP complete.")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    cmap = cm.get_cmap("tab20", n_cats)
    fig, ax = plt.subplots(figsize=(12, 9))

    for i, cat in enumerate(unique_cats):
        mask = color_idxs == i
        ax.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            s=4,
            alpha=0.6,
            color=cmap(i),
            label=f"{cat} ({mask.sum()})",
            linewidths=0,
        )

    ax.set_title(
        f"UMAP of {len(arxiv_ids):,} paper embeddings"
        + (f" (dim={args.dim})" if args.dim else ""),
        fontsize=13,
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(
        loc="upper right",
        markerscale=3,
        fontsize=8,
        framealpha=0.7,
        title="arXiv archive",
    )
    ax.set_aspect("equal", adjustable="datalim")

    # Annotate the 64 sampled papers with their titles at 4pt
    for idx in sample_indices:
        aid = arxiv_ids[idx]
        title = id_to_title.get(aid, aid)
        x, y = embedding_2d[idx, 0], embedding_2d[idx, 1]
        ax.annotate(
            title,
            xy=(x, y),
            xytext=(2, 2),
            textcoords="offset points",
            fontsize=4,
            alpha=0.85,
            clip_on=True,
        )

    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)

    print(f"\nPlot saved to: {out_path}")


if __name__ == "__main__":
    main()
