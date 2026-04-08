#!/usr/bin/env python3
"""
Generate embeddings for all papers that have a structured summary, using the
prompt:

    Title: <title>

    <structured summary>

Results are stored incrementally in a JSON file in the experiments/ directory
(arxiv_id → vector as a list of floats).  Re-running skips papers already
present in the JSON.  After all embeddings are complete, a 2D UMAP SVG is
generated alongside the JSON.

The production embedding DB (embeddings_cache.db) is never read from or
written to.

Requires:
    pip install umap-learn matplotlib

Usage:
    python experiments/embed_summaries.py
    python experiments/embed_summaries.py --config /path/to/alt_llm_config.json
    python experiments/embed_summaries.py --out-json my_embeddings.json --out-plot my_umap.svg
    python experiments/embed_summaries.py --n-neighbors 20 --min-dist 0.05
"""

import argparse
import glob
import json
import os
import sqlite3
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np

try:
    import umap
except ImportError:
    print("Error: umap-learn is not installed.  Run: pip install umap-learn", file=sys.stderr)
    sys.exit(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

try:
    from arxiv_to_prompt import count_tokens
except ImportError:
    def count_tokens(text: str) -> int:  # type: ignore[misc]
        return len(text) // 4  # rough fallback

from openai import OpenAI

from arxiv_lib import config as _config
from arxiv_lib.ingest import sanitize_old_style_arxiv_id

_EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_titles(arxiv_ids: list[str]) -> dict[str, str]:
    """Batch-fetch titles from app.db; missing IDs map to empty string."""
    if not arxiv_ids:
        return {}
    with sqlite3.connect(_config.APP_DB_PATH) as con:
        placeholders = ",".join("?" * len(arxiv_ids))
        rows = con.execute(
            f"SELECT arxiv_id, title FROM papers WHERE arxiv_id IN ({placeholders})",
            arxiv_ids,
        ).fetchall()
    result = {aid: "" for aid in arxiv_ids}
    for aid, title in rows:
        result[aid] = title or ""
    return result


def _load_primary_categories(arxiv_ids: list[str]) -> list[str]:
    """Return the archive prefix (e.g. 'astro-ph') for each arXiv ID."""
    if not arxiv_ids:
        return []
    with sqlite3.connect(_config.APP_DB_PATH) as con:
        placeholders = ",".join("?" * len(arxiv_ids))
        rows = con.execute(
            f"SELECT arxiv_id, categories FROM papers WHERE arxiv_id IN ({placeholders})",
            arxiv_ids,
        ).fetchall()
    id_to_cat: dict[str, str] = {}
    for aid, cats_json in rows:
        if cats_json:
            try:
                cats = json.loads(cats_json)
                if cats:
                    id_to_cat[aid] = cats[0].split(".")[0]
                    continue
            except (ValueError, IndexError):
                pass
        id_to_cat[aid] = "unknown"
    return [id_to_cat.get(aid, "unknown") for aid in arxiv_ids]


def _make_plot(
    arxiv_ids: list[str],
    matrix: np.ndarray,
    id_to_title: dict[str, str],
    out_path: str,
    n_neighbors: int,
    min_dist: float,
) -> None:
    categories  = _load_primary_categories(arxiv_ids)
    unique_cats = sorted(set(categories))
    cat_to_idx  = {c: i for i, c in enumerate(unique_cats)}
    color_idxs  = np.array([cat_to_idx[c] for c in categories])
    n_cats      = len(unique_cats)

    print(f"\nRunning UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=42,
        verbose=True,
    )
    embedding_2d = reducer.fit_transform(matrix)
    print("UMAP complete.")

    rng = np.random.default_rng(seed=42)
    sample_indices = rng.choice(len(arxiv_ids), size=min(128, len(arxiv_ids)), replace=False)

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

    ax.set_title(f"UMAP of {len(arxiv_ids):,} summary embeddings", fontsize=13)
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

    for idx in sample_indices:
        aid   = arxiv_ids[idx]
        title = id_to_title.get(aid, aid)
        x, y  = embedding_2d[idx, 0], embedding_2d[idx, 1]
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
    print(f"Plot saved to: {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed all papers with cached summaries and plot a UMAP."
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="Path to alternate llm_config.json.  Only the 'embedding' section is used.",
    )
    parser.add_argument(
        "--out-json",
        metavar="PATH",
        default=None,
        help="Output JSON path (default: summary_embeddings.json in experiments/).",
    )
    parser.add_argument(
        "--out-plot",
        metavar="PATH",
        default=None,
        help="Output SVG path (default: summary_embeddings_umap.svg in experiments/).",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        metavar="N",
        help="UMAP n_neighbors (default: 15).",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        metavar="F",
        help="UMAP min_dist (default: 0.1).",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of papers to embed (default: no limit).",
    )
    args = parser.parse_args()

    json_path = os.path.abspath(args.out_json) if args.out_json else os.path.join(_EXPERIMENTS_DIR, "summary_embeddings.json")
    plot_path = os.path.abspath(args.out_plot) if args.out_plot else os.path.join(_EXPERIMENTS_DIR, "summary_embeddings_umap.svg")

    # ------------------------------------------------------------------
    # Build effective embedding config
    # ------------------------------------------------------------------
    effective_cfg: dict = dict(_config.LLM_CONFIG.get("embedding", {}))

    if args.config:
        alt_path = os.path.abspath(args.config)
        with open(alt_path, "r", encoding="utf-8") as f:
            alt_cfg = json.load(f)
        if "embedding" not in alt_cfg:
            parser.error(f"Alternate config {alt_path!r} has no 'embedding' section.")
        effective_cfg.update(alt_cfg["embedding"])
        print(f"Using alternate LLM config from: {alt_path}")

    model    = effective_cfg.get("model", "")
    max_tok  = effective_cfg.get("max_input_tokens", 24576)
    if "api_key" in effective_cfg:
        api_key = effective_cfg["api_key"]
    else:
        api_key = _config.API_KEYS.get(effective_cfg.get("api_key_name", "embed_api_key"), "")
    base_url = effective_cfg.get("base_url", "https://router.huggingface.co/v1")

    print(f"Model    : {model}")
    print(f"Base URL : {base_url}")
    print()

    # ------------------------------------------------------------------
    # Discover summary files
    # ------------------------------------------------------------------
    summary_files = sorted(glob.glob(os.path.join(_config.SUMMARY_CACHE_DIR, "*.txt")))
    if not summary_files:
        print(f"No summary files found in {_config.SUMMARY_CACHE_DIR}.", file=sys.stderr)
        sys.exit(1)

    # Filename stem is already the sanitized arxiv_id (e.g. "2309.06676")
    all_ids = [os.path.splitext(os.path.basename(f))[0] for f in summary_files]
    print(f"Found {len(all_ids)} summary file(s).")

    if args.max_papers is not None:
        all_ids = all_ids[:args.max_papers]
        print(f"Limiting to {len(all_ids)} papers.")

    # ------------------------------------------------------------------
    # Load existing JSON (resume support)
    # ------------------------------------------------------------------
    embeddings: dict[str, list[float]] = {}
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            embeddings = json.load(f)
        print(f"Loaded {len(embeddings)} existing embedding(s) from {json_path}.")

    pending = [aid for aid in all_ids if aid not in embeddings]
    print(f"{len(pending)} paper(s) to embed.\n")

    if not pending and not embeddings:
        print("Nothing to do.", file=sys.stderr)
        sys.exit(0)

    # ------------------------------------------------------------------
    # Fetch titles for all IDs
    # ------------------------------------------------------------------
    id_to_title = _load_titles(all_ids)

    # ------------------------------------------------------------------
    # Embed pending papers
    # ------------------------------------------------------------------
    client = OpenAI(base_url=base_url, api_key=api_key)

    from tqdm.auto import tqdm

    for n, arxiv_id in enumerate(tqdm(pending), 1):
        summary_file = os.path.join(_config.SUMMARY_CACHE_DIR, f"{arxiv_id}.txt")
        with open(summary_file, "r", encoding="utf-8") as f:
            summary = f.read().strip()

        title   = id_to_title.get(arxiv_id, "")
        prompt  = f"Title: {title}\n\n{summary}" if title else summary

        n_tok = count_tokens(prompt)
        if n_tok > max_tok:
            chars_per_token = len(prompt) / max(n_tok, 1)
            chars_to_keep   = int(max_tok * chars_per_token)
            prompt = prompt[:chars_to_keep] + "\n\n[Truncated due to token limit]"
            print(f"  [{n}/{len(pending)}] {arxiv_id}: truncated from ~{n_tok} to ~{max_tok} tokens.")
        else:
            print(f"  [{n}/{len(pending)}] {arxiv_id} (~{n_tok} tokens)...")

        try:
            result  = client.embeddings.create(input=prompt, model=model)
            vector  = result.data[0].embedding
        except Exception as e:
            print(f"  Error embedding {arxiv_id}: {e} — skipping.", file=sys.stderr)
            continue

        embeddings[arxiv_id] = vector

        # Incremental save after every 128 papers
        if n % 128 == 0 or n == len(pending):
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(embeddings, f)

    print(f"\n{len(embeddings)} total embedding(s) saved to {json_path}.")

    if not embeddings:
        print("No embeddings available to plot.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # UMAP + plot
    # ------------------------------------------------------------------
    ordered_ids = sorted(embeddings.keys())
    matrix = np.array([embeddings[aid][:64] for aid in ordered_ids], dtype=np.float32)
    print(f"Embedding matrix shape: {matrix.shape}")

    _make_plot(
        arxiv_ids=ordered_ids,
        matrix=matrix,
        id_to_title=id_to_title,
        out_path=plot_path,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
    )


if __name__ == "__main__":
    main()
