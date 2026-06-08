#!/usr/bin/env python3
"""
Search score calibration experiment.

For a random sample of N papers that have both a cached summary and a search
embedding, this script:
  1. Extracts the LLM-generated keyword string from each paper's summary.
  2. Embeds the keyword string using the production search embedding pipeline
     (with caching via search_term_embeddings in embeddings_cache.db).
  3. Computes cosine similarity between the keyword embedding and:
       - The paper's own search embedding  ("self score")
       - M randomly sampled papers' search embeddings  ("random scores")
  4. Saves raw results to a JSON file.
  5. Builds a percentile lookup table (cosine → percentile rank in the random
     distribution) and writes it to search_score_percentiles.json at the
     project root, making the calibrated scorer in search.py live immediately.
  6. Prints summary statistics and saves a histogram plot.

Usage:
    python experiments/search_score_calibration.py
    python experiments/search_score_calibration.py --n-papers 200 --m-random 2048
    python experiments/search_score_calibration.py --output my_results.json
    python experiments/search_score_calibration.py --config alt_llm_config.json
    python experiments/search_score_calibration.py --no-percentile-table
"""

import argparse
import json
import os
import random
import sqlite3
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np

from arxiv_lib.config import (
    BASE_DIR,
    EMBEDDING_CACHE_DB,
    SEARCH_EMBEDDING_DIM,
    SUMMARY_CACHE_DIR,
    LLM_CONFIG,
    API_KEYS,
)
from arxiv_lib.ingest import load_search_term_embedding, store_search_term_embedding
from experiments.shash_distribution import fit_shash, shash_cdf, shash_pdf

_EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))

_INSTRUCT_PREFIX = (
    "Instruct: "
    "Given an astrophysics search query, retrieve relevant arXiv "
    "paper summaries that match the query.\n"
    "Query: "
)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _embed_query(query: str, embed_cfg: dict) -> np.ndarray:
    """Embed *query* using the configured embedding model."""
    import httpx
    from openai import OpenAI

    model    = embed_cfg.get("model", "")
    base_url = embed_cfg.get("base_url", "https://router.huggingface.co/v1")
    api_key  = API_KEYS.get(embed_cfg.get("api_key_name", "embed_api_key"), "")

    prompt = _INSTRUCT_PREFIX + query
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        http_client=httpx.Client(trust_env=False),
    )
    result = client.embeddings.create(input=prompt, model=model)
    vec = np.array(result.data[0].embedding, dtype=np.float32)
    return vec[:SEARCH_EMBEDDING_DIM]


def _get_or_embed(query: str, embed_cfg: dict) -> np.ndarray:
    """Return cached embedding or compute and cache it."""
    cached = load_search_term_embedding(query)
    if cached is not None:
        return cached[:SEARCH_EMBEDDING_DIM]
    vec = _embed_query(query, embed_cfg)
    try:
        store_search_term_embedding(query, vec)
    except Exception as exc:
        print(f"  Warning: could not cache embedding: {exc}", file=sys.stderr)
    return vec


# ---------------------------------------------------------------------------
# Search embedding DB helpers
# ---------------------------------------------------------------------------

def _load_all_search_vectors() -> dict[str, np.ndarray]:
    """Load all paper search embeddings from embeddings_cache.db."""
    vectors: dict[str, np.ndarray] = {}
    with sqlite3.connect(EMBEDDING_CACHE_DB()) as con:
        rows = con.execute(
            "SELECT arxiv_id, vector FROM search_embeddings"
        ).fetchall()
    for arxiv_id, blob in rows:
        full = np.frombuffer(blob, dtype=np.float32).copy()
        vectors[arxiv_id] = full[:SEARCH_EMBEDDING_DIM]
    return vectors


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

def _extract_keywords(arxiv_id: str) -> str | None:
    """
    Read the summary cache file for *arxiv_id* and extract the keywords line.
    Returns the keyword string (after 'Keywords: '), or None if not found.
    """
    summary_dir = SUMMARY_CACHE_DIR()
    # Normalise ID to strip version suffix if present
    clean_id = arxiv_id.replace("/", "_")
    path = os.path.join(summary_dir, f"{clean_id}.txt")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        first_line = f.readline().strip()
    if first_line.startswith("Keywords:"):
        return first_line[len("Keywords:"):].strip()
    return None


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between 1-D vector *a* and matrix *b* (N × D)."""
    a_norm  = a / (np.linalg.norm(a) + 1e-12)
    b_norms = np.linalg.norm(b, axis=1, keepdims=True) + 1e-12
    return (b @ a_norm) / b_norms[:, 0]


def _production_score(cosine_sim: float) -> float:
    """Replicate the production search score for a single cosine similarity value.

    Uses the calibrated percentile-based scorer from config if the percentile
    table exists; falls back to the legacy 3·log formula when running before
    the table has been generated (i.e. on first calibration run).
    """
    try:
        from arxiv_lib.search import cosine_to_search_score
        return float(cosine_to_search_score(np.array([cosine_sim]))[0])
    except FileNotFoundError:
        return float(3.0 * np.log(np.clip(cosine_sim, 1e-12, 1.0)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate search scores by comparing self vs. random paper similarities."
    )
    parser.add_argument(
        "--n-papers", type=int, default=100, metavar="N",
        help="Number of source papers to sample (default: 100).",
    )
    parser.add_argument(
        "--m-random", type=int, default=1024, metavar="M",
        help="Number of random papers to compare against per source paper (default: 1024).",
    )
    parser.add_argument(
        "--output", metavar="PATH",
        default=os.path.join(_EXPERIMENTS_DIR, "search_score_calibration.json"),
        help="Output JSON path (default: experiments/search_score_calibration.json).",
    )
    parser.add_argument(
        "--plot", metavar="PATH",
        default=os.path.join(_EXPERIMENTS_DIR, "search_score_calibration.png"),
        help="Output histogram PNG path (default: experiments/search_score_calibration.png).",
    )
    parser.add_argument(
        "--percentile-table", metavar="PATH",
        default=os.path.join(BASE_DIR, "search_score_percentiles.json"),
        help="Where to write the percentile lookup table used by the production scorer "
             "(default: <project_root>/search_score_percentiles.json).",
    )
    parser.add_argument(
        "--no-percentile-table", action="store_true",
        help="Skip writing the percentile lookup table.",
    )
    parser.add_argument(
        "--percentile", type=float, default=99.5, metavar="P",
        help="For each source paper, use the Pth percentile of its random cosine "
             "similarities to characterise the 'noise' distribution (default: 99.5).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="SEED",
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--incremental", action="store_true",
        help="Load existing results from --output (if present), skip papers already "
             "measured, and append new measurements. Useful for extending a previous run.",
    )
    parser.add_argument(
        "--config", metavar="PATH", default=None,
        help="Path to alternate llm_config.json (only the 'embedding' section is used).",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------
    # Embedding config
    # ------------------------------------------------------------------
    embed_cfg: dict = dict(LLM_CONFIG.get("embedding", {}))
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            alt = json.load(f)
        embed_cfg.update(alt.get("embedding", {}))

    # ------------------------------------------------------------------
    # Load all search embeddings
    # ------------------------------------------------------------------
    print("Loading search embeddings from DB…")
    all_vectors = _load_all_search_vectors()
    all_ids = list(all_vectors.keys())
    print(f"  {len(all_ids):,} papers with search embeddings.")

    if not all_ids:
        print("Error: no search embeddings found in embeddings_cache.db.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load existing results (incremental mode)
    # ------------------------------------------------------------------
    existing_results: list[dict] = []
    already_done: set[str] = set()
    if args.incremental and os.path.exists(args.output):
        with open(args.output, encoding="utf-8") as f:
            existing_data = json.load(f)
        existing_results = existing_data.get("results", [])
        already_done = {r["arxiv_id"] for r in existing_results}
        print(f"Incremental mode: loaded {len(existing_results)} existing results "
              f"from {args.output!r}.")

    # ------------------------------------------------------------------
    # Find papers that also have a summary with keywords
    # ------------------------------------------------------------------
    print("Finding papers with keyword summaries…")
    eligible: list[tuple[str, str]] = []  # (arxiv_id, keywords)
    for arxiv_id in all_ids:
        kw = _extract_keywords(arxiv_id)
        if kw:
            eligible.append((arxiv_id, kw))
    print(f"  {len(eligible):,} papers eligible (have embedding + keywords).")

    if not eligible:
        print("Error: no eligible papers found.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Sample N source papers (excluding already-measured ones)
    # ------------------------------------------------------------------
    eligible_new = [(aid, kw) for aid, kw in eligible if aid not in already_done]
    n_source = min(args.n_papers, len(eligible_new))
    if args.incremental:
        print(f"  {len(eligible_new):,} eligible papers not yet measured.")
    if n_source < args.n_papers and not args.incremental:
        print(f"  Warning: only {n_source} eligible papers; using all of them.")
    source_papers = rng.sample(eligible_new, n_source)
    print(f"Sampling {n_source} new source papers.")

    # ------------------------------------------------------------------
    # Build numpy matrix of all vectors for fast batch similarity
    # ------------------------------------------------------------------
    all_ids_arr = np.array(all_ids)
    all_matrix  = np.stack([all_vectors[aid] for aid in all_ids])  # (N_total, D)
    id_to_idx   = {aid: i for i, aid in enumerate(all_ids)}

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    new_results = []
    for i, (arxiv_id, keywords) in enumerate(source_papers):
        print(f"[{i+1}/{n_source}] {arxiv_id}: {keywords}")

        # Embed keywords
        try:
            query_vec = _get_or_embed(keywords, embed_cfg)
        except Exception as exc:
            print(f"  Skipping — embedding failed: {exc}", file=sys.stderr)
            continue

        # Self score
        self_vec  = all_vectors[arxiv_id]  # (D,)
        self_sim  = float(_cosine_similarity(query_vec, self_vec[np.newaxis, :])[0])
        self_score = _production_score(self_sim)

        # Random sample (excluding the source paper itself)
        pool_ids  = [aid for aid in all_ids if aid != arxiv_id]
        m_actual  = min(args.m_random, len(pool_ids))
        rand_ids  = rng.sample(pool_ids, m_actual)
        rand_idxs = [id_to_idx[aid] for aid in rand_ids]
        rand_mat  = all_matrix[rand_idxs]  # (m_actual, D)
        rand_sims = _cosine_similarity(query_vec, rand_mat).tolist()
        rand_scores = [_production_score(s) for s in rand_sims]

        new_results.append({
            "arxiv_id":      arxiv_id,
            "keywords":      keywords,
            "self_cosine":   self_sim,
            "self_score":    self_score,
            "random_cosines": rand_sims,
            "random_scores": rand_scores,
        })

    results = existing_results + new_results
    print(f"\n{len(new_results)} new measurements; {len(results)} total.")

    if not results:
        print("Error: no results produced.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    output_data = {
        "params": {
            "n_papers":            n_source,
            "m_random":            args.m_random,
            "seed":                args.seed,
            "search_embedding_dim": SEARCH_EMBEDDING_DIM,
        },
        "results": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {args.output}")

    # ------------------------------------------------------------------
    # Build and save percentile lookup table
    # ------------------------------------------------------------------
    self_cosines    = np.array([r["self_cosine"]   for r in results])
    all_rand_cos    = np.concatenate([r["random_cosines"] for r in results])
    self_scores     = np.array([r["self_score"]    for r in results])
    all_rand_scores = np.concatenate([r["random_scores"]  for r in results])

    # Per-paper pth-percentile cosine: the value at which only (100-P)% of
    # random papers score higher — used as a single representative "noise" value.
    per_paper_pct_cos = np.array(
        [np.percentile(r["random_cosines"], args.percentile) for r in results]
    )

    if not args.no_percentile_table:
        # Fit a SHASH distribution to the per-paper pth-percentile cosines.
        shash_params, shash_flags = fit_shash(per_paper_pct_cos)
        print(f"\nSHASH fit (P={args.percentile}): {shash_params}  flags={shash_flags}")

        # Generate the lookup table by evaluating the SHASH CDF over a cosine grid.
        # The table maps cosine_similarity → percentile fraction for use in
        # cosine_to_search_score() in search.py.
        pad   = 3 * per_paper_pct_cos.std()
        x_min = per_paper_pct_cos.min() - pad
        x_max = per_paper_pct_cos.max() + pad
        cosine_grid = np.linspace(x_min, x_max, 256)
        pct_fractions = shash_cdf(cosine_grid, **shash_params)  # values in [0, 1]
        percentile_data = {
            "percentile":        (pct_fractions * 100).tolist(),
            "cosine_similarity": cosine_grid.tolist(),
            "shash_params":      {k: float(v) for k, v in shash_params.items()},
            "percentile_p":      args.percentile,
        }
        with open(args.percentile_table, "w", encoding="utf-8") as f:
            json.dump(percentile_data, f, indent=2)
        print(f"Percentile table saved to: {args.percentile_table}")

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    print("\n--- Cosine similarity statistics ---")
    print(f"  Self           : mean={self_cosines.mean():.4f}  std={self_cosines.std():.4f}"
          f"  min={self_cosines.min():.4f}  max={self_cosines.max():.4f}")
    print(f"  Random (all)   : mean={all_rand_cos.mean():.4f}  std={all_rand_cos.std():.4f}"
          f"  min={all_rand_cos.min():.4f}  max={all_rand_cos.max():.4f}")
    print(f"  Random P={args.percentile:.1f}%: mean={per_paper_pct_cos.mean():.4f}  std={per_paper_pct_cos.std():.4f}"
          f"  min={per_paper_pct_cos.min():.4f}  max={per_paper_pct_cos.max():.4f}")

    print("\n--- Production score statistics ---")
    print(f"  Self   : mean={self_scores.mean():.4f}  std={self_scores.std():.4f}"
          f"  min={self_scores.min():.4f}  max={self_scores.max():.4f}")
    print(f"  Random : mean={all_rand_scores.mean():.4f}  std={all_rand_scores.std():.4f}"
          f"  min={all_rand_scores.min():.4f}  max={all_rand_scores.max():.4f}")

    # ------------------------------------------------------------------
    # Histogram plot
    # ------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: per-paper pth-percentile cosines with SHASH fit
    bins_pct = np.linspace(per_paper_pct_cos.min() - 0.01,
                           per_paper_pct_cos.max() + 0.01, 40)
    ax1.hist(per_paper_pct_cos, bins=bins_pct, density=True, alpha=0.6,
             label=f"P={args.percentile}% per paper (n={len(per_paper_pct_cos)})",
             color="steelblue")
    if not args.no_percentile_table:
        x_fit = np.linspace(per_paper_pct_cos.min() - 0.02,
                            per_paper_pct_cos.max() + 0.02, 400)
        ax1.plot(x_fit, shash_pdf(x_fit, **shash_params), "r-", lw=1.5,
                 label="SHASH fit")
    ax1.hist(self_cosines, bins=bins_pct, density=True, alpha=0.7,
             label=f"Self (n={len(self_cosines)})", color="tomato")
    ax1.set_xlabel("Cosine similarity")
    ax1.set_ylabel("Density")
    ax1.set_title(f"Cosine similarity:\nself vs. random P={args.percentile}% (SHASH fit)")
    ax1.legend(fontsize=8)

    # Right: production scores
    bins_score = np.linspace(
        min(all_rand_scores.min(), self_scores.min()) - 0.1,
        max(all_rand_scores.max(), self_scores.max()) + 0.1,
        60,
    )
    ax2.hist(all_rand_scores, bins=bins_score, density=True, alpha=0.6, label=f"Random (n={len(all_rand_scores):,})", color="steelblue")
    ax2.hist(self_scores,     bins=bins_score, density=True, alpha=0.8, label=f"Self (n={len(self_scores)})", color="tomato")
    ax2.set_xlabel("Production score  (ln percentile fraction)")
    ax2.set_ylabel("Density")
    ax2.set_title("Production score:\nkeywords vs. self / random paper")
    ax2.legend()

    fig.suptitle(
        f"Search score calibration  "
        f"(N={n_source} papers, M={args.m_random} random comparisons each, "
        f"dim={SEARCH_EMBEDDING_DIM})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(args.plot, dpi=150)
    print(f"Histogram saved to: {args.plot}")


if __name__ == "__main__":
    main()
