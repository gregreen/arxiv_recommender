#!/usr/bin/env python3
"""
Search summary_embeddings.json by free-text query.

The query is wrapped in the same Instruct/Query prompt that is used when
embedding papers with gen_arxiv_embedding(), then sent to the embedding
model from the standard configuration.  Cosine similarity is computed
against every vector in the JSON file and the top-N hits are printed with
their titles and full structured summaries.

Usage:
    python experiments/query_summaries.py "epoch of reionization, 21 cm cosmology"
    python experiments/query_summaries.py --top 20 "fast radio bursts"
    python experiments/query_summaries.py --embeddings /path/to/other.json "dark matter"
    python experiments/query_summaries.py --config alt_llm_config.json "lensing"
"""

import argparse
import json
import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
from openai import OpenAI

from arxiv_lib import config as _config

try:
    from arxiv_to_prompt import count_tokens
except ImportError:
    def count_tokens(text: str) -> int:  # type: ignore[misc]
        return len(text) // 4

_EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_EMBEDDINGS = os.path.join(_EXPERIMENTS_DIR, "summary_embeddings.json")

_INSTRUCT_PREFIX = (
    "Instruct: "
    "Given an astrophysics search query, retrieve relevant arXiv "
    "paper summaries that match the query.\n"
    "Query: "
)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between vector a (1-D) and matrix b (N × D)."""
    a_norm = a / (np.linalg.norm(a) + 1e-12)
    b_norms = np.linalg.norm(b, axis=1, keepdims=True) + 1e-12
    return b @ a_norm / b_norms[:, 0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query summary embeddings by free-text similarity search."
    )
    parser.add_argument(
        "query",
        nargs="+",
        help="Query text (all positional args are joined with spaces).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        metavar="N",
        help="Number of top results to display (default: 10).",
    )
    parser.add_argument(
        "--embeddings",
        metavar="PATH",
        default=_DEFAULT_EMBEDDINGS,
        help=f"Path to summary_embeddings.json (default: {_DEFAULT_EMBEDDINGS}).",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Path to alternate llm_config.json.  Only the 'embedding' section is used.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        metavar="D",
        help="Truncate embedding vectors to D dimensions before similarity search (default: use full vectors).",
    )
    args = parser.parse_args()

    query_text = " ".join(args.query)

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

    model    = effective_cfg.get("model", "")
    max_tok  = effective_cfg.get("max_input_tokens", 24576)
    if "api_key" in effective_cfg:
        api_key = effective_cfg["api_key"]
    else:
        api_key = _config.API_KEYS.get(effective_cfg.get("api_key_name", "embed_api_key"), "")
    base_url = effective_cfg.get("base_url", "https://router.huggingface.co/v1")

    # ------------------------------------------------------------------
    # Build Instruct/Query prompt
    # ------------------------------------------------------------------
    full_prompt = _INSTRUCT_PREFIX + query_text

    n_tok = count_tokens(full_prompt)
    if n_tok > max_tok:
        chars_per_token = len(full_prompt) / max(n_tok, 1)
        chars_to_keep   = int(max_tok * chars_per_token)
        full_prompt = full_prompt[:chars_to_keep] + "\n\n[Truncated due to token limit]"
        print(f"Query truncated to ~{max_tok} tokens.", file=sys.stderr)

    # ------------------------------------------------------------------
    # Embed the query
    # ------------------------------------------------------------------
    print(f"Embedding query via {base_url} / {model} …")
    client = OpenAI(base_url=base_url, api_key=api_key)
    result = client.embeddings.create(input=full_prompt, model=model)
    query_vec = np.array(result.data[0].embedding, dtype=np.float32)
    print(f"Query vector dim: {len(query_vec)}")

    # ------------------------------------------------------------------
    # Load summary embeddings
    # ------------------------------------------------------------------
    embeddings_path = os.path.abspath(args.embeddings)
    if not os.path.exists(embeddings_path):
        print(f"Error: embeddings file not found: {embeddings_path}", file=sys.stderr)
        sys.exit(1)

    with open(embeddings_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    arxiv_ids = sorted(raw.keys())
    matrix    = np.array([raw[aid] for aid in arxiv_ids], dtype=np.float32)
    print(f"Loaded {len(arxiv_ids)} embeddings (dim {matrix.shape[1]}).\n")

    if args.embedding_dim is not None:
        matrix = matrix[:, :args.embedding_dim]
        query_vec = query_vec[:args.embedding_dim]
        print(f"Truncated embeddings to {args.embedding_dim} dimensions.")

    # Trim query vector to corpus dimension if models differ
    d = matrix.shape[1]
    if len(query_vec) != d:
        if len(query_vec) > d:
            query_vec = query_vec[:d]
            print(f"Note: query vector trimmed to {d} dims to match corpus.", file=sys.stderr)
        else:
            print(
                f"Error: query vector dim {len(query_vec)} < corpus dim {d}.",
                file=sys.stderr,
            )
            sys.exit(1)

    # ------------------------------------------------------------------
    # Cosine similarity + ranking
    # ------------------------------------------------------------------
    sims    = _cosine_similarity(query_vec, matrix)
    top_n   = min(args.top, len(arxiv_ids))
    indices = np.argsort(sims)[::-1][:top_n]

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    summary_dir = _config.SUMMARY_CACHE_DIR()

    # Try to batch-fetch titles from app.db
    import sqlite3
    id_to_title: dict[str, str] = {}
    try:
        with sqlite3.connect(_config.APP_DB_PATH()) as con:
            top_ids = [arxiv_ids[i] for i in indices]
            placeholders = ",".join("?" * len(top_ids))
            rows = con.execute(
                f"SELECT arxiv_id, title FROM papers WHERE arxiv_id IN ({placeholders})",
                top_ids,
            ).fetchall()
        for aid, title in rows:
            id_to_title[aid] = title or ""
    except Exception:
        pass  # titles are optional; we'll fall back to the arxiv_id

    sep = "=" * 72

    for rank, idx in enumerate(indices, 1):
        aid   = arxiv_ids[idx]
        score = sims[idx]
        title = id_to_title.get(aid, aid)

        summary_path = os.path.join(summary_dir, f"{aid}.txt")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = f.read().strip()
        else:
            summary = "(structured summary not found)"

        print(sep)
        print(f"#{rank}  [{score:.4f}]  {aid}")
        print(f"Title: {title}")
        print()
        print(summary)
        print()

    print(sep)
    print(f"\nQuery: {query_text!r}")
    print(f"Top {top_n} of {len(arxiv_ids)} papers shown.")


if __name__ == "__main__":
    main()
