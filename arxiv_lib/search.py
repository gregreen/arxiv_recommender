"""
Semantic search: embed a free-text query and rank papers by cosine similarity.

Public API
----------
search_papers(con, user_id, query, window, limit) -> list[dict]
    Embed *query* with the Instruct/Query prefix, compute cosine similarity
    against all embedded papers in *window*, and return the top *limit*
    papers sorted by descending similarity.
"""

import json
import sqlite3
import httpx

import numpy as np
from openai import OpenAI

from arxiv_lib.config import (
    API_KEYS,
    EMBEDDING_STORAGE_DIM,
    LLM_CONFIG,
)
from arxiv_lib.recommend import _load_vectors, _window_cutoff

# Instruct/Query prefix used when embedding search queries.
# Matches the prefix used in experiments/query_summaries.py.
_INSTRUCT_PREFIX = (
    "Instruct: "
    "Given an astrophysics search query, retrieve relevant arXiv "
    "paper summaries that match the query.\n"
    "Query: "
)


class SearchEmbeddingError(RuntimeError):
    """Raised when the remote embedding API call fails."""


def _embed_query(query: str) -> np.ndarray:
    """Embed *query* using the Instruct/Query prefix and return a float32 vector."""
    cfg      = LLM_CONFIG.get("embedding", {})
    model    = cfg.get("model", "")
    base_url = cfg.get("base_url", "https://router.huggingface.co/v1")
    api_key  = API_KEYS.get(cfg.get("api_key_name", "embed_api_key"), "")

    prompt = _INSTRUCT_PREFIX + query

    try:
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=httpx.Client(trust_env=False)
        )
        result = client.embeddings.create(input=prompt, model=model)
    except Exception as exc:
        raise SearchEmbeddingError(f"Embedding API error: {exc}") from exc

    return np.array(result.data[0].embedding, dtype=np.float32)[:EMBEDDING_STORAGE_DIM]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between 1-D vector *a* and matrix *b* (N × D)."""
    a_norm  = a / (np.linalg.norm(a) + 1e-12)
    b_norms = np.linalg.norm(b, axis=1, keepdims=True) + 1e-12
    return (b @ a_norm) / b_norms[:, 0]


def search_papers(
    con: sqlite3.Connection,
    user_id: int,
    query: str,
    window: str,
    limit: int,
) -> list[dict]:
    """
    Embed *query* and return up to *limit* papers from the given time window,
    sorted by cosine similarity (descending).

    Parameters
    ----------
    con : sqlite3.Connection
        Open connection to app.db.
    user_id : int
        ID of the requesting user (used to populate the ``liked`` field).
    query : str
        Free-text search query.
    window : str
        Time window: "day", "week", or "month".
    limit : int
        Maximum number of results to return.

    Returns
    -------
    list[dict]
        Each dict has keys: arxiv_id, title, authors, published_date,
        score (cosine similarity), rank, liked, generated_at.

    Raises
    ------
    SearchEmbeddingError
        If the remote embedding API call fails.
    """
    query_vec = _embed_query(query)

    # Collect paper IDs in the requested time window.
    cutoff = _window_cutoff(window, con)
    rows = con.execute(
        "SELECT arxiv_id FROM papers "
        "WHERE published_date >= ? AND published_date IS NOT NULL",
        (cutoff,),
    ).fetchall()
    window_ids = [r[0] for r in rows]
    if not window_ids:
        return []

    # Load embedding vectors for those papers.
    vectors = _load_vectors(window_ids)
    if not vectors:
        return []

    # Compute cosine similarities.
    ids    = list(vectors.keys())
    matrix = np.stack([vectors[aid] for aid in ids])          # (N, D)
    q      = query_vec[:matrix.shape[1]]                       # align dims
    sims   = 3*np.log(np.clip(_cosine_similarity(q, matrix), 1e-12, 1.0))

    top_n   = min(limit, len(ids))
    indices = np.argsort(sims)[::-1][:top_n]
    top_ids  = [ids[i] for i in indices]
    top_sims = [float(sims[i]) for i in indices]

    # Fetch metadata + liked status from app.db in one query.
    placeholders = ",".join("?" * len(top_ids))
    meta_rows = con.execute(
        f"""
        SELECT p.arxiv_id, p.title, p.authors, p.published_date,
               ul.liked
          FROM papers p
          LEFT JOIN user_papers ul
                 ON ul.arxiv_id = p.arxiv_id AND ul.user_id = ?
         WHERE p.arxiv_id IN ({placeholders})
        """,
        [user_id] + top_ids,
    ).fetchall()

    meta: dict[str, dict] = {}
    for arxiv_id, title, authors_json, published_date, liked in meta_rows:
        try:
            authors = json.loads(authors_json) if authors_json else []
        except Exception:
            authors = []
        meta[arxiv_id] = {
            "title":          title or "",
            "authors":        authors,
            "published_date": published_date,
            "liked":          liked,
        }

    # Build result list preserving similarity rank order.
    results = []
    for rank, (aid, score) in enumerate(zip(top_ids, top_sims), 1):
        m = meta.get(aid, {})
        results.append({
            "arxiv_id":       aid,
            "title":          m.get("title", ""),
            "authors":        m.get("authors", []),
            "published_date": m.get("published_date"),
            "score":          score,
            "rank":           rank,
            "liked":          m.get("liked"),
            "generated_at":   None,
        })

    return results
