"""
Semantic search: embed a free-text query and rank papers by cosine similarity.

Public API
----------
search_papers(con, user_id, query) -> dict[str, list[dict]]
    Embed *query* with the Instruct/Query prefix, compute cosine similarity
    against all embedded papers across all time windows once, and return
    per-window result lists sorted by descending similarity.
"""

import json
import logging
import sqlite3
import httpx

import numpy as np
from openai import OpenAI

from arxiv_lib.config import (

    API_KEYS,
    EMBEDDING_STORAGE_DIM,
    LLM_CONFIG,
    ONBOARDING_BROWSE_LIMIT,
    RECOMMEND_TIME_WINDOWS,
    SEARCH_EMBEDDING_DIM,
)
from arxiv_lib.ingest import load_search_term_embedding, store_search_term_embedding

_log = logging.getLogger(__name__)
from arxiv_lib.recommend import _window_cutoff

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


def _load_search_vectors(arxiv_ids: list[str]) -> dict[str, np.ndarray]:
    """
    Load search embeddings for the given arXiv IDs from embeddings_cache.db.

    Returns a dict mapping arxiv_id → truncated float32 vector (length SEARCH_EMBEDDING_DIM).
    IDs not found in the DB are silently omitted.
    """
    import sqlite3 as _sqlite3
    from arxiv_lib.config import EMBEDDING_CACHE_DB
    if not arxiv_ids:
        return {}
    placeholders = ",".join("?" * len(arxiv_ids))
    vectors: dict[str, np.ndarray] = {}
    with _sqlite3.connect(EMBEDDING_CACHE_DB()) as emb_con:
        rows = emb_con.execute(
            f"SELECT arxiv_id, vector FROM search_embeddings WHERE arxiv_id IN ({placeholders})",
            arxiv_ids,
        ).fetchall()
    for arxiv_id, blob in rows:
        full = np.frombuffer(blob, dtype=np.float32)
        vectors[arxiv_id] = full[:SEARCH_EMBEDDING_DIM]
    return vectors


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between 1-D vector *a* and matrix *b* (N × D)."""
    a_norm  = a / (np.linalg.norm(a) + 1e-12)
    b_norms = np.linalg.norm(b, axis=1, keepdims=True) + 1e-12
    return (b @ a_norm) / b_norms[:, 0]


def search_papers(
    con: sqlite3.Connection,
    user_id: int,
    query: str,
) -> dict[str, list[dict]]:
    """
    Embed *query* and return per-window result lists sorted by cosine
    similarity (descending).

    The query embedding is looked up in ``search_term_embeddings`` first; the
    remote embedding API is only called on a cache miss.  After scoring, the
    query text and its embedding are persisted: the vector is stored in
    ``search_term_embeddings`` (keyed by query text, shared across users), and
    ``user_search_terms`` is upserted with ``last_searched_at = now`` for this
    user.  Both storage steps are non-fatal — a failure logs a warning and does
    not interrupt the search response.

    Similarity scores are computed against all papers whose published_date
    falls within the broadest time window (month).  Results are partitioned by
    the cutoff of each time window, and the top ONBOARDING_BROWSE_LIMIT papers
    are returned for each window.

    Parameters
    ----------
    con : sqlite3.Connection
        Open connection to app.db.
    user_id : int
        ID of the requesting user (used to populate the ``liked`` field and
        record the search in ``user_search_terms``).
    query : str
        Free-text search query.

    Returns
    -------
    dict[str, list[dict]]
        Keys are RECOMMEND_TIME_WINDOWS ("day", "week", "month").  Each
        value is a list of dicts with keys: arxiv_id, title, authors,
        published_date, score (log-scaled cosine similarity), rank, liked,
        generated_at.

    Raises
    ------
    SearchEmbeddingError
        If the remote embedding API call fails (only raised on a cache miss).
    """
    query_vec = load_search_term_embedding(query)
    cache_hit = query_vec is not None
    if not cache_hit:
        query_vec = _embed_query(query)

    # Persist the search term and its embedding (non-fatal on failure).
    try:
        if not cache_hit:
            store_search_term_embedding(query, query_vec)
        # Also store the search term for this user with a timestamp
        # (for potential future use in paper recommendations).
        con.execute(
            """
            INSERT INTO user_search_terms(user_id, query, last_searched_at)
            VALUES (?, ?, datetime('now'))
            ON CONFLICT(user_id, query)
            DO UPDATE SET last_searched_at=excluded.last_searched_at
            """,
            (user_id, query),
        )
        con.commit()
    except Exception:
        _log.warning("Failed to store search term for user %s", user_id, exc_info=True)

    # Load all papers within the broadest window (month).
    window_cutoffs = {w: _window_cutoff(w, con) for w in RECOMMEND_TIME_WINDOWS}
    earliest_cutoff = min(window_cutoffs.values())
    rows = con.execute(
        "SELECT arxiv_id, published_date FROM papers "
        "WHERE published_date >= ? AND published_date IS NOT NULL",
        (earliest_cutoff,),
    ).fetchall()
    if not rows:
        return {w: [] for w in RECOMMEND_TIME_WINDOWS}

    all_ids = [r[0] for r in rows]
    pub_dates = {r[0]: r[1] for r in rows}

    # Load search embedding vectors and compute cosine similarity once.
    vectors = _load_search_vectors(all_ids)
    if not vectors:
        return {w: [] for w in RECOMMEND_TIME_WINDOWS}

    ids    = list(vectors.keys())
    matrix = np.stack([vectors[aid] for aid in ids])   # (N, D)
    q      = query_vec[:SEARCH_EMBEDDING_DIM]           # align to search dim
    sims   = 3 * np.log(np.clip(_cosine_similarity(q, matrix), 1e-12, 1.0))
    scores = dict(zip(ids, sims.tolist()))

    # Fetch metadata + liked status for all candidate papers.
    placeholders = ",".join("?" * len(ids))
    meta_rows = con.execute(
        f"""
        SELECT p.arxiv_id, p.title, p.authors, p.published_date,
               ul.liked
          FROM papers p
          LEFT JOIN user_papers ul
                 ON ul.arxiv_id = p.arxiv_id AND ul.user_id = ?
         WHERE p.arxiv_id IN ({placeholders})
        """,
        [user_id] + ids,
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

    # Sort all papers by score descending, then partition per window.
    sorted_ids = sorted(ids, key=lambda aid: scores[aid], reverse=True)

    result: dict[str, list[dict]] = {}
    for window in RECOMMEND_TIME_WINDOWS:
        cutoff = window_cutoffs[window]
        window_papers = [
            aid for aid in sorted_ids
            if pub_dates.get(aid, "") >= cutoff
        ][:ONBOARDING_BROWSE_LIMIT]
        window_results = []
        for rank, aid in enumerate(window_papers, 1):
            m = meta.get(aid, {})
            window_results.append({
                "arxiv_id":       aid,
                "title":          m.get("title", ""),
                "authors":        m.get("authors", []),
                "published_date": m.get("published_date"),
                "score":          scores[aid],
                "rank":           rank,
                "liked":          m.get("liked"),
                "generated_at":   None,
            })
        result[window] = window_results

    return result


def lookup_paper_by_id(
    con: sqlite3.Connection,
    user_id: int,
    arxiv_id: str,
) -> dict | None:
    """
    Look up a single paper by its canonical arXiv ID.

    Returns a Recommendation-shaped dict (score=None, rank=1, generated_at=None)
    if the paper exists in the database, or None if it is not found.
    """
    row = con.execute(
        """
        SELECT p.arxiv_id, p.title, p.authors, p.published_date,
               ul.liked
          FROM papers p
          LEFT JOIN user_papers ul
                 ON ul.arxiv_id = p.arxiv_id AND ul.user_id = ?
         WHERE p.arxiv_id = ?
        """,
        (user_id, arxiv_id),
    ).fetchone()

    if row is None:
        return None

    aid, title, authors_json, published_date, liked = row
    try:
        authors = json.loads(authors_json) if authors_json else []
    except Exception:
        authors = []

    return {
        "arxiv_id":       aid,
        "title":          title or "",
        "authors":        authors,
        "published_date": published_date,
        "score":          None,
        "rank":           1,
        "liked":          liked,
        "generated_at":   None,
    }
