"""
Recommendation library — training, caching, and retrieval of per-user recommendations.

Public API
----------
get_recommendations(con, user_id, time_window) -> list[dict]
    Main entry point.  Returns a ranked list of papers for the given user and
    time window ('day', 'week', 'month').  Retrains the model and refreshes the
    recommendation cache if stale.

get_or_train_model(con, user_id) -> tuple[ScoringModel, str]
    Load a cached model if the hash matches; otherwise train a new one and
    persist it.  Raises NotEnoughDataError if the user has too few liked papers.

refresh_recommendations(con, user_id, model, model_hash) -> None
    Score all papers for all time windows and upsert the results into the
    recommendations table.

recommendations_are_stale(con, user_id, model_hash) -> bool
    True if the cache is missing, hash-mismatched, or new papers have arrived
    since the last generation.

All functions accept an open app.db sqlite3.Connection. The caller is
responsible for committing transactions when needed.
"""

import json
import random
import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np

from arxiv_lib.config import (
    APP_DB_PATH,
    BACKGROUND_NEGATIVE_COUNT,
    BACKGROUND_NEGATIVE_MIN_COUNT,
    EMBEDDING_CACHE_DB,
    RECOMMENDATION_EMBEDDING_DIM,
    MAX_DISLIKED_PAPERS_TO_USE,
    MAX_LIKED_PAPERS_TO_USE,
    MAX_MODEL_AGE_DAYS,
    MAX_QUERY_TERMS_TO_USE,
    RECOMMEND_MIN_LIKED,
    RECOMMEND_TIME_WINDOWS,
)
from arxiv_lib.ingest import load_search_term_embedding
from arxiv_lib.scoring import ScoringModel, compute_model_hash


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class NotEnoughDataError(ValueError):
    """Raised when a user has too few liked papers to train a model."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_vectors(arxiv_ids: list[str]) -> dict[str, np.ndarray]:
    """
    Load recommendation embeddings for the given arXiv IDs from embeddings_cache.db.

    Returns a dict mapping arxiv_id → truncated float32 vector (length RECOMMENDATION_EMBEDDING_DIM).
    IDs not found in the DB are silently omitted.
    """
    if not arxiv_ids:
        return {}
    placeholders = ",".join("?" * len(arxiv_ids))
    vectors: dict[str, np.ndarray] = {}
    with sqlite3.connect(EMBEDDING_CACHE_DB) as emb_con:
        rows = emb_con.execute(
            f"SELECT arxiv_id, vector FROM recommendation_embeddings WHERE arxiv_id IN ({placeholders})",
            arxiv_ids,
        ).fetchall()
    for arxiv_id, blob in rows:
        full = np.frombuffer(blob, dtype=np.float32)
        vectors[arxiv_id] = full[:RECOMMENDATION_EMBEDDING_DIM]
    return vectors


def _get_background_negative_ids(
    con: sqlite3.Connection,
    exclude_ids: set[str],
) -> list[str]:
    """
    Return up to BACKGROUND_NEGATIVE_COUNT arXiv IDs to use as background
    negative examples when training a scoring model.

    Selects a random sample of papers (excluding any the user has explicitly
    liked or disliked) so that the negative set evolves with the corpus.
    Only returns IDs that have both metadata and an embedding.
    """
    if exclude_ids:
        placeholders = ",".join("?" * len(exclude_ids))
        rows = con.execute(
            f"""
            SELECT p.arxiv_id
              FROM papers p
             WHERE p.published_date IS NOT NULL
               AND p.arxiv_id NOT IN ({placeholders})
             ORDER BY RANDOM()
             LIMIT ?
            """,
            list(exclude_ids) + [BACKGROUND_NEGATIVE_COUNT],
        ).fetchall()
    else:
        rows = con.execute(
            """
            SELECT p.arxiv_id
              FROM papers p
             WHERE p.published_date IS NOT NULL
             ORDER BY RANDOM()
             LIMIT ?
            """,
            (BACKGROUND_NEGATIVE_COUNT,),
        ).fetchall()
    # Filter to only IDs that have an embedding (cross-DB lookup done in Python).
    candidate_ids = [r[0] for r in rows]
    if not candidate_ids:
        return []
    with sqlite3.connect(EMBEDDING_CACHE_DB) as emb_con:
        placeholders = ",".join("?" * len(candidate_ids))
        embedded = {
            r[0] for r in emb_con.execute(
                f"SELECT arxiv_id FROM recommendation_embeddings WHERE arxiv_id IN ({placeholders})",
                candidate_ids,
            ).fetchall()
        }
    return [aid for aid in candidate_ids if aid in embedded]


def _window_cutoff(time_window: str, con: sqlite3.Connection) -> str:
    """Return a full ISO 8601 UTC datetime string for the start of the given
    time window, anchored at the most recent paper's published_date.

    1 second is added to the delta so papers at exactly the anchor boundary
    are included.
    """
    deltas = {"day": timedelta(days=1), "week": timedelta(weeks=1), "month": timedelta(days=30)}
    if time_window not in deltas:
        raise ValueError(f"Unknown time window: {time_window!r}. Expected one of {RECOMMEND_TIME_WINDOWS}")

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

    cutoff = anchor - deltas[time_window] + timedelta(seconds=1)
    return cutoff.strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def recommendations_are_stale(
    con: sqlite3.Connection,
    user_id: int,
    model_hash: str,
) -> bool:
    """
    Return True if the cached recommendations for *user_id* are stale.

    Stale if:
    - No recommendations exist for this user, OR
    - The stored model_hash differs from the current hash.
    """
    row = con.execute(
        "SELECT model_hash FROM recommendations WHERE user_id = ? LIMIT 1",
        (user_id,),
    ).fetchone()

    if row is None:
        return True  # no cached recommendations

    cached_hash = row[0]
    return cached_hash != model_hash


def get_or_train_model(
    con: sqlite3.Connection,
    user_id: int,
) -> tuple[ScoringModel, str]:
    """
    Return a trained ScoringModel for *user_id*, loading from cache if fresh.

    Raises
    ------
    NotEnoughDataError
        If the user has fewer than RECOMMEND_MIN_LIKED liked papers with embeddings.
    """
    # Load liked and disliked paper IDs, most-recently-added first, capped
    rows = con.execute(
        "SELECT arxiv_id, liked FROM user_papers WHERE user_id = ? AND liked != 0 ORDER BY added_at DESC",
        (user_id,),
    ).fetchall()
    liked_ids    = [r[0] for r in rows if r[1] ==  1][:MAX_LIKED_PAPERS_TO_USE]
    disliked_ids = [r[0] for r in rows if r[1] == -1][:MAX_DISLIKED_PAPERS_TO_USE]

    # Load query terms and their embeddings
    query_rows = con.execute(
        "SELECT query FROM user_search_terms WHERE user_id = ?"
        " ORDER BY last_searched_at DESC LIMIT ?",
        (user_id, MAX_QUERY_TERMS_TO_USE),
    ).fetchall()
    query_terms_with_embeddings = []
    query_vecs_list = []
    for (query,) in query_rows:
        vec = load_search_term_embedding(query)
        if vec is not None:
            query_terms_with_embeddings.append(query)
            query_vecs_list.append(vec[:RECOMMENDATION_EMBEDDING_DIM])
    query_vectors = (
        np.array(query_vecs_list, dtype=np.float32)
        if query_vecs_list else None
    )

    model_hash = compute_model_hash(liked_ids, disliked_ids, query_terms_with_embeddings)

    # Check for a cached model with a matching hash
    cached = con.execute(
        "SELECT model_blob, trained_at FROM user_models WHERE user_id = ? AND model_hash = ?",
        (user_id, model_hash),
    ).fetchone()
    if cached is not None:
        # Force retrain if the model is too old (background negatives may have drifted)
        trained_at = datetime.fromisoformat(cached[1].rstrip("Z")).replace(tzinfo=timezone.utc)
        age_days = (datetime.now(tz=timezone.utc) - trained_at).days
        if age_days <= MAX_MODEL_AGE_DAYS:
            model = ScoringModel.deserialize(json.loads(cached[0]))
            return model, model_hash

    # Need to retrain — load vectors
    liked_vectors    = _load_vectors(liked_ids)
    disliked_vectors = _load_vectors(disliked_ids)
    exclude_ids      = set(liked_ids) | set(disliked_ids)
    background_ids   = _get_background_negative_ids(con, exclude_ids)
    background_vectors = _load_vectors(background_ids)

    # Validate training set
    pos_ids_list = [aid for aid in liked_vectors]
    v_pos_list = [liked_vectors[aid] for aid in pos_ids_list]
    if len(v_pos_list) < RECOMMEND_MIN_LIKED:
        raise NotEnoughDataError(
            f"Need at least {RECOMMEND_MIN_LIKED} liked papers with embeddings to train "
            f"(have {len(v_pos_list)})."
        )

    neg_dict = {**background_vectors, **disliked_vectors}
    if len(neg_dict) < BACKGROUND_NEGATIVE_MIN_COUNT:
        raise NotEnoughDataError(
            f"Need at least {BACKGROUND_NEGATIVE_MIN_COUNT} negative papers with embeddings "
            f"(have {len(neg_dict)})."
        )

    v_pos = np.array(v_pos_list, dtype=np.float32)
    v_neg = np.array(list(neg_dict.values()), dtype=np.float32)
    model = ScoringModel.from_training_data(
        v_pos, v_neg,
        positive_ids=pos_ids_list,
        query_vectors=query_vectors
    )

    # Persist the trained model
    con.execute(
        """
        INSERT OR REPLACE INTO user_models
            (user_id, model_blob, model_hash, n_liked, n_disliked, trained_at)
        VALUES (?, ?, ?, ?, ?, datetime('now'))
        """,
        (user_id, json.dumps(model.serialize()), model_hash, len(liked_ids), len(disliked_ids)),
    )

    return model, model_hash


def refresh_recommendations(
    con: sqlite3.Connection,
    user_id: int,
    model: ScoringModel,
    model_hash: str,
) -> None:
    """
    Score all embedded papers across all time windows and replace the results
    in the recommendations table.

    Liked papers are scored with score_positive_embeddings() (avoids
    self-similarity bias); all other papers use score_embeddings().
    """
    # Pre-compute cutoffs and restrict paper loading to the earliest one
    window_cutoffs = {w: _window_cutoff(w, con) for w in RECOMMEND_TIME_WINDOWS}
    earliest_cutoff = min(window_cutoffs.values())

    papers = con.execute(
        "SELECT arxiv_id, published_date FROM papers "
        "WHERE published_date IS NOT NULL AND published_date >= ?",
        (earliest_cutoff,),
    ).fetchall()
    all_ids = [r[0] for r in papers]
    pub_dates = {r[0]: r[1] for r in papers}

    if not all_ids:
        return

    all_vectors = _load_vectors(all_ids)
    # Keep only IDs that have both metadata and an embedding
    scoreable_ids = [aid for aid in all_ids if aid in all_vectors]

    # Identify liked papers — use self-similarity-corrected scoring for them
    liked_set = set(model.positive_ids)

    non_liked_ids = [aid for aid in scoreable_ids if aid not in liked_set]
    liked_scoreable = [aid for aid in scoreable_ids if aid in liked_set]

    scores: dict[str, float] = {}
    if non_liked_ids:
        v_non_liked = np.array([all_vectors[aid] for aid in non_liked_ids], dtype=np.float32)
        s = model.score_embeddings(v_non_liked)
        for aid, sc in zip(non_liked_ids, s):
            scores[aid] = float(sc)

    if liked_scoreable:
        # score_positive_embeddings() returns scores in model.positive_ids order,
        # correcting for self-similarity (diagonal → median replacement in RBF).
        pos_scores = dict(zip(model.positive_ids, model.score_positive_embeddings()))
        # Some liked papers may not have been in the training set (embedding
        # was missing at train time).  Fall back to biased scoring for those only.
        stragglers = [aid for aid in liked_scoreable if aid not in pos_scores]
        for aid in liked_scoreable:
            if aid in pos_scores:
                scores[aid] = pos_scores[aid]
        if stragglers:
            v_stragglers = np.array([all_vectors[aid] for aid in stragglers], dtype=np.float32)
            s = model.score_embeddings(v_stragglers)
            for aid, sc in zip(stragglers, s):
                scores[aid] = float(sc)

    # Replace recommendations for each time window (DELETE then INSERT)
    generated_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for window in RECOMMEND_TIME_WINDOWS:
        cutoff = window_cutoffs[window]
        window_ids = [
            aid for aid in scoreable_ids
            if aid in scores and pub_dates.get(aid, "") >= cutoff
        ]
        if not window_ids:
            continue
        window_scores = [(aid, scores[aid]) for aid in window_ids]
        window_scores.sort(key=lambda x: x[1], reverse=True)
        rows_to_insert = [
            (user_id, aid, window, score, rank, model_hash, generated_at)
            for rank, (aid, score) in enumerate(window_scores, start=1)
        ]
        con.execute(
            "DELETE FROM recommendations WHERE user_id = ? AND time_window = ?",
            (user_id, window),
        )
        con.executemany(
            """
            INSERT INTO recommendations
                (user_id, arxiv_id, time_window, score, rank, model_hash, generated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows_to_insert,
        )


def get_recommendations(
    con: sqlite3.Connection,
    user_id: int,
    time_window: str,
) -> list[dict]:
    """
    Return a ranked list of recommended papers for *user_id* in *time_window*.

    Retrains the model and refreshes the cache if stale.  The returned list is
    ordered by rank (ascending) and includes all papers in the time window,
    including liked ones.

    Parameters
    ----------
    con : sqlite3.Connection
        Open app.db connection.
    user_id : int
        The user to generate recommendations for.
    time_window : str
        One of 'day', 'week', 'month'.

    Returns
    -------
    list[dict]
        Each dict has keys: arxiv_id, title, authors, published_date, score, rank,
        liked (1 / 0 / -1 or None).

    Raises
    ------
    NotEnoughDataError
        If the user has too few liked papers to produce recommendations.
    ValueError
        If time_window is not one of the supported values.
    """
    if time_window not in RECOMMEND_TIME_WINDOWS:
        raise ValueError(f"Unknown time window: {time_window!r}. Expected one of {RECOMMEND_TIME_WINDOWS}")

    # Compute current hash (load papersets again — cheap)
    rows = con.execute(
        "SELECT arxiv_id, liked FROM user_papers WHERE user_id = ? AND liked != 0",
        (user_id,),
    ).fetchall()
    liked_ids    = [r[0] for r in rows if r[1] == 1]
    disliked_ids = [r[0] for r in rows if r[1] == -1]
    model_hash   = compute_model_hash(liked_ids, disliked_ids)

    if recommendations_are_stale(con, user_id, model_hash):
        model, model_hash = get_or_train_model(con, user_id)
        refresh_recommendations(con, user_id, model, model_hash)
        con.commit()

    result_rows = con.execute(
        """
        SELECT r.arxiv_id, p.title, p.authors, p.published_date,
               r.score, r.rank, up.liked, r.generated_at
          FROM recommendations r
          JOIN papers p ON p.arxiv_id = r.arxiv_id
          LEFT JOIN user_papers up
               ON up.arxiv_id = r.arxiv_id AND up.user_id = r.user_id
         WHERE r.user_id = ? AND r.time_window = ?
         ORDER BY r.rank ASC
        """,
        (user_id, time_window),
    ).fetchall()

    return [
        {
            "arxiv_id":       row[0],
            "title":          row[1],
            "authors":        json.loads(row[2]) if row[2] else [],
            "published_date": row[3],
            "score":          row[4],
            "rank":           row[5],
            "liked":          row[6],
            "generated_at":   row[7],
        }
        for row in result_rows
    ]


def get_onboarding_papers(
    con: sqlite3.Connection,
    time_window: str,
    limit: int,
    seed: int = 0,
) -> list[dict]:
    """
    Return up to *limit* papers published within *time_window*, in a stable order.

    The order is determined by *seed* (typically the user_id), so the same user
    always sees the same ordering for a given set of papers.  Papers are fetched
    in arxiv_id order then shuffled in Python with the seed, which is cheaper
    than relying on SQLite's non-seedable RANDOM() and avoids reordering on
    every request.

    Used when the user has too few liked papers to generate scored recommendations.
    Returned dicts have the same keys as get_recommendations() but with
    score=None, rank=None, and generated_at=None.
    """
    cutoff = _window_cutoff(time_window, con)
    rows = con.execute(
        """
        SELECT arxiv_id, title, authors, published_date
          FROM papers
         WHERE published_date >= ? AND published_date IS NOT NULL
         ORDER BY arxiv_id
        """,
        (cutoff,),
    ).fetchall()
    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:limit]
    return [
        {
            "arxiv_id":       row[0],
            "title":          row[1] or "",
            "authors":        json.loads(row[2]) if row[2] else [],
            "published_date": row[3],
            "score":          None,
            "rank":           None,
            "liked":          None,
            "generated_at":   None,
        }
        for row in rows
    ]
