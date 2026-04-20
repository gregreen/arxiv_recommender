"""
Unit tests for arxiv_lib/recommend.py.

Uses the `app_db_con` + `data_dir` fixtures for DB isolation.
ScoringModel training uses synthetic vectors — no network calls needed.
"""

import json
import sqlite3
from datetime import date, timedelta

import numpy as np
import pytest

import arxiv_lib.ingest as ingest
from arxiv_lib.config import (
    EMBEDDING_CACHE_DB,
    RECOMMEND_MIN_LIKED,
    RECOMMENDATION_EMBEDDING_DIM,
    RECOMMEND_TIME_WINDOWS,
)
from arxiv_lib.recommend import (
    NotEnoughDataError,
    get_or_train_model,
    recommendations_are_stale,
    refresh_recommendations,
)
from arxiv_lib.scoring import ScoringModel, compute_model_hash

# ---------------------------------------------------------------------------
# DB setup helpers
# ---------------------------------------------------------------------------

_USER_ID = 1


def _insert_user(con: sqlite3.Connection) -> None:
    con.execute(
        "INSERT OR IGNORE INTO users (id, email, password_hash) VALUES (?, ?, ?)",
        (_USER_ID, "test@example.com", "hash"),
    )
    con.commit()


def _insert_paper(con: sqlite3.Connection, arxiv_id: str, pub_date: str) -> None:
    con.execute(
        "INSERT OR IGNORE INTO papers (arxiv_id, title, authors, published_date) "
        "VALUES (?, ?, ?, ?)",
        (arxiv_id, f"Paper {arxiv_id}", '["Author A"]', pub_date),
    )
    con.commit()


def _like_paper(con: sqlite3.Connection, arxiv_id: str) -> None:
    con.execute(
        "INSERT OR IGNORE INTO user_papers (user_id, arxiv_id, liked) VALUES (?, ?, 1)",
        (_USER_ID, arxiv_id),
    )
    con.commit()


def _insert_rec_embedding(data_dir, arxiv_id: str, vec: np.ndarray) -> None:
    ingest._init_embedding_db()
    with sqlite3.connect(EMBEDDING_CACHE_DB()) as con:
        con.execute(
            "INSERT OR REPLACE INTO recommendation_embeddings VALUES (?, ?)",
            (arxiv_id, vec.astype(np.float32).tobytes()),
        )


def _make_model(n: int = 20) -> ScoringModel:
    """Train a minimal ScoringModel on synthetic separable data."""
    rng = np.random.default_rng(seed=7)
    d = RECOMMENDATION_EMBEDDING_DIM
    center_pos = np.zeros(d); center_pos[0] =  3.0
    center_neg = np.zeros(d); center_neg[0] = -3.0
    pos = (center_pos + rng.normal(scale=0.1, size=(n, d))).astype(np.float32)
    neg = (center_neg + rng.normal(scale=0.1, size=(n, d))).astype(np.float32)
    return ScoringModel.from_training_data(pos, neg, positive_ids=[])


# ---------------------------------------------------------------------------
# recommendations_are_stale
# ---------------------------------------------------------------------------


class TestRecommendationsAreStale:
    def test_no_cache_is_stale(self, app_db_con):
        assert recommendations_are_stale(app_db_con, _USER_ID, "anyhash") is True

    def test_matching_hash_is_fresh(self, app_db_con):
        _insert_user(app_db_con)
        app_db_con.execute(
            "INSERT INTO recommendations "
            "(user_id, arxiv_id, time_window, score, rank, model_hash, generated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, datetime('now'))",
            (_USER_ID, "2309.06676", "month", -1.0, 1, "myhash"),
        )
        app_db_con.commit()
        assert recommendations_are_stale(app_db_con, _USER_ID, "myhash") is False

    def test_mismatched_hash_is_stale(self, app_db_con):
        _insert_user(app_db_con)
        app_db_con.execute(
            "INSERT INTO recommendations "
            "(user_id, arxiv_id, time_window, score, rank, model_hash, generated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, datetime('now'))",
            (_USER_ID, "2309.06676", "month", -1.0, 1, "oldhash"),
        )
        app_db_con.commit()
        assert recommendations_are_stale(app_db_con, _USER_ID, "newhash") is True


# ---------------------------------------------------------------------------
# get_or_train_model
# ---------------------------------------------------------------------------


class TestGetOrTrainModel:
    def test_raises_when_too_few_liked(self, app_db_con, data_dir):
        """Fewer than RECOMMEND_MIN_LIKED liked papers → NotEnoughDataError."""
        ingest._init_embedding_db()  # ensure embedding tables exist
        _insert_user(app_db_con)
        _insert_paper(app_db_con, "2309.06676", date.today().isoformat())
        _like_paper(app_db_con, "2309.06676")
        # 1 liked paper < RECOMMEND_MIN_LIKED (4) — no embedding needed
        with pytest.raises(NotEnoughDataError):
            get_or_train_model(app_db_con, _USER_ID)

    def test_returns_cached_model_without_retraining(self, app_db_con, data_dir):
        """A model with a matching hash is deserialized; user_models row count unchanged."""
        _insert_user(app_db_con)
        liked_ids = [f"2309.0{i:04d}" for i in range(RECOMMEND_MIN_LIKED)]
        for aid in liked_ids:
            _insert_paper(app_db_con, aid, date.today().isoformat())
            _like_paper(app_db_con, aid)
        model_hash = compute_model_hash(liked_ids, [], [])

        # Seed a pre-trained model in the cache
        original_model = _make_model()
        app_db_con.execute(
            "INSERT INTO user_models (user_id, model_blob, model_hash, n_liked, n_disliked) "
            "VALUES (?, ?, ?, ?, ?)",
            (_USER_ID, json.dumps(original_model.serialize()), model_hash, len(liked_ids), 0),
        )
        app_db_con.commit()
        row_count_before = app_db_con.execute(
            "SELECT COUNT(*) FROM user_models"
        ).fetchone()[0]

        returned_model, returned_hash = get_or_train_model(app_db_con, _USER_ID)

        # Same hash, no new row
        assert returned_hash == model_hash
        assert app_db_con.execute(
            "SELECT COUNT(*) FROM user_models"
        ).fetchone()[0] == row_count_before

        # Deserialized model produces identical scores to original
        rng = np.random.default_rng(seed=99)
        eval_vecs = rng.standard_normal((5, RECOMMENDATION_EMBEDDING_DIM)).astype(np.float32)
        np.testing.assert_allclose(
            original_model.score_embeddings(eval_vecs),
            returned_model.score_embeddings(eval_vecs),
            atol=1e-4,
        )


# ---------------------------------------------------------------------------
# refresh_recommendations
# ---------------------------------------------------------------------------


class TestRefreshRecommendations:
    def test_populates_all_windows_with_correct_partitioning(self, app_db_con, data_dir):
        """
        After refresh, the recommendations table has rows for all three windows.
        Papers appear only in windows whose cutoff they satisfy.
        Scores are all ≤ 0 (log-probabilities).
        Ranks start at 1.
        """
        _insert_user(app_db_con)
        today = date.today()
        papers = {
            "today":    today.isoformat(),
            "5daysago": (today - timedelta(days=5)).isoformat(),
            "20daysago": (today - timedelta(days=20)).isoformat(),
        }
        rng = np.random.default_rng(seed=42)
        unit = np.ones(RECOMMENDATION_EMBEDDING_DIM, dtype=np.float32)
        unit /= np.linalg.norm(unit)

        for arxiv_id, pub_date in papers.items():
            _insert_paper(app_db_con, arxiv_id, pub_date)
            # Give each paper a slightly different embedding so scores vary
            vec = unit + rng.normal(scale=0.01, size=RECOMMENDATION_EMBEDDING_DIM).astype(np.float32)
            _insert_rec_embedding(data_dir, arxiv_id, vec)

        model = _make_model()
        model_hash = "testhash"
        refresh_recommendations(app_db_con, _USER_ID, model, model_hash)
        app_db_con.commit()

        for window in RECOMMEND_TIME_WINDOWS:
            rows = app_db_con.execute(
                "SELECT arxiv_id, score, rank FROM recommendations "
                "WHERE user_id = ? AND time_window = ? ORDER BY rank",
                (_USER_ID, window),
            ).fetchall()
            assert len(rows) > 0, f"No recommendations for window {window!r}"
            assert rows[0][2] == 1, "Rank should start at 1"
            assert all(r[1] <= 0 for r in rows), "All scores should be log-probabilities ≤ 0"

        # "today" paper must appear in all windows; "20daysago" only in "month"
        day_ids   = {r[0] for r in app_db_con.execute(
            "SELECT arxiv_id FROM recommendations WHERE user_id=? AND time_window='day'",
            (_USER_ID,),
        ).fetchall()}
        month_ids = {r[0] for r in app_db_con.execute(
            "SELECT arxiv_id FROM recommendations WHERE user_id=? AND time_window='month'",
            (_USER_ID,),
        ).fetchall()}

        assert "today"     in day_ids
        assert "20daysago" in month_ids
        assert "20daysago" not in day_ids

    def test_refresh_replaces_old_recommendations(self, app_db_con, data_dir):
        """A second call to refresh_recommendations replaces the old rows (no duplicates)."""
        _insert_user(app_db_con)
        today = date.today().isoformat()
        _insert_paper(app_db_con, "2309.06676", today)
        vec = np.ones(RECOMMENDATION_EMBEDDING_DIM, dtype=np.float32)
        _insert_rec_embedding(data_dir, "2309.06676", vec)

        model = _make_model()
        refresh_recommendations(app_db_con, _USER_ID, model, "hash1")
        app_db_con.commit()
        refresh_recommendations(app_db_con, _USER_ID, model, "hash2")
        app_db_con.commit()

        # No duplicate rows: each (user_id, arxiv_id, time_window) should appear once
        duplicates = app_db_con.execute(
            "SELECT COUNT(*) FROM ("
            "  SELECT user_id, arxiv_id, time_window, COUNT(*) AS c "
            "  FROM recommendations WHERE user_id=? "
            "  GROUP BY user_id, arxiv_id, time_window HAVING c > 1"
            ")",
            (_USER_ID,),
        ).fetchone()[0]
        assert duplicates == 0
