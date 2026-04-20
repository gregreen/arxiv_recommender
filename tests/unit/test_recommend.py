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
    aggregate_group_scores,
    get_group_recommendations,
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
# aggregate_group_scores
# ---------------------------------------------------------------------------


class TestAggregateGroupScores:
    def test_empty_list_returns_empty(self):
        """An empty member list produces an empty result."""
        assert aggregate_group_scores([]) == {}

    def test_all_empty_dicts_returns_empty(self):
        """A list of empty member dicts (no scores) produces an empty result."""
        assert aggregate_group_scores([{}, {}, {}]) == {}

    def test_output_keys_are_union_of_inputs(self):
        """Every paper that appears in at least one member's scores must appear
        in the output; no extra keys are added."""
        m1 = {"p1": -1.0, "p2": -2.0}
        m2 = {"p2": -0.5, "p3": -3.0}
        m3 = {"p1": -0.8, "p3": -1.5, "p4": -4.0}
        result = aggregate_group_scores([m1, m2, m3])
        assert set(result.keys()) == {"p1", "p2", "p3", "p4"}

    def test_all_output_values_in_valid_range(self):
        """For every paper, math.log(v) must be <= 0 (up to a small epsilon for
        float rounding).  Input scores are realistic log-probabilities in [-20, 0]."""
        import math
        rng = np.random.default_rng(seed=42)
        paper_ids = [f"p{i}" for i in range(20)]
        members = [
            {aid: float(rng.uniform(-20.0, 0.0)) for aid in paper_ids}
            for _ in range(5)
        ]
        result = aggregate_group_scores(members)
        assert set(result.keys()) == set(paper_ids)
        for aid, v in result.items():
            assert math.log(v) <= 1e-9, (
                f"score for {aid!r} out of range: log({v}) = {math.log(v)}"
            )

    def test_consistent_ordering_preserved(self):
        """When every member ranks papers in the same order, the aggregated scores
        must preserve that consensus ranking.

        Setup: for each of 4 members, pick a random range [lo, hi] with hi <= 0,
        draw 5 scores uniformly from that range, sort them descending, and assign
        them to papers [p0, p1, p2, p3, p4] in that order so p0 is always ranked
        first and p4 last."""
        rng = np.random.default_rng(seed=7)
        paper_ids = ["p0", "p1", "p2", "p3", "p4"]
        members = []
        for _ in range(4):
            # pick lo < hi <= 0
            hi = float(rng.uniform(-2.0, 0.0))
            lo = float(rng.uniform(-20.0, hi - 0.1))
            raw = rng.uniform(lo, hi, size=5)
            scores = sorted(raw, reverse=True)  # largest first → p0 gets highest score
            members.append(dict(zip(paper_ids, scores)))

        result = aggregate_group_scores(members)
        ranked = sorted(paper_ids, key=lambda aid: result[aid], reverse=True)
        assert ranked == paper_ids, (
            f"Expected ordering p0>p1>p2>p3>p4, got {ranked}\n"
            f"Scores: { {aid: result[aid] for aid in paper_ids} }"
        )


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


# ---------------------------------------------------------------------------
# get_group_recommendations
# ---------------------------------------------------------------------------

# --- helpers for multi-user group scenarios ---

def _insert_user_n(con: sqlite3.Connection, user_id: int, email: str) -> None:
    con.execute(
        "INSERT OR IGNORE INTO users (id, email, password_hash, is_active, email_verified) "
        "VALUES (?, ?, 'hash', 1, 1)",
        (user_id, email),
    )
    con.commit()


def _like_paper_as(con: sqlite3.Connection, arxiv_id: str, user_id: int) -> None:
    con.execute(
        "INSERT OR IGNORE INTO user_papers (user_id, arxiv_id, liked) VALUES (?, ?, 1)",
        (user_id, arxiv_id),
    )
    con.commit()


def _insert_group(con: sqlite3.Connection, *member_user_ids: int) -> int:
    """Create a group, add members (first is admin), return group_id."""
    con.execute("INSERT INTO groups (name) VALUES ('Test Group')")
    group_id = con.execute("SELECT last_insert_rowid()").fetchone()[0]
    for i, uid in enumerate(member_user_ids):
        con.execute(
            "INSERT INTO group_members (group_id, user_id, is_admin) VALUES (?, ?, ?)",
            (group_id, uid, 1 if i == 0 else 0),
        )
    con.commit()
    return group_id


def _build_corpus(
    con: sqlite3.Connection, data_dir, n: int = 70
) -> list[str]:
    """Insert n background papers with neg-cluster embeddings; return their arxiv_ids."""
    ingest._init_embedding_db()
    rng = np.random.default_rng(seed=0)
    d = RECOMMENDATION_EMBEDDING_DIM
    center_neg = np.zeros(d, dtype=np.float32)
    center_neg[0] = -3.0
    today = date.today().isoformat()
    arxiv_ids = [f"bg{i:04d}" for i in range(n)]
    for i, aid in enumerate(arxiv_ids):
        con.execute(
            "INSERT OR IGNORE INTO papers (arxiv_id, title, authors, published_date) "
            "VALUES (?, ?, '[]', ?)",
            (aid, f"Background paper {i}", today),
        )
        vec = (center_neg + rng.normal(scale=0.05, size=d)).astype(np.float32)
        with sqlite3.connect(EMBEDDING_CACHE_DB()) as emb:
            emb.execute(
                "INSERT OR REPLACE INTO recommendation_embeddings VALUES (?, ?)",
                (aid, vec.tobytes()),
            )
    con.commit()
    return arxiv_ids


def _setup_trainable_user(
    con: sqlite3.Connection,
    data_dir,
    user_id: int,
    email: str,
    corpus_ids: list[str],
) -> list[str]:
    """Insert user, 4 liked papers with pos-cluster embeddings; return liked_ids."""
    ingest._init_embedding_db()
    _insert_user_n(con, user_id, email)
    rng = np.random.default_rng(seed=user_id * 100)
    d = RECOMMENDATION_EMBEDDING_DIM
    center_pos = np.zeros(d, dtype=np.float32)
    center_pos[0] = 3.0
    today = date.today().isoformat()
    liked_ids = [f"u{user_id}_{i}" for i in range(RECOMMEND_MIN_LIKED)]
    for aid in liked_ids:
        con.execute(
            "INSERT OR IGNORE INTO papers (arxiv_id, title, authors, published_date) "
            "VALUES (?, ?, '[]', ?)",
            (aid, f"Liked paper {aid}", today),
        )
        _like_paper_as(con, aid, user_id)
        vec = (center_pos + rng.normal(scale=0.05, size=d)).astype(np.float32)
        with sqlite3.connect(EMBEDDING_CACHE_DB()) as emb:
            emb.execute(
                "INSERT OR REPLACE INTO recommendation_embeddings VALUES (?, ?)",
                (aid, vec.tobytes()),
            )
    con.commit()
    return liked_ids


class TestGetGroupRecommendations:
    def test_invalid_time_window_raises(self, app_db_con, data_dir):
        """Passing an unrecognised time_window string must raise ValueError immediately."""
        _insert_user_n(app_db_con, 1, "u1@test.com")
        group_id = _insert_group(app_db_con, 1)
        with pytest.raises(ValueError, match="Unknown time window"):
            get_group_recommendations(app_db_con, group_id, "badwindow", 1)

    def test_empty_group_returns_empty(self, app_db_con, data_dir):
        """A group with no members returns empty results and zero counts."""
        group_id = _insert_group(app_db_con)
        results, group_member_count, active_member_count = get_group_recommendations(
            app_db_con, group_id, "week", 1
        )
        assert results == []
        assert group_member_count == 0
        assert active_member_count == 0

    def test_all_members_too_few_liked_returns_empty(self, app_db_con, data_dir):
        """When every group member has too few liked papers to train a model,
        results are empty, active_member_count is 0, but group_member_count is accurate."""
        ingest._init_embedding_db()
        for uid, email in [(1, "u1@test.com"), (2, "u2@test.com")]:
            _insert_user_n(app_db_con, uid, email)
            # 1 liked paper with no embedding → NotEnoughDataError for each user
            _insert_paper(app_db_con, f"sparse{uid}", date.today().isoformat())
            _like_paper_as(app_db_con, f"sparse{uid}", uid)
        group_id = _insert_group(app_db_con, 1, 2)
        results, group_member_count, active_member_count = get_group_recommendations(
            app_db_con, group_id, "week", 1
        )
        assert results == []
        assert group_member_count == 2
        assert active_member_count == 0

    def test_single_member_scores_in_valid_range(self, app_db_con, data_dir):
        """With one fully-trained member, all returned scores must be ≤ 0
        (they are log-probabilities; a small epsilon handles float rounding)."""
        corpus = _build_corpus(app_db_con, data_dir)
        _setup_trainable_user(app_db_con, data_dir, 1, "u1@test.com", corpus)
        group_id = _insert_group(app_db_con, 1)
        results, group_member_count, active_member_count = get_group_recommendations(
            app_db_con, group_id, "week", 1
        )
        assert len(results) > 0
        assert group_member_count == 1
        assert active_member_count == 1
        assert all(r["score"] <= 1e-9 for r in results), (
            f"Score out of range: {max(r['score'] for r in results)}"
        )

    def test_two_members_scores_in_valid_range(self, app_db_con, data_dir):
        """With two fully-trained members, all aggregated scores must still be ≤ 0
        and both members must appear in active_member_count."""
        corpus = _build_corpus(app_db_con, data_dir)
        _setup_trainable_user(app_db_con, data_dir, 1, "u1@test.com", corpus)
        _setup_trainable_user(app_db_con, data_dir, 2, "u2@test.com", corpus)
        group_id = _insert_group(app_db_con, 1, 2)
        results, group_member_count, active_member_count = get_group_recommendations(
            app_db_con, group_id, "week", 1
        )
        assert group_member_count == 2
        assert active_member_count == 2
        assert len(results) > 0
        assert all(r["score"] <= 1e-9 for r in results), (
            f"Score out of range: {max(r['score'] for r in results)}"
        )

    def test_partial_exclusion_active_member_count(self, app_db_con, data_dir):
        """Members with too few liked papers are silently excluded: they contribute
        to group_member_count but not active_member_count, and the remaining members
        still produce valid results with scores ≤ 0."""
        corpus = _build_corpus(app_db_con, data_dir)
        _setup_trainable_user(app_db_con, data_dir, 1, "u1@test.com", corpus)
        _setup_trainable_user(app_db_con, data_dir, 2, "u2@test.com", corpus)
        # User 3: 1 liked paper, no embedding → NotEnoughDataError
        ingest._init_embedding_db()
        _insert_user_n(app_db_con, 3, "u3@test.com")
        _insert_paper(app_db_con, "sparse3", date.today().isoformat())
        _like_paper_as(app_db_con, "sparse3", 3)
        group_id = _insert_group(app_db_con, 1, 2, 3)
        results, group_member_count, active_member_count = get_group_recommendations(
            app_db_con, group_id, "week", 1
        )
        assert group_member_count == 3
        assert active_member_count == 2
        assert len(results) > 0
        assert all(r["score"] <= 1e-9 for r in results), (
            f"Score out of range: {max(r['score'] for r in results)}"
        )

    def test_requesting_user_liked_status_reflected(self, app_db_con, data_dir):
        """The 'liked' field in each result reflects the requesting user's personal
        like status: 1 for papers they liked, None for papers they have no opinion on."""
        corpus = _build_corpus(app_db_con, data_dir)
        liked_ids = _setup_trainable_user(app_db_con, data_dir, 1, "u1@test.com", corpus)
        group_id = _insert_group(app_db_con, 1)
        results, _, _ = get_group_recommendations(app_db_con, group_id, "week", 1)
        assert len(results) > 0

        result_map = {r["arxiv_id"]: r for r in results}

        # At least one of the user's liked papers should appear in the results
        liked_in_results = [aid for aid in liked_ids if aid in result_map]
        assert liked_in_results, "Expected at least one liked paper in results"
        for aid in liked_in_results:
            assert result_map[aid]["liked"] == 1

        # Corpus (background) papers were not liked; their liked field should be None
        corpus_in_results = [aid for aid in corpus if aid in result_map]
        assert corpus_in_results, "Expected at least one corpus paper in results"
        for aid in corpus_in_results:
            assert result_map[aid]["liked"] is None
