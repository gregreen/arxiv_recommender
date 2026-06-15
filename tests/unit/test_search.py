"""
Unit tests for arxiv_lib/search.py.

Network calls (OpenAI embedding API) are mocked.
DB tests use the `app_db_con` + `data_dir` fixtures from conftest.py.
"""

import sqlite3
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import arxiv_lib.ingest as ingest
from arxiv_lib.config import (
    EMBEDDING_CACHE_DB,
    EMBEDDING_STORAGE_DIM,
    QUERY_VECTOR_DIM,
    RECOMMENDATION_EMBEDDING_DIM,
    SEARCH_EMBEDDING_DIM,
)
from arxiv_lib.ingest import store_search_term_embedding
from arxiv_lib.search import (
    SearchEmbeddingError,
    _cosine_similarity,
    _embed_query,
    cosine_to_search_score,
    lookup_paper_by_id,
    search_papers,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_QUERY = "dark matter halos"
_FAKE_EMBEDDING_4096 = [float(i) / 4096.0 for i in range(4096)]

def _make_openai_mock():
    mock_client = MagicMock()
    mock_item = MagicMock()
    mock_item.embedding = _FAKE_EMBEDDING_4096
    mock_client.embeddings.create.return_value.data = [mock_item]
    return mock_client


def _insert_paper(con: sqlite3.Connection, arxiv_id: str, published_date: str) -> None:
    con.execute(
        "INSERT OR IGNORE INTO papers (arxiv_id, title, authors, published_date) "
        "VALUES (?, ?, ?, ?)",
        (arxiv_id, f"Paper {arxiv_id}", '["Author A"]', published_date),
    )
    con.commit()


def _insert_search_embedding(data_dir, arxiv_id: str, vec: np.ndarray) -> None:
    ingest._init_embedding_db()
    with sqlite3.connect(EMBEDDING_CACHE_DB()) as con:
        con.execute(
            "INSERT OR REPLACE INTO search_embeddings VALUES (?, ?)",
            (arxiv_id, vec.astype(np.float32).tobytes()),
        )


# ---------------------------------------------------------------------------
# _embed_query
# ---------------------------------------------------------------------------


class TestEmbedQuery:
    def test_returns_truncated_float32_vector(self):
        """_embed_query returns float32 ndarray of length EMBEDDING_STORAGE_DIM."""
        client_mock = _make_openai_mock()
        with patch("arxiv_lib.search.OpenAI", return_value=client_mock):
            result = _embed_query(_QUERY)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (EMBEDDING_STORAGE_DIM,)

    def test_instruct_prefix_prepended(self):
        """The embedding API receives the query with the instruct prefix, not the raw query."""
        client_mock = _make_openai_mock()
        with patch("arxiv_lib.search.OpenAI", return_value=client_mock):
            _embed_query(_QUERY)

        call_args = client_mock.embeddings.create.call_args
        sent_input = call_args.kwargs.get("input") or call_args.args[0]
        assert sent_input.startswith("Instruct:")
        assert _QUERY in sent_input

    def test_api_failure_raises_search_embedding_error(self):
        """Network / API failure is wrapped in SearchEmbeddingError."""
        client_mock = MagicMock()
        client_mock.embeddings.create.side_effect = Exception("timeout")
        with patch("arxiv_lib.search.OpenAI", return_value=client_mock):
            with pytest.raises(SearchEmbeddingError):
                _embed_query(_QUERY)


# ---------------------------------------------------------------------------
# _cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_shape(self):
        rng = np.random.default_rng(0)
        a = rng.standard_normal(32).astype(np.float32)
        b = rng.standard_normal((20, 32)).astype(np.float32)
        result = _cosine_similarity(a, b)
        assert result.shape == (20,)

    def test_self_similarity_is_one(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = _cosine_similarity(a, a.reshape(1, -1))
        np.testing.assert_allclose(result[0], 1.0, atol=1e-6)

    def test_values_in_range(self):
        rng = np.random.default_rng(1)
        a = rng.standard_normal(64).astype(np.float32)
        b = rng.standard_normal((50, 64)).astype(np.float32)
        result = _cosine_similarity(a, b)
        assert np.all(result >= -1.0 - 1e-6)
        assert np.all(result <=  1.0 + 1e-6)


# ---------------------------------------------------------------------------
# search_papers
# ---------------------------------------------------------------------------


class TestSearchPapers:
    def test_cache_hit_skips_api(self, app_db_con, data_dir):
        """Pre-cached query vector is used; OpenAI never called."""
        today = date.today().isoformat()
        arxiv_id = "2309.06676"
        _insert_paper(app_db_con, arxiv_id, today)

        unit_vec = np.ones(EMBEDDING_STORAGE_DIM, dtype=np.float32)
        unit_vec /= np.linalg.norm(unit_vec)
        _insert_search_embedding(data_dir, arxiv_id, unit_vec)
        store_search_term_embedding(_QUERY, unit_vec)

        with patch("arxiv_lib.search.OpenAI") as mock_openai_cls:
            result = search_papers(app_db_con, user_id=1, query=_QUERY)

        mock_openai_cls.assert_not_called()
        assert set(result.keys()) == {"day", "week", "month"}
        month_ids = [r["arxiv_id"] for r in result["month"]]
        assert arxiv_id in month_ids

    def test_window_partitioning(self, app_db_con, data_dir):
        """Papers appear only in the windows their published_date falls within."""
        today = date.today()
        papers = {
            "today":    (today.isoformat(),              "day"),
            "5daysago": ((today - timedelta(days=5)).isoformat(), "week"),
            "20daysago": ((today - timedelta(days=20)).isoformat(), "month"),
        }

        unit_vec = np.ones(EMBEDDING_STORAGE_DIM, dtype=np.float32)
        unit_vec /= np.linalg.norm(unit_vec)

        for arxiv_id, (pub_date, _) in papers.items():
            _insert_paper(app_db_con, arxiv_id, pub_date)
            _insert_search_embedding(data_dir, arxiv_id, unit_vec)

        store_search_term_embedding(_QUERY, unit_vec)

        with patch("arxiv_lib.search.OpenAI"):
            result = search_papers(app_db_con, user_id=1, query=_QUERY)

        day_ids   = {r["arxiv_id"] for r in result["day"]}
        week_ids  = {r["arxiv_id"] for r in result["week"]}
        month_ids = {r["arxiv_id"] for r in result["month"]}

        assert "today"     in day_ids
        assert "5daysago"  in week_ids
        assert "5daysago"  not in day_ids
        assert "20daysago" in month_ids
        assert "20daysago" not in week_ids

    def test_empty_db_returns_empty_windows(self, app_db_con, data_dir):
        """No papers in DB → all windows return empty lists."""
        unit_vec = np.ones(EMBEDDING_STORAGE_DIM, dtype=np.float32)
        store_search_term_embedding(_QUERY, unit_vec)

        with patch("arxiv_lib.search.OpenAI"):
            result = search_papers(app_db_con, user_id=1, query=_QUERY)

        assert all(v == [] for v in result.values())


# ---------------------------------------------------------------------------
# lookup_paper_by_id
# ---------------------------------------------------------------------------


def _make_mock_model(*, has_query_vectors: bool) -> MagicMock:
    model = MagicMock()
    if has_query_vectors:
        model.query_vectors = np.random.default_rng(0).standard_normal(
            (3, QUERY_VECTOR_DIM)
        ).astype(np.float32)
    else:
        model.query_vectors = None
    model.score_embeddings.return_value = np.array([0.75], dtype=np.float32)
    return model


class TestLookupPaperById:
    _AID = "2309.99999"
    _REC_VEC = np.ones(RECOMMENDATION_EMBEDDING_DIM, dtype=np.float32)
    _SEARCH_VEC = np.ones(QUERY_VECTOR_DIM, dtype=np.float32)

    def _insert(self, con: sqlite3.Connection) -> None:
        con.execute(
            "INSERT OR IGNORE INTO papers (arxiv_id, title, authors, published_date) "
            "VALUES (?, ?, ?, ?)",
            (self._AID, "Test Paper", '["Author"]', date.today().isoformat()),
        )
        con.commit()

    def test_returns_none_for_unknown_paper(self, app_db_con, data_dir):
        result = lookup_paper_by_id(app_db_con, user_id=1, arxiv_id="9999.00000")
        assert result is None

    def test_score_is_none_when_no_model(self, app_db_con, data_dir):
        """NotEnoughDataError → score field is None, paper is still returned."""
        from arxiv_lib.recommend import NotEnoughDataError
        self._insert(app_db_con)
        with patch("arxiv_lib.search.get_or_train_model", side_effect=NotEnoughDataError):
            result = lookup_paper_by_id(app_db_con, user_id=1, arxiv_id=self._AID)
        assert result is not None
        assert result["score"] is None

    def test_scores_paper_without_query_vectors(self, app_db_con, data_dir):
        """When model.query_vectors is None, score_embeddings is called without
        query_vectors_for_papers and the returned score is stored on the paper."""
        self._insert(app_db_con)
        model = _make_mock_model(has_query_vectors=False)
        rec_vecs = {self._AID: self._REC_VEC}

        with (
            patch("arxiv_lib.search.get_or_train_model", return_value=(model, None)),
            patch("arxiv_lib.search._load_vectors", return_value=rec_vecs),
        ):
            result = lookup_paper_by_id(app_db_con, user_id=1, arxiv_id=self._AID)

        model.score_embeddings.assert_called_once()
        kwargs = model.score_embeddings.call_args.kwargs or {}
        assert "query_vectors_for_papers" not in kwargs
        assert result["score"] == pytest.approx(0.75)

    def test_scores_paper_with_query_vectors(self, app_db_con, data_dir):
        """When model.query_vectors is set, the search-space embedding is loaded
        and forwarded as query_vectors_for_papers to avoid a cdist dimension mismatch."""
        self._insert(app_db_con)
        model = _make_mock_model(has_query_vectors=True)
        rec_vecs = {self._AID: self._REC_VEC}
        search_vecs = {self._AID: self._SEARCH_VEC}

        with (
            patch("arxiv_lib.search.get_or_train_model", return_value=(model, None)),
            patch("arxiv_lib.search._load_vectors", return_value=rec_vecs),
            patch("arxiv_lib.search._load_search_paper_vectors", return_value=search_vecs),
        ):
            result = lookup_paper_by_id(app_db_con, user_id=1, arxiv_id=self._AID)

        model.score_embeddings.assert_called_once()
        kwargs = model.score_embeddings.call_args.kwargs or {}
        q_passed = kwargs.get("query_vectors_for_papers")
        assert q_passed is not None
        assert q_passed.shape == (1, QUERY_VECTOR_DIM)
        assert result["score"] == pytest.approx(0.75)

    def test_score_is_none_when_search_embedding_missing(self, app_db_con, data_dir):
        """When model has query_vectors but the paper has no search embedding,
        scoring is skipped (score=None) rather than crashing."""
        self._insert(app_db_con)
        model = _make_mock_model(has_query_vectors=True)
        rec_vecs = {self._AID: self._REC_VEC}

        with (
            patch("arxiv_lib.search.get_or_train_model", return_value=(model, None)),
            patch("arxiv_lib.search._load_vectors", return_value=rec_vecs),
            patch("arxiv_lib.search._load_search_paper_vectors", return_value={}),
        ):
            result = lookup_paper_by_id(app_db_con, user_id=1, arxiv_id=self._AID)

        model.score_embeddings.assert_not_called()
        assert result["score"] is None


# ---------------------------------------------------------------------------
# cosine_to_search_score
# ---------------------------------------------------------------------------

class TestCosineToSearchScore:
    """Tests for the SHASH-based cosine→search-score conversion."""

    @staticmethod
    def _score(val) -> np.ndarray:
        return cosine_to_search_score(np.array(val))

    # -- shape preservation -------------------------------------------------

    def test_scalar_in_scalar_out(self):
        result = self._score(0.5)
        # scipy.stats.norm.cdf may return a numpy scalar rather than a 0-d
        # ndarray, but the shape should still be ().
        assert np.shape(result) == ()
        assert isinstance(result, (np.ndarray, np.floating))

    def test_1d_in_1d_out(self):
        result = self._score([0.1, 0.5, 0.9])
        assert result.shape == (3,)

    def test_2d_in_2d_out(self):
        result = self._score([[0.1, 0.5], [0.3, 0.7]])
        assert result.shape == (2, 2)

    # -- no NaN / Inf in normal range ---------------------------------------

    def test_no_nan_inf_normal_range(self):
        cos = np.linspace(-1, 1, 10001)
        result = cosine_to_search_score(cos)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_no_nan_inf_edges(self):
        result = self._score([-1.0, 0.0, 1.0])
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    # -- out-of-range clipping -----------------------------------------------

    def test_above_one_clipped(self):
        assert self._score(1.0) == self._score(1.0001)

    def test_below_neg_one_clipped(self):
        assert self._score(-1.0) == self._score(-1.0001)

    def test_out_of_range_no_nan(self):
        result = self._score([-2.0, -1.0001, 1.0001, 2.0])
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    # -- monotonicity --------------------------------------------------------

    def test_monotonic(self):
        cos = np.linspace(-1, 1, 1000)
        scores = cosine_to_search_score(cos)
        # Non-decreasing: at very low cosines the CDF fraction may be
        # clipped to 1e-9, producing a flat floor.
        assert np.all(np.diff(scores) >= 0)

    # -- score range ---------------------------------------------------------

    def test_all_scores_le_zero(self):
        cos = np.linspace(-1, 1, 10001)
        scores = cosine_to_search_score(cos)
        assert np.all(scores <= 0.0)

    # -- vectorised ----------------------------------------------------------

    def test_vectorised_same_as_elementwise(self):
        cos = np.array([-0.5, 0.0, 0.3, 0.7, 1.0])
        batch = cosine_to_search_score(cos)
        elem  = np.array([float(self._score(c)) for c in cos])
        assert np.allclose(batch, elem)
