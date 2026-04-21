"""
Unit tests for the summary / embedding pipeline in arxiv_lib/ingest.py.

All OpenAI API calls are mocked; no network access or real keys are needed.
Tests use the `data_dir` fixture from conftest.py to isolate all file I/O.
"""

import json
import os
import sqlite3
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import arxiv_lib.ingest as ingest
from arxiv_lib.config import EMBEDDING_CACHE_DB, EMBEDDING_STORAGE_DIM, SUMMARY_CACHE_DIR, SUMMARY_REQUIRED_HEADINGS
from arxiv_lib.ingest import (
    _validate_summary,
    fetch_recommendation_embedding,
    fetch_search_embedding,
    summarize_arxiv_paper,
)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

_ARXIV_ID = "2309.06676"

_FAKE_METADATA = {
    _ARXIV_ID: {
        "title": "A Test Paper",
        "authors": ["Alice A.", "Bob B."],
        "abstract": "We test things.",
        "published_date": "2023-09-13",
        "categories": ["cs.LG"],
    }
}

_FAKE_SUMMARY = (
    "Keywords: testing\n"
    "Scientific Questions: Does mocking work?\n"
    "Data: synthetic\n"
    "Methods: patching\n"
    "Results: yes\n"
    "Conclusions: it works\n"
    "Key takeaway: mock everything"
)

# A 4096-element embedding (matches real API output length)
_FAKE_EMBEDDING_4096 = [float(i) / 4096.0 for i in range(4096)]


def _make_embedding_mock():
    """Return a mock that looks like openai.OpenAI() with a working embeddings.create."""
    mock_client = MagicMock()
    mock_data_item = MagicMock()
    mock_data_item.embedding = _FAKE_EMBEDDING_4096
    mock_client.embeddings.create.return_value.data = [mock_data_item]
    return mock_client


def _make_openai_constructor_mock(client_mock):
    """Return a patch-ready mock for `arxiv_lib.ingest.OpenAI` that returns *client_mock*."""
    mock_openai = MagicMock(return_value=client_mock)
    return mock_openai


# ---------------------------------------------------------------------------
# summarize_arxiv_paper
# ---------------------------------------------------------------------------


class TestSummarizeArxivPaper:
    def test_cache_hit_skips_api(self, data_dir):
        """If a cached .txt file exists, return it without touching the OpenAI client."""
        cache_dir = SUMMARY_CACHE_DIR()
        cache_file = os.path.join(cache_dir, f"{_ARXIV_ID}.txt")
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, "w") as f:
            f.write(_FAKE_SUMMARY)

        with patch("arxiv_lib.ingest.OpenAI") as mock_openai_cls:
            result = summarize_arxiv_paper(_ARXIV_ID)

        assert result == _FAKE_SUMMARY
        mock_openai_cls.assert_not_called()

    def test_strips_cot_think_tag(self, data_dir):
        """Content before (and including) </think> should be stripped from the response."""
        cot_response = "<think>chain of thought</think>\n" + _FAKE_SUMMARY

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content=cot_response))
        ]

        with (
            patch("arxiv_lib.ingest.OpenAI", return_value=mock_client),
            patch("arxiv_lib.ingest.get_arxiv_metadata", return_value=_FAKE_METADATA),
            patch("arxiv_lib.ingest.get_arxiv_source", return_value=r"\section{Intro} Hello"),
            patch("arxiv_lib.ingest.LLM_CONFIG", {
                "summary": {
                    "model": "test-model",
                    "max_input_tokens": 98304,
                    "base_url": "https://example.com/v1",
                    "api_key_name": "summary_api_key",
                    "cot_closing_tags": ["</think>", "</reasoning>"],
                    "completion_kwargs": {},
                }
            }),
        ):
            result = summarize_arxiv_paper(_ARXIV_ID)

        assert result == _FAKE_SUMMARY
        assert "<think>" not in result

    def test_summary_cached_to_disk(self, data_dir):
        """A freshly generated summary should be written to the cache dir."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content=_FAKE_SUMMARY))
        ]

        with (
            patch("arxiv_lib.ingest.OpenAI", return_value=mock_client),
            patch("arxiv_lib.ingest.get_arxiv_metadata", return_value=_FAKE_METADATA),
            patch("arxiv_lib.ingest.get_arxiv_source", return_value=r"\section{Intro} Hello"),
            patch("arxiv_lib.ingest.LLM_CONFIG", {
                "summary": {
                    "model": "test-model",
                    "max_input_tokens": 98304,
                    "base_url": "https://example.com/v1",
                    "api_key_name": "summary_api_key",
                    "cot_closing_tags": [],
                    "completion_kwargs": {},
                }
            }),
        ):
            result = summarize_arxiv_paper(_ARXIV_ID)

        cache_file = os.path.join(SUMMARY_CACHE_DIR(), f"{_ARXIV_ID}.txt")
        assert os.path.exists(cache_file)
        with open(cache_file) as f:
            assert f.read() == result

    def test_api_error_raises_runtime_error(self, data_dir):
        """A failing LLM call should raise RuntimeError (not leak the raw exception)."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("connection refused")

        with (
            patch("arxiv_lib.ingest.OpenAI", return_value=mock_client),
            patch("arxiv_lib.ingest.get_arxiv_metadata", return_value=_FAKE_METADATA),
            patch("arxiv_lib.ingest.get_arxiv_source", return_value=r"\section{Intro} Hello"),
            patch("arxiv_lib.ingest.LLM_CONFIG", {
                "summary": {
                    "model": "test-model",
                    "max_input_tokens": 98304,
                    "base_url": "https://example.com/v1",
                    "api_key_name": "summary_api_key",
                    "cot_closing_tags": [],
                    "completion_kwargs": {},
                }
            }),
        ):
            with pytest.raises(RuntimeError, match="Summary API call failed"):
                summarize_arxiv_paper(_ARXIV_ID)

    def test_truncated_llm_response_raises_and_does_not_cache(self, data_dir):
        """A summary missing required headings should raise RuntimeError and not be cached."""
        truncated = "Keywords: testing\nScientific Questions: Does mocking work?\n"
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content=truncated))
        ]

        with (
            patch("arxiv_lib.ingest.OpenAI", return_value=mock_client),
            patch("arxiv_lib.ingest.get_arxiv_metadata", return_value=_FAKE_METADATA),
            patch("arxiv_lib.ingest.get_arxiv_source", return_value=r"\section{Intro} Hello"),
            patch("arxiv_lib.ingest.LLM_CONFIG", {
                "summary": {
                    "model": "test-model",
                    "max_input_tokens": 98304,
                    "base_url": "https://example.com/v1",
                    "api_key_name": "summary_api_key",
                    "cot_closing_tags": [],
                    "completion_kwargs": {},
                }
            }),
        ):
            with pytest.raises(RuntimeError, match="Incomplete summary"):
                summarize_arxiv_paper(_ARXIV_ID)

        cache_file = os.path.join(SUMMARY_CACHE_DIR(), f"{_ARXIV_ID}.txt")
        assert not os.path.exists(cache_file), "Truncated summary must not be written to cache"


# ---------------------------------------------------------------------------
# _validate_summary
# ---------------------------------------------------------------------------


class TestValidateSummary:
    def test_valid_summary_passes(self):
        """A summary containing all required headings must not raise."""
        _validate_summary(_ARXIV_ID, _FAKE_SUMMARY)  # no exception

    def test_missing_one_heading_raises(self):
        """A summary missing one heading raises RuntimeError naming that heading."""
        incomplete = _FAKE_SUMMARY.replace("Key takeaway:", "")
        with pytest.raises(RuntimeError) as exc_info:
            _validate_summary(_ARXIV_ID, incomplete)
        assert "Key takeaway:" in str(exc_info.value)

    def test_truncated_summary_raises_listing_all_missing(self):
        """A severely truncated summary raises RuntimeError listing every missing heading."""
        truncated = "Keywords: foo\nScientific Questions: bar"
        with pytest.raises(RuntimeError) as exc_info:
            _validate_summary("1805.03653", truncated)
        msg = str(exc_info.value)
        for heading in ["Data:", "Methods:", "Results:", "Conclusions:", "Key takeaway:"]:
            assert heading in msg, f"Expected '{heading}' to appear in error message"

    def test_error_message_includes_length(self):
        """RuntimeError message should report the total summary length."""
        with pytest.raises(RuntimeError, match=r"total length: \d+ chars"):
            _validate_summary(_ARXIV_ID, "Keywords: only")

    def test_headings_parsed_from_prompt_file(self):
        """SUMMARY_REQUIRED_HEADINGS must contain exactly the 7 expected section labels."""
        expected = [
            "Keywords:", "Scientific Questions:", "Data:", "Methods:",
            "Results:", "Conclusions:", "Key takeaway:",
        ]
        assert SUMMARY_REQUIRED_HEADINGS == expected


# ---------------------------------------------------------------------------
# fetch_search_embedding
# ---------------------------------------------------------------------------


class TestFetchSearchEmbedding:
    def test_cache_hit_skips_api(self, data_dir):
        """If a vector already exists in search_embeddings, return it without API calls."""
        ingest._init_embedding_db()
        fake_vec = np.ones(EMBEDDING_STORAGE_DIM, dtype=np.float32)
        with sqlite3.connect(EMBEDDING_CACHE_DB()) as con:
            con.execute(
                "INSERT INTO search_embeddings VALUES (?, ?)",
                (_ARXIV_ID, fake_vec.tobytes()),
            )

        with patch("arxiv_lib.ingest.OpenAI") as mock_openai_cls:
            result = fetch_search_embedding(_ARXIV_ID)

        mock_openai_cls.assert_not_called()
        np.testing.assert_array_equal(result, fake_vec)

    def test_full_pipeline_truncates_to_storage_dim(self, data_dir):
        """End-to-end: API returns 4096 floats → stored and returned as 512-dim float32."""
        client_mock = _make_embedding_mock()

        with (
            patch("arxiv_lib.ingest.OpenAI", return_value=client_mock),
            patch("arxiv_lib.ingest.get_arxiv_metadata", return_value=_FAKE_METADATA),
            patch("arxiv_lib.ingest.summarize_arxiv_paper", return_value=_FAKE_SUMMARY),
            patch("arxiv_lib.ingest.LLM_CONFIG", {
                "embedding": {
                    "model": "test-embed-model",
                    "max_input_tokens": 24576,
                    "base_url": "https://example.com/v1",
                    "api_key_name": "embed_api_key",
                }
            }),
        ):
            result = fetch_search_embedding(_ARXIV_ID)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (EMBEDDING_STORAGE_DIM,)

    def test_vector_stored_in_db_and_reused(self, data_dir):
        """Second call should hit the DB cache; embedding API called only once."""
        client_mock = _make_embedding_mock()

        patch_kwargs = dict(
            get_arxiv_metadata=_FAKE_METADATA,
            summarize_paper=_FAKE_SUMMARY,
        )

        with (
            patch("arxiv_lib.ingest.OpenAI", return_value=client_mock),
            patch("arxiv_lib.ingest.get_arxiv_metadata", return_value=_FAKE_METADATA),
            patch("arxiv_lib.ingest.summarize_arxiv_paper", return_value=_FAKE_SUMMARY),
            patch("arxiv_lib.ingest.LLM_CONFIG", {
                "embedding": {
                    "model": "test-embed-model",
                    "max_input_tokens": 24576,
                    "base_url": "https://example.com/v1",
                    "api_key_name": "embed_api_key",
                }
            }),
        ):
            result1 = fetch_search_embedding(_ARXIV_ID)
            result2 = fetch_search_embedding(_ARXIV_ID)

        # API called exactly once (second call used DB cache)
        assert client_mock.embeddings.create.call_count == 1
        np.testing.assert_array_equal(result1, result2)


# ---------------------------------------------------------------------------
# fetch_recommendation_embedding
# ---------------------------------------------------------------------------


class TestFetchRecommendationEmbedding:
    def test_cache_hit_skips_api(self, data_dir):
        """If a vector already exists in recommendation_embeddings, return it without API calls."""
        ingest._init_embedding_db()
        fake_vec = np.full(EMBEDDING_STORAGE_DIM, 0.5, dtype=np.float32)
        with sqlite3.connect(EMBEDDING_CACHE_DB()) as con:
            con.execute(
                "INSERT INTO recommendation_embeddings VALUES (?, ?)",
                (_ARXIV_ID, fake_vec.tobytes()),
            )

        with patch("arxiv_lib.ingest.OpenAI") as mock_openai_cls:
            result = fetch_recommendation_embedding(_ARXIV_ID)

        mock_openai_cls.assert_not_called()
        np.testing.assert_array_equal(result, fake_vec)

    def test_full_pipeline(self, data_dir):
        """End-to-end: produces a float32 vector of length EMBEDDING_STORAGE_DIM."""
        client_mock = _make_embedding_mock()

        with (
            patch("arxiv_lib.ingest.OpenAI", return_value=client_mock),
            patch("arxiv_lib.ingest.get_arxiv_metadata", return_value=_FAKE_METADATA),
            patch("arxiv_lib.ingest.summarize_arxiv_paper", return_value=_FAKE_SUMMARY),
            patch("arxiv_lib.ingest.LLM_CONFIG", {
                "embedding": {
                    "model": "test-embed-model",
                    "max_input_tokens": 24576,
                    "base_url": "https://example.com/v1",
                    "api_key_name": "embed_api_key",
                }
            }),
        ):
            result = fetch_recommendation_embedding(_ARXIV_ID)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (EMBEDDING_STORAGE_DIM,)


# ---------------------------------------------------------------------------
# Old-style arXiv ID handling (slash IDs, e.g. "hep-th/9901001")
# ---------------------------------------------------------------------------

_OLD_ID = "hep-th/9901001"
_OLD_ID_CLEAN = "hep-th_9901001"   # sanitized form used in filenames

_OLD_FAKE_METADATA = {
    _OLD_ID: {
        "title": "Old-Style Paper",
        "authors": ["Carol C."],
        "abstract": "Old stuff.",
        "published_date": "1999-01-01",
        "categories": ["hep-th"],
    }
}


class TestOldStyleArxivId:
    def test_summarize_paper_cache_hit_old_style_id(self, data_dir):
        """Cache-read uses underscore filename; slash ID is resolved correctly."""
        cache_dir = SUMMARY_CACHE_DIR()
        cache_file = os.path.join(cache_dir, f"{_OLD_ID_CLEAN}.txt")
        with open(cache_file, "w") as f:
            f.write(_FAKE_SUMMARY)

        with patch("arxiv_lib.ingest.OpenAI") as mock_openai_cls:
            result = summarize_arxiv_paper(_OLD_ID)

        assert result == _FAKE_SUMMARY
        mock_openai_cls.assert_not_called()

    def test_summarize_paper_writes_underscore_filename(self, data_dir):
        """A generated summary for a slash ID is stored with underscores, not a slash path."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content=_FAKE_SUMMARY))
        ]

        with (
            patch("arxiv_lib.ingest.OpenAI", return_value=mock_client),
            patch("arxiv_lib.ingest.get_arxiv_metadata", return_value=_OLD_FAKE_METADATA),
            patch("arxiv_lib.ingest.get_arxiv_source", return_value=r"\section{Intro} Hello"),
            patch("arxiv_lib.ingest.LLM_CONFIG", {
                "summary": {
                    "model": "test-model",
                    "max_input_tokens": 98304,
                    "base_url": "https://example.com/v1",
                    "api_key_name": "summary_api_key",
                    "cot_closing_tags": [],
                    "completion_kwargs": {},
                }
            }),
        ):
            summarize_arxiv_paper(_OLD_ID)

        cache_dir = SUMMARY_CACHE_DIR()
        assert os.path.exists(os.path.join(cache_dir, f"{_OLD_ID_CLEAN}.txt"))
        # A slash path would create a subdirectory — that must not happen
        assert not os.path.exists(os.path.join(cache_dir, "hep-th", "9901001.txt"))

    def test_fetch_search_embedding_old_style_id_uses_raw_key_in_db(self, data_dir):
        """The DB key for a slash ID is stored as-is (with slash), not sanitized."""
        client_mock = _make_embedding_mock()

        with (
            patch("arxiv_lib.ingest.OpenAI", return_value=client_mock),
            patch("arxiv_lib.ingest.get_arxiv_metadata", return_value=_OLD_FAKE_METADATA),
            patch("arxiv_lib.ingest.summarize_arxiv_paper", return_value=_FAKE_SUMMARY),
            patch("arxiv_lib.ingest.LLM_CONFIG", {
                "embedding": {
                    "model": "test-embed-model",
                    "max_input_tokens": 24576,
                    "base_url": "https://example.com/v1",
                    "api_key_name": "embed_api_key",
                }
            }),
        ):
            fetch_search_embedding(_OLD_ID)

        with sqlite3.connect(EMBEDDING_CACHE_DB()) as con:
            row = con.execute(
                "SELECT 1 FROM search_embeddings WHERE arxiv_id = ?", (_OLD_ID,)
            ).fetchone()
        assert row is not None, "Row should be stored under the raw slash key"

    def test_fetch_search_embedding_old_style_cache_hit(self, data_dir):
        """Pre-inserted slash-key row is found and returned; OpenAI is never called."""
        ingest._init_embedding_db()
        fake_vec = np.full(EMBEDDING_STORAGE_DIM, 0.7, dtype=np.float32)
        with sqlite3.connect(EMBEDDING_CACHE_DB()) as con:
            con.execute(
                "INSERT INTO search_embeddings VALUES (?, ?)",
                (_OLD_ID, fake_vec.tobytes()),
            )

        with patch("arxiv_lib.ingest.OpenAI") as mock_openai_cls:
            result = fetch_search_embedding(_OLD_ID)

        mock_openai_cls.assert_not_called()
        np.testing.assert_array_equal(result, fake_vec)
