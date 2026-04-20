"""
Integration tests for daemons/embed_daemon.py — process_one_task.

Uses the real app.db schema (via app_db_con fixture) and the real embedding
DB (via data_dir fixture).  All OpenAI API calls are mocked so no network
access is required.
"""

import json
import sqlite3
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from arxiv_lib.appdb import enqueue_task
from arxiv_lib.config import EMBEDDING_CACHE_DB
import arxiv_lib.ingest as ingest

# Import the daemon module so we can patch its namespace
import daemons.embed_daemon as embed_daemon
from daemons.embed_daemon import process_one_task

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_ARXIV_ID = "2309.06676"

_DUMMY_VECTOR = np.zeros(512, dtype=np.float32)


def _enqueue_embed(con: sqlite3.Connection) -> int:
    task_id = enqueue_task(con, "embed", {"arxiv_id": _ARXIV_ID}, priority=1)
    con.commit()
    return task_id


def _task_status(con: sqlite3.Connection, task_id: int) -> str:
    row = con.execute("SELECT status FROM task_queue WHERE id = ?", (task_id,)).fetchone()
    return row[0]


def _task_error(con: sqlite3.Connection, task_id: int) -> str | None:
    row = con.execute("SELECT error FROM task_queue WHERE id = ?", (task_id,)).fetchone()
    return row[0]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestProcessOneTask:
    def test_happy_path(self, app_db_con):
        """Enqueue a task; both fetchers succeed; task ends up 'done'."""
        _enqueue_embed(app_db_con)

        with (
            patch.object(embed_daemon, "fetch_search_embedding", return_value=_DUMMY_VECTOR) as mock_search,
            patch.object(embed_daemon, "fetch_recommendation_embedding", return_value=_DUMMY_VECTOR) as mock_rec,
        ):
            found = process_one_task(app_db_con)

        assert found is True
        task_id = app_db_con.execute("SELECT id FROM task_queue LIMIT 1").fetchone()[0]
        assert _task_status(app_db_con, task_id) == "done"
        mock_search.assert_called_once_with(_ARXIV_ID)
        mock_rec.assert_called_once_with(_ARXIV_ID)

    def test_empty_queue_returns_false(self, app_db_con):
        """Returns False immediately when the queue is empty."""
        result = process_one_task(app_db_con)
        assert result is False

    def test_already_ingested_skips_fetchers(self, app_db_con, data_dir):
        """If paper + both embeddings already exist, task completes without calling fetchers."""
        # Insert the paper into papers table
        app_db_con.execute(
            "INSERT INTO papers (arxiv_id, title) VALUES (?, ?)",
            (_ARXIV_ID, "A Test Paper"),
        )
        app_db_con.commit()

        # Insert fake vectors into the embedding DB
        ingest._init_embedding_db()
        vec_bytes = _DUMMY_VECTOR.tobytes()
        with sqlite3.connect(EMBEDDING_CACHE_DB()) as emb_con:
            emb_con.execute(
                "INSERT INTO search_embeddings VALUES (?, ?)", (_ARXIV_ID, vec_bytes)
            )
            emb_con.execute(
                "INSERT INTO recommendation_embeddings VALUES (?, ?)", (_ARXIV_ID, vec_bytes)
            )

        task_id = _enqueue_embed(app_db_con)

        with (
            patch.object(embed_daemon, "fetch_search_embedding") as mock_search,
            patch.object(embed_daemon, "fetch_recommendation_embedding") as mock_rec,
        ):
            found = process_one_task(app_db_con)

        assert found is True
        assert _task_status(app_db_con, task_id) == "done"
        mock_search.assert_not_called()
        mock_rec.assert_not_called()

    def test_fetcher_failure_retries_then_fails(self, app_db_con):
        """
        fail_task re-queues the task as 'pending' until max_attempts (3) is reached,
        after which the status permanently becomes 'failed'.
        """
        task_id = _enqueue_embed(app_db_con)

        with patch.object(
            embed_daemon, "fetch_search_embedding", side_effect=RuntimeError("API down")
        ):
            # Three attempts exhaust the retry budget (max_attempts=3 default)
            for _ in range(3):
                found = process_one_task(app_db_con)
                assert found is True

        assert _task_status(app_db_con, task_id) == "failed"
        error_text = _task_error(app_db_con, task_id)
        assert error_text is not None
        assert "API down" in error_text

    def test_fetcher_first_failure_reschedules(self, app_db_con):
        """After the first failure (attempt 1/3) the task should be reset to 'pending'."""
        task_id = _enqueue_embed(app_db_con)

        with patch.object(
            embed_daemon, "fetch_search_embedding", side_effect=RuntimeError("transient")
        ):
            found = process_one_task(app_db_con)

        assert found is True
        # Still pending — will be retried
        assert _task_status(app_db_con, task_id) == "pending"
        # But the error column is populated
        assert _task_error(app_db_con, task_id) is not None
