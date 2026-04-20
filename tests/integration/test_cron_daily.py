"""Integration tests for scripts/cron_daily.py :: run().

Uses a real temporary SQLite DB (via the app_db_con fixture) and patches only
fetch_oaipmh_metadata so no network calls are made.

Covers:
  1. New papers are written to the DB and embed tasks enqueued at priority=1.
  2. Already-known papers are NOT re-enqueued (partition happens before write).
  3. A fetch failure for one category is logged and skipped; the next category
     still runs and its papers are enqueued.
"""

import json
from unittest.mock import patch

from arxiv_lib.config import APP_DB_PATH
from scripts.cron_daily import run


# ---------------------------------------------------------------------------
# Anonymised metadata payloads
# ---------------------------------------------------------------------------

_PAPER_A = {
    "title": "Magnetic reconnection rates in low-density stellar coronae",
    "authors": ["Jane Smith", "Robert Chen"],
    "abstract": "We investigate magnetic reconnection in the outer layers of stellar coronae.",
    "published_date": "2026-04-10T04:00:00Z",
    "categories": ["cs.LG"],
}

_PAPER_B = {
    "title": "Temporal bias and cultural memory in generative sequence models",
    "authors": ["Maria Torres", "David Kim"],
    "abstract": "Generative sequence models encode biases present at data collection time.",
    "published_date": "2026-04-10T04:00:00Z",
    "categories": ["cs.LG"],
}


def _pending_tasks(con):
    """Return all pending task_queue rows as a list of sqlite3.Row."""
    return con.execute(
        "SELECT * FROM task_queue WHERE status = 'pending'"
    ).fetchall()


def _paper_ids_in_db(con):
    """Return the set of arxiv_ids currently in the papers table."""
    return {row[0] for row in con.execute("SELECT arxiv_id FROM papers").fetchall()}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@patch("scripts.cron_daily.fetch_oaipmh_metadata")
def test_run_enqueues_new_papers(mock_fetch, app_db_con):
    """New papers are written to the DB and get an embed task at priority=1."""
    mock_fetch.return_value = {
        "2604.00001": _PAPER_A,
        "2604.00002": _PAPER_B,
    }

    total = run(["cs.LG"], db_path=APP_DB_PATH(), date="2026-04-10")

    assert total == 2

    # Both papers persisted
    assert _paper_ids_in_db(app_db_con) == {"2604.00001", "2604.00002"}

    # Two pending embed tasks
    tasks = _pending_tasks(app_db_con)
    assert len(tasks) == 2
    task_arxiv_ids = {json.loads(t["payload"])["arxiv_id"] for t in tasks}
    assert task_arxiv_ids == {"2604.00001", "2604.00002"}

    # Daily ingest uses priority=1 (lower = higher priority)
    assert all(t["priority"] == 1 for t in tasks)
    assert all(t["type"] == "embed" for t in tasks)


@patch("scripts.cron_daily.fetch_oaipmh_metadata")
def test_run_skips_already_known_papers(mock_fetch, app_db_con):
    """Already-known papers are not re-enqueued; metadata is refreshed for all."""
    # Pre-insert paper A to simulate it being known from a previous day
    app_db_con.execute(
        "INSERT INTO papers (arxiv_id, title) VALUES (?, ?)",
        ("2604.00001", "Old cached title"),
    )
    app_db_con.commit()

    mock_fetch.return_value = {
        "2604.00001": _PAPER_A,  # already known
        "2604.00002": _PAPER_B,  # new
    }

    total = run(["cs.LG"], db_path=APP_DB_PATH(), date="2026-04-10")

    # Only the new paper counts
    assert total == 1

    # Only one embed task enqueued (for the new paper)
    tasks = _pending_tasks(app_db_con)
    assert len(tasks) == 1
    assert json.loads(tasks[0]["payload"])["arxiv_id"] == "2604.00002"

    # Both papers written to DB (metadata refresh still happens for all)
    assert _paper_ids_in_db(app_db_con) == {"2604.00001", "2604.00002"}

    # The pre-existing paper's title is refreshed with the new fetched value
    row = app_db_con.execute(
        "SELECT title FROM papers WHERE arxiv_id = ?", ("2604.00001",)
    ).fetchone()
    assert row["title"] == _PAPER_A["title"]


@patch("scripts.cron_daily.fetch_oaipmh_metadata")
def test_run_fetch_failure_continues_to_next_category(mock_fetch, app_db_con):
    """A RuntimeError from one category is logged and skipped; subsequent categories run."""
    mock_fetch.side_effect = [
        RuntimeError("OAI-PMH request failed: 503"),  # cs.LG fails
        {"2604.00002": _PAPER_B},                     # stat.ML succeeds
    ]

    total = run(["cs.LG", "stat.ML"], db_path=APP_DB_PATH(), date="2026-04-10")

    # Only the paper from the succeeding category is enqueued
    assert total == 1
    tasks = _pending_tasks(app_db_con)
    assert len(tasks) == 1
    assert json.loads(tasks[0]["payload"])["arxiv_id"] == "2604.00002"
