"""
Unit tests for the embed task retry schedule (not_before delays, max attempts).
"""

import sqlite3
from datetime import datetime

import pytest

from arxiv_lib.appdb import (
    claim_next_task,
    enqueue_task,
    fail_task,
    get_connection,
    init_app_db,
)
from arxiv_lib.config import APP_DB_PATH, EMBED_RETRY_DELAYS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _enqueue_and_claim_embed(con: sqlite3.Connection, arxiv_id: str) -> sqlite3.Row:
    """Enqueue an embed task, claim it (increments attempts), return the row."""
    enqueue_task(con, "embed", {"arxiv_id": arxiv_id})
    con.commit()
    row = claim_next_task(con, "embed")
    con.commit()
    return row


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEmbedRetrySchedule:
    @pytest.fixture()
    def con(self, data_dir):
        init_app_db(APP_DB_PATH())
        c = get_connection(APP_DB_PATH())
        c.execute(
            "INSERT INTO users (email, password_hash, is_active, email_verified) "
            "VALUES ('u@example.com', 'x', 1, 1)"
        )
        c.commit()
        yield c
        c.close()

    def test_retry_not_before_per_attempt(self, con):
        """Each embed retry delay sets not_before correctly."""
        task = _enqueue_and_claim_embed(con, "2309.00030")
        for i, delay in enumerate(EMBED_RETRY_DELAYS):
            con.execute(
                "UPDATE task_queue SET attempts=? WHERE id=?", (i + 1, task["id"])
            )
            fail_task(con, task["id"], "err", max_attempts=5,
                      retry_delay_seconds=delay)
            con.commit()
            row = con.execute(
                "SELECT status, not_before FROM task_queue WHERE id=?",
                (task["id"],),
            ).fetchone()
            assert row["status"] == "pending", f"attempt {i + 1}"
            if delay == 0:
                assert row["not_before"] is None, "delay=0 → not_before NULL"
            else:
                assert row["not_before"] is not None, \
                    f"delay={delay} → not_before set"

    def test_permanent_failure_after_5_attempts(self, con):
        """After 5 failures (max_attempts=5), embed task is permanently failed."""
        task = _enqueue_and_claim_embed(con, "2309.00031")
        for attempt in range(1, 5):
            con.execute(
                "UPDATE task_queue SET attempts=? WHERE id=?", (attempt, task["id"])
            )
            result = fail_task(con, task["id"], "err", max_attempts=5)
            con.commit()
            assert not result, f"attempt {attempt}"
        con.execute("UPDATE task_queue SET attempts=5 WHERE id=?", (task["id"],))
        result = fail_task(con, task["id"], "final err", max_attempts=5)
        con.commit()
        assert result
        row = con.execute(
            "SELECT status FROM task_queue WHERE id=?", (task["id"],)
        ).fetchone()
        assert row["status"] == "failed"

    def test_not_before_prevents_claim(self, con):
        """claim_next_task('embed') skips tasks with future not_before."""
        task = _enqueue_and_claim_embed(con, "2309.00032")
        con.execute("UPDATE task_queue SET attempts=1 WHERE id=?", (task["id"],))
        fail_task(con, task["id"], "err", max_attempts=5,
                  retry_delay_seconds=3600)
        con.commit()
        claimed = claim_next_task(con, "embed")
        con.commit()
        assert claimed is None
