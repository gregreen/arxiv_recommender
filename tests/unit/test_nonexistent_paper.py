"""
Unit tests for arxiv_lib/appdb.py additions:
  - fail_task return value (bool)
  - fail_task not_before / retry delay
  - claim_next_tasks_batch respects not_before
  - remove_nonexistent_paper
And for daemons/meta_daemon.py:
  - exponential backoff on fetch_meta failure
  - auto-removal after permanent failure (no HEAD check)
"""

import json
import sqlite3
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from arxiv_lib.appdb import (
    claim_next_tasks_batch,
    enqueue_task,
    fail_task,
    get_connection,
    init_app_db,
    remove_nonexistent_paper,
)
from arxiv_lib.config import APP_DB_PATH


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def con(data_dir):
    """Initialised in-memory app.db connection, isolated per test."""
    init_app_db(APP_DB_PATH())
    c = get_connection(APP_DB_PATH())
    # Insert a dummy user so foreign-key constraints on user_papers pass.
    c.execute(
        "INSERT INTO users (email, password_hash, is_active, email_verified) "
        "VALUES ('u@example.com', 'x', 1, 1)"
    )
    c.commit()
    yield c
    c.close()


def _user_id(con) -> int:
    return con.execute("SELECT id FROM users LIMIT 1").fetchone()[0]


def _enqueue_and_claim(con, arxiv_id: str) -> sqlite3.Row:
    """Enqueue a fetch_meta task, claim it (increments attempts), return the row."""
    enqueue_task(con, "fetch_meta", {"arxiv_id": arxiv_id})
    con.commit()
    rows = claim_next_tasks_batch(con, "fetch_meta", 1)
    con.commit()
    return rows[0]


# ---------------------------------------------------------------------------
# fail_task — return value
# ---------------------------------------------------------------------------


class TestFailTaskReturnValue:
    def test_returns_false_while_retrying(self, con):
        """fail_task should return False while attempts < max_attempts."""
        task = _enqueue_and_claim(con, "2309.00001")
        # After 1 attempt, max_attempts=3 → should retry (False)
        result = fail_task(con, task["id"], "oops", max_attempts=3)
        con.commit()
        assert result is False

    def test_returns_true_on_final_attempt(self, con):
        """fail_task should return True when attempts == max_attempts."""
        arxiv_id = "2309.00001"
        task = _enqueue_and_claim(con, arxiv_id)
        task_id = task["id"]
        # Exhaust retries: fail max_attempts-1 times (re-claim each time)
        max_attempts = 3
        for i in range(max_attempts - 1):
            result = fail_task(con, task_id, "oops", max_attempts=max_attempts)
            con.commit()
            assert result is False, f"Expected False on attempt {i + 1}"
            rows = claim_next_tasks_batch(con, "fetch_meta", 1)
            con.commit()
            task_id = rows[0]["id"]

        # Final attempt → permanently failed
        result = fail_task(con, task_id, "oops", max_attempts=max_attempts)
        con.commit()
        assert result is True

    def test_returns_false_for_missing_task(self, con):
        """fail_task on a non-existent task_id should return False (no-op)."""
        result = fail_task(con, 99999, "oops")
        assert result is False

    def test_status_is_failed_after_true_return(self, con):
        """When fail_task returns True, the task row must have status='failed'."""
        task = _enqueue_and_claim(con, "2309.00001")
        fail_task(con, task["id"], "oops", max_attempts=1)
        con.commit()
        row = con.execute(
            "SELECT status FROM task_queue WHERE id=?", (task["id"],)
        ).fetchone()
        assert row["status"] == "failed"


# ---------------------------------------------------------------------------
# remove_nonexistent_paper
# ---------------------------------------------------------------------------


class TestRemoveNonexistentPaper:
    def _add_user_paper(self, con, arxiv_id: str) -> None:
        uid = _user_id(con)
        con.execute(
            "INSERT INTO user_papers (user_id, arxiv_id, liked) VALUES (?, ?, 1)",
            (uid, arxiv_id),
        )
        con.commit()

    def test_removes_user_papers_rows(self, con):
        self._add_user_paper(con, "2309.00001")
        n = remove_nonexistent_paper(con, "2309.00001")
        con.commit()
        assert n == 1
        row = con.execute(
            "SELECT 1 FROM user_papers WHERE arxiv_id='2309.00001'"
        ).fetchone()
        assert row is None

    def test_returns_count_of_affected_users(self, con):
        # Add a second user and give both users the same paper
        con.execute(
            "INSERT INTO users (email, password_hash, is_active, email_verified) "
            "VALUES ('u2@example.com', 'x', 1, 1)"
        )
        uid1 = con.execute("SELECT id FROM users WHERE email='u@example.com'").fetchone()[0]
        uid2 = con.execute("SELECT id FROM users WHERE email='u2@example.com'").fetchone()[0]
        con.execute(
            "INSERT INTO user_papers (user_id, arxiv_id, liked) VALUES (?, ?, 1)",
            (uid1, "2309.00001"),
        )
        con.execute(
            "INSERT INTO user_papers (user_id, arxiv_id, liked) VALUES (?, ?, 1)",
            (uid2, "2309.00001"),
        )
        con.commit()
        n = remove_nonexistent_paper(con, "2309.00001")
        con.commit()
        assert n == 2

    def test_returns_zero_when_no_rows(self, con):
        n = remove_nonexistent_paper(con, "2309.99999")
        con.commit()
        assert n == 0

    def test_removes_papers_table_row(self, con):
        con.execute(
            "INSERT INTO papers (arxiv_id, title, abstract, authors, published_date, categories) "
            "VALUES ('2309.00001', 'T', 'A', '[]', '2023-09-01', '[]')"
        )
        con.commit()
        remove_nonexistent_paper(con, "2309.00001")
        con.commit()
        row = con.execute("SELECT 1 FROM papers WHERE arxiv_id='2309.00001'").fetchone()
        assert row is None


# ---------------------------------------------------------------------------
# fail_task — not_before / retry delay
# ---------------------------------------------------------------------------


class TestRetryDelay:
    def test_sets_not_before_when_delay_given(self, con):
        """fail_task with retry_delay_seconds > 0 must set a future not_before."""
        task = _enqueue_and_claim(con, "2309.00010")
        fail_task(con, task["id"], "oops", max_attempts=3, retry_delay_seconds=3600)
        con.commit()
        row = con.execute(
            "SELECT status, not_before FROM task_queue WHERE id=?", (task["id"],)
        ).fetchone()
        assert row["status"] == "pending"
        assert row["not_before"] is not None
        nb = datetime.fromisoformat(row["not_before"])
        assert nb > datetime.utcnow()

    def test_not_before_null_when_no_delay(self, con):
        """fail_task with retry_delay_seconds=0 (default) must leave not_before NULL."""
        task = _enqueue_and_claim(con, "2309.00011")
        fail_task(con, task["id"], "oops", max_attempts=3, retry_delay_seconds=0)
        con.commit()
        row = con.execute(
            "SELECT not_before FROM task_queue WHERE id=?", (task["id"],)
        ).fetchone()
        assert row["not_before"] is None

    def test_not_before_null_on_permanent_failure(self, con):
        """Permanently failed tasks must have not_before=NULL."""
        task = _enqueue_and_claim(con, "2309.00012")
        fail_task(con, task["id"], "oops", max_attempts=1, retry_delay_seconds=3600)
        con.commit()
        row = con.execute(
            "SELECT status, not_before FROM task_queue WHERE id=?", (task["id"],)
        ).fetchone()
        assert row["status"] == "failed"
        assert row["not_before"] is None

    def test_claim_skips_future_not_before(self, con):
        """claim_next_tasks_batch must not return tasks whose not_before is in the future."""
        task = _enqueue_and_claim(con, "2309.00013")
        fail_task(con, task["id"], "oops", max_attempts=5, retry_delay_seconds=3600)
        con.commit()
        # Task is 'pending' but not_before is ~1h from now — should not be claimed
        claimed = claim_next_tasks_batch(con, "fetch_meta", 10)
        con.commit()
        assert claimed == []

    def test_claim_picks_up_past_not_before(self, con):
        """claim_next_tasks_batch must return tasks whose not_before is in the past."""
        task = _enqueue_and_claim(con, "2309.00014")
        # Set not_before to 1 second in the past directly
        con.execute(
            "UPDATE task_queue SET status='pending', not_before=datetime('now', '-1 seconds') WHERE id=?",
            (task["id"],),
        )
        con.commit()
        claimed = claim_next_tasks_batch(con, "fetch_meta", 10)
        con.commit()
        assert len(claimed) == 1
        assert claimed[0]["id"] == task["id"]


# ---------------------------------------------------------------------------
# meta_daemon — permanent failure triggers auto-removal (no HEAD check)
# ---------------------------------------------------------------------------


class TestMetaDaemonPermanentFailure:
    def test_permanent_failure_removes_from_user_papers(self, con):
        """After exhausting all retries, process_meta_batch removes the paper from user_papers."""
        import daemons.meta_daemon as daemon

        arxiv_id = "2309.00020"
        uid = _user_id(con)
        con.execute(
            "INSERT INTO user_papers (user_id, arxiv_id, liked) VALUES (?, ?, 1)",
            (uid, arxiv_id),
        )
        enqueue_task(con, "fetch_meta", {"arxiv_id": arxiv_id})
        con.commit()

        with patch.object(daemon, "get_arxiv_metadata", return_value={}):
            # Drive all 5 attempts; bypass not_before by resetting it after each fail
            for _ in range(5):
                daemon.process_meta_batch(con)
                con.commit()
                # Clear not_before so the task is immediately claimable
                con.execute(
                    "UPDATE task_queue SET not_before=NULL WHERE status='pending'"
                )
                con.commit()

        row = con.execute(
            "SELECT 1 FROM user_papers WHERE arxiv_id=?", (arxiv_id,)
        ).fetchone()
        assert row is None, "user_papers row should have been removed after permanent failure"

    def test_no_removal_while_retrying(self, con):
        """user_papers row must be preserved while the task is still retrying."""
        import daemons.meta_daemon as daemon

        arxiv_id = "2309.00021"
        uid = _user_id(con)
        con.execute(
            "INSERT INTO user_papers (user_id, arxiv_id, liked) VALUES (?, ?, 1)",
            (uid, arxiv_id),
        )
        enqueue_task(con, "fetch_meta", {"arxiv_id": arxiv_id})
        con.commit()

        with patch.object(daemon, "get_arxiv_metadata", return_value={}):
            # Run only 1 batch (attempt 1 of 5 — still retrying)
            daemon.process_meta_batch(con)
            con.commit()

        row = con.execute(
            "SELECT 1 FROM user_papers WHERE arxiv_id=?", (arxiv_id,)
        ).fetchone()
        assert row is not None, "user_papers row must remain while task is still retrying"

    def test_retry_delay_increases_with_attempts(self, con):
        """Each retry should set a not_before that is farther in the future than the last."""
        import daemons.meta_daemon as daemon
        from arxiv_lib.config import META_FETCH_RETRY_DELAYS

        arxiv_id = "2309.00022"
        enqueue_task(con, "fetch_meta", {"arxiv_id": arxiv_id})
        con.commit()

        delays_seen = []
        with patch.object(daemon, "get_arxiv_metadata", return_value={}):
            for _ in range(len(META_FETCH_RETRY_DELAYS)):
                daemon.process_meta_batch(con)
                con.commit()
                row = con.execute(
                    "SELECT not_before FROM task_queue WHERE status='pending'"
                ).fetchone()
                if row and row["not_before"]:
                    delays_seen.append(row["not_before"])
                # Clear not_before for next iteration
                con.execute("UPDATE task_queue SET not_before=NULL WHERE status='pending'")
                con.commit()

        # We should have recorded one not_before per retry (4 retries before permanent failure)
        assert len(delays_seen) == len(META_FETCH_RETRY_DELAYS)
        # Each successive not_before should be farther in the future
        for i in range(1, len(delays_seen)):
            assert delays_seen[i] > delays_seen[i - 1]
