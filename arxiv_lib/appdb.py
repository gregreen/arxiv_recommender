"""
App database helpers for app.db (SQLite, WAL mode).

This module owns:
  - Schema creation / migration (init_app_db)
  - Connection management (get_connection)
  - Task queue operations (enqueue_task, claim_next_task, complete_task, fail_task)

All SQL uses parameterised queries.  The caller is responsible for committing
transactions when doing multi-step operations that must be atomic.
"""

import json
import sqlite3
from typing import Any

from arxiv_lib.config import APP_DB_PATH

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY,
    email         TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    is_active     INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Per-user arXiv category subscriptions (used by daily cron to decide what to ingest)
CREATE TABLE IF NOT EXISTS user_categories (
    user_id  INTEGER NOT NULL REFERENCES users(id),
    category TEXT NOT NULL,   -- e.g. "astro-ph", "cs.LG"
    PRIMARY KEY (user_id, category)
);

-- Per-user paper feedback
-- liked: 1=liked (positive), 0=neutral/removed, -1=disliked (negative)
CREATE TABLE IF NOT EXISTS user_papers (
    user_id  INTEGER NOT NULL REFERENCES users(id),
    arxiv_id TEXT NOT NULL,
    liked    INTEGER NOT NULL DEFAULT 1,  -- 1 / 0 / -1
    added_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, arxiv_id)
);

-- Papers known to the system (populated by ingest daemon / migration script)
-- categories: JSON array of strings, e.g. ["astro-ph.GA", "astro-ph.SR"]
-- authors:    JSON array of strings
-- published_date and categories may be NULL for backfilled / legacy papers
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id       TEXT PRIMARY KEY,
    title          TEXT,
    abstract       TEXT,
    authors        TEXT,           -- JSON array
    published_date TEXT,           -- YYYY-MM-DD; NULL for legacy backfilled rows
    categories     TEXT,           -- JSON array; NULL for legacy backfilled rows
    embedded_at    TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS papers_embedded_at ON papers(embedded_at);

-- Per-user fitted recommendation model
-- model_blob: JSON produced by ScoringModel.serialize() — NOT a pickle blob
-- model_hash: compute_model_hash(liked_ids) — detect stale models
CREATE TABLE IF NOT EXISTS user_models (
    user_id    INTEGER PRIMARY KEY REFERENCES users(id),
    model_blob TEXT NOT NULL,       -- JSON from ScoringModel.serialize()
    model_hash TEXT NOT NULL,       -- from compute_model_hash(liked_ids)
    n_liked    INTEGER NOT NULL,
    n_disliked INTEGER NOT NULL,
    trained_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Recommendation cache
-- time_window: 'day' | 'week' | 'month' | 'year'
-- score:       log-probability from logistic regression (higher = more relevant)
-- rank:        1-based rank within (user_id, time_window)
CREATE TABLE IF NOT EXISTS recommendations (
    user_id      INTEGER NOT NULL REFERENCES users(id),
    arxiv_id     TEXT NOT NULL,
    time_window  TEXT NOT NULL,
    score        REAL NOT NULL,
    rank         INTEGER NOT NULL,
    model_hash   TEXT NOT NULL,
    generated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, arxiv_id, time_window)
);
CREATE INDEX IF NOT EXISTS recommendations_user_window
    ON recommendations(user_id, time_window, rank);

-- Task queue (shared by meta, embed, and recommend daemons)
-- type:    'fetch_meta' | 'embed' | 'recommend'
-- payload: JSON, e.g. {"arxiv_id": "2309.06676"} or {"user_id": 3}
-- status:  'pending' | 'running' | 'done' | 'failed'
CREATE TABLE IF NOT EXISTS task_queue (
    id           INTEGER PRIMARY KEY,
    type         TEXT NOT NULL,
    payload      TEXT NOT NULL,     -- JSON
    status       TEXT NOT NULL DEFAULT 'pending',
    attempts     INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    started_at   TEXT,
    completed_at TEXT,
    error        TEXT
);
CREATE INDEX IF NOT EXISTS task_queue_pending
    ON task_queue(type, status, created_at)
    WHERE status = 'pending';
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_connection(path: str = APP_DB_PATH) -> sqlite3.Connection:
    """
    Open and return a WAL-enabled connection to app.db (or the given path).

    The caller owns the connection lifetime and must close it when done.
    Use as a context manager for automatic commit/rollback:

        with get_connection() as con:
            con.execute(...)
    """
    con = sqlite3.connect(path, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    con.row_factory = sqlite3.Row
    return con


def init_app_db(path: str = APP_DB_PATH) -> None:
    """
    Create all app.db tables if they do not already exist.

    Safe to call repeatedly (all DDL uses IF NOT EXISTS).
    Also runs incremental column migrations for existing databases.
    """
    con = sqlite3.connect(path)
    try:
        con.executescript(_SCHEMA_SQL)
        # Incremental migrations for existing databases
        cols = {row[1] for row in con.execute("PRAGMA table_info(users)")}
        if "is_active" not in cols:
            # Existing users default to active so they aren't locked out
            con.execute("ALTER TABLE users ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1")
        con.commit()
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Task queue helpers
# ---------------------------------------------------------------------------

def enqueue_task(con: sqlite3.Connection, task_type: str, payload: dict[str, Any]) -> int:
    """
    Insert a new 'pending' task into task_queue.

    Parameters
    ----------
    con : sqlite3.Connection
        Open app.db connection.  Caller must commit.
    task_type : str
        'embed' or 'recommend'.
    payload : dict
        JSON-serialisable payload dict (e.g. {"arxiv_id": "2309.06676"}).

    Returns
    -------
    int
        Row id of the newly created task.
    """
    cur = con.execute(
        "INSERT INTO task_queue (type, payload) VALUES (?, ?)",
        (task_type, json.dumps(payload)),
    )
    return cur.lastrowid


def claim_next_task(con: sqlite3.Connection, task_type: str) -> sqlite3.Row | None:
    """
    Atomically claim the oldest pending task of the given type.

    Sets status='running', increments attempts, records started_at.

    Parameters
    ----------
    con : sqlite3.Connection
        Open app.db connection.  Caller should commit after this call.
    task_type : str
        'embed' or 'recommend'.

    Returns
    -------
    sqlite3.Row or None
        The claimed task row (with id, type, payload, attempts, …), or None if
        no pending tasks of the requested type exist.
    """
    # UPDATE … RETURNING * is atomic and avoids a second SELECT.
    # Requires SQLite ≥ 3.35 (ships with Python 3.12+).
    row = con.execute(
        """
        UPDATE task_queue
           SET status     = 'running',
               attempts   = attempts + 1,
               started_at = datetime('now')
         WHERE id = (
               SELECT id FROM task_queue
                WHERE type = ? AND status = 'pending'
                ORDER BY created_at
                LIMIT 1
         )
         RETURNING *
        """,
        (task_type,),
    ).fetchone()
    return row


def claim_next_tasks_batch(
    con: sqlite3.Connection,
    task_type: str,
    limit: int,
) -> list[sqlite3.Row]:
    """
    Atomically claim up to *limit* pending tasks of the given type.

    Sets status='running', increments attempts, records started_at for each.

    Parameters
    ----------
    con : sqlite3.Connection
        Open app.db connection.  Caller should commit after this call.
    task_type : str
        'fetch_meta', 'embed', or 'recommend'.
    limit : int
        Maximum number of tasks to claim in one call.

    Returns
    -------
    list[sqlite3.Row]
        The claimed task rows, or an empty list if no pending tasks exist.
    """
    rows = con.execute(
        """
        UPDATE task_queue
           SET status     = 'running',
               attempts   = attempts + 1,
               started_at = datetime('now')
         WHERE id IN (
               SELECT id FROM task_queue
                WHERE type = ? AND status = 'pending'
                ORDER BY created_at ASC
                LIMIT ?
         )
         RETURNING *
        """,
        (task_type, limit),
    ).fetchall()
    return rows


def complete_task(con: sqlite3.Connection, task_id: int) -> None:
    """
    Mark a task as done.

    Parameters
    ----------
    con : sqlite3.Connection
        Open app.db connection.  Caller must commit.
    task_id : int
        id of the task to complete.
    """
    con.execute(
        "UPDATE task_queue SET status='done', completed_at=datetime('now') WHERE id=?",
        (task_id,),
    )


def fail_task(con: sqlite3.Connection, task_id: int, error: str, max_attempts: int = 3) -> None:
    """
    Mark a task as failed.

    If attempts < max_attempts, reset to 'pending' so the daemon retries.
    Otherwise leave as 'failed'.

    Parameters
    ----------
    con : sqlite3.Connection
        Open app.db connection.  Caller must commit.
    task_id : int
        id of the task that failed.
    error : str
        Error message / traceback to store.
    max_attempts : int
        Maximum number of attempts before permanently failing.
    """
    row = con.execute("SELECT attempts FROM task_queue WHERE id=?", (task_id,)).fetchone()
    if row is None:
        return
    new_status = "pending" if row["attempts"] < max_attempts else "failed"
    con.execute(
        "UPDATE task_queue SET status=?, error=?, completed_at=datetime('now') WHERE id=?",
        (new_status, error, task_id),
    )
