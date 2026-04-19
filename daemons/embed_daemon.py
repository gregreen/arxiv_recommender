#!/usr/bin/env python3
"""
Embed daemon — processes 'embed' tasks from the task queue.

'embed' tasks are produced by the meta daemon after metadata has been written
to the papers table.  For each task:
  1. Skip if the paper is already fully ingested (papers table + embeddings DB).
  2. Call fetch_arxiv_embedding(), which runs:
       summarise (LLM) → embed (embedding model) → save to embeddings_cache.db
  3. Mark the task done (or failed on error, with automatic retry up to 3 attempts).

The embed daemon and meta daemon are independent and can run in parallel.

Usage:
    python3 daemons/embed_daemon.py           # run forever (poll loop)
    python3 daemons/embed_daemon.py --once    # process one task and exit
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
import traceback

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from arxiv_lib.appdb import (
    complete_task,
    claim_next_task,
    fail_task,
    get_connection,
    init_app_db,
)
from arxiv_lib.config import (
    APP_DB_PATH,
    EMBEDDING_CACHE_DB,
    EMBED_INGEST_POLL_INTERVAL,
)
from arxiv_lib.ingest import fetch_search_embedding, fetch_recommendation_embedding, _init_embedding_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [embed] %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _paper_already_ingested(app_con: sqlite3.Connection, arxiv_id: str) -> bool:
    """Return True if the paper exists in the papers table and both embedding tables."""
    in_papers = app_con.execute(
        "SELECT 1 FROM papers WHERE arxiv_id = ?", (arxiv_id,)
    ).fetchone() is not None

    if not in_papers:
        return False

    with sqlite3.connect(EMBEDDING_CACHE_DB()) as emb_con:
        in_search = emb_con.execute(
            "SELECT 1 FROM search_embeddings WHERE arxiv_id = ?", (arxiv_id,)
        ).fetchone() is not None
        in_recommendation = emb_con.execute(
            "SELECT 1 FROM recommendation_embeddings WHERE arxiv_id = ?", (arxiv_id,)
        ).fetchone() is not None

    return in_search and in_recommendation


# ---------------------------------------------------------------------------
# Core task processor
# ---------------------------------------------------------------------------

def process_one_task(app_con: sqlite3.Connection) -> bool:
    """
    Claim and process one 'embed' task.

    Returns True if a task was found and processed (success or failure),
    False if the queue was empty.
    """
    task = claim_next_task(app_con, "embed")
    app_con.commit()
    if task is None:
        return False

    task_id = task["id"]
    payload = json.loads(task["payload"])
    arxiv_id = payload.get("arxiv_id", "")

    log.info("Processing embed task %d: arxiv_id=%s (attempt %d)", task_id, arxiv_id, task["attempts"])

    try:
        if _paper_already_ingested(app_con, arxiv_id):
            log.info("  %s already ingested — skipping.", arxiv_id)
            complete_task(app_con, task_id)
            app_con.commit()
            return True

        # Summarise (LLM) → embed (both types) → save to embeddings_cache.db
        fetch_search_embedding(arxiv_id)
        fetch_recommendation_embedding(arxiv_id)

        complete_task(app_con, task_id)
        app_con.commit()
        log.info("  Task %d complete.", task_id)

    except Exception:
        tb = traceback.format_exc()
        log.error("  Task %d failed:\n%s", task_id, tb)
        fail_task(app_con, task_id, tb)
        app_con.commit()

    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process one task (or report empty queue) then exit.",
    )
    parser.add_argument(
        "--db", default=APP_DB_PATH(),
        help=f"Path to app.db (default: {APP_DB_PATH()})",
    )
    args = parser.parse_args()

    log.info("Starting embed daemon (db=%s)", args.db)
    init_app_db(args.db)
    _init_embedding_db()

    app_con = get_connection(args.db)
    try:
        while True:
            found = process_one_task(app_con)
            if args.once:
                if not found:
                    log.info("Queue empty — nothing to do.")
                break
            if not found:
                time.sleep(EMBED_INGEST_POLL_INTERVAL)
    finally:
        app_con.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
