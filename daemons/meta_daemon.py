#!/usr/bin/env python3
"""
Meta daemon — processes 'fetch_meta' tasks from the task queue in batches.

For each batch:
  1. Claim up to INGEST_META_BATCH_SIZE pending 'fetch_meta' tasks atomically.
  2. Filter out papers already in the papers table (metadata already known).
  3. Fetch metadata for the remainder via Semantic Scholar batch API
     (with arXiv Atom API fallback), writing results to the papers table.
  4. For each successfully fetched paper: enqueue an 'embed' task.
     For papers whose metadata could not be retrieved: fail the task (auto-retry
     up to 3 attempts).

The meta daemon and embed daemon are independent and can run in parallel.
cron_daily.py enqueues 'fetch_meta' tasks; this daemon converts them into
'embed' tasks that the embed daemon picks up.

Usage:
    python3 daemons/meta_daemon.py           # run forever (poll loop)
    python3 daemons/meta_daemon.py --once    # process one batch and exit
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from arxiv_lib.appdb import (
    claim_next_tasks_batch,
    complete_task,
    enqueue_task,
    fail_task,
    get_connection,
    init_app_db,
)
from arxiv_lib.config import (
    APP_DB_PATH,
    INGEST_META_BATCH_SIZE,
    META_INGEST_POLL_INTERVAL,
)
from arxiv_lib.ingest import get_arxiv_metadata, load_tokens

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [meta] %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _paper_has_metadata(app_con, arxiv_id: str) -> bool:
    """Return True if the paper already exists in the papers table."""
    return app_con.execute(
        "SELECT 1 FROM papers WHERE arxiv_id = ?", (arxiv_id,)
    ).fetchone() is not None


# ---------------------------------------------------------------------------
# Core batch processor
# ---------------------------------------------------------------------------

def process_meta_batch(app_con, tokens: dict) -> bool:
    """
    Claim and process one batch of 'fetch_meta' tasks.

    Returns True if any tasks were claimed (batch found), False if the queue
    was empty.
    """
    tasks = claim_next_tasks_batch(app_con, "fetch_meta", INGEST_META_BATCH_SIZE)
    app_con.commit()
    if not tasks:
        return False

    id_to_task = {json.loads(t["payload"])["arxiv_id"]: t for t in tasks}
    all_ids = list(id_to_task)
    log.info("Processing meta batch of %d task(s).", len(tasks))

    needs_fetch = [aid for aid in all_ids if not _paper_has_metadata(app_con, aid)]
    already_known = len(all_ids) - len(needs_fetch)
    if already_known:
        log.info("  %d already in papers table — skipping fetch.", already_known)

    if needs_fetch:
        try:
            get_arxiv_metadata(needs_fetch, s2_token=tokens.get("semantic_scholar"))
        except Exception:
            # Log but continue — we check per-paper below; partial success is fine.
            log.error("  Metadata fetch raised an exception:\n%s", traceback.format_exc())

    enqueued = 0
    failed = 0
    for arxiv_id, task in id_to_task.items():
        task_id = task["id"]
        if _paper_has_metadata(app_con, arxiv_id):
            enqueue_task(app_con, "embed", {"arxiv_id": arxiv_id})
            complete_task(app_con, task_id)
            enqueued += 1
        else:
            fail_task(app_con, task_id, "metadata not available after fetch attempt")
            log.warning("  Metadata unavailable for %s — task failed/retried.", arxiv_id)
            failed += 1

    app_con.commit()
    log.info(
        "  Batch done: %d embed task(s) enqueued, %d task(s) failed/retried.",
        enqueued, failed,
    )
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process one batch (or report empty queue) then exit.",
    )
    parser.add_argument(
        "--db", default=APP_DB_PATH,
        help=f"Path to app.db (default: {APP_DB_PATH})",
    )
    args = parser.parse_args()

    log.info("Starting meta daemon (db=%s)", args.db)
    init_app_db(args.db)
    tokens = load_tokens()

    app_con = get_connection(args.db)
    try:
        while True:
            found = process_meta_batch(app_con, tokens)
            if args.once:
                if not found:
                    log.info("Queue empty — nothing to do.")
                break
            if not found:
                time.sleep(META_INGEST_POLL_INTERVAL)
    finally:
        app_con.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
