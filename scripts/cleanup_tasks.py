#!/usr/bin/env python3
"""
Daily maintenance script: remove old, successfully-completed tasks from
task_queue to keep the database tidy.

What it does
------------
Deletes rows from task_queue where ALL of the following are true:
  - status  = 'done'
  - error   IS NULL   (tasks that showed a traceback are never auto-deleted)
  - completed_at < datetime('now', '-N days')  (both sides are UTC — SQLite's
    datetime('now') always returns UTC, matching how completed_at is stored)

Tasks with status 'failed', 'running', or 'pending', and any 'done' task that
recorded a non-null error, are left completely untouched.

After deletion the database is VACUUMed to reclaim free pages.

Cron example
------------
Run at 04:00 every day, logging output to /var/log/arxiv-cleanup-tasks.log:

    0 4 * * * /home/<user>/arxiv_recommender/.venv/bin/python3 \\
        /home/<user>/arxiv_recommender/scripts/cleanup_tasks.py \\
        >> /var/log/arxiv-cleanup-tasks.log 2>&1

Usage
-----
    python scripts/cleanup_tasks.py
    python scripts/cleanup_tasks.py --dry-run
    python scripts/cleanup_tasks.py --max-age-days 7
"""

import argparse
import os
import sqlite3
import sys
from datetime import datetime

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from arxiv_lib import config as _config

# Rows matching this WHERE clause are eligible for deletion.
# datetime('now', '-? days') is evaluated by SQLite in UTC, matching the UTC
# values stored in completed_at — no Python timezone arithmetic needed.
_WHERE = (
    "status = 'done' "
    "AND error IS NULL "
    "AND completed_at < datetime('now', :cutoff)"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Remove old successfully-completed tasks from task_queue. "
            "Tasks with errors are never auto-deleted."
        )
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=4,
        metavar="N",
        help="Delete 'done' (no error) tasks completed more than N days ago (default: 4).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be done without deleting anything.",
    )
    args = parser.parse_args()

    cutoff = f"-{args.max_age_days} days"
    dry_run = args.dry_run
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"[{now_str}] cleanup_tasks  max_age={args.max_age_days}d  dry_run={dry_run}")

    db_path = _config.APP_DB_PATH()
    if not os.path.exists(db_path):
        print(f"Note: {db_path} does not exist — nothing to do.")
        sys.exit(0)

    size_before = os.path.getsize(db_path)
    print(f"app.db  —  {size_before:,} bytes on disk")

    with sqlite3.connect(db_path) as con:
        total = con.execute("SELECT COUNT(*) FROM task_queue").fetchone()[0]
        eligible = con.execute(
            f"SELECT COUNT(*) FROM task_queue WHERE {_WHERE}",
            {"cutoff": cutoff},
        ).fetchone()[0]

        print(
            f"\n[task_queue]"
            f"\n  Total rows            : {total:,}"
            f"\n  Eligible for deletion : {eligible:,}  "
            f"(done, no error, completed >{args.max_age_days}d ago)"
        )

        if dry_run:
            print("  Would delete          : (dry run — no changes made)")
            return

        if eligible == 0:
            print("  Nothing to delete.")
            return

        con.execute(
            f"DELETE FROM task_queue WHERE {_WHERE}",
            {"cutoff": cutoff},
        )
        con.commit()
        print(f"  Deleted               : {eligible:,} rows")

    # VACUUM outside the transaction (cannot run inside one)
    with sqlite3.connect(db_path) as con:
        con.execute("VACUUM")

    size_after = os.path.getsize(db_path)
    saved = size_before - size_after
    print(
        f"\napp.db after VACUUM  —  {size_after:,} bytes on disk"
        f"  ({saved:+,} bytes)"
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
