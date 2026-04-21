#!/usr/bin/env python3
"""
Retroactively clean up permanently-failed fetch_meta tasks.

What it does
------------
1. Queries task_queue for all rows where type='fetch_meta' AND status='failed'.
2. For each, sends a HEAD request to export.arxiv.org/abs/{id} to confirm
   whether the paper actually exists.
3. If the paper returns HTTP 404 (confirmed non-existent):
   - Removes it from user_papers (all users' libraries).
   - Removes any partial row from the papers table.
4. Papers where the existence check is inconclusive (timeout, 5xx, unexpected
   status) are reported but NOT deleted — safety first.

Run this after deploying the meta daemon update to clean up any bad IDs that
accumulated before the new validation was in place.

Usage
-----
    python scripts/remove_failed_imports.py              # check and fix
    python scripts/remove_failed_imports.py --dry-run    # report only
    python scripts/remove_failed_imports.py --db /path/to/app.db
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from arxiv_lib import config as _config
from arxiv_lib.appdb import get_connection, remove_nonexistent_paper
from arxiv_lib.ingest import check_arxiv_exists

# Be polite: space HEAD requests apart so we don't hammer arxiv.org.
_HEAD_DELAY_SECONDS = 5.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check permanently-failed fetch_meta tasks and remove confirmed "
            "non-existent papers from all user libraries."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be removed without making any changes.",
    )
    parser.add_argument(
        "--db",
        default=_config.APP_DB_PATH(),
        metavar="PATH",
        help=f"Path to app.db (default: {_config.APP_DB_PATH()})",
    )
    args = parser.parse_args()

    dry_run = args.dry_run
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now_str}] remove_failed_imports  dry_run={dry_run}")

    with get_connection(args.db) as con:
        rows = con.execute(
            "SELECT id, payload FROM task_queue WHERE type='fetch_meta' AND status='failed'"
        ).fetchall()

    print(f"\n[APP_DB]  {args.db}")
    print(f"  Permanently-failed fetch_meta tasks: {len(rows):,}")

    if not rows:
        print("\nNothing to do.")
        return

    # Deduplicate by arxiv_id (multiple failed tasks for the same ID are possible).
    seen: set[str] = set()
    arxiv_ids: list[str] = []
    for row in rows:
        try:
            arxiv_id = json.loads(row["payload"])["arxiv_id"]
        except (json.JSONDecodeError, KeyError):
            print(f"  Warning: could not parse payload {row['payload']!r} — skipping.")
            continue
        if arxiv_id not in seen:
            seen.add(arxiv_id)
            arxiv_ids.append(arxiv_id)

    print(f"  Unique arXiv IDs to check  : {len(arxiv_ids):,}")
    print()

    # ── HEAD check each ID ────────────────────────────────────────────────────

    confirmed_absent: list[str] = []
    inconclusive: list[str] = []
    confirmed_present: list[str] = []

    for i, arxiv_id in enumerate(arxiv_ids):
        if i > 0:
            time.sleep(_HEAD_DELAY_SECONDS)
        result = check_arxiv_exists(arxiv_id)
        if result is False:
            confirmed_absent.append(arxiv_id)
            print(f"  404  {arxiv_id}")
        elif result is True:
            confirmed_present.append(arxiv_id)
            print(f"  200  {arxiv_id}  (exists — skipping)")
        else:
            inconclusive.append(arxiv_id)
            print(f"  ???  {arxiv_id}  (inconclusive — skipping)")

    print()
    print(f"  Confirmed absent   : {len(confirmed_absent):,}")
    print(f"  Confirmed present  : {len(confirmed_present):,}")
    print(f"  Inconclusive       : {len(inconclusive):,}")

    if not confirmed_absent:
        print("\nNo confirmed-absent papers to remove.")
        return

    if dry_run:
        print(
            f"\n[dry-run] Would remove {len(confirmed_absent)} paper(s) from user libraries:"
        )
        for arxiv_id in confirmed_absent:
            print(f"  {arxiv_id}")
        return

    # ── Remove confirmed-absent papers ────────────────────────────────────────

    print()
    total_users_affected = 0
    with get_connection(args.db) as con:
        for arxiv_id in confirmed_absent:
            n = remove_nonexistent_paper(con, arxiv_id)
            total_users_affected += n
            print(f"  Removed  {arxiv_id}  ({n} user library row(s))")
        con.commit()

    print()
    print(f"  Total user_papers rows deleted: {total_users_affected:,}")
    print("\nDone.")


if __name__ == "__main__":
    main()
