#!/usr/bin/env python3
"""
Validate all cached LLM summaries against the required section headings.

What it does
------------
1. Scans every .txt file in SUMMARY_CACHE_DIR.
2. Validates each summary against SUMMARY_REQUIRED_HEADINGS (parsed from
   system_prompt_summary.txt — stays in sync with the prompt automatically).
3. For each invalid summary:
   a. Deletes the .txt file from SUMMARY_CACHE_DIR.
   b. Deletes the matching rows from search_embeddings and
      recommendation_embeddings in the embeddings cache DB.
   c. Enqueues a fresh 'embed' task (priority 1 = daily-ingest priority)
      so the embed daemon will regenerate the summary and embeddings.

Run this script manually after changing the summary prompt, or when the embed
daemon logs 'Incomplete summary' errors to clean up any previously cached bad
summaries.

Usage
-----
    python scripts/validate_summaries.py              # validate and fix
    python scripts/validate_summaries.py --dry-run    # report only, no changes
    python scripts/validate_summaries.py --db /path/to/app.db
"""

import argparse
import glob
import os
import sqlite3
import sys
from datetime import datetime

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from arxiv_lib import config as _config
from arxiv_lib.appdb import enqueue_task, get_connection
from arxiv_lib.ingest import _validate_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate cached LLM summaries and requeue those that are missing "
            "required section headings."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report invalid summaries without deleting or requeuing anything.",
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
    print(f"[{now_str}] validate_summaries  dry_run={dry_run}")

    summary_dir = _config.SUMMARY_CACHE_DIR()
    txt_files = sorted(glob.glob(os.path.join(summary_dir, "*.txt")))

    print(f"\n[SUMMARY_CACHE_DIR]  {summary_dir}")
    print(f"  Total summaries  : {len(txt_files):,}")

    # ── Validate each summary ─────────────────────────────────────────────────

    invalid: list[tuple[str, str, str]] = []  # (arxiv_id, filepath, error_msg)

    for fpath in txt_files:
        arxiv_id = os.path.splitext(os.path.basename(fpath))[0]
        try:
            with open(fpath, encoding="utf-8") as f:
                summary = f.read()
            _validate_summary(arxiv_id, summary)
        except RuntimeError as exc:
            invalid.append((arxiv_id, fpath, str(exc)))
        except OSError as exc:
            print(f"  Warning: could not read {fpath}: {exc}")

    print(f"  Invalid summaries: {len(invalid):,}")

    if not invalid:
        print("\nAll summaries are valid — nothing to do.")
        return

    # ── Report ────────────────────────────────────────────────────────────────

    print()
    for arxiv_id, _fpath, error_msg in invalid:
        # Truncate long error messages for readability
        display = error_msg if len(error_msg) <= 120 else error_msg[:117] + "..."
        print(f"  INVALID  {arxiv_id}")
        print(f"           {display}")

    if dry_run:
        print(
            f"\n[dry-run] Would delete {len(invalid)} summary file(s), "
            f"remove embeddings, and requeue {len(invalid)} embed task(s)."
        )
        return

    # ── Delete summaries ──────────────────────────────────────────────────────

    print()
    deleted_summaries = 0
    for arxiv_id, fpath, _err in invalid:
        try:
            os.remove(fpath)
            deleted_summaries += 1
        except OSError as exc:
            print(f"  Warning: could not delete {fpath}: {exc}")

    print(f"  Deleted {deleted_summaries} summary file(s).")

    # ── Delete embeddings ─────────────────────────────────────────────────────

    emb_db = _config.EMBEDDING_CACHE_DB()
    deleted_emb_rows = 0
    try:
        with sqlite3.connect(emb_db) as emb_con:
            for arxiv_id, _fpath, _err in invalid:
                for table in ("search_embeddings", "recommendation_embeddings"):
                    cur = emb_con.execute(
                        f"DELETE FROM {table} WHERE arxiv_id = ?", (arxiv_id,)
                    )
                    deleted_emb_rows += cur.rowcount
    except sqlite3.Error as exc:
        print(f"  Warning: embedding DB error: {exc}")

    print(f"  Deleted {deleted_emb_rows} embedding row(s) from {emb_db}.")

    # ── Requeue embed tasks ───────────────────────────────────────────────────

    enqueued = 0
    try:
        with get_connection(args.db) as app_con:
            for arxiv_id, _fpath, _err in invalid:
                enqueue_task(app_con, "embed", {"arxiv_id": arxiv_id}, priority=1)
                enqueued += 1
            app_con.commit()
    except sqlite3.Error as exc:
        print(f"  Warning: app DB error while requeuing: {exc}")

    print(f"  Enqueued {enqueued} embed task(s) at priority 1.")
    print("\nDone.")


if __name__ == "__main__":
    main()
