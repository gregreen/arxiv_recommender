#!/usr/bin/env python3
"""
Daily ingest cron script — fetches metadata from the arXiv RSS Atom feed and
enqueues 'embed' tasks for new papers.

For each category in DAILY_INGEST_CATEGORIES (from config.py), fetches the
arXiv daily mailing via the official Atom feed, writes metadata directly to
the papers table, and enqueues an 'embed' task for every new paper.  This
gives full ISO 8601 UTC timestamps on published_date (as opposed to the
date-only values returned by Semantic Scholar).

Run this once per day after the arXiv mailing is published (typically ~14:00
US Eastern on weekdays).  The embed daemon will then process the tasks.

Usage:
    python3 scripts/cron_daily.py                    # uses DAILY_INGEST_CATEGORIES
    python3 scripts/cron_daily.py astro-ph.GA cs.LG  # override categories
    python3 scripts/cron_daily.py --db /path/to/app.db

Cron example (runs at 19:00 UTC Mon–Fri, after the 14:00 ET cutoff):
    0 19 * * 1-5  cd /path/to/arxiv_recommender && python3 scripts/cron_daily.py
"""

import argparse
import logging
import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from arxiv_lib.appdb import enqueue_task, get_connection, init_app_db
from arxiv_lib.config import APP_DB_PATH, DAILY_INGEST_CATEGORIES
from arxiv_lib.ingest import fetch_daily_mailing_metadata, write_to_arxiv_metadata_cache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [cron_daily] %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)


def _already_known(con, arxiv_id: str) -> bool:
    return con.execute(
        "SELECT 1 FROM papers WHERE arxiv_id = ?", (arxiv_id,)
    ).fetchone() is not None


def run(categories: list[str], db_path: str) -> int:
    """
    Fetch metadata from the arXiv RSS Atom feed for each category and enqueue
    embed tasks for new papers.

    Returns the total number of newly enqueued embed tasks.
    """
    init_app_db(db_path)
    con = get_connection(db_path)

    total_enqueued = 0
    total_skipped = 0

    try:
        for category in categories:
            log.info("Fetching daily mailing for category: %s", category)
            try:
                metadata = fetch_daily_mailing_metadata(category)
            except RuntimeError as e:
                log.error("  Failed to fetch Atom feed for %s: %s", category, e)
                continue

            log.info("  Feed returned %d new paper(s).", len(metadata))

            # Write all fetched metadata to the papers table in one batch
            write_to_arxiv_metadata_cache(metadata)

            enqueued = 0
            skipped = 0
            for arxiv_id in metadata:
                if _already_known(con, arxiv_id):
                    # write_to_arxiv_metadata_cache uses INSERT OR REPLACE, so
                    # metadata is refreshed; but we skip re-enqueueing embed.
                    skipped += 1
                else:
                    enqueue_task(con, "embed", {"arxiv_id": arxiv_id})
                    enqueued += 1

            con.commit()
            log.info(
                "  %s: %d embed task(s) enqueued, %d already known → skipped.",
                category, enqueued, skipped,
            )
            total_enqueued += enqueued
            total_skipped += skipped

    finally:
        con.close()

    log.info(
        "Done. Total: %d task(s) enqueued, %d skipped.",
        total_enqueued, total_skipped,
    )
    return total_enqueued


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "categories",
        nargs="*",
        help=(
            "arXiv categories to ingest (e.g. astro-ph cs.LG). "
            f"Defaults to DAILY_INGEST_CATEGORIES from config: {DAILY_INGEST_CATEGORIES}"
        ),
    )
    parser.add_argument(
        "--db", default=APP_DB_PATH,
        help=f"Path to app.db (default: {APP_DB_PATH})",
    )
    args = parser.parse_args()

    categories = args.categories if args.categories else DAILY_INGEST_CATEGORIES
    run(categories, args.db)
    return 0


if __name__ == "__main__":
    sys.exit(main())
