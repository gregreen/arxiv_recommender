#!/usr/bin/env python3
"""
One-shot script: re-fetch published_date for all papers in app.db from the
arXiv Atom API to obtain full ISO 8601 timestamps (YYYY-MM-DDTHH:MM:SSZ).

Usage:
    python scripts/refetch_timestamps.py [--batch-size N] [--sleep S]

arXiv rate-limit guidance: keep batch size ≤ 50 and sleep ≥ 10s between
requests.  The script uses exponential backoff (up to 120s) on failure.
"""

import argparse
import logging
import random
import sys
import time

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__file__)))

from arxiv_lib.appdb import get_connection, init_app_db
from arxiv_lib.ingest import fetch_arxiv_metadata, write_to_arxiv_metadata_cache

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("refetch_timestamps")

MAX_BACKOFF  = 120.0
SLEEP_JITTER = 2.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-fetch arXiv timestamps for all papers.")
    parser.add_argument("--batch-size", type=int, default=50, help="IDs per Atom API request (default: 50)")
    parser.add_argument("--sleep", type=float, default=10.0, help="Seconds to sleep between batches (default: 10)")
    args = parser.parse_args()

    init_app_db()
    con = get_connection()
    arxiv_ids = [r[0] for r in con.execute(
        "SELECT arxiv_id FROM papers ORDER BY arxiv_id"
    ).fetchall()]
    con.close()

    n_batches = -(-len(arxiv_ids) // args.batch_size)  # ceiling division
    log.info(f"{len(arxiv_ids)} papers  |  batch_size={args.batch_size}  |  {n_batches} batches  |  sleep={args.sleep}s")

    total_updated = 0
    failed_ids: list[str] = []

    for i in range(0, len(arxiv_ids), args.batch_size):
        batch = arxiv_ids[i : i + args.batch_size]
        batch_num = i // args.batch_size + 1
        attempt = 0
        backoff = args.sleep

        while True:
            attempt += 1
            log.info(f"Batch {batch_num}/{n_batches}  [{i}–{i+len(batch)-1}]  attempt {attempt}")
            try:
                results = fetch_arxiv_metadata(batch)
                write_to_arxiv_metadata_cache(results)
                total_updated += len(results)
                log.info(f"  → {len(results)} written")
                break
            except Exception as exc:
                log.warning(f"  ! failed: {exc}")
                if attempt >= 5:
                    log.error(f"  Giving up on batch {batch_num}")
                    failed_ids.extend(batch)
                    break
                sleep_t = min(backoff, MAX_BACKOFF) + random.uniform(0, SLEEP_JITTER)
                log.info(f"  waiting {sleep_t:.1f}s before retry ...")
                time.sleep(sleep_t)
                backoff *= 2

        if i + args.batch_size < len(arxiv_ids):
            sleep_t = args.sleep + random.uniform(0, SLEEP_JITTER)
            log.info(f"  sleeping {sleep_t:.1f}s ...")
            time.sleep(sleep_t)

    log.info(f"\nFinished. Updated {total_updated} papers.")
    if failed_ids:
        log.warning(f"Failed ({len(failed_ids)} papers): {failed_ids}")

    con = get_connection()
    ts_count = con.execute(
        "SELECT COUNT(*) FROM papers WHERE published_date LIKE '%T%'"
    ).fetchone()[0]
    total = con.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    log.info(f"{ts_count}/{total} papers now have full timestamps.")
    sample = con.execute(
        "SELECT arxiv_id, published_date FROM papers "
        "WHERE published_date LIKE '%T%' ORDER BY published_date DESC LIMIT 5"
    ).fetchall()
    if sample:
        log.info("Most recent:")
        for r in sample:
            log.info(f"  {r[0]}: {r[1]}")
    con.close()


if __name__ == "__main__":
    main()
