#!/usr/bin/env python3
"""
Delete page_stats_daily_users rows older than 90 days.

The dedup table covers the longest analytics window (90 days).  Rows older
than that are no longer needed for distinct-user counts.
page_stats_daily (visit counters) is kept indefinitely — it is negligible in size.

Usage:
    python scripts/cleanup_analytics.py

Recommended schedule: daily cron / Ofelia job.
"""

import sys

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__file__)))

from arxiv_lib.appdb import get_connection, init_app_db


def main():
    init_app_db()
    con = get_connection()
    cur = con.execute(
        "DELETE FROM page_stats_daily_users WHERE date < date('now', '-90 days')"
    )
    con.commit()
    con.close()
    print(f"Pruned {cur.rowcount} analytics dedup row(s) older than 90 days.")


if __name__ == "__main__":
    main()
