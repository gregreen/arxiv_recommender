#!/usr/bin/env python3
"""
Delete page_events rows older than 365 days to honour the analytics data
retention policy described in the Privacy notice.

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
        "DELETE FROM page_events WHERE ts < datetime('now', '-365 days')"
    )
    con.commit()
    con.close()
    print(f"Pruned {cur.rowcount} page event(s) older than 365 days.")


if __name__ == "__main__":
    main()
