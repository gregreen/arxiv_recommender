#!/usr/bin/env python3
"""
NULL out stale email verification tokens for accounts that have already been
verified and whose 7-day retention window has passed.

Usage:
    python scripts/cleanup_tokens.py

Recommended schedule: daily or weekly cron.
"""

import sys

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__file__)))

from datetime import datetime, timezone

from arxiv_lib.appdb import get_connection, init_app_db


def main():
    init_app_db()
    con = get_connection()
    now = datetime.now(timezone.utc).isoformat()
    cur = con.execute(
        """UPDATE users
           SET email_verify_token = NULL,
               email_verify_token_expires_at = NULL
           WHERE email_verified = 1
             AND email_verify_token IS NOT NULL
             AND email_verify_token_expires_at < ?""",
        (now,),
    )
    con.commit()
    con.close()
    print(f"Cleaned up {cur.rowcount} expired verification token(s).")


if __name__ == "__main__":
    main()
