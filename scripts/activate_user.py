#!/usr/bin/env python3
"""
Activate (or deactivate) a user account in app.db, or list pending registrations.

Usage:
    python scripts/activate_user.py --list                 # list inactive users
    python scripts/activate_user.py <email>                # activate
    python scripts/activate_user.py <email> --deactivate   # deactivate
"""

import argparse
import sys

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__file__)))

from arxiv_lib.appdb import get_connection, init_app_db


def cmd_list(con):
    rows = con.execute(
        "SELECT id, email, created_at FROM users WHERE is_active = 0 ORDER BY created_at"
    ).fetchall()
    if not rows:
        print("No pending registration requests.")
        return
    print(f"{'ID':<6}  {'Email':<40}  {'Registered'}")
    print("-" * 70)
    for row in rows:
        created = row["created_at"] or ""
        print(f"{row['id']:<6}  {row['email']:<40}  {created}")


def cmd_activate(con, email, deactivate):
    row = con.execute("SELECT id, email, is_active FROM users WHERE email = ?", (email,)).fetchone()
    if row is None:
        print(f"Error: no user with email {email!r} found.", file=sys.stderr)
        sys.exit(1)

    new_state = 0 if deactivate else 1
    action = "Deactivated" if deactivate else "Activated"

    con.execute("UPDATE users SET is_active = ? WHERE id = ?", (new_state, row["id"]))
    con.commit()
    print(f"{action} user {row['email']!r} (id={row['id']}).")


def main():
    parser = argparse.ArgumentParser(description="Manage pending user accounts.")
    parser.add_argument("email", nargs="?", help="Email address of the user to activate/deactivate.")
    parser.add_argument("--deactivate", action="store_true", help="Deactivate instead of activate.")
    parser.add_argument("--list", action="store_true", help="List all inactive (pending) users.")
    args = parser.parse_args()

    if not args.list and not args.email:
        parser.error("Provide an email address, or use --list.")

    init_app_db()
    con = get_connection()

    if args.list:
        cmd_list(con)
    else:
        cmd_activate(con, args.email, args.deactivate)

    con.close()


if __name__ == "__main__":
    main()
