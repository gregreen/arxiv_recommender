#!/usr/bin/env python3
"""
Activate (or deactivate) a user account in app.db.

Usage:
    python scripts/activate_user.py <email>            # activate
    python scripts/activate_user.py <email> --deactivate
"""

import argparse
import sys

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__file__)))

from arxiv_lib.appdb import get_connection, init_app_db


def main():
    parser = argparse.ArgumentParser(description="Activate a pending user account.")
    parser.add_argument("email", help="Email address of the user to activate.")
    parser.add_argument("--deactivate", action="store_true", help="Deactivate instead of activate.")
    args = parser.parse_args()

    init_app_db()
    con = get_connection()

    row = con.execute("SELECT id, email, is_active FROM users WHERE email = ?", (args.email,)).fetchone()
    if row is None:
        print(f"Error: no user with email {args.email!r} found.", file=sys.stderr)
        sys.exit(1)

    new_state = 0 if args.deactivate else 1
    action = "Deactivated" if args.deactivate else "Activated"

    con.execute("UPDATE users SET is_active = ? WHERE id = ?", (new_state, row["id"]))
    con.commit()
    con.close()

    print(f"{action} user {row['email']!r} (id={row['id']}).")


if __name__ == "__main__":
    main()
