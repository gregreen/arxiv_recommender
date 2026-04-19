#!/usr/bin/env python3
"""
Daily maintenance script: remove malformed embedding vectors, remove orphaned
search-term embeddings, and compact the embeddings_cache.db SQLite database.

What it does
------------
1. Deletes any rows in search_embeddings or recommendation_embeddings whose
   BLOB size does not match the expected size for EMBEDDING_STORAGE_DIM float32
   values (i.e. EMBEDDING_STORAGE_DIM * 4 bytes).
2. Deletes any rows in search_term_embeddings whose query text no longer
   appears in user_search_terms (orphaned after user deletion).
3. Runs VACUUM to reclaim free pages and shrink the file on disk.
4. Reports row counts and file sizes before / after.

A timestamped backup of embeddings_cache.db is always created before any
writes (unless --dry-run is given).  Nothing is done (and no backup is
written) when no issues are found.

Cron example
------------
Run at 03:00 every day, logging output to /var/log/arxiv-cleanup-embeddings.log:

    0 3 * * * /home/<user>/arxiv_recommender/.venv/bin/python3 \\
        /home/<user>/arxiv_recommender/scripts/cleanup_embeddings.py \\
        >> /var/log/arxiv-cleanup-embeddings.log 2>&1

Usage
-----
    python scripts/cleanup_embeddings.py
    python scripts/cleanup_embeddings.py --dry-run
"""

import argparse
import os
import shutil
import sqlite3
import sys
from datetime import datetime

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from arxiv_lib import config as _config

# Tables in embeddings_cache.db that store per-paper vectors (keyed by arxiv_id).
_TABLES = ("search_embeddings", "recommendation_embeddings")
_EXPECTED_BYTES = _config.EMBEDDING_STORAGE_DIM * 4   # float32 → 4 bytes per element


def _file_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _print_stats(con: sqlite3.Connection, table: str) -> None:
    total = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"  Total rows : {total:,}")
    rows = con.execute(
        f"SELECT length(vector) AS sz, COUNT(*) AS cnt "
        f"FROM {table} GROUP BY sz ORDER BY sz"
    ).fetchall()
    for sz, cnt in rows:
        dims = sz // 4
        marker = "  ✓" if sz == _EXPECTED_BYTES else "  ✗ MALFORMED"
        print(f"  {sz:>8} bytes ({dims} float32s) — {cnt:,} rows{marker}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove malformed embedding vectors and VACUUM embeddings_cache.db."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be done without writing anything.",
    )
    args = parser.parse_args()

    db_path = _config.EMBEDDING_CACHE_DB()

    if not os.path.exists(db_path):
        print(f"Note: {db_path} does not exist — nothing to do.")
        sys.exit(0)

    size_before = _file_size(db_path)
    print(f"embeddings_cache.db  —  {size_before:,} bytes on disk")

    # ------------------------------------------------------------------
    # Check for malformed rows across both tables
    # ------------------------------------------------------------------
    with sqlite3.connect(db_path) as con:
        # Ensure tables exist (triggers migration if needed)
        tables = {row[0] for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}

        total_malformed = 0
        for table in _TABLES:
            if table not in tables:
                print(f"\nTable '{table}' not found — skipping.")
                continue
            count = con.execute(
                f"SELECT COUNT(*) FROM {table} WHERE length(vector) != ?",
                (_EXPECTED_BYTES,),
            ).fetchone()[0]
            total_malformed += count
            print(f"\n[{table}]")
            print(f"  Expected blob size : {_EXPECTED_BYTES} bytes "
                  f"({_config.EMBEDDING_STORAGE_DIM} dims × 4 bytes / float32)")
            print(f"  Malformed rows     : {count:,}")
            print("  Pre-cleanup stats:")
            _print_stats(con, table)

    # ------------------------------------------------------------------
    # Count orphaned search-term embeddings
    # ------------------------------------------------------------------
    app_db_path = _config.APP_DB_PATH()
    n_orphans = 0
    with sqlite3.connect(db_path) as con:
        tables = {row[0] for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        if "search_term_embeddings" in tables and os.path.exists(app_db_path):
            con.execute(f"ATTACH DATABASE ? AS appdb", (app_db_path,))
            n_orphans = con.execute(
                """
                SELECT COUNT(*) FROM search_term_embeddings
                WHERE query NOT IN (
                    SELECT DISTINCT query FROM appdb.user_search_terms
                )
                """
            ).fetchone()[0]
            print(f"\n[search_term_embeddings]")
            print(f"  Orphaned rows (no matching user_search_terms entry) : {n_orphans:,}")
        elif "search_term_embeddings" not in tables:
            print("\n[search_term_embeddings] Table not found — skipping.")

    if total_malformed == 0 and n_orphans == 0:
        print("\nNothing to clean up — database is already healthy.")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Dry-run exit
    # ------------------------------------------------------------------
    if args.dry_run:
        print("\nDry run — no files will be written.")
        print(f"Would back up : {db_path}  →  {db_path}.bak.<timestamp>")
        print(f"Would delete  : {total_malformed:,} malformed rows across {len(_TABLES)} tables")
        print(f"Would delete  : {n_orphans:,} orphaned search-term embedding rows")
        print("Would VACUUM  : yes")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Backup
    # ------------------------------------------------------------------
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{db_path}.bak.{timestamp}"
    print(f"\nBacking up {db_path}\n  → {backup_path} ...")
    shutil.copy2(db_path, backup_path)
    print(f"  Backup written ({os.path.getsize(backup_path):,} bytes).")

    # ------------------------------------------------------------------
    # Delete malformed rows from both tables
    # ------------------------------------------------------------------
    with sqlite3.connect(db_path) as con:
        con.execute("PRAGMA journal_mode=WAL")
        tables = {row[0] for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        total_deleted = 0
        for table in _TABLES:
            if table not in tables:
                continue
            deleted = con.execute(
                f"DELETE FROM {table} WHERE length(vector) != ?",
                (_EXPECTED_BYTES,),
            ).rowcount
            total_deleted += deleted
            print(f"  Deleted {deleted:,} malformed rows from '{table}'.")

        # Remove orphaned search-term embeddings.
        if "search_term_embeddings" in tables and os.path.exists(app_db_path):
            con.execute("ATTACH DATABASE ? AS appdb", (app_db_path,))
            orphans_deleted = con.execute(
                """
                DELETE FROM search_term_embeddings
                WHERE query NOT IN (
                    SELECT DISTINCT query FROM appdb.user_search_terms
                )
                """
            ).rowcount
            total_deleted += orphans_deleted
            print(f"  Deleted {orphans_deleted:,} orphaned rows from 'search_term_embeddings'.")

        con.commit()
        print(f"\nTotal deleted: {total_deleted:,} rows.")

        for table in _TABLES:
            if table not in tables:
                continue
            print(f"\nPost-delete stats [{table}]:")
            _print_stats(con, table)

    # ------------------------------------------------------------------
    # VACUUM (must run outside any active transaction)
    # ------------------------------------------------------------------
    print("\nRunning VACUUM ...")
    with sqlite3.connect(db_path) as con:
        con.execute("VACUUM")
    print("  VACUUM complete.")

    size_after = _file_size(db_path)
    saved      = size_before - size_after
    print(
        f"\nDone.\n"
        f"  File size : {size_before:,} bytes  →  {size_after:,} bytes  "
        f"(saved {saved:,} bytes / {saved / 1024 / 1024:.2f} MB)"
    )


if __name__ == "__main__":
    main()
