#!/usr/bin/env python3
"""
Daily maintenance script: remove malformed embedding vectors and compact the
embeddings_cache.db SQLite database.

What it does
------------
1. Deletes any rows whose BLOB size does not match the expected size for
   config.EMBEDDING_DIM float32 values (i.e. EMBEDDING_DIM * 4 bytes).
2. Runs VACUUM to reclaim free pages and shrink the file on disk.
3. Reports row counts and file sizes before / after.

A timestamped backup of embeddings_cache.db is always created before any
writes (unless --dry-run is given).  Nothing is done (and no backup is
written) when no malformed rows are found.

Cron example
------------
Run at 03:00 every day, logging output to /var/log/arxiv-cleanup-embeddings.log:

    0 3 * * * /home/<user>/arxiv_recommender/.venv/bin/python3 \
        /home/<user>/arxiv_recommender/scripts/cleanup_embeddings.py \
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

_EXPECTED_BYTES = _config.EMBEDDING_STORAGE_DIM * 4   # float32 → 4 bytes per element


def _file_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _print_stats(con: sqlite3.Connection) -> None:
    total = con.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    print(f"  Total rows : {total:,}")
    rows = con.execute(
        "SELECT length(vector) AS sz, COUNT(*) AS cnt "
        "FROM embeddings GROUP BY sz ORDER BY sz"
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

    db_path = _config.EMBEDDING_CACHE_DB

    if not os.path.exists(db_path):
        print(f"Note: {db_path} does not exist — nothing to do.")
        sys.exit(0)

    size_before = _file_size(db_path)
    print(f"embeddings_cache.db  —  {size_before:,} bytes on disk")

    # ------------------------------------------------------------------
    # Check for malformed rows
    # ------------------------------------------------------------------
    with sqlite3.connect(db_path) as con:
        malformed_count = con.execute(
            "SELECT COUNT(*) FROM embeddings WHERE length(vector) != ?",
            (_EXPECTED_BYTES,),
        ).fetchone()[0]

        print(f"\nExpected blob size : {_EXPECTED_BYTES} bytes "
              f"({_config.EMBEDDING_STORAGE_DIM} dims × 4 bytes / float32)")
        print(f"Malformed rows     : {malformed_count:,}")

        print("\nPre-cleanup stats:")
        _print_stats(con)

    if malformed_count == 0:
        print("\nNothing to clean up — database is already healthy.")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Dry-run exit
    # ------------------------------------------------------------------
    if args.dry_run:
        print("\nDry run — no files will be written.")
        print(f"Would back up : {db_path}  →  {db_path}.bak.<timestamp>")
        print(f"Would delete  : {malformed_count:,} malformed rows")
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
    # Delete malformed rows
    # ------------------------------------------------------------------
    with sqlite3.connect(db_path) as con:
        con.execute("PRAGMA journal_mode=WAL")
        deleted = con.execute(
            "DELETE FROM embeddings WHERE length(vector) != ?",
            (_EXPECTED_BYTES,),
        ).rowcount
        con.commit()
        print(f"\nDeleted {deleted:,} malformed rows.")

        print("\nPost-delete stats:")
        _print_stats(con)

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
