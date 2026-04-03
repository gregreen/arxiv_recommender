#!/usr/bin/env python3
"""
One-time migration: populate the `papers` table in app.db from the existing
embedding cache and metadata cache.

Safe to run multiple times — uses INSERT OR IGNORE so already-migrated rows
are skipped.

Usage:
    python3 scripts/migrate_legacy.py [--db PATH]

The script never deletes or modifies embeddings_cache.db, arxiv_metadata_cache/,
or any other existing data file.
"""

import argparse
import json
import sqlite3
import sys
import os

# Allow running from the project root without installing the package.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from arxiv_lib.appdb import init_app_db, get_connection
from arxiv_lib.config import APP_DB_PATH, EMBEDDING_CACHE_DB
from arxiv_lib.ingest import load_from_arxiv_metadata_cache


def load_all_embedding_ids(embedding_db_path: str) -> list[str]:
    """Return all arXiv IDs present in embeddings_cache.db."""
    with sqlite3.connect(embedding_db_path) as con:
        rows = con.execute("SELECT arxiv_id FROM embeddings ORDER BY arxiv_id").fetchall()
    return [row[0] for row in rows]


def migrate(app_db_path: str, embedding_db_path: str) -> None:
    # Ensure schema exists (safe if already created)
    init_app_db(app_db_path)

    print(f"Reading arXiv IDs from {embedding_db_path} ...")
    arxiv_ids = load_all_embedding_ids(embedding_db_path)
    print(f"  Found {len(arxiv_ids)} embedded papers.")

    print("Loading metadata from arxiv_metadata_cache/ ...")
    metadata = load_from_arxiv_metadata_cache(arxiv_ids)
    print(f"  Metadata found for {len(metadata)} / {len(arxiv_ids)} papers.")

    rows = []
    for arxiv_id in arxiv_ids:
        meta = metadata.get(arxiv_id, {})
        rows.append((
            arxiv_id,
            meta.get("title"),
            meta.get("abstract"),
            json.dumps(meta["authors"]) if "authors" in meta else None,
            None,   # published_date — not stored in legacy cache
            None,   # categories — not stored in legacy cache
        ))

    print(f"Inserting {len(rows)} rows into papers table (INSERT OR IGNORE) ...")
    with get_connection(app_db_path) as con:
        con.executemany(
            """
            INSERT OR IGNORE INTO papers
                (arxiv_id, title, abstract, authors, published_date, categories)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    # Verify
    with get_connection(app_db_path) as con:
        count = con.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    print(f"Done. papers table now has {count} rows.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db", default=APP_DB_PATH,
        help=f"Path to app.db (default: {APP_DB_PATH})",
    )
    parser.add_argument(
        "--embedding-db", default=EMBEDDING_CACHE_DB,
        help=f"Path to embeddings_cache.db (default: {EMBEDDING_CACHE_DB})",
    )
    args = parser.parse_args()

    migrate(args.db, args.embedding_db)
    return 0


if __name__ == "__main__":
    sys.exit(main())
