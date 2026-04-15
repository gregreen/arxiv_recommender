#!/usr/bin/env python3
"""
Daily maintenance script: remove stale arXiv source files and arxiv-to-prompt
download cache entries that have not been accessed recently.

What it does
------------
1. Deletes individual .tex files in SOURCE_CACHE_DIR that are older than
   --max-age-days days.  These are re-downloadable on demand; they are NOT
   LLM-generated and are cheap to regenerate.
2. Deletes per-paper subdirectories inside the arxiv-to-prompt download cache
   (~/.cache/arxiv-to-prompt/ by default, or $XDG_CACHE_HOME/arxiv-to-prompt/)
   that are older than --max-age-days days.

SUMMARY_CACHE_DIR is intentionally NOT touched; those files are produced by
LLMs and are expensive to regenerate.

Cron example
------------
Run at 03:30 every day, logging output to /var/log/arxiv-source-cleanup.log:

    30 3 * * * /home/<user>/arxiv_recommender/.venv/bin/python3 \\
        /home/<user>/arxiv_recommender/scripts/arxiv_source_cleanup.py \\
        >> /var/log/arxiv-source-cleanup.log 2>&1

Usage
-----
    python scripts/arxiv_source_cleanup.py
    python scripts/arxiv_source_cleanup.py --dry-run
    python scripts/arxiv_source_cleanup.py --max-age-days 14
"""

import argparse
import glob
import os
import shutil
import sys
import time
from datetime import datetime

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from arxiv_lib import config as _config


def _is_stale(path: str, max_age_secs: float) -> bool:
    try:
        return time.time() - os.stat(path).st_mtime > max_age_secs
    except OSError:
        return False


def _dir_size(path: str) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for fname in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, fname))
            except OSError:
                pass
    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Remove stale arXiv .tex source files and arxiv-to-prompt download "
            "cache entries older than --max-age-days days."
        )
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=7,
        metavar="N",
        help="Delete files/dirs older than N days (default: 7).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be done without deleting anything.",
    )
    args = parser.parse_args()

    max_age_secs = args.max_age_days * 86400
    dry_run = args.dry_run
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"[{now_str}] arxiv_source_cleanup  max_age={args.max_age_days}d  dry_run={dry_run}")

    # ── Section 1: SOURCE_CACHE_DIR .tex files ────────────────────────────────

    source_dir = _config.SOURCE_CACHE_DIR
    tex_files = sorted(glob.glob(os.path.join(source_dir, "*.tex")))
    stale_tex = [f for f in tex_files if _is_stale(f, max_age_secs)]

    bytes_tex = sum(os.path.getsize(f) for f in stale_tex if os.path.exists(f))
    print(
        f"\n[SOURCE_CACHE_DIR]  {source_dir}"
        f"\n  Total .tex files : {len(tex_files):,}"
        f"\n  Stale (>{args.max_age_days}d)  : {len(stale_tex):,}  ({bytes_tex:,} bytes)"
    )

    if not dry_run:
        deleted_tex = 0
        for fpath in stale_tex:
            try:
                os.remove(fpath)
                deleted_tex += 1
            except OSError as exc:
                print(f"  Warning: could not delete {fpath}: {exc}")
        print(f"  Deleted          : {deleted_tex:,} files  ({bytes_tex:,} bytes freed)")
    else:
        print(f"  Would delete     : {len(stale_tex):,} files  ({bytes_tex:,} bytes)")

    # ── Section 2: arxiv-to-prompt download cache (subdirectories) ───────────

    xdg_cache = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    atp_cache_dir = os.path.join(xdg_cache, "arxiv-to-prompt")

    print(f"\n[ARXIV-TO-PROMPT CACHE]  {atp_cache_dir}")

    if not os.path.isdir(atp_cache_dir):
        print("  Directory does not exist — skipping.")
    else:
        subdirs = []
        with os.scandir(atp_cache_dir) as it:
            for entry in it:
                if entry.is_dir(follow_symlinks=False):
                    subdirs.append(entry.path)
        subdirs.sort()

        stale_dirs = [d for d in subdirs if _is_stale(d, max_age_secs)]
        bytes_dirs = sum(_dir_size(d) for d in stale_dirs)

        print(
            f"  Total subdirs    : {len(subdirs):,}"
            f"\n  Stale (>{args.max_age_days}d)  : {len(stale_dirs):,}  ({bytes_dirs:,} bytes)"
        )

        if not dry_run:
            deleted_dirs = 0
            for dpath in stale_dirs:
                try:
                    shutil.rmtree(dpath)
                    deleted_dirs += 1
                except OSError as exc:
                    print(f"  Warning: could not delete {dpath}: {exc}")
            print(f"  Deleted          : {deleted_dirs:,} dirs  ({bytes_dirs:,} bytes freed)")
        else:
            print(f"  Would delete     : {len(stale_dirs):,} dirs  ({bytes_dirs:,} bytes)")

    print("\nDone.")


if __name__ == "__main__":
    main()
