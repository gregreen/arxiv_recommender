#!/bin/bash
# restore.sh — Restore arXiv Recommender data from a restic backup.
#
# ─────────────────────────────────────────────────────────────────────────────
# WHAT IS RESTORED
# ─────────────────────────────────────────────────────────────────────────────
#   app.db               — user accounts, preferences, groups, likes
#   embeddings_cache.db  — precomputed paper embeddings
#   arxiv_summary_cache/ — LLM-generated paper summaries
#
# ─────────────────────────────────────────────────────────────────────────────
# SAFETY NOTES
# ─────────────────────────────────────────────────────────────────────────────
# • Stop the application services BEFORE running this script to avoid
#   in-flight writes being overwritten:
#     systemctl stop arxiv-recommender arxiv-embed-daemon arxiv-meta-daemon
#   Restart them afterwards:
#     systemctl start arxiv-recommender arxiv-embed-daemon arxiv-meta-daemon
#
# • Database files are written to a temporary path then atomically renamed
#   into place, so there is no window where a partial file is live.
#
# • arxiv_summary_cache/ is replaced via a two-step mv: the old directory is
#   moved aside first, then the restored one is moved in, and only then the
#   old copy is deleted. If the second mv fails the old data is still intact.
#
# • With --merge-papers, papers that exist in the live data but are absent
#   from the backup snapshot are preserved: their app.db row, both embeddings,
#   and their summary file are copied into the restored copies BEFORE the swap.
#
# ─────────────────────────────────────────────────────────────────────────────
# USAGE
# ─────────────────────────────────────────────────────────────────────────────
#   # Restore latest snapshot:
#   bash scripts/restore.sh
#
#   # Restore a specific snapshot:
#   bash scripts/restore.sh --snapshot abc12345
#
#   # Dry run (restores into a temp dir, prints what would be swapped, no live
#   # files are changed):
#   bash scripts/restore.sh --dry-run
#
#   # List available snapshots:
#   restic snapshots
#
#   # Preserve papers not present in the backup:
#   bash scripts/restore.sh --merge-papers
#
# ─────────────────────────────────────────────────────────────────────────────
# REQUIRED ENVIRONMENT VARIABLES
# ─────────────────────────────────────────────────────────────────────────────
#   RESTIC_PASSWORD     — Encryption passphrase for the restic repository.
#   RESTIC_REPOSITORY   — Repository URL (e.g. b2:<bucket>:/arxiv-recommender).
#   + backend credentials: B2_ACCOUNT_ID + B2_ACCOUNT_KEY (Backblaze B2), or
#                           AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY (S3/OSS/R2).
#
# See scripts/backup.sh for full backend setup instructions.
#
# ─────────────────────────────────────────────────────────────────────────────
# AWS S3 — IAM POLICY (restore-only)
# ─────────────────────────────────────────────────────────────────────────────
# Create an IAM user or role with the following inline policy
# (replace <bucket-name> with your bucket name):
#
#   {
#     "Version": "2012-10-17",
#     "Statement": [
#       {
#         "Effect": "Allow",
#         "Action": [
#           "s3:GetObject",
#           "s3:HeadObject",
#           "s3:ListBucket",
#           "s3:GetBucketLocation"
#         ],
#         "Resource": [
#           "arn:aws:s3:::<bucket-name>",
#           "arn:aws:s3:::<bucket-name>/*"
#         ]
#       },
#       {
#         "Effect": "Allow",
#         "Action": [
#           "s3:PutObject",
#           "s3:DeleteObject"
#         ],
#         "Resource": ["arn:aws:s3:::<bucket-name>/locks/*"]
#       }
#     ]
#   }
#
# s3:PutObject + s3:DeleteObject are scoped to locks/* only so restic can
# create and clean up its repository lock. No snapshot data can be modified.
#
# ─────────────────────────────────────────────────────────────────────────────
# ALIBABA CLOUD OSS — RAM USER PERMISSIONS (restore-only)
# ─────────────────────────────────────────────────────────────────────────────
# Create a RAM user with OpenAPI access and the following inline policy
# (replace <bucket> with your bucket name).
#
# Restic requires PutObject + DeleteObject on the locks/ prefix to create and
# clean up its repository lock, even for read-only operations like restore.
# Scoping these to locks/* prevents the restore user from touching any snapshot
# data (packs, index, snapshots, keys).
#
#   {
#     "Version": "1",
#     "Statement": [
#       {
#         "Effect": "Allow",
#         "Action": [
#           "oss:GetObject",
#           "oss:HeadObject",
#           "oss:ListObjects",
#           "oss:GetBucketInfo"
#         ],
#         "Resource": [
#           "acs:oss:*:*:<bucket>",
#           "acs:oss:*:*:<bucket>/*"
#         ]
#       },
#       {
#         "Effect": "Allow",
#         "Action": [
#           "oss:PutObject",
#           "oss:DeleteObject"
#         ],
#         "Resource": [
#           "acs:oss:*:*:<bucket>/locks/*"
#         ]
#       }
#     ]
#   }
#
# Keep backup (write-only) and restore users separate where possible —
# a compromised restore credential cannot overwrite or delete snapshot data.

set -euo pipefail

DATA_DIR="${DATA_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────
SNAPSHOT="latest"
DRY_RUN=0
MERGE_PAPERS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --snapshot)
            [[ $# -lt 2 ]] && { echo "ERROR: --snapshot requires an argument." >&2; exit 1; }
            SNAPSHOT="$2"; shift 2 ;;
        --dry-run)
            DRY_RUN=1; shift ;;
        --merge-papers)
            MERGE_PAPERS=1; shift ;;
        *)
            echo "ERROR: Unknown argument: $1" >&2
            echo "Usage: $0 [--snapshot <id>] [--dry-run] [--merge-papers]" >&2
            exit 1 ;;
    esac
done

# ─────────────────────────────────────────────────────────────────────────────
# Environment checks
# ─────────────────────────────────────────────────────────────────────────────
for var in RESTIC_PASSWORD RESTIC_REPOSITORY; do
    if [[ -z "${!var:-}" ]]; then
        echo "ERROR: Environment variable $var is not set." >&2
        exit 1
    fi
done

# ─────────────────────────────────────────────────────────────────────────────
# Restic restore into a work directory
# ─────────────────────────────────────────────────────────────────────────────
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

echo "==> Restoring snapshot '$SNAPSHOT' from $RESTIC_REPOSITORY ..."
restic restore "$SNAPSHOT" --target "$WORK_DIR"
echo "==> Restic restore complete."

# ─────────────────────────────────────────────────────────────────────────────
# Locate restored files (restic recreates the original absolute path under
# --target, so e.g. /home/user/project/app.db lands at $WORK_DIR/home/user/…)
# ─────────────────────────────────────────────────────────────────────────────
find_restored() {
    # Usage: find_restored <filename-or-dirname>
    find "$WORK_DIR" -maxdepth 6 -name "$1" | head -1
}

RESTORED_APP_DB="$(find_restored app.db)"
RESTORED_EMB_DB="$(find_restored embeddings_cache.db)"
RESTORED_SUMMARY_DIR="$(find_restored arxiv_summary_cache)"

# ─────────────────────────────────────────────────────────────────────────────
# --dry-run: report what would be swapped, then exit without touching live data
# ─────────────────────────────────────────────────────────────────────────────
if [[ $DRY_RUN -eq 1 ]]; then
    echo "==> Dry run: the following would be swapped into $DATA_DIR:"
    [[ -n "$RESTORED_APP_DB" ]]      && echo "      app.db"               || echo "      WARNING: app.db not found in snapshot"
    [[ -n "$RESTORED_EMB_DB" ]]      && echo "      embeddings_cache.db"  || echo "      WARNING: embeddings_cache.db not found in snapshot"
    [[ -n "$RESTORED_SUMMARY_DIR" ]] && echo "      arxiv_summary_cache/" || echo "      WARNING: arxiv_summary_cache not found in snapshot"
    echo "==> Dry run complete. No files were changed."
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# --merge-papers: copy papers absent from the backup into the restored copies
# ─────────────────────────────────────────────────────────────────────────────
if [[ $MERGE_PAPERS -eq 1 ]]; then
    echo "==> Merging papers not present in the snapshot ..."

    LIVE_APP_DB="$DATA_DIR/app.db"
    LIVE_EMB_DB="$DATA_DIR/embeddings_cache.db"
    LIVE_SUMMARY_DIR="$DATA_DIR/arxiv_summary_cache"

    MERGED_PAPERS=0
    MERGED_REC_EMBEDDINGS=0
    MERGED_SEARCH_EMBEDDINGS=0
    MERGED_SUMMARIES=0

    # Merge papers table: rows in live but absent from restored.
    if [[ -n "$RESTORED_APP_DB" && -f "$LIVE_APP_DB" ]]; then
        MERGED_PAPERS="$(sqlite3 "$RESTORED_APP_DB" "
            ATTACH '$LIVE_APP_DB' AS live;
            INSERT OR IGNORE INTO papers
                SELECT * FROM live.papers
                WHERE arxiv_id NOT IN (SELECT arxiv_id FROM papers);
            SELECT changes();
        ")"
    fi

    # Merge both embedding tables.
    if [[ -n "$RESTORED_EMB_DB" && -f "$LIVE_EMB_DB" ]]; then
        MERGED_REC_EMBEDDINGS="$(sqlite3 "$RESTORED_EMB_DB" "
            ATTACH '$LIVE_EMB_DB' AS live;
            INSERT OR IGNORE INTO recommendation_embeddings
                SELECT * FROM live.recommendation_embeddings
                WHERE arxiv_id NOT IN (SELECT arxiv_id FROM recommendation_embeddings);
            SELECT changes();
        ")"
        MERGED_SEARCH_EMBEDDINGS="$(sqlite3 "$RESTORED_EMB_DB" "
            ATTACH '$LIVE_EMB_DB' AS live;
            INSERT OR IGNORE INTO search_embeddings
                SELECT * FROM live.search_embeddings
                WHERE arxiv_id NOT IN (SELECT arxiv_id FROM search_embeddings);
            SELECT changes();
        ")"
    fi

    # Merge summary files: copy live files that don't exist in the restored dir.
    if [[ -n "$RESTORED_SUMMARY_DIR" && -d "$LIVE_SUMMARY_DIR" ]]; then
        while IFS= read -r -d '' src; do
            fname="$(basename "$src")"
            dest="$RESTORED_SUMMARY_DIR/$fname"
            if [[ ! -f "$dest" ]]; then
                cp "$src" "$dest"
                (( MERGED_SUMMARIES++ )) || true
            fi
        done < <(find "$LIVE_SUMMARY_DIR" -maxdepth 1 -name '*.txt' -print0)
    fi

    echo "    Papers (app.db):               $MERGED_PAPERS merged"
    echo "    Recommendation embeddings:     $MERGED_REC_EMBEDDINGS merged"
    echo "    Search embeddings:             $MERGED_SEARCH_EMBEDDINGS merged"
    echo "    Summary files:                 $MERGED_SUMMARIES merged"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Swap phase: atomically move restored files into the live DATA_DIR
# ─────────────────────────────────────────────────────────────────────────────
RESTORED_COUNT=0

# Databases: write to a .tmp file beside the target, then rename atomically.
for db in app.db embeddings_cache.db; do
    restored_db="$(find_restored "$db")"
    if [[ -z "$restored_db" ]]; then
        echo "WARNING: $db not found in snapshot; skipping." >&2
        continue
    fi
    tmp_dest="$DATA_DIR/$db.tmp"
    cp "$restored_db" "$tmp_dest"
    mv "$tmp_dest" "$DATA_DIR/$db"
    echo "==> Restored $db"
    (( RESTORED_COUNT++ )) || true
done

# arxiv_summary_cache: two-step mv to keep old data safe until new is in place.
if [[ -z "$RESTORED_SUMMARY_DIR" ]]; then
    echo "WARNING: arxiv_summary_cache not found in snapshot; skipping." >&2
else
    LIVE_SUMMARY="$DATA_DIR/arxiv_summary_cache"
    OLD_SUMMARY="$WORK_DIR/arxiv_summary_cache.old"

    if [[ -d "$LIVE_SUMMARY" ]]; then
        mv "$LIVE_SUMMARY" "$OLD_SUMMARY"
    fi
    mv "$RESTORED_SUMMARY_DIR" "$LIVE_SUMMARY"
    # Old copy is now safe to discard; failure before this point leaves it intact.
    if [[ -d "$OLD_SUMMARY" ]]; then
        rm -rf "$OLD_SUMMARY"
    fi
    echo "==> Restored arxiv_summary_cache/"
    (( RESTORED_COUNT++ )) || true
fi

echo "==> Restore complete: $RESTORED_COUNT item(s) written to $DATA_DIR"
