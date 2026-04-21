#!/bin/bash
# backup.sh — Back up critical arXiv Recommender data using restic.
#
# ─────────────────────────────────────────────────────────────────────────────
# WHAT IS BACKED UP
# ─────────────────────────────────────────────────────────────────────────────
#   app.db               — user accounts, preferences, groups, likes
#   embeddings_cache.db  — precomputed paper embeddings
#   arxiv_summary_cache/ — LLM-generated paper summaries
#
# NOT backed up:
#   arxiv_source_cache/   — large .tex files; re-downloadable from arXiv
#   arxiv_metadata_cache/ — no longer used
#
# ─────────────────────────────────────────────────────────────────────────────
# HOW DATABASE SNAPSHOTS WORK
# ─────────────────────────────────────────────────────────────────────────────
# Each SQLite database is snapshotted using the built-in Online Backup API
# (`sqlite3 source.db ".backup dest.db"`). This is safe under concurrent
# writes: it reads a consistent snapshot page-by-page, automatically checkpoints
# WAL-mode journals, and produces a single self-contained file. Raw file copies
# of a live SQLite database risk partial writes and are not used here.
#
# ─────────────────────────────────────────────────────────────────────────────
# RESTIC OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
# restic encrypts all data client-side before uploading. The repository
# password (RESTIC_PASSWORD) never leaves the machine — without it, backup
# data is useless to anyone with storage access. restic also deduplicates at
# the chunk level, so daily snapshots of unchanged data cost very little space.
#
# ─────────────────────────────────────────────────────────────────────────────
# SECURITY STRATEGY: WRITE-ONLY CREDENTIALS + IMMUTABLE STORAGE
# ─────────────────────────────────────────────────────────────────────────────
# A compromised VM should not be able to delete or overwrite existing backups.
# Achieve this with two independent controls:
#
# 1. Write-only credentials — the key stored on the VM has no delete permission.
#    Even if the VM is fully compromised, past snapshots cannot be purged.
#
# 2. Object Lock / WORM (Write Once Read Many) — a bucket-level policy that
#    prevents deletion or overwrite of objects for a fixed period (e.g. 30 days)
#    regardless of credentials, including the root/master account.
#    Both controls are redundant — either one alone is protective. Together they
#    provide defense-in-depth.
#
# ─────────────────────────────────────────────────────────────────────────────
# BACKEND SETUP: BACKBLAZE B2
# ─────────────────────────────────────────────────────────────────────────────
# Credentials env vars:
#   B2_ACCOUNT_ID       — Application Key ID from the Backblaze console
#   B2_ACCOUNT_KEY      — Application Key (see below)
#
# RESTIC_REPOSITORY format:
#   b2:<bucket-name>:/arxiv-recommender
#
# 1. Create a private bucket (e.g. "my-arxiv-recommender-backup").
#    Enable Object Lock at creation time (cannot be added later on B2) with a
#    default retention period (e.g. 30 days, Compliance mode).
#
# 2. Create an Application Key scoped to that bucket only, with capabilities:
#      listFiles, readFiles, writeFiles
#    Do NOT grant deleteFiles. With Object Lock enabled, this key cannot
#    circumvent the retention policy even if stolen.
#
# 3. One-time repository init (use master key or a key with listBuckets):
#      export B2_ACCOUNT_ID=<key-id>
#      export B2_ACCOUNT_KEY=<application-key>
#      export RESTIC_PASSWORD=<passphrase>
#      export RESTIC_REPOSITORY=b2:<bucket-name>:/arxiv-recommender
#      restic init
#    Switch to the restricted write-only key for all subsequent daily runs.
#
# ─────────────────────────────────────────────────────────────────────────────
# BACKEND SETUP: AWS S3 (and any S3-compatible service)
# ─────────────────────────────────────────────────────────────────────────────
# Credentials env vars (restic uses AWS var names for all S3-compatible backends):
#   AWS_ACCESS_KEY_ID     — IAM access key ID
#   AWS_SECRET_ACCESS_KEY — IAM secret access key
#
# RESTIC_REPOSITORY format:
#   s3:s3.amazonaws.com/<bucket>/arxiv-recommender          (AWS)
#   s3:https://oss-<region>.aliyuncs.com/<bucket>/path      (Alibaba OSS)
#   s3:https://<account>.r2.cloudflarestorage.com/<bucket>  (Cloudflare R2)
#   s3:https://<endpoint>/<bucket>                          (any other)
#
# AWS S3 setup:
# -------------
# 
# 1. Create a private S3 bucket. Enable Object Lock at creation time (cannot
#    be added later). Set a default retention period (e.g. 30 days, Compliance).
#
# 2. Create an IAM policy granting read+write but NOT delete:
#    {
#      "Version": "2012-10-17",
#      "Statement": [{
#        "Effect": "Allow",
#        "Action": [
#          "s3:PutObject", "s3:GetObject", "s3:HeadObject",
#          "s3:ListBucket", "s3:GetBucketLocation",
#          "s3:AbortMultipartUpload", "s3:ListMultipartUploadParts",
#          "s3:ListBucketMultipartUploads"
#        ],
#        "Resource": [
#          "arn:aws:s3:::<bucket-name>",
#          "arn:aws:s3:::<bucket-name>/*"
#        ]
#      }]
#    }
#    Notably absent: s3:DeleteObject, s3:DeleteBucket.
#    Attach this policy to a dedicated IAM user or role used only for backups.
#
# 3. One-time repository init:
#      export AWS_ACCESS_KEY_ID=<key-id>
#      export AWS_SECRET_ACCESS_KEY=<secret-key>
#      export RESTIC_PASSWORD=<passphrase>
#      export RESTIC_REPOSITORY=s3:s3.amazonaws.com/<bucket>/arxiv-recommender
#      restic init
#
# Alibaba Cloud OSS setup (S3-compatible, same credential env vars):
# ------------------------------------------------------------------
# 
# 1. Create a private OSS bucket. Enable versioning (Basic Settings → Versioning).
#    Enable a Retention Policy (WORM): Basic Settings → Retention Policy →
#    Create Policy → set period (e.g. 30 days) → Lock (permanent, Compliance mode).
#
# 2. Create a RAM user with OpenAPI access. Attach an inline policy:
#    {
#      "Version": "1",
#      "Statement": [{
#        "Effect": "Allow",
#        "Action": [
#          "oss:PutObject", "oss:GetObject", "oss:HeadObject",
#          "oss:ListObjects", "oss:ListObjectVersions",
#          "oss:GetBucketLocation", "oss:GetBucketInfo", "oss:GetBucketVersioning",
#          "oss:ListMultipartUploads", "oss:AbortMultipartUpload", "oss:ListParts"
#        ],
#        "Resource": [
#          "acs:oss:*:*:<bucket-name>",
#          "acs:oss:*:*:<bucket-name>/*"
#        ]
#      }]
#    }
#    Notably absent: oss:DeleteObject, oss:DeleteBucket, oss:AbortBucketWorm.
#
# 3. One-time repository init:
#      export AWS_ACCESS_KEY_ID=<ram-access-key-id>
#      export AWS_SECRET_ACCESS_KEY=<ram-access-key-secret>
#      export RESTIC_PASSWORD=<passphrase>
#      export RESTIC_REPOSITORY=s3:https://oss-us-west-1.aliyuncs.com/<bucket>/arxiv-recommender
#      restic init
#
# ─────────────────────────────────────────────────────────────────────────────
# REQUIRED ENVIRONMENT VARIABLES (all backends)
# ─────────────────────────────────────────────────────────────────────────────
# Add to .env (Docker) or <project_dir>/.env (systemd):
#
#   RESTIC_PASSWORD     — Encryption passphrase for the restic repository.
#                         Store offline (password manager). Without it,
#                         backups cannot be decrypted even if storage is accessible.
#   RESTIC_REPOSITORY   — Repository URL (format depends on backend; see above).
#   + backend credentials — B2_ACCOUNT_ID + B2_ACCOUNT_KEY, or
#                           AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY.
#
# ─────────────────────────────────────────────────────────────────────────────
# RESTORING
# ─────────────────────────────────────────────────────────────────────────────
#   List snapshots:   restic snapshots
#   Restore latest:   restic restore latest --target /tmp/restore
#   Restore specific: restic restore <snapshot-id> --target /tmp/restore
#
# ─────────────────────────────────────────────────────────────────────────────
# NOTE ON PRUNING
# ─────────────────────────────────────────────────────────────────────────────
# This script does NOT call `restic forget --prune`. With write-only credentials
# and Object Lock / WORM enabled, pruning requires delete permissions that the
# backup key intentionally lacks. Old snapshots age out automatically when the
# retention period expires. For active pruning, run `restic forget --prune`
# separately with a more-privileged key stored off-VM (e.g. on your laptop).
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Resolve DATA_DIR ──────────────────────────────────────────────────────────
# In Docker: DATA_DIR=/app/data (set in docker-compose.yml)
# In systemd: DATA_DIR defaults to the project root if unset
DATA_DIR="${DATA_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

# ── Validate always-required env vars ────────────────────────────────────────
for var in RESTIC_PASSWORD RESTIC_REPOSITORY; do
    if [[ -z "${!var:-}" ]]; then
        echo "ERROR: Environment variable $var is not set." >&2
        echo "       See the setup instructions at the top of this script." >&2
        exit 1
    fi
done

# ── Create a temp dir and register cleanup ────────────────────────────────────
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

# ── Snapshot each SQLite database via the Online Backup API ──────────────────
# sqlite3 ".backup" reads a consistent snapshot even under concurrent writes
# and fully checkpoints WAL mode. The resulting file is self-contained — no
# need to copy -wal or -shm companions.
BACKUP_PATHS=()

for db in app.db embeddings_cache.db; do
    db_path="$DATA_DIR/$db"
    dest="$WORK_DIR/$db"
    if [[ -f "$db_path" ]]; then
        echo "==> Snapshotting $db ..."
        sqlite3 "$db_path" ".backup $dest"
        BACKUP_PATHS+=("$dest")
    else
        echo "WARNING: $db_path not found; skipping." >&2
    fi
done

# ── Add summary cache directly (plain files, no sqlite treatment needed) ──────
summary_cache="$DATA_DIR/arxiv_summary_cache"
if [[ -d "$summary_cache" ]]; then
    BACKUP_PATHS+=("$summary_cache")
else
    echo "WARNING: $summary_cache not found; skipping." >&2
fi

if [[ ${#BACKUP_PATHS[@]} -eq 0 ]]; then
    echo "ERROR: No backup paths found under DATA_DIR=$DATA_DIR" >&2
    exit 1
fi

# ── Run backup ────────────────────────────────────────────────────────────────
echo "==> Starting restic backup ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
echo "    Repository: $RESTIC_REPOSITORY"
echo "    Paths: ${BACKUP_PATHS[*]}"

restic backup "${BACKUP_PATHS[@]}"

echo "==> Backup complete ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
