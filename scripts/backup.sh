#!/bin/bash
# backup.sh — Back up critical arXiv Recommender data to Backblaze B2 using restic.
#
# Backed-up paths:
#   app.db (+ WAL companions)
#   embeddings_cache.db (+ WAL companions)
#   arxiv_summary_cache/
#
# NOT backed up:
#   arxiv_source_cache/  — large .tex files; re-downloadable
#   arxiv_metadata_cache/ — no longer used
#
# ─────────────────────────────────────────────────────────────────────────────
# B2 BUCKET SETUP (one-time, in the Backblaze console)
# ─────────────────────────────────────────────────────────────────────────────
# 1. Create a private bucket (e.g. "my-arxiv-recommender-backup").
#
# 2. Enable Object Lock on the bucket with a default retention period
#    (e.g. 30 days, Compliance mode). This prevents ANY key — including the
#    master key — from deleting or overwriting objects within that window,
#    providing ransomware / compromised-VM resistance.
#    Note: Object Lock must be enabled at bucket creation time on B2.
#
# 3. Create an Application Key scoped ONLY to that bucket with capabilities:
#      listFiles, readFiles, writeFiles
#    Do NOT grant deleteFiles or listBuckets. With Object Lock enabled, this
#    key cannot circumvent the retention policy even if it were stolen.
#
# 4. One-time repository initialisation (run manually once, using a key that
#    also has listBuckets for the init call, or use the master key just this
#    once):
#      export B2_ACCOUNT_ID=<key-id>
#      export B2_ACCOUNT_KEY=<application-key>
#      export RESTIC_PASSWORD=<your-strong-passphrase>
#      export RESTIC_REPOSITORY=b2:<bucket-name>:/arxiv-recommender
#      restic init
#    After init, switch to the restricted write-only key for all future runs.
#
# ─────────────────────────────────────────────────────────────────────────────
# REQUIRED ENVIRONMENT VARIABLES
# ─────────────────────────────────────────────────────────────────────────────
# Add these to .env (Docker) or <project_dir>/.env (systemd):
#
#   B2_ACCOUNT_ID       — Application Key ID from the Backblaze console
#   B2_ACCOUNT_KEY      — Application Key (write-only, scoped to backup bucket)
#   RESTIC_PASSWORD     — Passphrase used to encrypt all backup data.
#                         Store this offline (password manager). Without it,
#                         backups cannot be decrypted even if B2 is accessible.
#   RESTIC_REPOSITORY   — e.g. b2:my-arxiv-recommender-backup:/arxiv-recommender
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
# This script does NOT call `restic forget --prune`. With a write-only key and
# Object Lock enabled on the bucket, pruning requires delete permissions that
# this key intentionally lacks. Old snapshots age out automatically when the
# Object Lock retention period expires. If you want active pruning, run
# `restic forget --prune` separately using a more-privileged key stored
# off-VM (e.g. on your laptop).
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Resolve DATA_DIR ──────────────────────────────────────────────────────────
# In Docker: DATA_DIR=/app/data (set in docker-compose.yml)
# In systemd: DATA_DIR defaults to the project root if unset
DATA_DIR="${DATA_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

# ── Validate required env vars ────────────────────────────────────────────────
for var in B2_ACCOUNT_ID B2_ACCOUNT_KEY RESTIC_PASSWORD RESTIC_REPOSITORY; do
    if [[ -z "${!var:-}" ]]; then
        echo "ERROR: Environment variable $var is not set." >&2
        echo "       See the setup instructions at the top of this script." >&2
        exit 1
    fi
done

# ── Build list of paths to back up ───────────────────────────────────────────
BACKUP_PATHS=()

for db in app.db embeddings_cache.db; do
    db_path="$DATA_DIR/$db"
    if [[ -f "$db_path" ]]; then
        BACKUP_PATHS+=("$db_path")
        # Include WAL-mode companion files if present
        [[ -f "${db_path}-wal" ]] && BACKUP_PATHS+=("${db_path}-wal")
        [[ -f "${db_path}-shm" ]] && BACKUP_PATHS+=("${db_path}-shm")
    else
        echo "WARNING: $db_path not found; skipping." >&2
    fi
done

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
