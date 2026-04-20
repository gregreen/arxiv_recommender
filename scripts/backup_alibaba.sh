#!/bin/bash
# backup_alibaba.sh — Back up critical arXiv Recommender data to Alibaba Cloud
# OSS using restic (S3-compatible backend).
#
# This is a drop-in replacement for scripts/backup.sh (Backblaze B2).
# To switch, update the volume mount and command in docker-compose.yml:
#   ofelia.job-exec.backup.command: "/bin/sh /app/scripts/backup_alibaba.sh"
# and mount this file instead of backup.sh:
#   - ./scripts/backup_alibaba.sh:/app/scripts/backup_alibaba.sh:ro
# For systemd, update deploy/arxiv-backup.service ExecStart accordingly.
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
# ALIBABA CLOUD OSS BUCKET SETUP (one-time, in the Alibaba Cloud console)
# ─────────────────────────────────────────────────────────────────────────────
# 1. Create a private OSS bucket in your chosen region
#    (e.g. "my-arxiv-recommender-backup" in oss-us-west-1).
#    Enable versioning: Bucket → Basic Settings → Versioning → Enabled.
#
# 2. Enable a Retention Policy (WORM) for ransomware / compromised-VM
#    resistance:
#      Bucket → Basic Settings → Retention Policy → Create Policy
#      Set a retention period (e.g. 30 days) and click "Lock" to make it
#      permanent. In Compliance mode, NO key — including the root account —
#      can delete or overwrite objects within that window.
#    Note: Like B2 Object Lock, this must be decided early; an unlocked policy
#    can be discarded, but a locked policy cannot be shortened.
#
# 3. Find your region endpoint. Examples:
#      US West 1:     oss-us-west-1.aliyuncs.com
#      US East 1:     oss-us-east-1.aliyuncs.com
#      EU Central 1:  oss-eu-central-1.aliyuncs.com
#      AP Southeast 1: oss-ap-southeast-1.aliyuncs.com
#    Full list: https://www.alibabacloud.com/help/en/oss/user-guide/regions-and-endpoints
#
# ─────────────────────────────────────────────────────────────────────────────
# RAM USER + POLICY SETUP (minimal write-only permissions)
# ─────────────────────────────────────────────────────────────────────────────
# 1. In the Alibaba Cloud console, go to RAM → Users → Create User.
#    Enable "OpenAPI Access" (programmatic access); save the AccessKey ID and
#    AccessKey Secret immediately — they are shown only once.
#
# 2. Attach an inline policy that grants read+write but NOT delete access,
#    scoped to your backup bucket only. Create a custom policy with this JSON
#    (replace <bucket-name> with your actual bucket name):
#
#    {
#      "Version": "1",
#      "Statement": [
#        {
#          "Effect": "Allow",
#          "Action": [
#            "oss:PutObject",
#            "oss:GetObject",
#            "oss:HeadObject",
#            "oss:ListObjects",
#            "oss:ListMultipartUploads",
#            "oss:AbortMultipartUpload",
#            "oss:ListParts",
#            "oss:GetBucketLocation",
#            "oss:GetBucketInfo",
#            "oss:GetBucketVersioning",
#            "oss:ListObjectVersions"
#          ],
#          "Resource": [
#            "acs:oss:*:*:<bucket-name>",
#            "acs:oss:*:*:<bucket-name>/*"
#          ]
#        }
#      ]
#    }
#
#    Notably absent: oss:DeleteObject, oss:DeleteBucket, oss:AbortBucketWorm.
#    With the Retention Policy locked, even the root account cannot delete
#    objects inside the retention window — this key cannot either.
#
# 3. For the one-time `restic init`, you may need to temporarily add
#    "oss:PutBucketVersioning" or run init with the root/admin key if you
#    encounter permission errors. After init, the restricted key above is
#    sufficient for all subsequent backup runs.
#
# ─────────────────────────────────────────────────────────────────────────────
# ONE-TIME RESTIC REPOSITORY INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────
# Run this once manually (can use root/admin credentials just for init):
#
#   export AWS_ACCESS_KEY_ID=<your-access-key-id>
#   export AWS_SECRET_ACCESS_KEY=<your-access-key-secret>
#   export RESTIC_PASSWORD=<your-strong-passphrase>
#   export RESTIC_REPOSITORY=s3:https://oss-<region>.aliyuncs.com/<bucket>/arxiv-recommender
#   restic init
#
# After init, switch to the restricted RAM user key for all daily runs.
#
# ─────────────────────────────────────────────────────────────────────────────
# REQUIRED ENVIRONMENT VARIABLES
# ─────────────────────────────────────────────────────────────────────────────
# Add these to .env (Docker) or <project_dir>/.env (systemd):
#
#   AWS_ACCESS_KEY_ID      — RAM user AccessKey ID (restic uses AWS var names
#                            for all S3-compatible backends, including OSS)
#   AWS_SECRET_ACCESS_KEY  — RAM user AccessKey Secret
#   RESTIC_PASSWORD        — Passphrase used to encrypt all backup data.
#                            Store this offline (password manager). Without it,
#                            backups cannot be decrypted even if OSS is accessible.
#   RESTIC_REPOSITORY      — e.g.
#                            s3:https://oss-us-west-1.aliyuncs.com/my-bucket/arxiv-recommender
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
# This script does NOT call `restic forget --prune`. With a write-only RAM key
# and a locked Retention Policy on the bucket, pruning requires delete
# permissions this key intentionally lacks. Old snapshots age out automatically
# when the retention period expires. If you want active pruning, run
# `restic forget --prune` separately using a more-privileged key stored
# off-VM (e.g. on your laptop).
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Resolve DATA_DIR ──────────────────────────────────────────────────────────
# In Docker: DATA_DIR=/app/data (set in docker-compose.yml)
# In systemd: DATA_DIR defaults to the project root if unset
DATA_DIR="${DATA_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

# ── Validate required env vars ────────────────────────────────────────────────
for var in AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY RESTIC_PASSWORD RESTIC_REPOSITORY; do
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
