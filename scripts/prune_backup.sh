#!/bin/bash
# prune_backup.sh — Prune arXiv Recommender restic backups.
# 
# This script can be run regularly (e.g., as a cron job) to prune old
# backups while keeping a reasonable number of snapshots (more recent
# snapshots, trailing off approximately exponentially as they get older).
# The --group-by host option ensures that the databases are pruned correctly,
# since each snapshot puts the databases in a uniquely named temporary
# directory.
# 
# ─────────────────────────────────────────────────────────────────────────────
# !! IMPORTANT SAFETY NOTE !!
# ─────────────────────────────────────────────────────────────────────────────
# Do not store the cloud credentials for the prune script on the web server
# that hosts the web application. These credentials allow the deletion of
# backup snapshots. If the web server is compromised, an attacker could delete
# all backups.
# 
# Instead, store the credentials and run this prune script on a separate
# machine.
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
# AWS S3 — IAM POLICY (prune-only)
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
#           "s3:GetBucketLocation",
#           "s3:PutObject",
#           "s3:DeleteObject",
#           "s3:AbortMultipartUpload",
#           "s3:ListMultipartUploadParts",
#           "s3:ListBucketMultipartUploads"
#         ],
#         "Resource": [
#           "arn:aws:s3:::<bucket-name>",
#           "arn:aws:s3:::<bucket-name>/*"
#         ]
#       }
#     ]
#   }
#
# Unlike the backup user (which scopes DeleteObject to locks/* only), the prune
# user needs DeleteObject on /* to remove unreferenced pack files after
# `restic forget --prune`. Do NOT store these credentials on the web server —
# see the safety note above.
#
# NOTE: If your bucket has Object Lock / WORM enabled, DeleteObject will fail on
# objects still within their retention window. This is expected and desirable —
# those objects are protected by design. Pruning will succeed for objects whose
# retention has expired.
#
# ─────────────────────────────────────────────────────────────────────────────
# ALIBABA CLOUD OSS — RAM USER PERMISSIONS (prune-only)
# ─────────────────────────────────────────────────────────────────────────────
# Create a RAM user with OpenAPI access and the following inline policy
# (replace <bucket> with your bucket name).
# 
# Restic requires read access to inspect the repository index and pack files,
# write access to update the index after pruning, and delete access to remove
# unreferenced pack files. Unlike the backup user (which scopes DeleteObject to
# locks/* only), the prune user needs DeleteObject on /* to actually free space.
# Do NOT store these credentials on the web server — see the safety note above.
#
# NOTE: If your bucket has a WORM Retention Policy enabled, DeleteObject will
# fail on objects still within their retention window. This is expected and
# desirable — those objects are protected by design. Pruning will succeed for
# objects whose retention has expired.
#
# {
#   "Version": "1",
#   "Statement": [
#     {
#       "Effect": "Allow",
#       "Action": [
#         "oss:GetObject",
#         "oss:HeadObject",
#         "oss:ListObjects",
#         "oss:GetBucketInfo",
#         "oss:PutObject",
#         "oss:DeleteObject"
#       ],
#       "Resource": [
#         "acs:oss:*:*:<bucket>",
#         "acs:oss:*:*:<bucket>/*"
#       ]
#     }
#   ]
# }

restic forget --group-by host \
              --keep-daily 7 --keep-weekly 3 \
              --keep-monthly 3 --keep-yearly 1 \
              --prune