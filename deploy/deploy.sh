#!/usr/bin/env bash
# deploy.sh — one-shot deployment script for arXiv Recommender
#
# Fill in the three variables below, then run as root:
#   sudo bash deploy/deploy.sh
#
# Safe to re-run for updates (idempotent). The .env file is never overwritten
# once created, so SECRET_KEY is preserved across re-deployments.

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
USER="<user>"           # Linux user that owns the project and runs services
PROJECT_DIR="<project_dir>"  # Absolute path to project root, no trailing slash
DOMAIN="<domain>"       # Production hostname (e.g. arxiv.example.com)
# ───────────────────────────────────────────────────────────────────────────────

# ── Validation ─────────────────────────────────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
    echo "ERROR: This script must be run as root (sudo bash deploy/deploy.sh)." >&2
    exit 1
fi

for var in USER PROJECT_DIR DOMAIN; do
    val="${!var}"
    if [[ "$val" == "<"* ]]; then
        echo "ERROR: \$$var is still set to its placeholder value '$val'." >&2
        echo "       Edit the variables at the top of this script before running." >&2
        exit 1
    fi
done

for cmd in python3 node npm caddy systemctl; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: '$cmd' not found on PATH. Please install it and retry." >&2
        exit 1
    fi
done

if ! id -u "$USER" &>/dev/null; then
    echo "ERROR: Linux user '$USER' does not exist." >&2
    exit 1
fi

echo "==> Deploying arXiv Recommender"
echo "    User:        $USER"
echo "    Project dir: $PROJECT_DIR"
echo "    Domain:      $DOMAIN"
echo

# ── Python virtualenv + dependencies ──────────────────────────────────────────
echo "==> Setting up Python virtualenv..."
if [[ ! -d "$PROJECT_DIR/.venv" ]]; then
    python3 -m venv "$PROJECT_DIR/.venv"
    chown -R "$USER":"$USER" "$PROJECT_DIR/.venv"
fi
sudo -u "$USER" "$PROJECT_DIR/.venv/bin/pip" install --quiet -r "$PROJECT_DIR/requirements.txt"

# ── Frontend build ─────────────────────────────────────────────────────────────
echo "==> Building frontend..."
sudo -u "$USER" bash -c "cd '$PROJECT_DIR/web/frontend' && npm ci --silent && npm run build"

# ── Cache directories ──────────────────────────────────────────────────────────
echo "==> Creating cache directories..."
for dir in arxiv_source_cache arxiv_metadata_cache arxiv_summary_cache; do
    mkdir -p "$PROJECT_DIR/$dir"
done
chown -R "$USER":"$USER" \
    "$PROJECT_DIR/arxiv_source_cache" \
    "$PROJECT_DIR/arxiv_metadata_cache" \
    "$PROJECT_DIR/arxiv_summary_cache"
chmod 750 \
    "$PROJECT_DIR/arxiv_source_cache" \
    "$PROJECT_DIR/arxiv_metadata_cache" \
    "$PROJECT_DIR/arxiv_summary_cache"

# ── File permissions ───────────────────────────────────────────────────────────
echo "==> Setting file permissions..."

# Project root: owner-only read/write/execute on dirs, no world access
chown "$USER":"$USER" "$PROJECT_DIR"
chmod 750 "$PROJECT_DIR"

# Static frontend dist: world-readable so Caddy (running as its own user) can serve it
find "$PROJECT_DIR/web/frontend/dist" -type d -exec chmod 755 {} +
find "$PROJECT_DIR/web/frontend/dist" -type f -exec chmod 644 {} +

# app.db: owner-only (contains user credentials)
if [[ -f "$PROJECT_DIR/app.db" ]]; then
    chown "$USER":"$USER" "$PROJECT_DIR/app.db"
    chmod 600 "$PROJECT_DIR/app.db"
    # Secure WAL-mode companion files if present
    for f in "$PROJECT_DIR/app.db-wal" "$PROJECT_DIR/app.db-shm"; do
        [[ -f "$f" ]] && chown "$USER":"$USER" "$f" && chmod 600 "$f"
    done
fi

# api_keys.json, llm_config.json: owner-only (contain secrets)
for f in api_keys.json llm_config.json; do
    if [[ -f "$PROJECT_DIR/$f" ]]; then
        chown "$USER":"$USER" "$PROJECT_DIR/$f"
        chmod 600 "$PROJECT_DIR/$f"
    fi
done

# ── SECRET_KEY + CORS_ALLOW_ORIGINS (.env) ────────────────────────────────────
if [[ ! -f "$PROJECT_DIR/.env" ]]; then
    echo "==> Generating .env..."
    secret=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    printf "SECRET_KEY=%s\nCORS_ALLOW_ORIGINS=https://%s\n" "$secret" "$DOMAIN" > "$PROJECT_DIR/.env"
    chown "$USER":"$USER" "$PROJECT_DIR/.env"
    chmod 600 "$PROJECT_DIR/.env"
    echo
    echo "  *** IMPORTANT: A new SECRET_KEY has been written to $PROJECT_DIR/.env ***"
    echo "  *** Back this file up now. Losing it will invalidate all user sessions. ***"
    echo
else
    echo "==> .env already exists — preserving existing SECRET_KEY."
    # Add CORS_ALLOW_ORIGINS if not already present
    if ! grep -q "^CORS_ALLOW_ORIGINS=" "$PROJECT_DIR/.env"; then
        echo "CORS_ALLOW_ORIGINS=https://$DOMAIN" >> "$PROJECT_DIR/.env"
        echo "    Added CORS_ALLOW_ORIGINS=https://$DOMAIN to .env"
    fi
fi

# ── systemd unit files ─────────────────────────────────────────────────────────
echo "==> Installing systemd unit files..."
for unit_file in \
    arxiv-recommender.service \
    arxiv-embed-daemon.service \
    arxiv-meta-daemon.service \
    arxiv-daily-ingest.service \
    arxiv-daily-ingest.timer \
    arxiv-cleanup-embeddings.service \
    arxiv-cleanup-embeddings.timer
do
    sed \
        -e "s|<user>|$USER|g" \
        -e "s|<project_dir>|$PROJECT_DIR|g" \
        "$PROJECT_DIR/deploy/$unit_file" \
        > "/etc/systemd/system/$unit_file"
    chmod 644 "/etc/systemd/system/$unit_file"
done

# ── Caddyfile ──────────────────────────────────────────────────────────────────
echo "==> Installing Caddyfile..."
mkdir -p /etc/caddy
sed \
    -e "s|<DOMAIN>|$DOMAIN|g" \
    -e "s|<project_dir>|$PROJECT_DIR|g" \
    "$PROJECT_DIR/Caddyfile" \
    > /etc/caddy/Caddyfile
chown root:root /etc/caddy/Caddyfile
chmod 644 /etc/caddy/Caddyfile

echo "==> Validating Caddyfile..."
if ! caddy validate --config /etc/caddy/Caddyfile; then
    echo "ERROR: Caddy config validation failed. Check /etc/caddy/Caddyfile." >&2
    exit 1
fi

# ── Enable and start services ──────────────────────────────────────────────────
echo "==> Enabling and starting services..."
systemctl daemon-reload
systemctl enable \
    arxiv-recommender.service \
    arxiv-embed-daemon.service \
    arxiv-meta-daemon.service \
    arxiv-daily-ingest.timer \
    arxiv-cleanup-embeddings.timer
systemctl restart \
    arxiv-recommender.service \
    arxiv-embed-daemon.service \
    arxiv-meta-daemon.service
systemctl restart caddy

# ── Post-install checklist ─────────────────────────────────────────────────────
cat <<EOF

════════════════════════════════════════════════════════════
  Deployment complete — post-install checklist
════════════════════════════════════════════════════════════

1. Verify config files exist in $PROJECT_DIR:

   api_keys.json   — required keys: "summary_api_key", "embed_api_key", "semantic_scholar"
                     if email verification enabled: also "resend_api_key"
                     example: {"summary_api_key": "sk-...", "embed_api_key": "sk-...",
                               "semantic_scholar": "...", "resend_api_key": "re_..."}

   llm_config.json — required keys: "embedding_model", "summary_model", "base_url"
                     example: {"embedding_model": "...", "summary_model": "...", "base_url": "..."}

   email_config.json — optional; create to enable email verification:
                     {"verification": {
                         "enabled": true,
                         "email_from": "noreply@mail.$DOMAIN",
                         "app_base_url": "https://$DOMAIN"
                       }
                     }

   If any config file is missing, create it and restart:
     systemctl restart arxiv-recommender arxiv-embed-daemon arxiv-meta-daemon

2. Back up $PROJECT_DIR/.env
   Losing SECRET_KEY will invalidate all user sessions.

3. Run the first ingest manually (optional, won't wait until tonight):
     sudo -u $USER $PROJECT_DIR/.venv/bin/python3 $PROJECT_DIR/scripts/cron_daily.py

4. Register via the web UI at https://$DOMAIN, then promote to admin:
     sudo -u $USER $PROJECT_DIR/.venv/bin/python3 $PROJECT_DIR/scripts/activate_user.py <email> --make-admin

5. Check service status:
     systemctl status arxiv-recommender arxiv-embed-daemon arxiv-meta-daemon
     systemctl list-timers arxiv-daily-ingest.timer

6. Check logs:
     journalctl -u arxiv-recommender -n 50
     journalctl -u arxiv-embed-daemon -n 50

════════════════════════════════════════════════════════════
EOF
