# ── Stage 1: Build the React frontend ─────────────────────────────────────────
FROM node:22-slim AS frontend-builder

WORKDIR /build

# Install dependencies first (better layer caching)
COPY web/frontend/package.json web/frontend/package-lock.json ./
RUN npm ci --silent

# Build
COPY web/frontend/ ./
RUN npm run build


# ── Stage 2: Python runtime ────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY arxiv_lib/ ./arxiv_lib/
COPY web/ ./web/
COPY daemons/ ./daemons/
COPY scripts/ ./scripts/

# Copy static configuration files (not secrets — those are bind-mounted at runtime)
COPY system_prompt_summary.txt \
     search_embedding_prompt.txt \
     recommendation_embedding_prompt.txt \
     llm_config.json \
     ./

# Copy built frontend from builder stage
COPY --from=frontend-builder /build/dist ./web/frontend/dist/

# Default command: run the API server.
# Override with 'command:' in docker-compose.yml for daemon containers.
# --proxy-headers        trust X-Forwarded-For from Caddy so slowapi rate-limits
#                        by real client IP rather than Caddy's internal bridge IP.
# --forwarded-allow-ips  accept proxy headers from any source; safe because port
#                        8000 is bound only within the Docker internal network and
#                        is never exposed to the public internet.
CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--proxy-headers", "--forwarded-allow-ips", "*"]
