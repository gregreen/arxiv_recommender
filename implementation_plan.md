# arxiv Recommender — Web Service Implementation Plan

**Last updated:** 2026-04-02  
**Status:** Phase 1 not yet started. `arxiv_embedding.py` is the current monolithic entry point.

---

## Background

`arxiv_embedding.py` is a working research script (~1500 lines) that:
- Fetches arXiv paper metadata (title, authors, abstract) via Semantic Scholar and the arXiv Atom API
- Downloads and processes LaTeX source
- Generates LLM-based structured paper summaries via HuggingFace InferenceClient (Qwen3-Next-80B-A3B-Thinking via Novita)
- Embeds papers using Qwen3-Embedding-8B (4096-dim Matryoshka vectors, truncated to 64 dims for scoring)
- Scores/ranks papers for a single hardcoded user using RBF+SVD features + logistic regression
- Stores embeddings in `embeddings_cache.db` (SQLite, WAL mode, migrated from `.npz`)
- Stores metadata in `arxiv_metadata_cache/{month}.json` (monthly JSON bundles)
- Stores LLM summaries in `arxiv_summary_cache/{arxiv_id}.txt` (one file per paper)
- Stores raw LaTeX in `arxiv_source_cache/{arxiv_id}.tex` (one file per paper; expendable)

The goal is to turn this into a multi-user web service.

---

## Architecture Overview

```
  Browser
    │  HTTP/REST + JWT (HttpOnly cookie)
    ▼
  FastAPI web server  (web/app.py)
    ├── Auth endpoints
    ├── Paper search / browse endpoints
    ├── User liked-papers CRUD
    └── Recommendations endpoint (reads from cache; queues if stale)
         │
         ├── app.db  (SQLite WAL — all user/app state)
         │    ├── users
         │    ├── user_categories
         │    ├── user_papers
         │    ├── papers
         │    ├── user_models
         │    ├── recommendations
         │    └── task_queue
         │
         └── embeddings_cache.db  (existing SQLite — read by web server, written by ingest daemon)
              arxiv_summary_cache/   (read by web server, written by ingest daemon)
              arxiv_metadata_cache/  (read by web server, written by ingest daemon)
              arxiv_source_cache/    (written and read by ingest daemon; expendable)

  Ingest Daemon  (daemons/ingest_daemon.py)
    Polls task_queue WHERE type='embed'
    Per paper: get_arxiv_metadata → summarize_arxiv_paper → gen_arxiv_embedding
    On completion: inserts row into papers table; enqueues 'recommend' tasks for relevant users

  Recommend Daemon  (daemons/recommend_daemon.py)
    Polls task_queue WHERE type='recommend'
    Per user: loads liked+disliked paper vectors, scores all candidate papers,
              persists ranked results and fitted model to app.db

  Daily cron  (scripts/cron_daily.py)
    Runs fetch_latest_mailing_ids() for all subscribed categories
    Enqueues 'embed' tasks for any IDs not already in the papers table
```

---

## Tech Stack

| Layer | Choice | Notes |
|---|---|---|
| Web framework | FastAPI | Python-native, async, easy to test |
| App DB | SQLite (WAL) | Upgrade path to PostgreSQL is mechanical |
| Embedding DB | `embeddings_cache.db` (existing) | Single-writer (ingest daemon), multi-reader |
| Frontend | React + TypeScript (Vite) | Two-pane layout |
| Auth | JWT in HttpOnly cookies | Stateless; `bcrypt` for password hashing |
| Task queue | Table in `app.db` | Avoids Redis/Celery dependency |

---

## App Database Schema (`app.db`)

```sql
-- Users
CREATE TABLE users (
    id          INTEGER PRIMARY KEY,
    email       TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Per-user arXiv category subscriptions (used by daily cron to decide what to ingest)
CREATE TABLE user_categories (
    user_id     INTEGER NOT NULL REFERENCES users(id),
    category    TEXT NOT NULL,   -- e.g. "astro-ph", "cs.LG"
    PRIMARY KEY (user_id, category)
);

-- Per-user paper feedback
-- liked: 1=liked (positive), 0=neutral/removed, -1=disliked (negative / "don't show me this")
CREATE TABLE user_papers (
    user_id     INTEGER NOT NULL REFERENCES users(id),
    arxiv_id    TEXT NOT NULL,
    liked       INTEGER NOT NULL DEFAULT 1,  -- 1 / 0 / -1
    added_at    TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, arxiv_id)
);

-- Papers known to the system (added by ingest daemon after embedding)
-- categories: JSON array of strings, e.g. ["astro-ph.GA", "astro-ph.SR"]
-- authors: JSON array of strings
CREATE TABLE papers (
    arxiv_id        TEXT PRIMARY KEY,
    title           TEXT,
    abstract        TEXT,
    authors         TEXT,   -- JSON
    published_date  TEXT,   -- YYYY-MM-DD from arXiv (may be NULL for older backfilled papers)
    categories      TEXT,   -- JSON
    embedded_at     TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX papers_embedded_at ON papers(embedded_at);

-- Per-user fitted recommendation model
-- model_hash: sha256(sorted liked_arxiv_ids + str(EMBEDDING_DIM) + SCORING_VERSION)[:16]
--   Used to detect when the model is stale relative to the user's current liked set.
-- n_liked / n_disliked: stored for informational purposes; NOT used for staleness checks
--   (use model_hash for that).
-- model_blob: pickle of dict:
--   {
--     "scaler": StandardScaler,
--     "pca_projection": np.ndarray (D, n_components),      -- from positive vectors SVD
--     "pca_residual":   np.ndarray (D, D-n_components),
--     "score_scaler":   StandardScaler,                    -- fitted on RBF features
--     "logistic":       LogisticRegression,
--     "gammas":         np.ndarray,
--     "embedding_dim":  int,
--     "scoring_version": str,
--   }
CREATE TABLE user_models (
    user_id         INTEGER PRIMARY KEY REFERENCES users(id),
    model_blob      BLOB NOT NULL,
    model_hash      TEXT NOT NULL,
    n_liked         INTEGER NOT NULL,
    n_disliked      INTEGER NOT NULL,
    trained_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Recommendation cache
-- time_window: 'day' | 'week' | 'month' | 'year'
-- model_hash: hash of the model used to generate these recommendations
--   (allows checking whether recs are stale when user's liked set changes)
-- score: log-probability from logistic regression (higher = more relevant)
-- rank: 1-based rank within (user_id, time_window)
-- Designed for future culling: keep top N rows per (user_id, time_window)
CREATE TABLE recommendations (
    user_id         INTEGER NOT NULL REFERENCES users(id),
    arxiv_id        TEXT NOT NULL,
    time_window     TEXT NOT NULL,
    score           REAL NOT NULL,
    rank            INTEGER NOT NULL,
    model_hash      TEXT NOT NULL,
    generated_at    TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, arxiv_id, time_window)
);
CREATE INDEX recommendations_user_window ON recommendations(user_id, time_window, rank);

-- Task queue (shared by both daemons)
-- type: 'embed' | 'recommend'
-- payload: JSON, e.g. {"arxiv_id": "2309.06676"} or {"user_id": 3, "time_window": "all"}
-- status: 'pending' | 'running' | 'done' | 'failed'
-- attempts: retry counter (max 3)
CREATE TABLE task_queue (
    id          INTEGER PRIMARY KEY,
    type        TEXT NOT NULL,
    payload     TEXT NOT NULL,  -- JSON
    status      TEXT NOT NULL DEFAULT 'pending',
    attempts    INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    started_at  TEXT,
    completed_at TEXT,
    error       TEXT
);
CREATE INDEX task_queue_pending ON task_queue(type, status, created_at)
    WHERE status = 'pending';
```

### Staleness / model invalidation logic

A recommendation cache entry for a user is considered stale if either:
1. `model_hash` in `recommendations` does not match the current `sha256(sorted(liked_ids) + str(EMBEDDING_DIM) + SCORING_VERSION)[:16]`, **or**
2. Any paper has been added to the `papers` table with `embedded_at > recommendations.generated_at`

The recommend daemon re-trains the model only if condition (1) is true OR
if `user_models.trained_at` is more than `RECOMMEND_MIN_RETRAIN_INTERVAL` (default: 60 minutes) ago.

Future culling: delete rows from `recommendations` WHERE `rank > N` AND `time_window = 'week'`, etc. The `rank` column makes this easy. Do not implement now.

---

## Scoring Algorithm (RBF+SVD logistic regression)

The core algorithm is already implemented in `rbf_svd_example()` in `arxiv_embedding.py`. It will be extracted into `arxiv_lib/scoring.py` with the following interface:

```python
def score_papers_for_user(
    liked_ids: list[str],           # user's explicitly liked papers (positive set)
    disliked_ids: list[str],        # user's explicitly disliked papers (always negative)
    candidate_ids: list[str],       # papers to rank (e.g. papers from last N days)
    embedding_cache: dict[str, np.ndarray],
    embedding_dim: int = EMBEDDING_DIM,    # config constant, default 64
    n_pca_components: int = 4,
    gammas: np.ndarray = None,      # defaults to np.logspace(-6, 6, num=6, base=2)
) -> dict[str, float]:              # {arxiv_id: log-probability score}
```

**Algorithm summary:**
1. Load truncated embeddings (`[:embedding_dim]`) for liked + disliked + candidate papers.
2. `StandardScaler` fit on liked+disliked+candidate, transform all.
3. SVD on liked-paper vectors to get `P` (top `n_pca_components`) and `P_residual`.
4. For both the full-dim and residual-dim projections, compute RBF kernel features at each gamma.
   Features: `[max_similarity, logsumexp at gamma_0, ..., logsumexp at gamma_N]` per paper.
5. Concatenate RBF features from both projections.
6. `StandardScaler` on feature matrix.
7. Logistic regression trained on liked (positive) vs. disliked+unrelated (negative) papers.
   Disliked papers are always in the negative set regardless of sample balancing.
8. Return `predict_log_proba(candidate_features)[:, 1]` as scores.

**Model caching:** The fitted (scaler, PCA projections, score scaler, logistic model) is pickled and stored in `user_models.model_blob`. The recommend daemon calls `get_or_train_user_model()` which:
- Computes `model_hash` from current liked IDs + config.
- Returns cached model if hash matches AND `trained_at` is within `RECOMMEND_MIN_RETRAIN_INTERVAL`.
- Otherwise retrains and persists.

**Key constants** (`arxiv_lib/config.py`):
```python
EMBEDDING_DIM = 64            # Matryoshka truncation. Max useful: 128.
SCORING_VERSION = "v1"        # Bump when algorithm changes to invalidate all cached models.
RECOMMEND_MIN_RETRAIN_INTERVAL = 3600   # seconds; don't retrain more often than this
```

---

## Project Structure

```
arxiv_recommender/
  arxiv_embedding.py          ← kept as-is; thin shim once lib exists (for manual use)
  arxiv_to_prompt.py          ← existing helper (LaTeX processing)
  implementation_plan.md      ← this file
  tokens.json                 ← API tokens (not committed)
  embeddings_cache.db         ← existing embedding store
  arxiv_metadata_cache/       ← existing monthly JSON bundles
  arxiv_summary_cache/        ← existing per-paper LLM summary .txt files
  arxiv_source_cache/         ← existing per-paper LaTeX .tex files (expendable)

  arxiv_lib/
    __init__.py
    config.py                 ← all constants (paths, EMBEDDING_DIM, SCORING_VERSION, etc.)
    ingest.py                 ← all fetch/summarize/embed logic from arxiv_embedding.py
    scoring.py                ← score_papers_for_user(), train_logistic_model(), rbf helpers
    appdb.py                  ← app.db schema creation, connection helpers, task queue helpers

  daemons/
    __init__.py
    ingest_daemon.py          ← polls task_queue for 'embed' tasks
    recommend_daemon.py       ← polls task_queue for 'recommend' tasks

  web/
    app.py                    ← FastAPI application factory
    auth.py                   ← JWT creation/verification, bcrypt helpers
    dependencies.py           ← FastAPI dependency injection (get_db, get_current_user)
    routers/
      auth.py                 ← POST /auth/register, POST /auth/login, POST /auth/logout
      papers.py               ← GET /papers/{arxiv_id}, GET /papers/search
      users.py                ← GET/POST/DELETE/PATCH /users/me/papers, POST /users/me/papers/import/ads
      recommendations.py      ← GET /recommendations?window=day|week|month|year
    frontend/                 ← React + TypeScript (Vite)
      src/
        App.tsx
        components/
          PaperList.tsx        ← left pane: list of recommendations
          PaperDetail.tsx      ← right pane: summary display
          RelevanceButtons.tsx ← "Relevant" / "Not relevant"
          OnboardingFlow.tsx   ← category selection + paper seeding
        api/                  ← typed fetch wrappers

  scripts/
    migrate_legacy.py         ← one-time: populate papers table from existing caches
    cron_daily.py             ← called by cron/systemd; enqueues embed tasks for today's mailing
```

---

## Phase-by-Phase Roadmap

### Phase 1 — Project structure + library refactor
**Goal:** Extract `arxiv_embedding.py` into `arxiv_lib/` without changing behaviour.

1. Create `arxiv_lib/config.py` with all constants. Replace hardcoded `[:64]`, path strings, and category lists with config references throughout.
2. Create `arxiv_lib/ingest.py` by copying all fetch/summarize/embed functions from `arxiv_embedding.py`:
   - `load_from_arxiv_metadata_cache`, `write_to_arxiv_metadata_cache`, `get_arxiv_metadata`
   - `fetch_arxiv_metadata`, `fetch_arxiv_metadata_s2`, `fetch_arxiv_metadata_html`
   - `get_arxiv_source`, `compress_latex_whitespace`
   - `summarize_arxiv_paper`
   - `gen_arxiv_embedding`, `fetch_arxiv_embedding`, `embed_arxiv_ids`, `embed_latest_mailing`
   - `_init_embedding_db`, `load_embedding_cache`, `save_embedding_cache`
   - `load_tokens`
   - Remove the hardcoded category validation in `raise_on_arxiv_category`; replace with config-driven list.
3. Create `arxiv_lib/scoring.py` with:
   - `rbf_scoring`, `train_logistic_model`, `calculate_projection_matrices`, `project_to_subspace`
   - New: `score_papers_for_user()` — the clean version of `rbf_svd_example()` (see algorithm above)
   - New: `get_or_train_user_model()` — handles model caching
4. Update `arxiv_embedding.py` to import from `arxiv_lib` (thin shim for backward compat).

**Verification:** `python arxiv_embedding.py` still runs and produces same output as before.

---

### Phase 2 — App database
**Goal:** Create `app.db` with full schema; run legacy migration.

1. Create `arxiv_lib/appdb.py` with:
   - `init_app_db(path)` — creates all tables (see schema above)
   - `get_connection(path)` — returns WAL-enabled connection
   - `enqueue_task(con, type, payload)` — insert into task_queue
   - `claim_next_task(con, type)` — atomic SELECT + UPDATE to set status='running'
   - `complete_task(con, id)` / `fail_task(con, id, error)` — update task status
2. Create `scripts/migrate_legacy.py`:
   - Read all `arxiv_id`s from `embeddings_cache.db`.
   - For each, load metadata from `arxiv_metadata_cache/`.
   - Insert row into `papers` table with `embedded_at = datetime('now')` (real timestamp not recoverable).
3. Run migration; verify 589 rows in `papers` table.

---

### Phase 3 — Ingest daemon
**Goal:** Working daemon that processes 'embed' tasks end-to-end.

`daemons/ingest_daemon.py` main loop:
1. `claim_next_task(con, 'embed')` — returns task or None.
2. If None: sleep (poll interval, e.g. 10 seconds) and retry.
3. Extract `arxiv_id` from payload.
4. Check if `arxiv_id` already in `papers` table; if yes, mark done and continue.
5. Run `ingest.summarize_arxiv_paper()` then `ingest.gen_arxiv_embedding()`.
6. Insert row into `papers` table.
7. Enqueue 'recommend' tasks for all users whose `user_categories` overlaps with this paper's categories.
8. Mark task done.
9. On any exception: mark task failed (increment attempts); if attempts < 3, reset to pending.

Startup: check for any `arxiv_id`s in `embeddings_cache.db` not yet in `papers` — backfill `papers` table (not re-embed; just copy metadata).

---

### Phase 4 — Recommendation engine
**Goal:** Clean, tested `score_papers_for_user()` function.

Extract from `rbf_svd_example()` into `arxiv_lib/scoring.py`:
- Takes: `liked_ids`, `disliked_ids`, `candidate_ids`, `embedding_cache`, and config kwargs.
- Disliked papers are always included in the negative set (not sampled away).
- Returns `{arxiv_id: log_prob_score}` for all candidates.
- Also: `get_or_train_user_model(user_id, liked_ids, disliked_ids, con)` — loads from DB if hash matches and not stale; otherwise trains and persists.

**Verification:** Running `score_papers_for_user(my_papers, [], all_other_ids, cache)` should reproduce the same top-10 results as `rbf_svd_example()`.

---

### Phase 5 — Recommend daemon
**Goal:** Working daemon that generates and caches recommendation lists.

`daemons/recommend_daemon.py` main loop:
1. `claim_next_task(con, 'recommend')` — returns task or None.
2. Extract `user_id` from payload.
3. Load user's liked and disliked papers from `user_papers` table.
4. Compute `model_hash`.
5. Check staleness: if existing recommendations are fresh (hash matches AND no new papers since `generated_at`), skip.
6. Load candidate paper IDs from `papers` table (all embedded papers; time-window filtering done at query time by the web server).
7. Load all vectors from `embeddings_cache.db`.
8. Call `score_papers_for_user()`.
9. Write results to `recommendations` table (upsert), with rank computed per `time_window`.
   - Write four sets: 'day', 'week', 'month', 'year' (filter `papers.embedded_at` accordingly).
10. Mark task done.

---

### Phase 6 — FastAPI backend
**Goal:** Working API that the frontend can call.

Key endpoints (all require JWT auth except register/login):

| Method | Path | Description |
|---|---|---|
| POST | `/auth/register` | Create account; return JWT cookie |
| POST | `/auth/login` | Verify credentials; return JWT cookie |
| POST | `/auth/logout` | Clear cookie |
| GET | `/papers/{arxiv_id}` | Metadata + LLM summary for detail pane |
| GET | `/papers/search?q=` | Wraps arXiv Atom API; returns title/authors/abstract |
| GET | `/users/me/papers` | List user's liked/disliked papers with metadata |
| POST | `/users/me/papers` | Add paper to liked set; enqueue embed+recommend tasks |
| PATCH | `/users/me/papers/{arxiv_id}` | Update liked flag (+1/0/-1) |
| DELETE | `/users/me/papers/{arxiv_id}` | Remove from set |
| POST | `/users/me/papers/import/ads` | Parse NASA ADS export for `arXiv:XXXX.XXXXX` lines; bulk-add |
| GET | `/recommendations?window=day` | Return ranked recommendations from cache; `{"status": "generating"}` if stale |
| GET | `/users/me/categories` | Get user's subscribed categories |
| PUT | `/users/me/categories` | Update subscribed categories |

NASA ADS bulk import format: ADS custom export with format `%X` produces one `arXiv:XXXX.XXXXX` per line. Parse with `re.findall(r'arXiv:(\d{4}\.\d{4,5})', text)`.

Security notes:
- Passwords: `bcrypt` via `passlib`.
- JWT: `python-jose` or `PyJWT`; short expiry (1 hour) with refresh token in separate HttpOnly cookie.
- All SQL via parameterised queries (already the pattern in existing code).
- Rate-limit the paper import endpoint (max 500 IDs per request).

---

### Phase 7 — React frontend
**Goal:** Working two-pane UI.

Stack: Vite + React + TypeScript. No heavy component library — plain CSS or Tailwind.

**Onboarding flow:**
1. Register / login screen.
2. Category selector (checkboxes for common arXiv categories).
3. Paper seeding via: (a) paste arXiv IDs or NASA ADS export text; (b) search form.
4. On submit: show "We're preparing your first recommendations" loading state.

**Main view:**
```
┌──────────────────────┬──────────────────────────────────────────────┐
│ Recommendations      │ Title: ...                                    │
│ [Day|Week|Month|Yr]  │ Authors: ...                                  │
│ ─────────────────    │ Abstract: ...                                 │
│ > Paper A  ████ 0.94 │                                               │
│   Paper B  ███  0.91 │ Summary:                                      │
│   Paper C  ██   0.88 │   Keywords: ...                               │
│   ...                │   Scientific Questions: ...                   │
│                      │   Data: ...                                   │
│                      │   Methods: ...                                │
│                      │   Results: ...                                │
│                      │   Key takeaway: ...                           │
│                      │                                               │
│                      │  [✓ Relevant]  [✗ Not relevant]               │
│                      │  arxiv.org/abs/XXXX.XXXXX ↗                  │
└──────────────────────┴──────────────────────────────────────────────┘
```

Clicking Relevant/Not relevant calls `PATCH /users/me/papers/{arxiv_id}` and triggers a background recommend refresh (the next poll will show updated results).

Stale recommendations: show spinner/banner while `status == "generating"`. Poll `/recommendations` every 30 seconds until fresh results appear.

**Liked papers manager:** separate `/library` route; table of saved papers with remove buttons.

---

### Phase 8 — Ops / deployment
**Goal:** Reliable unattended operation.

1. **Daily cron** (`scripts/cron_daily.py`):
   - For each unique category in `user_categories`, call `fetch_latest_mailing_ids()`.
   - For each ID not already in `papers` table, insert an 'embed' task into `task_queue`.
   - Run via `cron` (e.g. `0 14 * * 1-5`) or `systemd.timer`.

2. **systemd service files** for ingest daemon and recommend daemon (auto-restart on failure).

3. **Environment config** (`.env` file, never committed):
   - `HF_TOKEN`, `S2_TOKEN` — API credentials
   - `SECRET_KEY` — JWT signing key
   - `APP_DB_PATH`, `EMBEDDING_DB_PATH` — override defaults from config.py
   - `SUMMARY_CACHE_DIR`, `METADATA_CACHE_DIR`, `SOURCE_CACHE_DIR`

4. **nginx** reverse proxy:
   - Static React build served directly from nginx.
   - `/api/*` proxied to FastAPI (uvicorn on localhost).

5. **Backups**: nightly copy of `app.db` and `embeddings_cache.db`; `arxiv_summary_cache/` is the most valuable (expensive to regenerate) and should be backed up too.

---

## Key Design Decisions (for future reference)

**Single app DB, not per-daemon DBs.** The task queue, user data, and recommendations are all in `app.db`. Only the embedding store is separate (`embeddings_cache.db`) because it is write-heavy from the ingest daemon.

**Disliked papers are always in the negative set.** Papers marked with `liked = -1` are always included as negative examples when training the logistic model, regardless of sample weighting. This ensures that a paper the user explicitly rejected cannot end up ranked highly.

**Model staleness uses a hash, not a timestamp alone.** `model_hash = sha256(sorted(liked_arxiv_ids) + str(EMBEDDING_DIM) + SCORING_VERSION)[:16]`. The `SCORING_VERSION` string lets you invalidate all cached models at once when the algorithm changes.

**Retraining throttle.** Even if the hash changes, the recommend daemon will not retrain more often than `RECOMMEND_MIN_RETRAIN_INTERVAL` (default: 60 minutes). This prevents a flurry of retrains when a user adds many papers quickly.

**Embedding dimension is a config constant.** `EMBEDDING_DIM = 64` in `config.py`. All `[:64]` slices in the original code should be replaced with `[:EMBEDDING_DIM]`. The model stores `embedding_dim` in its blob so stale models from a different dim setting are automatically invalidated (hash will differ because SCORING_VERSION or the dim is encoded).

**LLM prompt language.** The current system prompt says "You are an expert astrophysics researcher." This should be made configurable or replaced with a field-agnostic version when the user base expands beyond astronomers. For now it is fine, as early users will almost all be astronomers.

**Matryoshka truncation.** Qwen3-Embedding-8B produces 4096-dim vectors. Truncating to 64 improves scoring speed dramatically and has been verified to work well empirically. 128 dims is the current upper bound planned. Do not store truncated vectors in `embeddings_cache.db` — always store full 4096-dim and truncate at inference time.

**SQLite vs PostgreSQL.** SQLite (WAL mode) is sufficient for ~1000 users. The upgrade path to PostgreSQL requires: changing the connection strings, replacing SQLite-specific PRAGMA statements, and potentially replacing `AUTOINCREMENT`/`INTEGER PRIMARY KEY` idioms. Design the schema to be compatible with both (avoid SQLite-specific features beyond WAL).

**Recommendation culling (future).** The `rank` column in `recommendations` is there specifically to support future culling queries like `DELETE FROM recommendations WHERE rank > 100 AND time_window = 'week'`. Do not implement yet.

---

## Current Status

- [x] `embeddings_cache.db`: SQLite embedding store, 589 papers, WAL mode, working
- [x] `arxiv_summary_cache/`: ~300+ LLM summary files, working
- [x] `arxiv_metadata_cache/`: monthly JSON bundles, working
- [ ] Phase 1: library refactor — not started
- [ ] Phase 2: app database — not started
- [ ] Phase 3: ingest daemon — not started
- [ ] Phase 4: recommendation engine — not started
- [ ] Phase 5: recommend daemon — not started
- [ ] Phase 6: FastAPI backend — not started
- [ ] Phase 7: React frontend — not started
- [ ] Phase 8: ops/deployment — not started
