# Security Audit — Proposed Changes

Changes are ordered by severity.  Each section states the file(s) affected,
the exact change needed, and why.

---

## Changes to implement

### 1. Hard-fail on empty or short `SECRET_KEY` at startup

**File:** `web/app.py`

**Current code (lines ~29–35):**
```python
if not SECRET_KEY:
    import warnings
    warnings.warn(
        "SECRET_KEY is not set. JWTs will be signed with an empty key — "
        "do not run this in production without setting SECRET_KEY.",
        stacklevel=2,
    )
```

**Change to:**
```python
if not SECRET_KEY or len(SECRET_KEY) < 32:
    raise RuntimeError(
        "SECRET_KEY must be set to a random string of at least 32 characters. "
        "Generate one with: python3 -c \"import secrets; print(secrets.token_hex(32))\""
    )
```

**Why:** A warning is ignored silently; the app starts and issues JWTs signed with
an empty key, which can be trivially forged.  A startup crash is the only safe
behaviour.  The 32-character minimum enforces meaningful entropy.

---

### 2. Rate-limit `/register`, `/verify-email`, and `/resend-verification`

**File:** `web/routers/auth.py`

**Change 1 — `register`:**
```python
# Before:
@router.post("/register", status_code=status.HTTP_202_ACCEPTED)
def register(body: RegisterRequest, db: sqlite3.Connection = Depends(get_db)):

# After:
@router.post("/register", status_code=status.HTTP_202_ACCEPTED)
@limiter.limit("5/hour")
def register(request: Request, body: RegisterRequest, db: sqlite3.Connection = Depends(get_db)):
```

**Change 2 — `verify_email`:**
```python
# Before:
@router.get("/verify-email")
def verify_email(token: str, db: sqlite3.Connection = Depends(get_db)):

# After:
@router.get("/verify-email")
@limiter.limit("5/hour")
def verify_email(request: Request, token: str, db: sqlite3.Connection = Depends(get_db)):
```

**Change 3 — `resend_verification`:**
```python
# Before:
@router.post("/resend-verification", status_code=status.HTTP_200_OK)
def resend_verification(body: ResendVerificationRequest, db: sqlite3.Connection = Depends(get_db)):

# After:
@router.post("/resend-verification", status_code=status.HTTP_200_OK)
@limiter.limit("5/hour")
def resend_verification(request: Request, body: ResendVerificationRequest, db: sqlite3.Connection = Depends(get_db)):
```

`Request` is already imported and `limiter` is already defined; no new imports
needed.

**Why:** These endpoints have no rate limit, enabling account-creation flooding
and email-bombing via the resend endpoint.

---

### 3. Remove localhost CORS origins and read production origin from environment

**File:** `web/app.py`

**Current code (lines ~49–55):**
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Change to:**
```python
_cors_origins_env = os.environ.get("CORS_ALLOW_ORIGINS", "")
_cors_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
if not _cors_origins:
    # Development fallback only — never reached in production if env var is set
    _cors_origins = ["http://localhost:5173", "http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)
```

Add `import os` if not already present (check existing imports).

In production, set `CORS_ALLOW_ORIGINS=https://yourdomain.com` in the
systemd service environment file.

**Why:** `allow_origins=["http://localhost:*"]` with `allow_credentials=True`
means any page running on localhost (e.g. a locally-opened malicious HTML file)
can make credentialed requests to the API.  The wildcard `allow_methods` and
`allow_headers` are broader than necessary.

---

### 4. Return a generic response for duplicate registration (prevent account enumeration)

**File:** `web/routers/auth.py`

**Current code (lines ~65–70):**
```python
existing = db.execute(
    "SELECT id FROM users WHERE email = ?", (body.email,)
).fetchone()
if existing:
    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail="An account with that email already exists.",
    )
```

**Change to:**
```python
existing = db.execute(
    "SELECT id FROM users WHERE email = ?", (body.email,)
).fetchone()
if existing:
    # Return 202 with the same generic message to prevent email enumeration.
    return {"message": "Registration received. Please check your email to verify your account."}
```

**Why:** HTTP 409 with a specific message tells an attacker which email addresses
are registered.  Returning the same 202 response as a successful registration
prevents this.

---

### 5. Don't leak exception details from the search endpoint

**File:** `web/routers/search.py`

**Current code (lines ~37–39):**
```python
except SearchEmbeddingError as exc:
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=f"Embedding service unavailable: {exc}",
    ) from exc
```

**Change to:**
```python
except SearchEmbeddingError as exc:
    log.error("search: embedding service error for user %s: %s", user["id"], exc)
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Embedding service temporarily unavailable.",
    ) from exc
```

Add `log = logging.getLogger(__name__)` near the top of the file if not
already present, and `import logging`.

**Why:** `f"Embedding service unavailable: {exc}"` sends the full exception
string to the client, which may include internal API endpoint URLs, model names,
or error response bodies from the embedding provider.

---

### 6. Add security response headers to Caddyfile

**File:** `Caddyfile`

**Change:** Add a `header` block inside the site block, before the `handle`
directives:

```
<DOMAIN> {
    header {
        X-Frame-Options "SAMEORIGIN"
        X-Content-Type-Options "nosniff"
        Referrer-Policy "strict-origin-when-cross-origin"
        # KaTeX requires unsafe-inline for styles; tighten if switching to
        # external KaTeX CSS.
        Content-Security-Policy "default-src 'self'; style-src 'self' 'unsafe-inline'; font-src 'self' data:; img-src 'self' data:"
    }

    handle /api/* {
        ...
    }
    ...
}
```

**Why:**
- `X-Frame-Options: SAMEORIGIN` — prevents clickjacking (the page cannot be
  embedded in an `<iframe>` on another origin).
- `X-Content-Type-Options: nosniff` — prevents browsers from MIME-sniffing
  response bodies, blocking a class of content-injection attacks.
- `Referrer-Policy` — limits what URL information is sent to third parties.
- `Content-Security-Policy` — restricts which origins can load scripts, styles,
  and other resources, significantly limiting XSS impact.

---

### 7. Sanitize KaTeX HTML output with DOMPurify

**File:** `web/frontend/src/components/MathText.tsx`

**Install:** `npm install dompurify @types/dompurify`

**Current code (lines ~22–27):**
```tsx
function renderMath(src: string, display: boolean): string {
  try {
    return katex.renderToString(src, { displayMode: display, throwOnError: false });
  } catch {
    return src;
  }
}
```

**Change to:**
```tsx
import DOMPurify from "dompurify";

function renderMath(src: string, display: boolean): string {
  try {
    const html = katex.renderToString(src, { displayMode: display, throwOnError: false });
    return DOMPurify.sanitize(html, { USE_PROFILES: { mathMl: true, svg: true } });
  } catch {
    return src;
  }
}
```

**Why:** KaTeX's output is inserted via `dangerouslySetInnerHTML`.  KaTeX itself
does not produce malicious HTML, but sanitizing the output is a cheap defence-in-
depth measure: if a KaTeX vulnerability is ever discovered, or if the summary
data pipeline is ever compromised and injects non-LaTeX content that survives
the `$...$` tokenizer, DOMPurify will strip any executable HTML before it
reaches the DOM.

---

### 8. Add admin action audit logging

**File:** `web/routers/admin.py` (and schema)

**Schema change** — add to `appdb.py` `init_app_db()`:
```sql
CREATE TABLE IF NOT EXISTS admin_audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    admin_id    INTEGER NOT NULL REFERENCES users(id),
    action      TEXT    NOT NULL,
    target_id   INTEGER,
    detail      TEXT,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
```

**Code change** — in the `PATCH /admin/users/{user_id}` handler, after the
`UPDATE`, insert a row:
```python
db.execute(
    "INSERT INTO admin_audit_log (admin_id, action, target_id, detail) VALUES (?, ?, ?, ?)",
    (admin["id"], "patch_user", user_id, json.dumps({"is_active": body.is_active})),
)
```

**Why:** Without an audit log there is no record of who activated or deactivated
accounts or when.  If the admin account were compromised, the actions taken
would be invisible.

---

### 9. Secure SQLite WAL/SHM companion files in deploy script

**File:** `deploy/deploy.sh`

**Current code (sets permissions on `app.db` only):**
```bash
if [[ -f "$PROJECT_DIR/app.db" ]]; then
    chown "$USER":"$USER" "$PROJECT_DIR/app.db"
    chmod 600 "$PROJECT_DIR/app.db"
fi
```

**Change to:**
```bash
if [[ -f "$PROJECT_DIR/app.db" ]]; then
    chown "$USER":"$USER" "$PROJECT_DIR/app.db"
    chmod 600 "$PROJECT_DIR/app.db"
    # Secure WAL-mode companion files if present
    for f in "$PROJECT_DIR/app.db-wal" "$PROJECT_DIR/app.db-shm"; do
        [[ -f "$f" ]] && chown "$USER":"$USER" "$f" && chmod 600 "$f"
    done
fi
```

**Why:** SQLite in WAL mode creates `-wal` and `-shm` files alongside the main
database.  These contain live transaction data and are effectively part of the
database.  If they are created with looser permissions (determined by the
process umask at creation time), the database contents could be readable by
other local users.

---

## Rejected / retracted findings

### R1. API keys exposed in git history

**Retracted.** `*.json` is line 1 of `.gitignore` and `api_keys.json` has never
been committed to the repository.  The concern was based on a hallucinated claim
by the subagent used during the audit.  No action needed.

---

### R2. Verification token retained for 7 days post-verification

**Retracted.** The original concern was that a retained token could be reused to
verify a second account.  This was incorrect: the `verify_email` endpoint looks
up `WHERE email_verify_token = ?`, so a token is cryptographically bound to
exactly one user row.  An attacker who reads the token after verification can
only receive `"already_verified"` — they cannot affect any other account.  The
7-day retention is intentional and correct: without it, after the token is
deleted the server cannot distinguish "token never existed / wrong token" from
"token already used", and the frontend cannot show a helpful message to a user
who clicks the link twice.

---

### R3. Pin all package versions in `requirements.txt`

**Rejected by user.** The application does not have known tight version
dependencies on its packages.  The user will pin specific packages if a version
sensitivity is discovered.  No action needed at this time.
