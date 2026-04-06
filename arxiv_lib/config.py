"""
Central configuration for the arxiv recommender.

All path constants, model names, and algorithm hyperparameters live here.
Import this module instead of hardcoding values anywhere else.
"""

import json
import os
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# BASE_DIR is the project root (one level above this file's package directory).
BASE_DIR             = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EMBEDDING_CACHE_DB   = os.path.join(BASE_DIR, "embeddings_cache.db")
EMBEDDING_CACHE_FILE = os.path.join(BASE_DIR, "embeddings_cache.npz")  # legacy; migration only
APP_DB_PATH          = os.path.join(BASE_DIR, "app.db")
SOURCE_CACHE_DIR     = os.path.join(BASE_DIR, "arxiv_source_cache")
METADATA_CACHE_DIR   = os.path.join(BASE_DIR, "arxiv_metadata_cache")
SUMMARY_CACHE_DIR    = os.path.join(BASE_DIR, "arxiv_summary_cache")

# Tokens JSON file (not committed to source control; kept for reference)
TOKENS_FILE    = os.path.join(BASE_DIR, "tokens.json")
# API keys and LLM configuration (not committed to source control)
API_KEYS_FILE  = os.path.join(BASE_DIR, "api_keys.json")
LLM_CONFIG_FILE = os.path.join(BASE_DIR, "llm_config.json")

# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
USER_AGENT = "arxiv-recommender/1.0"

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
# Qwen3-Embedding-8B produces 4096-dim Matryoshka vectors.
# Always store the FULL 4096-dim vector in the DB; truncate to EMBEDDING_DIM
# only at scoring time.  Increasing EMBEDDING_DIM up to 128 is safe.
EMBEDDING_DIM = 64

# ---------------------------------------------------------------------------
# Scoring / recommendation
# ---------------------------------------------------------------------------
# Bump SCORING_VERSION whenever the scoring algorithm changes in a way that
# would make previously cached user models produce wrong results.  This
# causes all cached models in user_models to be re-trained on next use.
SCORING_VERSION = "v1"

# RBF kernel: gammas are spaced logarithmically.
RBF_GAMMAS = np.logspace(-6, 6, num=6, base=2)

# Number of SVD components extracted from the positive-vector matrix.
RBF_PCA_COMPONENTS = 8

# Background negative papers used for training all users' scoring models.
# A random sample of up to BACKGROUND_NEGATIVE_COUNT papers (excluding any
# the user has explicitly liked or disliked) is drawn at each retrain so the
# negative set evolves with the corpus.
# BACKGROUND_NEGATIVE_MIN_COUNT: reject training if fewer than this are available.
BACKGROUND_NEGATIVE_COUNT = 512
BACKGROUND_NEGATIVE_MIN_COUNT = 64

# Cap on how many of the user's most-recently-added liked/disliked papers are
# used for training.  Prevents training time growing unboundedly as the library
# grows and keeps the model focused on recent interests.
MAX_LIKED_PAPERS_TO_USE    = 256
MAX_DISLIKED_PAPERS_TO_USE = 128

# Models older than this are retrained unconditionally (even if the hash
# matches) so that drifting background negatives are refreshed.
MAX_MODEL_AGE_DAYS = 90

# Minimum number of liked papers required before attempting to train a model.
RECOMMEND_MIN_LIKED = 4

# Maximum number of papers returned by the onboarding browse (shown to new users
# who have not yet liked enough papers to generate scored recommendations).
# Returned in random order so the user sees a varied sample each visit.
ONBOARDING_BROWSE_LIMIT = 256

# Time windows exposed by the recommendations endpoint.
RECOMMEND_TIME_WINDOWS = ("day", "week", "month")

# ---------------------------------------------------------------------------
# Import rate limiting
# ---------------------------------------------------------------------------
# Users with fewer than IMPORT_TIER_THRESHOLD lifetime imports are Tier A
# (higher daily limit); users at or above are Tier B (lower daily limit).
# Limits are enforced over a rolling 24-hour window.
IMPORT_TIER_THRESHOLD      = 32
IMPORT_DAILY_LIMIT_TIER_A  = 16
IMPORT_DAILY_LIMIT_TIER_B  = 4

# How long the ingest daemons sleep between polls when the task queue is empty.
META_INGEST_POLL_INTERVAL = 5   # seconds
EMBED_INGEST_POLL_INTERVAL = 0.1 # seconds

# Maximum number of 'fetch_meta' tasks claimed and processed in one S2 batch call.
INGEST_META_BATCH_SIZE = 256


def _load_json_file(path: str, label: str) -> dict:
    """Load a JSON file and return its contents as a dict.

    Prints a warning and returns {} if the file is missing or malformed.
    """
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {label} not found at {path!r}. API calls will fail.")
        return {}
    except json.JSONDecodeError as e:
        print(f"Warning: {label} at {path!r} is malformed: {e}")
        return {}


# API keys (from api_keys.json) and LLM configuration (from llm_config.json)
# are loaded once at import time so all modules share the same objects.
API_KEYS:   dict[str, str]  = _load_json_file(API_KEYS_FILE,   "api_keys.json")
LLM_CONFIG: dict[str, dict] = _load_json_file(LLM_CONFIG_FILE, "llm_config.json")

# ---------------------------------------------------------------------------
# LLM / embedding model identifiers
# ---------------------------------------------------------------------------
# Model identity and provider settings live in llm_config.json.
# EMBEDDING_DIM is an algorithm parameter and stays here.

# ---------------------------------------------------------------------------
# arXiv categories
# ---------------------------------------------------------------------------
# fetch_latest_mailing_ids() validates its category argument against this set.
# Add new categories here as users from other fields join; no code changes needed.
ARXIV_CATEGORIES = {
    "astro-ph",
    "astro-ph.CO", "astro-ph.EP", "astro-ph.GA",
    "astro-ph.HE", "astro-ph.IM", "astro-ph.SR",
    "cond-mat",
    "cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.NE",
    "gr-qc",
    "hep-ex", "hep-ph", "hep-th",
    "math.PR", "math.ST",
    "physics.data-an",
    "quant-ph",
    "stat.ML",
}

# Categories fetched automatically by scripts/cron_daily.py each day.
# Use top-level categories (e.g. "astro-ph") to ingest all sub-categories via
# the arXiv new-submissions listing, or specific sub-categories for finer control.
DAILY_INGEST_CATEGORIES: list[str] = ["astro-ph"]

# ---------------------------------------------------------------------------
# Web / auth
# ---------------------------------------------------------------------------
# SECRET_KEY must be set in the environment before starting the API server.
# Generate one with: python3 -c "import secrets; print(secrets.token_hex(32))"
SECRET_KEY = os.environ.get("SECRET_KEY", "")

JWT_ALGORITHM   = "HS256"
JWT_EXPIRE_HOURS = 24

# ---------------------------------------------------------------------------
# Email verification
# ---------------------------------------------------------------------------
# Set EMAIL_VERIFICATION_ENABLED=1 in the environment to enable email
# verification on registration.  Requires resend_api_key in api_keys.json,
# EMAIL_FROM (the sending address), and APP_BASE_URL (the public HTTPS root).
#
# When disabled (default), new accounts are inactive until an admin activates
# them manually via scripts/activate_user.py.
#
# Toggling this flag is safe: existing rows are unaffected because the login
# check uses email_verify_token IS NOT NULL (set only when verification was
# requested) rather than the flag value.
EMAIL_VERIFICATION_ENABLED: bool = os.environ.get("EMAIL_VERIFICATION_ENABLED", "0") == "1"
EMAIL_FROM:     str = os.environ.get("EMAIL_FROM", "")
APP_BASE_URL:   str = os.environ.get("APP_BASE_URL", "").rstrip("/")
