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

# Allow DATA_DIR to be overridden via environment variable (e.g. for Docker,
# where persistent data lives on a mounted volume rather than the project root).
_DATA_DIR            = os.environ.get("DATA_DIR", BASE_DIR)

EMBEDDING_CACHE_DB   = os.path.join(_DATA_DIR, "embeddings_cache.db")
APP_DB_PATH          = os.path.join(_DATA_DIR, "app.db")
SOURCE_CACHE_DIR     = os.path.join(_DATA_DIR, "arxiv_source_cache")
METADATA_CACHE_DIR   = os.path.join(_DATA_DIR, "arxiv_metadata_cache")
SUMMARY_CACHE_DIR    = os.path.join(_DATA_DIR, "arxiv_summary_cache")

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
# LLM summarization prompt
# ---------------------------------------------------------------------------
with open(os.path.join(BASE_DIR, "system_prompt_summary.txt")) as f:
    SUMMARIZE_SYSTEM_PROMPT = f.read().strip()

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
# Qwen3-Embedding-8B produces 4096-dim Matryoshka vectors.
# Embeddings are stored at EMBEDDING_STORAGE_DIM dimensions.
# SEARCH_EMBEDDING_DIM: dimensions used at search time (higher = more precise).
# RECOMMENDATION_EMBEDDING_DIM: dimensions used at recommendation scoring time.
EMBEDDING_STORAGE_DIM        = 512
SEARCH_EMBEDDING_DIM         = 512
RECOMMENDATION_EMBEDDING_DIM = 128
# Dimensionality to which query and paper-search vectors are truncated before
# being passed to ScoringModel.  Search embeddings are more informative than
# recommendation embeddings for query-vs-paper comparisons, but higher dimensions
# increase RAM and training time.  Must be ≤ EMBEDDING_STORAGE_DIM.
QUERY_VECTOR_DIM             = 512

# Search embedding prompt template (used for semantic search queries).
# Supports {title}, {summary}, {abstract}, and {authors} placeholders.
# Missing fields should be passed as "Unavailable".
with open(os.path.join(BASE_DIR, "search_embedding_prompt.txt")) as f:
    SEARCH_EMBEDDING_PROMPT = f.read().strip()

# Recommendation embedding prompt template (used for scoring/recommendations).
with open(os.path.join(BASE_DIR, "recommendation_embedding_prompt.txt")) as f:
    RECOMMENDATION_EMBEDDING_PROMPT = f.read().strip()

# ---------------------------------------------------------------------------
# Scoring / recommendation
# ---------------------------------------------------------------------------
# Bump SCORING_VERSION whenever the scoring algorithm changes in a way that
# would make previously cached user models produce wrong results.  This
# causes all cached models in user_models to be re-trained on next use.
SCORING_VERSION = "v4"

# RBF kernel: gammas are spaced logarithmically.
RBF_GAMMAS = np.logspace(-4, 4, num=4, base=2)

# Number of SVD components extracted from the positive-vector matrix.
RBF_PCA_COMPONENTS = 8

# Number of dimensions to use when calculating fraction of vector length
# that resides in the high-variance subspace defined by the positive examples.
SUBSPACE_FRACTION_DIMS = [8, 16, 32]

# Background negative papers used for training all users' scoring models.
# A random sample of up to BACKGROUND_NEGATIVE_COUNT papers (excluding any
# the user has explicitly liked or disliked) is drawn at each retrain so the
# negative set evolves with the corpus.
# BACKGROUND_NEGATIVE_MIN_COUNT: reject training if fewer than this are available.
BACKGROUND_NEGATIVE_COUNT = 512
BACKGROUND_NEGATIVE_MIN_COUNT = 64

# Cap on how many of the user's most-recently-added liked/disliked
# papers + query terms are used for training. Prevents training time
# from growing unboundedly as the library grows, and keeps the model
# focused on recent interests.
MAX_LIKED_PAPERS_TO_USE    = 256
MAX_DISLIKED_PAPERS_TO_USE = 128
MAX_QUERY_TERMS_TO_USE     = 128

# Models older than this are retrained unconditionally (even if the hash
# matches) so that drifting background negatives are refreshed.
MAX_MODEL_AGE_DAYS = 90

# Minimum number of liked papers required before attempting to train a model.
RECOMMEND_MIN_LIKED = 4

# Maximum number of papers returned by the onboarding browse (shown to new users
# who have not yet liked enough papers to generate scored recommendations).
# Returned in random order so the user sees a varied sample each visit.
ONBOARDING_BROWSE_LIMIT = 256

# Maximum number of recommendations stored (and returned) per user per time window.
# Applied at write time in refresh_recommendations(); only the top-scoring papers
# are written to the recommendations table.
MAX_RECOMMENDATIONS_PER_WINDOW = 256

# Maximum number of papers to be scored at once, in order to limit RAM usage. If more
# than this number of papers need to be scored, they are processed in batches by the
# scoring model, and the results are concatenated.
RECOMMENDATION_SCORING_BATCH_SIZE = 512

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
META_INGEST_POLL_INTERVAL  = 5.0 # seconds
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
# SEARCH_EMBEDDING_DIM and RECOMMENDATION_EMBEDDING_DIM are algorithm parameters and stay here.

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
# Email verification is enabled when email_config.json exists in the project
# root and contains a non-empty "verification.email_from" value.
# Requires resend_api_key in api_keys.json.
#
# email_config.json example:
#   {
#     "verification": {
#       "email_from":   "noreply@mail.yourdomain.com",
#       "app_base_url": "https://yourdomain.com"
#     }
#   }
#
# Toggling (adding/removing the file or emptying email_from) is safe: existing
# rows are unaffected because the login check uses email_verify_token IS NOT
# NULL (set only when verification was requested) rather than this flag.
EMAIL_CONFIG_FILE = os.path.join(BASE_DIR, "email_config.json")
_email_config: dict = _load_json_file(EMAIL_CONFIG_FILE, "email_config.json").get("verification", {})

EMAIL_VERIFICATION_ENABLED: bool = _email_config.get("enabled", False)
VERIFICATION_EMAIL_FROM: str  = _email_config.get("email_from", "").strip()
APP_BASE_URL:            str  = _email_config.get("app_base_url", "").rstrip("/")
