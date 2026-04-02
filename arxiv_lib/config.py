"""
Central configuration for the arxiv recommender.

All path constants, model names, and algorithm hyperparameters live here.
Import this module instead of hardcoding values anywhere else.
"""

import os
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# BASE_DIR is the project root (one level above this file's package directory).
BASE_DIR             = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EMBEDDING_CACHE_DB   = os.path.join(BASE_DIR, "embeddings_cache.db")
EMBEDDING_CACHE_FILE = os.path.join(BASE_DIR, "embeddings_cache.npz")  # legacy; migration only
SOURCE_CACHE_DIR     = os.path.join(BASE_DIR, "arxiv_source_cache")
METADATA_CACHE_DIR   = os.path.join(BASE_DIR, "arxiv_metadata_cache")
SUMMARY_CACHE_DIR    = os.path.join(BASE_DIR, "arxiv_summary_cache")

# Tokens JSON file (not committed to source control)
TOKENS_FILE = os.path.join(BASE_DIR, "tokens.json")

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

# Minimum interval (seconds) between model retrains for a given user,
# even if their liked-paper set has changed.
RECOMMEND_MIN_RETRAIN_INTERVAL = 3600

# ---------------------------------------------------------------------------
# LLM / embedding model identifiers
# ---------------------------------------------------------------------------
# Context limit note (Novita + Qwen3-Next-80B-A3B-Thinking):
#   Input tokens > ~110K causes a 400 Bad Request.
#   Empirically: 109,929 tokens OK, 111,476 tokens FAIL.
#   SUMMARY_MAX_TOKENS = 96 * 1024 leaves ~3K headroom for metadata
#   and ~16K for the model's output.
SUMMARY_PROVIDER   = "novita"
SUMMARY_MODEL      = "Qwen/Qwen3-Next-80B-A3B-Thinking"
SUMMARY_MAX_TOKENS = 96 * 1024

EMBEDDING_PROVIDER   = "scaleway"
EMBEDDING_MODEL      = "Qwen/Qwen3-Embedding-8B"
EMBEDDING_MAX_TOKENS = 24 * 1024

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
