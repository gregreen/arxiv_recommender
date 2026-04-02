"""
Scoring pipeline: feature engineering, model training, and per-user recommendation.

The recommended workflow (implemented by score_papers_for_user) is:

  StandardScaler
    → SVD on positive vectors (RBF_PCA_COMPONENTS)
    → RBF features in full-dim space AND residual subspace
    → concatenate
    → StandardScaler
    → LogisticRegression(C=0.2, class_weight='balanced')
    → log-proba[:, 1] as relevance score

Pure numpy / sklearn; no network calls.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from arxiv_lib.config import (
    EMBEDDING_DIM,
    RBF_GAMMAS,
    RBF_PCA_COMPONENTS,
)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def rbf_scoring(
    gamma: np.ndarray,
    positive_vectors: np.ndarray,
    search_vectors: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute RBF kernel aggregate scores for *search_vectors* relative to *positive_vectors*.

    Parameters
    ----------
    gamma : np.ndarray, shape (G,)
        Log-space scale parameters.
    positive_vectors : np.ndarray, shape (N_pos, D)
        Already-standardised reference embedding vectors.
    search_vectors : np.ndarray, shape (N_search, D) | None
        Vectors to score.  When *None*, positive_vectors are scored against
        themselves (self-similarity excluded via diagonal replacement with row median).

    Returns
    -------
    np.ndarray, shape (N_search_or_pos, G + 1)
        Column 0: max similarity to any positive vector.
        Columns 1..G: logsumexp of RBF(gamma_i) across positive vectors.
    """
    gamma_0 = 1.0 / positive_vectors.shape[1]

    if search_vectors is None:
        sq = -gamma_0 * cdist(positive_vectors, positive_vectors, "sqeuclidean")
        diag = np.diag_indices_from(sq)
        sq[diag] = np.median(sq, axis=1)
    else:
        sq = -gamma_0 * cdist(search_vectors, positive_vectors, "sqeuclidean")

    n_rows = sq.shape[0]
    features = np.empty((n_rows, len(gamma) + 1), dtype=np.float64)
    features[:, 0] = np.max(sq, axis=1)
    for i, g in enumerate(gamma):
        features[:, i + 1] = logsumexp(g * sq, axis=1)
    return features


def calculate_projection_matrices(
    vectors: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    PCA via SVD on *vectors*.

    Returns
    -------
    P : np.ndarray, shape (D, n_components)
        Top-component projection matrix.
    P_residual : np.ndarray, shape (D, D - n_components)
        Residual (low-variance) projection matrix.
    """
    X_centered = vectors - vectors.mean(axis=0)
    _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
    P          = Vt[:n_components].T
    P_residual = Vt[n_components:].T
    return P, P_residual


def project_to_subspace(vectors: np.ndarray, projection_matrix: np.ndarray) -> np.ndarray:
    """Project *vectors* into the subspace spanned by *projection_matrix*."""
    return vectors @ projection_matrix


def train_logistic_model(
    v_positive: np.ndarray,
    v_negative: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Train a balanced logistic regression to separate positive from negative examples.

    Returns
    -------
    model : LogisticRegression
    X_test : np.ndarray
    y_test : np.ndarray
    """
    X = np.vstack((v_positive, v_negative))
    y = np.concatenate((np.ones(len(v_positive)), np.zeros(len(v_negative))))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    model = LogisticRegression(
        random_state=random_state,
        class_weight="balanced",
        C=0.2,
    )
    model.fit(X_train, y_train)
    return model, X_test, y_test


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def build_rbf_features(
    v_positive: np.ndarray,
    v_eval: np.ndarray,
    gammas: np.ndarray = RBF_GAMMAS,
    n_pca_components: int = RBF_PCA_COMPONENTS,
) -> np.ndarray:
    """
    Build a raw RBF feature matrix for *v_eval* relative to *v_positive*.

    Both arrays must already be standardised by the caller.  No scaling is
    applied here; the caller is responsible for fitting and storing the
    StandardScaler instances (they are part of the user model).

    Steps
    -----
    1. SVD on *v_positive* → residual projection matrix.
    2. RBF features in the full space.
    3. RBF features in the residual (low-variance) subspace.
    4. Concatenate → raw feature matrix.

    Parameters
    ----------
    v_positive : np.ndarray, shape (N_pos, D)
        Standardised embedding vectors for the positive (liked) papers.
    v_eval : np.ndarray, shape (N_eval, D)
        Standardised embedding vectors for all papers to score.
    gammas : np.ndarray
        RBF kernel scale parameters.
    n_pca_components : int
        Number of SVD components extracted from *v_positive*.

    Returns
    -------
    np.ndarray, shape (N_eval, F)
        Raw (unscaled) RBF feature matrix, in the same row order as *v_eval*.
    """
    n_components = min(n_pca_components, v_positive.shape[0] - 1, v_positive.shape[1] - 1)
    _, P_residual = calculate_projection_matrices(v_positive, n_components)

    rbf_full = rbf_scoring(gammas, v_positive, v_eval)

    v_pos_res  = project_to_subspace(v_positive, P_residual)
    v_eval_res = project_to_subspace(v_eval, P_residual)
    rbf_res    = rbf_scoring(gammas, v_pos_res, v_eval_res)

    return np.hstack([rbf_full, rbf_res])


def fit_scoring_model(
    features: np.ndarray,
    idx_pos:  np.ndarray,
    idx_neg:  np.ndarray,
) -> LogisticRegression:
    """
    Train a logistic regression on pre-built RBF features.

    The training set is the explicitly labelled papers (positive + disliked).
    If there are too few explicit negatives, the entire unlabelled pool is used
    as the negative class instead.

    Parameters
    ----------
    features : np.ndarray, shape (N, F)
        Output of build_rbf_features().
    idx_pos : np.ndarray of bool, shape (N,)
        True for liked papers.
    idx_neg : np.ndarray of bool, shape (N,)
        True for explicitly disliked papers.

    Returns
    -------
    LogisticRegression
    """
    idx_unlabelled = ~idx_pos
    idx_train = idx_pos | idx_neg

    if idx_train.sum() >= 4 and idx_pos.sum() >= 2 and idx_neg.sum() >= 2:
        v_train_pos = features[idx_pos]
        v_train_neg = features[idx_neg]
    else:
        v_train_pos = features[idx_pos]
        v_train_neg = features[idx_unlabelled]

    model = LogisticRegression(
        random_state=42,
        class_weight="balanced",
        C=0.2,
        max_iter=1000,
    )
    X_train = np.vstack((v_train_pos, v_train_neg))
    y_train = np.concatenate((
        np.ones(len(v_train_pos)),
        np.zeros(len(v_train_neg)),
    ))
    model.fit(X_train, y_train)
    return model


def apply_scoring_model(
    model:     LogisticRegression,
    features:  np.ndarray,
    arxiv_ids: np.ndarray,
) -> dict[str, float]:
    """
    Apply a fitted logistic regression model to produce per-paper scores.

    Parameters
    ----------
    model : LogisticRegression
        Fitted model, typically the output of fit_scoring_model().
    features : np.ndarray, shape (N, F)
        Feature matrix from build_rbf_features().
    arxiv_ids : np.ndarray of str, shape (N,)
        Paper IDs in the same row order as *features*.

    Returns
    -------
    dict[str, float]
        Maps arXiv ID → log P(relevant).  Higher = more recommended.
    """
    log_proba = model.predict_log_proba(features)[:, 1]
    return {aid: float(log_proba[i]) for i, aid in enumerate(arxiv_ids)}


def score_papers_for_user(
    liked_ids:    list[str],
    disliked_ids: list[str],
    all_ids:      list[str],
    embeddings:   dict[str, np.ndarray],
    embedding_dim: int = EMBEDDING_DIM,
    gammas: np.ndarray = RBF_GAMMAS,
    n_pca_components: int = RBF_PCA_COMPONENTS,
) -> dict[str, float]:
    """
    Convenience wrapper: truncate embeddings, build features, train, and score.

    Equivalent to::

        arxiv_ids = ...                          # IDs present in embeddings
        vectors   = embeddings[:embedding_dim]   # truncate
        features  = build_rbf_features(vectors[idx_pos], vectors, gammas, n_pca_components)
        model     = fit_scoring_model(features, idx_pos, idx_neg)
        return    apply_scoring_model(model, features, arxiv_ids)

    Returns a zero-score dict when there are fewer than 2 liked papers with embeddings.

    Returns
    -------
    dict[str, float]
        Maps arXiv ID → log P(relevant).  Higher = more recommended.
        Only IDs present in *embeddings* are included.
    """
    valid_ids = [aid for aid in all_ids if aid in embeddings]
    if not valid_ids:
        return {}

    arxiv_ids = np.array(valid_ids)
    vectors = np.array(
        [embeddings[aid][:embedding_dim] for aid in arxiv_ids], dtype=np.float32
    )

    liked_set    = set(liked_ids)
    disliked_set = set(disliked_ids)
    idx_pos = np.array([aid in liked_set    for aid in arxiv_ids], dtype=bool)
    idx_neg = np.array([aid in disliked_set for aid in arxiv_ids], dtype=bool)

    if idx_pos.sum() < 2:
        return {aid: 0.0 for aid in arxiv_ids}

    # Both scalers are part of the user model and must be stored alongside the
    # logistic regression weights when persisting per-user models.
    embedding_scaler = StandardScaler()
    vectors_s = embedding_scaler.fit_transform(vectors)

    features_raw = build_rbf_features(vectors_s[idx_pos], vectors_s, gammas, n_pca_components)

    feature_scaler = StandardScaler()
    features = feature_scaler.fit_transform(features_raw)

    model = fit_scoring_model(features, idx_pos, idx_neg)
    return apply_scoring_model(model, features, arxiv_ids)
