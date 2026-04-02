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
    gammas: np.ndarray,
    positive_vectors: np.ndarray,
    search_vectors: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute RBF kernel aggregate scores for *search_vectors* relative to *positive_vectors*.

    Parameters
    ----------
    gammas : np.ndarray, shape (G,)
        Log-space scale parameters.
    positive_vectors : np.ndarray, shape (N_pos, D)
        Already-standardised reference embedding vectors.
    search_vectors : np.ndarray, shape (N_search, D) | None
        Standardized vectors to score.  When *None*, positive_vectors are scored against
        themselves (self-similarity excluded via diagonal replacement with row median).

    Returns
    -------
    np.ndarray, shape (N_search_or_pos, G + 1)
        Column 0: max similarity to any positive vector.
        Columns 1..G: logsumexp of RBF(gamma_i) across positive vectors.
    """
    gamma_0 = 1.0 / positive_vectors.shape[1]

    if search_vectors is None:
        # Exclude comparisons to self by replacing diagonal with row median
        sq = -gamma_0 * cdist(positive_vectors, positive_vectors, "sqeuclidean")
        diag = np.diag_indices_from(sq)
        sq[diag] = np.median(sq, axis=1)
    else:
        sq = -gamma_0 * cdist(search_vectors, positive_vectors, "sqeuclidean")

    n_rows = sq.shape[0]
    features = np.empty((n_rows, len(gammas)+1), dtype=np.float64)
    features[:, 0] = np.max(sq, axis=1)
    for i, g in enumerate(gammas):
        features[:, i+1] = logsumexp(g * sq, axis=1)
    return features


def calculate_projection_matrices(
    vectors: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    PCA via SVD on *vectors*. Returns two matrices: one for projecting into
    the high-variance subspace, and the other for projecting into the residual
    (low-variance) subspace.

    Parameters
    ----------
    vectors : np.ndarray, shape (N, D)
        Input vectors for PCA. Must already be standardised by the caller.
    n_components : int
        Number of PCA components to extract. Must be less than D.

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
    X = np.concatenate((v_positive, v_negative), axis=0)
    y = np.concatenate((np.ones(len(v_positive)), np.zeros(len(v_negative))))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    model = LogisticRegression(
        random_state=random_state,
        class_weight="balanced",
        C=0.2,
        max_iter=1000
    )
    model.fit(X_train, y_train)
    return model, X_test, y_test


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def calculate_rbf_features(
    v_positive: np.ndarray,
    v_eval: np.ndarray | None = None,
    gammas: np.ndarray = RBF_GAMMAS,
    n_pca_components: int = RBF_PCA_COMPONENTS,
    P_residual: np.ndarray | None = None
) -> np.ndarray:
    """
    Calculate raw RBF feature matrix for *v_eval* relative to *v_positive*.

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
    v_eval : np.ndarray | None, shape (N_eval, D)
        Standardised embedding vectors for all papers to score. If *None*,
        defaults to *v_positive* and avoids self-similarity comparisons.
    gammas : np.ndarray
        RBF kernel scale parameters.
    n_pca_components : int
        Number of SVD components extracted from *v_positive*.
    P_residual : np.ndarray | None, shape (D, D - n_pca_components)
        Optional pre-computed residual projection matrix.  If provided, this is
        used instead of re-computing the SVD and projection matrices from *v_positive*.

    Returns
    -------
    np.ndarray, shape (N_eval, F)
        Raw (unscaled) RBF feature matrix, in the same row order as *v_eval*.
    np.ndarray, shape (D, D - n_pca_components)
        Projection matrix for the residual projection matrix (for optional caching).
    """
    if P_residual is None:
        # Calculate projection matrices from low-variance components of v_positive
        n_components = min(n_pca_components, v_positive.shape[0]-1, v_positive.shape[1]-1)
        _, P_proj = calculate_projection_matrices(v_positive, n_components)
    else:
        P_proj = P_residual

    # Features in the full space
    rbf_full = rbf_scoring(gammas, v_positive, v_eval)

    # Features in the residual (low-variance) subspace
    v_pos_res = project_to_subspace(v_positive, P_proj)

    if v_eval is None:
        v_eval_res = None
    else:
        v_eval_res = project_to_subspace(v_eval, P_proj)
    
    rbf_res = rbf_scoring(gammas, v_pos_res, v_eval_res)

    # Concatenate the features from both the full and subspace
    # features = np.hstack([rbf_full, rbf_res])
    # rbf_full[:,0] = np.random.normal(size=rbf_full.shape[0]) # Hack to remove this feature
    # rbf_res[:] = np.random.normal(size=rbf_res.shape) # Hack to remove this feature
    features = np.concatenate([rbf_full, rbf_res], axis=1)

    return features, P_proj


class ScoringModel(object):
    """
    Container for all components of a fitted user scoring model.

    This includes the logistic regression weights, plus the embedding and
    feature scalers, plus the SVD projection matrices.  All of these are
    necessary to persist when caching per-user models.
    """
    def __init__(
        self,
        logistic_model: LogisticRegression,
        residual_projection_matrix: np.ndarray,
        mu_features: np.ndarray,
        sigma_features: np.ndarray,
        mu_vectors: np.ndarray,
        sigma_vectors: np.ndarray,
        positive_vectors: np.ndarray
    ):
        self.logistic_model = logistic_model
        self.P_residual = residual_projection_matrix
        self.mu_features = mu_features
        self.sigma_features = sigma_features
        self.mu_vectors = mu_vectors
        self.sigma_vectors = sigma_vectors
        self.positive_vectors = positive_vectors

    @classmethod
    def from_training_data(cls, positive_vectors: np.ndarray,
                                negative_vectors: np.ndarray) -> "ScoringModel":
        """
        Train a ScoringModel from the given positive and negative embedding vectors.

        Parameters
        ----------
        positive_vectors : np.ndarray, shape (N_pos, D)
            Original (unscaled) embedding vectors for the positive (liked) papers.
        negative_vectors : np.ndarray, shape (N_neg, D)
            Original (unscaled) embedding vectors for the negative (disliked) papers.
        
        Returns
        -------
        ScoringModel
            Trained model instance, ready to score new embedding vectors.
        """
        # Create model with placeholder inputs
        model = cls(
            logistic_model=None,
            residual_projection_matrix=None,
            mu_features=None,
            sigma_features=None,
            mu_vectors=None,
            sigma_vectors=None,
            positive_vectors=None
        )
        model.fit(positive_vectors, negative_vectors)
        return model

    def fit(self, positive_vectors: np.ndarray, negative_vectors: np.ndarray):
        """
        Fit the logistic regression model given positive and negative vectors.

        The embedding and feature scalers, plus the projection matrices, must
        already be fitted by the caller and stored in this instance before calling
        fit().

        Parameters
        ----------
        positive_vectors : np.ndarray, shape (N_pos, D)
            Original (unscaled) embedding vectors for the positive (liked) papers.
        negative_vectors : np.ndarray, shape (N_neg, D)
            Original (unscaled) embedding vectors for the negative (disliked) papers.
        """
        # Scale vectors (zero mean, unit variance)
        vectors = np.concatenate((positive_vectors, negative_vectors), axis=0)
        self.mu_vectors = np.mean(vectors, axis=0)
        self.sigma_vectors = np.std(vectors, axis=0)
        # print(f"mu_vectors = {self.mu_vectors}")
        # print(f"sigma_vectors = {self.sigma_vectors}")
        v_pos = self.scale_vectors(positive_vectors)
        v_neg = self.scale_vectors(negative_vectors)

        # Calculate RBF features for negative vectors
        features_neg, self.P_residual = calculate_rbf_features(
            v_pos, v_eval=v_neg,
            gammas=RBF_GAMMAS,
            n_pca_components=RBF_PCA_COMPONENTS
        )
        # Calculate RBF features for positive vectors (avoiding self-comparisons)
        features_pos, _ = calculate_rbf_features(
            v_pos, v_eval=v_pos,
            gammas=RBF_GAMMAS,
            n_pca_components=RBF_PCA_COMPONENTS,
            P_residual=self.P_residual
        )
        # Concatenate features
        features = np.concatenate((features_pos, features_neg), axis=0)
        
        # Scale features (zero mean, unit variance)
        self.mu_features = features.mean(axis=0)
        self.sigma_features = features.std(axis=0)
        features = self.scale_features(features)

        # print(f'mu_features = {self.mu_features}')
        # print(f'sigma_features = {self.sigma_features}')

        # print('Positive features (first 5 rows):')
        # print(self.scale_features(features_pos)[:5])

        # print('Negative features (first 5 rows):')
        # print(self.scale_features(features_neg)[:5])

        # Fit logistic regression
        y = np.concatenate((
            np.ones(len(positive_vectors)),
            np.zeros(len(negative_vectors))
        ))
        self.logistic_model = LogisticRegression(
            random_state=42,
            class_weight="balanced",
            C=0.2,
            max_iter=1000
        )
        self.logistic_model.fit(features, y)

        # Store positive vectors for later use in scoring
        self.positive_vectors = positive_vectors
    
    def scale_features(self, features: np.ndarray) -> np.ndarray:
        """Scale features using the mean and std from the training data."""
        # print(f"Scaling features with mu.shape = {self.mu_features.shape}, "
        #       f"sigma.shape = {self.sigma_features.shape}, "
        #       f"features.shape = {features.shape}")
        features_scaled = (features - self.mu_features[None]) / self.sigma_features[None]
        return features_scaled
    
    def scale_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Scale embedding vectors using the mean and std from the training data."""
        # print(f"Scaling vectors with mu.shape = {self.mu_vectors.shape}, "
        #       f"sigma.shape = {self.sigma_vectors.shape}, "
        #       f"vectors.shape = {vectors.shape}")
        vectors_scaled = (vectors - self.mu_vectors[None]) / self.sigma_vectors[None]
        return vectors_scaled
    
    def score_embeddings(self, vectors: np.ndarray) -> np.ndarray:
        """
        Apply the fitted model to score the given embedding vectors.

        The input *vectors* must be in the original (unscaled) embedding space.

        Parameters
        ----------
        vectors : np.ndarray, shape (N_eval, D)
            Original (unscaled) embedding vectors for the papers to score.
        
        Returns
        -------
        np.ndarray, shape (N_eval,)
            ln P(relevant) scores for each input vector. Higher = more recommended.
        """
        print('v_pos: ', self.scale_vectors(self.positive_vectors)[:5])
        print('v_eval: ', self.scale_vectors(vectors)[:5])
        features, _ = calculate_rbf_features(
            self.scale_vectors(self.positive_vectors),
            v_eval=self.scale_vectors(vectors),
            gammas=RBF_GAMMAS,
            P_residual=self.P_residual
        )
        features = self.scale_features(features)
        ln_prob = self.logistic_model.predict_log_proba(features)[:, 1]
        return ln_prob
    
    def score_positive_embeddings(self) -> np.ndarray:
        """
        Score the positive (liked) embedding vectors using the fitted model.

        This is a convenience method that applies the model to the original
        positive vectors, which are stored in this instance.  This is used to
        calculate the "self-similarity" feature for liked papers.

        Returns
        -------
        np.ndarray, shape (N_pos,)
            ln P(relevant) scores for each positive vector. Higher = more recommended.
        """
        features, _ = calculate_rbf_features(
            self.scale_vectors(self.positive_vectors),
            v_eval=None, # Avoid self-comparisons by passing None
            gammas=RBF_GAMMAS,
            P_residual=self.P_residual
        )
        features = self.scale_features(features)
        ln_prob = self.logistic_model.predict_log_proba(features)[:, 1]
        return ln_prob
    
    def serialize(self) -> dict:
        """
        Serialize the model components into a dictionary for caching.

        Returns
        -------
        dict
            Dictionary containing all necessary components to reconstruct this model.
        """
        return {
            "logistic_model": serialize_logistic_regression_model(self.logistic_model),
            "residual_projection_matrix": self.P_residual.tolist(),
            "mu_features": self.mu_features.tolist(),
            "sigma_features": self.sigma_features.tolist(),
            "mu_vectors": self.mu_vectors.tolist(),
            "sigma_vectors": self.sigma_vectors.tolist(),
            "positive_vectors": self.positive_vectors.tolist(),
        }
    
    @classmethod
    def deserialize(cls, data: dict) -> "ScoringModel":
        """
        Deserialize a ScoringModel from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing the serialized model components, as produced by serialize().

        Returns
        -------
        ScoringModel
            Reconstructed ScoringModel instance.
        """
        return cls(
            logistic_model=deserialize_logistic_regression_model(data["logistic_model"]),
            residual_projection_matrix=np.array(data["residual_projection_matrix"]),
            mu_features=np.array(data["mu_features"]),
            sigma_features=np.array(data["sigma_features"]),
            mu_vectors=np.array(data["mu_vectors"]),
            sigma_vectors=np.array(data["sigma_vectors"]),
            positive_vectors=np.array(data["positive_vectors"])
        )



def serialize_logistic_regression_model(model: LogisticRegression) -> dict:
    """Serialize a LogisticRegression model into a dictionary."""
    return {
        "coef": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
        "classes": model.classes_.tolist(),
        "C": model.C,
        "class_weight": model.class_weight,
        "random_state": model.random_state,
    }

def deserialize_logistic_regression_model(data: dict) -> LogisticRegression:
    """Deserialize a LogisticRegression model from a dictionary."""
    model = LogisticRegression()
    model.coef_ = np.array(data["coef"])
    model.intercept_ = np.array(data["intercept"])
    model.classes_ = np.array(data["classes"])
    model.C = data["C"]
    model.class_weight = data["class_weight"]
    model.random_state = data["random_state"]
    return model


def fit_scoring_model(
    positive_embeddings: np.ndarray,
    negative_embeddings: np.ndarray
) -> ScoringModel:
    """
    Train a logistic regression on a set of embedding vectors.

    The training set is the explicitly labelled papers (positive + disliked).
    All papers not explicitly in the positive set are assumed to be in the negative set.

    Parameters
    ----------
    features : np.ndarray, shape (N, F)
        Output of build_rbf_features().
    idx_pos : np.ndarray of bool, shape (N,)
        True for liked papers.

    Returns
    -------
    LogisticRegression
    """
    idx_neg = ~idx_pos
    X_train = np.vstack((features[idx_pos], features[idx_neg]))
    y_train = np.concatenate((
        np.ones(len(features[idx_pos])),
        np.zeros(len(features[idx_neg])),
    ))
    model = LogisticRegression(
        random_state=42,
        class_weight="balanced",
        C=0.2,
        max_iter=1000
    )
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
