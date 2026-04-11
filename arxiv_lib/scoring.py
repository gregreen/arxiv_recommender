"""
Scoring pipeline: feature engineering, model training, and per-user recommendation.

The recommended workflow (see `ScoringModel` and `recommend.py`) is:

  StandardScaler
    → SVD on positive vectors (RBF_PCA_COMPONENTS)
    → RBF features in full-dim space AND residual subspace
    → concatenate
    → StandardScaler
    → LogisticRegression(C=0.2, class_weight='balanced')
    → log-proba[:, 1] as relevance score

Pure numpy / sklearn; no network calls.
"""

import hashlib
import json

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from arxiv_lib.config import (
    QUERY_VECTOR_DIM,
    RECOMMENDATION_EMBEDDING_DIM,
    RBF_GAMMAS,
    RBF_PCA_COMPONENTS,
    SCORING_VERSION,
)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def rbf_scoring(
    gammas: np.ndarray,
    positive_vectors: np.ndarray,
    search_vectors: np.ndarray | None = None,
    metric='sqeuclidean'
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
        Standardized vectors to score.  When *None*, positive_vectors are
        scored against themselves (self-similarity excluded via diagonal
        replacement with row median).
    metric : str
        Distance metric to use for cdist. Default is 'sqeuclidean' (squared
        Euclidean distance).

    Returns
    -------
    np.ndarray, shape (N_search, G + 1)
        Column 0: max similarity to any positive vector.
        Columns 1..G: logsumexp of RBF(gamma_i) across positive vectors.
    """
    gamma_0 = 1.0 / positive_vectors.shape[1]

    if search_vectors is None:
        # Exclude comparisons to self by replacing diagonal with row median
        sq = -gamma_0 * cdist(positive_vectors, positive_vectors, metric)
        diag = np.diag_indices_from(sq)
        sq[diag] = np.median(sq, axis=1)
    else:
        sq = -gamma_0 * cdist(search_vectors, positive_vectors, metric)

    n_rows = sq.shape[0]
    features = np.empty((n_rows, len(gammas)+1), dtype=np.float64)
    features[:, 0] = np.max(sq, axis=1)
    for i, g in enumerate(gammas):
        features[:, i+1] = logsumexp(g * sq, axis=1)
    
    # Convert sum into mean
    features[:,1:] -= np.log(positive_vectors.shape[0])

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


def score_query_terms(positive_vectors: np.ndarray,
                      negative_vectors: np.ndarray,
                      query_vectors: np.ndarray) -> np.ndarray:
    """Determine which query terms are most likely to be relevant."""

    dist_pos = np.min(
        cdist(positive_vectors, query_vectors, 'cosine'),
        axis=0
    )
    dist_neg = np.min(
        cdist(negative_vectors, query_vectors, 'cosine'),
        axis=0
    )

    return np.log(dist_pos/(dist_neg+1e-12))


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
        positive_vectors: np.ndarray,
        positive_ids: list[str] | None = None,
        query_vectors: np.ndarray | None = None,
        positive_query_vectors: np.ndarray | None = None,
        query_terms: list[str] | None = None
    ):
        self.logistic_model = logistic_model
        self.P_residual = residual_projection_matrix
        self.mu_features = mu_features
        self.sigma_features = sigma_features
        self.mu_vectors = mu_vectors
        self.sigma_vectors = sigma_vectors
        self.positive_vectors = positive_vectors
        self.positive_ids: list[str] = positive_ids or []
        self.query_vectors = query_vectors
        self.positive_query_vectors = positive_query_vectors
        self.query_terms = query_terms

    @classmethod
    def from_training_data(cls, positive_vectors: np.ndarray,
                                negative_vectors: np.ndarray,
                                positive_ids: list[str] | None = None,
                                query_vectors: np.ndarray | None = None,
                                positive_query_vectors: np.ndarray | None = None,
                                negative_query_vectors: np.ndarray | None = None,
                                query_terms: list[str] | None = None
                          ) -> "ScoringModel":
        """
        Train a ScoringModel from the given positive and negative embedding vectors.

        Parameters
        ----------
        positive_vectors : np.ndarray, shape (N_pos, D_rec)
            Original (unscaled) recommendation embedding vectors for positive (liked) papers.
        negative_vectors : np.ndarray, shape (N_neg, D_rec)
            Original (unscaled) recommendation embedding vectors for negative (disliked) papers.
        positive_ids : list[str] | None
            arXiv IDs corresponding to each row of *positive_vectors*, in the same
            order.  When provided, score_positive_embeddings() results can be mapped
            back to paper IDs without self-similarity bias.
        query_vectors : np.ndarray, shape (N_query, D_query) | None
            Optional original (unscaled) embedding vectors for the user's search query terms.
        positive_query_vectors : np.ndarray, shape (N_pos, D_query) | None
            Optional search-space embedding vectors for the positive papers, aligned
            row-for-row with *positive_vectors*.  When provided, query features use
            these instead of the rec-space vectors for cosine comparison against
            *query_vectors*.
        negative_query_vectors : np.ndarray, shape (N_neg, D_query) | None
            Analogous search-space vectors for the negative papers.
        
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
            positive_vectors=None,
            positive_ids=list(positive_ids) if positive_ids is not None else [],
            query_vectors=None,
            positive_query_vectors=None,
            query_terms=None
        )
        model.fit(positive_vectors, negative_vectors,
                  query_vectors=query_vectors,
                  positive_query_vectors=positive_query_vectors,
                  negative_query_vectors=negative_query_vectors,
                  query_terms=query_terms)
        return model

    def fit(self, positive_vectors: np.ndarray,
                  negative_vectors: np.ndarray,
                  query_vectors: np.ndarray | None = None,
                  positive_query_vectors: np.ndarray | None = None,
                  negative_query_vectors: np.ndarray | None = None,
                  query_terms: list[str] | None = None
           ) -> None:
        """
        Fit the logistic regression model given positive and negative vectors.

        The embedding and feature scalers, plus the projection matrices, must
        already be fitted by the caller and stored in this instance before calling
        fit().

        Parameters
        ----------
        positive_vectors : np.ndarray, shape (N_pos, D_rec)
            Original (unscaled) recommendation embedding vectors for positive (liked) papers.
        negative_vectors : np.ndarray, shape (N_neg, D_rec)
            Original (unscaled) recommendation embedding vectors for negative (disliked) papers.
        query_vectors : np.ndarray, shape (N_query, D_query) | None
            Optional original (unscaled) embedding vectors for the user's search query terms.
        positive_query_vectors : np.ndarray, shape (N_pos, D_query) | None
            Optional search-space embedding vectors for the positive papers, aligned
            row-for-row with *positive_vectors*.  Used instead of rec-space vectors
            for cosine comparison against *query_vectors* when provided.
        negative_query_vectors : np.ndarray, shape (N_neg, D_query) | None
            Analogous search-space vectors for the negative papers.
        query_terms : list[str] | None
            Optional list of the user's search query terms corresponding to *query_vectors*.  Used for interpretability of the query-term relevance scores.
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

        # Calculate query features for positive and negative vectors.
        # Use search-space paper vectors (positive_query_vectors) for the cosine
        # comparison if provided; otherwise fall back to rec-space vectors.
        if query_vectors is not None:
            # Fall back to rec-space vectors for the RBF features if search-space query vectors aren't provided, since the RBF features are less sensitive to the domain shift and we want to preserve them for scoring even when search-space vectors are unavailable.
            pos_paper_vecs = positive_query_vectors if positive_query_vectors is not None else positive_vectors
            neg_paper_vecs = negative_query_vectors if negative_query_vectors is not None else negative_vectors

            # Score the relevance of each query term to the user's positive and negative papers
            query_relevance = score_query_terms(
                positive_query_vectors,
                negative_query_vectors,
                query_vectors
            )
            keep_idx = np.where(query_relevance < 0)[0] # Keep terms that are more similar to the positive papers than the negative papers
            print(keep_idx)

            query_vectors = query_vectors[keep_idx]
            query_terms = [query_terms[i] for i in keep_idx]
            print(query_terms)

            query_features_pos = rbf_scoring(
                RBF_GAMMAS,
                query_vectors,
                pos_paper_vecs,
                metric='cosine'
            )
            query_features_neg = rbf_scoring(
                RBF_GAMMAS,
                query_vectors,
                neg_paper_vecs,
                metric='cosine'
            )
            # features_pos = query_features_pos
            # features_neg = query_features_neg
            features_pos = np.concatenate(
                (features_pos, query_features_pos),
                axis=1
            )
            features_neg = np.concatenate(
                (features_neg, query_features_neg),
                axis=1
            )
        
        # Concatenate features along the sample dimension
        features = np.concatenate((features_pos, features_neg), axis=0)
        
        # Scale features (zero mean, unit variance)
        self.mu_features = features.mean(axis=0)
        self.sigma_features = features.std(axis=0)
        features = self.scale_features(features)

        print(f'mu_features = {self.mu_features}')
        print(f'sigma_features = {self.sigma_features}')

        # features[:,:2*(len(RBF_GAMMAS)+1)] = 0. # Hack to remove RBF features and force the model to rely on query features for debugging

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
            C=1.0,
            max_iter=1000
        )
        self.logistic_model.fit(features, y)

        # Store positive and query vectors for later use in scoring
        self.positive_vectors = positive_vectors
        self.query_vectors = query_vectors
        self.positive_query_vectors = positive_query_vectors
        self.query_terms = query_terms

        self.print_coefficients() # Debugging
    
    def print_coefficients(self):
        # Print regression coefficients for debugging
        # Need to show both the max_sim and logsumexp features for each gamma
        # and the full-space and residual features separately to understand what the model is doing.
        print("Logistic regression coefficients:")
        print('gamma  coeff_full  coeff_res   coeff_query')
        print('-----  ----------  ----------  ------------')

        # Extract different categories of coefficients for clarity
        n_f = 1 + len(RBF_GAMMAS)  # Number of full-space features
        coeff_full = self.logistic_model.coef_[0, :n_f]
        coeff_res = self.logistic_model.coef_[0, n_f:2*n_f]
        coeff_query = (
            np.full_like(coeff_res, np.nan)
            if self.query_vectors is None else
            self.logistic_model.coef_[0, 2*n_f:]
        )

        # Print features in order from largest to smallest scales
        for i, gamma in enumerate(RBF_GAMMAS):
            print(
                f'{np.log(gamma): >+5.2f}  '
               +f'{coeff_full[i+1]: >+10.5f}  '
               +f'{coeff_res[i+1]: >+10.5f}  '
               +f'{coeff_query[i+1]: >+10.5f}'
            )
        # Nearest-neighbor features (gamma -> inf)
        print(
            f' inf   '
           +f'{coeff_full[0]: >+10.5f}  '
           +f'{coeff_res[0]: >+10.5f}  '
           +f'{coeff_query[0]: >+10.5f}'
        )
    
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
    
    def _calc_query_features(
        self,
        vectors: np.ndarray,
        query_vectors_for_papers: np.ndarray | None = None
    ) -> np.ndarray | None:
        if self.query_vectors is not None:
            # No scaling applied for query-vector comparison, since
            # embeddings are trained to maximize cosine similarity for
            # query vs. relevant documents.
            # Use search-space paper vectors when provided; fall back to
            # rec-space vectors otherwise.
            paper_vecs = query_vectors_for_papers if query_vectors_for_papers is not None else vectors
            features = rbf_scoring(
                RBF_GAMMAS,
                self.query_vectors,
                paper_vecs,
                metric='cosine'
            )
            return features
        return None
    
    def score_embeddings(
        self,
        vectors: np.ndarray,
        query_vectors_for_papers: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Apply the fitted model to score the given embedding vectors.

        Parameters
        ----------
        vectors : np.ndarray, shape (N_eval, D_rec)
            Original (unscaled) recommendation embedding vectors for the papers to score.
        query_vectors_for_papers : np.ndarray, shape (N_eval, D_query) | None
            Optional search-space embedding vectors for the same papers, aligned
            row-for-row with *vectors*.  When provided, query features use these
            instead of *vectors* for cosine comparison against the stored query terms.
        
        Returns
        -------
        np.ndarray, shape (N_eval,)
            ln P(relevant) scores for each input vector. Higher = more recommended.
        """
        # Calculate features vs. positive vectors
        features, _ = calculate_rbf_features(
            self.scale_vectors(self.positive_vectors),
            v_eval=self.scale_vectors(vectors),
            gammas=RBF_GAMMAS,
            P_residual=self.P_residual
        )

        # If user query vectors are available, calculate features
        if self.query_vectors is not None:
            query_features = self._calc_query_features(vectors, query_vectors_for_papers)
            features = np.concatenate((features, query_features), axis=1)
            # features = query_features
        
        # Scale features using the training data statistics
        features = self.scale_features(features)
        
        # print(np.any(~np.isfinite(features))) # Debugging: check for NaNs or infs in features

        # features[:,:2*(len(RBF_GAMMAS)+1)] = 0. # Hack to remove RBF features and force the model to rely on query features for debugging

        # Input features into Logistic Regression model
        ln_prob = self.logistic_model.predict_log_proba(features)[:, 1]

        return ln_prob
    
    def score_positive_embeddings(self) -> np.ndarray:
        """
        Score the positive (liked) embedding vectors using the fitted model.

        This is a convenience method that applies the model to the original
        positive vectors, which are stored in this instance. This is used to
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

        # If user query vectors are available, calculate features.
        # Use stored positive_query_vectors (search-space) when available.
        if self.query_vectors is not None:
            query_features = self._calc_query_features(
                self.positive_vectors, self.positive_query_vectors
            )
            features = np.concatenate((features, query_features), axis=1)
        
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
            "positive_ids": self.positive_ids,
            "query_vectors": (
                self.query_vectors.tolist()
                if self.query_vectors is not None else None
            ),
            "positive_query_vectors": (
                self.positive_query_vectors.tolist()
                if self.positive_query_vectors is not None else None
            ),
            "query_terms": self.query_terms
        }
    
    @classmethod
    def deserialize(cls, data: dict) -> "ScoringModel":
        """
        Deserialize a ScoringModel from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing the serialized model components, as produced
            by serialize().

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
            positive_vectors=np.array(data["positive_vectors"]),
            positive_ids=data.get("positive_ids", []),
            query_vectors=(
                np.array(data["query_vectors"])
                if data.get("query_vectors") is not None else None
            ),
            positive_query_vectors=(
                np.array(data["positive_query_vectors"])
                if data.get("positive_query_vectors") is not None else None
            ),
            query_terms=data.get("query_terms", None)
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


def compute_model_hash(liked_ids: list[str],
                       disliked_ids: list[str] = (),
                       query_terms: list[str] = ()) -> str:
    """
    Compute a short hash that uniquely identifies the combination
    of liked/disliked papers, user query terms, and the current scoring
    configuration.

    The hash changes whenever the liked or disliked set, the query set,
    the embedding dimension, or scoring version changes, allowing callers
    to detect stale cached models.

    Parameters
    ----------
    liked_ids : list[str]
        arXiv IDs of the user's liked papers.  Order does not matter.
    disliked_ids : list[str]
        arXiv IDs of the user's explicitly disliked papers.  Order does not matter.
    query_terms : list[str]
        query terms that the user has entered, which may influence the model.  Order does not matter.

    Returns
    -------
    str
        First 16 hex characters of the SHA-256 digest.
    """
    payload = (
        json.dumps(sorted(liked_ids))
        + json.dumps(sorted(disliked_ids))
        + json.dumps(sorted(query_terms))
        + str(RECOMMENDATION_EMBEDDING_DIM)
        + str(QUERY_VECTOR_DIM)
        + json.dumps(RBF_GAMMAS.tolist())
        + SCORING_VERSION
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]

