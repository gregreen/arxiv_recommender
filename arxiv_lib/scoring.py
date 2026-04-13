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
    SUBSPACE_FRACTION_DIMS,
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
    full_matrices: bool = True
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
    full_matrices : bool
        Passed to np.linalg.svd. If True, the returned projection matrices are
        full D×D; if False, they are D×n_components and D×(D-n_components)
        respectively. The former is allows one to calculate the projection
        into the full residual space, while the former is more efficient if
        one only needs the projection into the high-variance subspace.

    Returns
    -------
    P : np.ndarray, shape (D, n_components)
        Top-component projection matrix.
    P_residual : np.ndarray, shape (D, D - n_components)
        Residual (low-variance) projection matrix. If full_matrices=False, then
        the second axis may be truncated to a lower number of dimensions.
    """
    # Use SVD decomposition to calculate the principal components.
    X_centered = vectors - vectors.mean(axis=0)
    _, S, Vt = np.linalg.svd(X_centered, full_matrices=full_matrices)

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
    """
    Determine which query terms are most likely to be relevant.

    Parameters
    ----------
    positive_vectors : np.ndarray, shape (N_pos, D)
        Raw embedding vectors for the positive (liked) papers.
    negative_vectors : np.ndarray, shape (N_neg, D)
        Raw embedding vectors for the negative (disliked) papers.
    query_vectors : np.ndarray, shape (N_query, D)
        Raw embedding vectors for the query terms.

    Returns
    -------
    np.ndarray, shape (N_query,)
        Relevance score for each query term, where more negative values indicate
        higher relevance (greater similarity to positive vectors and/or greater
        dissimilarity to negative vectors).
    """

    dist_pos = np.min(
        cdist(positive_vectors, query_vectors, 'cosine'),
        axis=0
    )
    dist_neg = np.min(
        cdist(negative_vectors, query_vectors, 'cosine'),
        axis=0
    )

    return np.log(dist_pos/(dist_neg+1e-12))


def cosine_dist_from_center(center: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    Calculate cosine distance from a center vector.
    
    Parameters
    ----------
    center : np.ndarray, shape (D,)
        Center vector to calculate distances from.
    vectors : np.ndarray, shape (N, D)

    Returns
    -------
    np.ndarray, shape (N,)
        Cosine distance from the center for each vector, where smaller values
        indicate greater similarity to the center.
    """
    center_norm = max(np.linalg.norm(center), 1e-12)
    vectors_norm = np.clip(
        np.linalg.norm(vectors, axis=1),
        a_min=1e-12, a_max=None
    )
    cosine_sim = (vectors @ center) / (vectors_norm * center_norm)
    cosine_sim.shape = (-1,1)
    return 1 - cosine_sim


def subspace_fraction(P_proj: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    Calculate the fraction of each vector's length captured by the projection.
    For a vector v, this fraction is defined by projecting the vector into
    the subspace (v_proj = v @ P_proj) and calculating the ratio of the
    projected vector's norm to the original vector's norm: ||v_proj|| / ||v||.

    Parameters
    ----------
    P_proj : np.ndarray, shape (D, d)
        Projection matrix for the d-dimensonal subspace of interest (e.g.
        the high-variance subspace, as determined by the positive vectors).
    vectors : np.ndarray, shape (N, D)
        Vectors to project onto the subspace.

    Returns
    -------
    np.ndarray, shape (N,)
        Fraction of each vector's length captured by the projection.
    """
    v_proj = project_to_subspace(vectors, P_proj)
    frac = np.linalg.norm(v_proj, axis=1) / (np.linalg.norm(vectors, axis=1) + 1e-12)
    return frac


def subspace_fraction_features(P_proj: np.ndarray,
                               vectors: np.ndarray,
                               dims: list[int] = SUBSPACE_FRACTION_DIMS) -> np.ndarray:
    """
    Calculate subspace fraction features for multiple subspace dimensionalities.

    For each specified number of dimensions k in *dims*, the projection matrix
    is truncated to the top k components, and the fraction of each vector's
    length captured by that projection is calculated.

    Parameters
    ----------
    P_proj : np.ndarray, shape (D, max(dims))
        Projection matrix for the maximum-dimensional subspace of interest
        (e.g. the high-variance subspace, as determined by the positive
        vectors).
    vectors : np.ndarray, shape (N, D)
        Vectors to project onto the subspace.
    dims : list[int]
        List of numbers of dimensions to use for calculating the fraction
        features.  For each k in *dims*, the projection matrix is truncated
        to the top k components before calculating the fraction.
    
    Returns
    -------
    np.ndarray, shape (N, len(dims)+1)
        Subspace fraction features for each vector and each specified number of
        dimensions, plus one final feature for the fraction of length^2 captured
        by the left-over dimensions.
    """
    features = np.empty((vectors.shape[0], len(dims)+1), dtype=np.float32)

    # Squared norms of the original vectors
    v_norm2 = np.sum(vectors**2, axis=1) + 1e-12

    # Project into largest subspace of interest
    v_proj = project_to_subspace(vectors, P_proj)

    # Calculate norm^2 in each subspace, and divide by original norm^2
    dims = [0] + sorted(dims)
    for i,(d0,d1) in enumerate(zip(dims[:-1], dims[1:])):
        proj_norm2 = np.sum(v_proj[:, d0:d1]**2, axis=1)
        features[:, i] = proj_norm2 / v_norm2

    features[:, -1] = 1 - np.sum(features[:, :-1], axis=1)
    return features


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
        highvar_projection_matrix: np.ndarray,
        mu_features: np.ndarray,
        sigma_features: np.ndarray,
        mu_vectors: np.ndarray,
        sigma_vectors: np.ndarray,
        positive_vectors: np.ndarray,
        positive_ids: list[str] | None = None,
        query_vectors: np.ndarray | None = None,
        positive_query_vectors: np.ndarray | None = None,
        explicit_negative_vectors: np.ndarray | None = None,
        query_terms: list[str] | None = None,
        pos_center: np.ndarray | None = None,
        neg_center: np.ndarray | None = None
    ):
        self.logistic_model = logistic_model
        self.P_residual = residual_projection_matrix
        self.P = highvar_projection_matrix
        self.mu_features = mu_features
        self.sigma_features = sigma_features
        self.mu_vectors = mu_vectors
        self.sigma_vectors = sigma_vectors
        self.positive_vectors = positive_vectors
        self.positive_ids: list[str] = positive_ids or []
        self.query_vectors = query_vectors
        self.positive_query_vectors = positive_query_vectors
        self.explicit_negative_vectors = explicit_negative_vectors
        self.query_terms = query_terms
        self.pos_center = pos_center
        self.neg_center = neg_center

    @classmethod
    def from_training_data(cls, positive_vectors: np.ndarray,
                                negative_vectors: np.ndarray,
                                positive_ids: list[str] | None = None,
                                query_vectors: np.ndarray | None = None,
                                positive_query_vectors: np.ndarray | None = None,
                                negative_query_vectors: np.ndarray | None = None,
                                n_explicit_negatives: int = 0,
                                query_terms: list[str] | None = None,
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
        n_explicit_negatives : int
            Number of rows at the start of *negative_vectors* that are
            user-disliked papers (as opposed to background negatives).  The
            corresponding vectors are stored as ``explicit_negative_vectors``
            for use in downstream feature engineering.  Defaults to 0.
        
        Returns
        -------
        ScoringModel
            Trained model instance, ready to score new embedding vectors.
        """
        # Create model with placeholder inputs
        model = cls(
            logistic_model=None,
            residual_projection_matrix=None,
            highvar_projection_matrix=None,
            mu_features=None,
            sigma_features=None,
            mu_vectors=None,
            sigma_vectors=None,
            positive_vectors=None,
            positive_ids=list(positive_ids) if positive_ids is not None else [],
            query_vectors=None,
            positive_query_vectors=None,
            explicit_negative_vectors=None,
            query_terms=None,
            pos_center=None,
            neg_center=None
        )
        model.fit(positive_vectors, negative_vectors,
                  query_vectors=query_vectors,
                  positive_query_vectors=positive_query_vectors,
                  negative_query_vectors=negative_query_vectors,
                  n_explicit_negatives=n_explicit_negatives,
                  query_terms=query_terms)
        return model

    def fit(self, positive_vectors: np.ndarray,
                  negative_vectors: np.ndarray,
                  query_vectors: np.ndarray | None = None,
                  positive_query_vectors: np.ndarray | None = None,
                  negative_query_vectors: np.ndarray | None = None,
                  n_explicit_negatives: int = 0,
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
        n_explicit_negatives : int
            Number of rows at the start of *negative_vectors* that are
            user-disliked papers (as opposed to background negatives).  The
            corresponding vectors are stored as ``explicit_negative_vectors``
            for use in downstream feature engineering.  Defaults to 0.
        query_terms : list[str] | None
            Optional list of the user's search query terms corresponding to
            *query_vectors*.  Used for interpretability of the query-term
            relevance scores.
        """

        pos_features = () # Start with empty set and add in features
        neg_features = ()

        # Calculate cosine distance from mean positive and negative vectors
        self.pos_center = np.mean(positive_vectors, axis=0)
        self.neg_center = np.mean(negative_vectors, axis=0)

        pos_features = pos_features + self._calc_center_features(positive_vectors)
        neg_features = neg_features + self._calc_center_features(negative_vectors)

        # Fraction of variance inside high-variance subspace
        self.P,_ = calculate_projection_matrices(positive_vectors, max(SUBSPACE_FRACTION_DIMS))
        pos_features = pos_features + (self._calc_subspace_fraction_features(positive_vectors),)
        neg_features = neg_features + (self._calc_subspace_fraction_features(negative_vectors),)

        # Scale vectors (zero mean, unit variance)
        vectors = np.concatenate((positive_vectors, negative_vectors), axis=0)
        self.mu_vectors = np.mean(vectors, axis=0)
        self.sigma_vectors = np.std(vectors, axis=0)
        v_pos = self.scale_vectors(positive_vectors)
        v_neg = self.scale_vectors(negative_vectors)

        # Calculate RBF features for negative vectors
        l2_features_neg, self.P_residual = calculate_rbf_features(
            v_pos, v_eval=v_neg,
            gammas=RBF_GAMMAS,
            n_pca_components=RBF_PCA_COMPONENTS
        )
        # Calculate RBF features for positive vectors (avoiding self-comparisons)
        l2_features_pos, _ = calculate_rbf_features(
            v_pos, v_eval=v_pos,
            gammas=RBF_GAMMAS,
            n_pca_components=RBF_PCA_COMPONENTS,
            P_residual=self.P_residual
        )

        pos_features = pos_features + (l2_features_pos,)
        neg_features = neg_features + (l2_features_neg,)

        # # RBF features calculated with respect to the explicit negative vectors
        # if n_explicit_negatives > 0:
        #     self.explicit_negative_vectors = negative_vectors[:n_explicit_negatives]
        #     features_disliked_pos = rbf_scoring(
        #         RBF_GAMMAS,
        #         self.explicit_negative_vectors,
        #         positive_vectors,
        #         metric='cosine'
        #     )
        #     features_disliked_neg_explicit = rbf_scoring( # Avoid self-comparison
        #         RBF_GAMMAS,
        #         self.explicit_negative_vectors,
        #         metric='cosine'
        #     )
        #     features_disliked_neg_bg = rbf_scoring(
        #         RBF_GAMMAS,
        #         self.explicit_negative_vectors,
        #         negative_vectors[n_explicit_negatives:],
        #         metric='cosine'
        #     )
        #     features_disliked_neg = np.concatenate(
        #         (features_disliked_neg_explicit, features_disliked_neg_bg),
        #         axis=0
        #     )
        #     pos_features = pos_features + (features_disliked_pos,)
        #     neg_features = neg_features + (features_disliked_neg,)
        # else:
        #     self.explicit_negative_vectors = None

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
            # Keep terms that are more similar to the positive papers than the negative papers
            keep_idx = np.where(query_relevance < 0)[0]

            query_vectors = query_vectors[keep_idx]
            query_terms = [query_terms[i] for i in keep_idx]

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
            pos_features = pos_features + (query_features_pos,)
            neg_features = neg_features + (query_features_neg,)
        
        pos_features = np.concatenate(pos_features, axis=1)
        neg_features = np.concatenate(neg_features, axis=1)
        
        # Concatenate features along the sample dimension
        features = np.concatenate((pos_features, neg_features), axis=0)
        
        # Scale features (zero mean, unit variance)
        self.mu_features = features.mean(axis=0)
        self.sigma_features = features.std(axis=0)
        features = self.scale_features(features)

        # print(f'mu_features = {self.mu_features}')
        # print(f'sigma_features = {self.sigma_features}')

        # # Zero-out residual l2 features, for debugging purposes.
        # n_f = 1 + len(RBF_GAMMAS)  # Number of full-space features
        # n_s = 1 + len(SUBSPACE_FRACTION_DIMS) # Number of subspace fraction features
        # features[:, 2+n_s+n_f:2+n_s+2*n_f] = 0.

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
            C=0.1,
            max_iter=1000,
            # solver='saga',
            # penalty='elasticnet',
            # l1_ratio=0.1
        )
        self.logistic_model.fit(features, y)

        # Report model accuracy on the training data for debugging (should be close to 100%)
        train_acc = self.logistic_model.score(features, y)
        print(f"Training accuracy: {train_acc: >7.4%}")

        # Store positive and query vectors for later use in scoring
        self.positive_vectors = positive_vectors
        self.query_vectors = query_vectors
        self.positive_query_vectors = positive_query_vectors
        self.query_terms = query_terms

        self.print_coefficients() # Debugging
    
    def print_coefficients(self):
        """
        Print the logistic regression coefficients in a human-readable
        format for debugging and interpretability.
        """
        coeffs = self.logistic_model.coef_[0]

        print("Logistic regression coefficients:")
        print(f"positive center cosine dist: {coeffs[0]: >+9.5f}")
        print(f"negative center cosine dist: {coeffs[1]: >+9.5f}")
        coeffs = coeffs[2:] # Skip center cosine distance coefficients

        print('Subspace fraction coefficients:')
        print('dim   coeff')
        print('----  --------')
        for i,d in enumerate(SUBSPACE_FRACTION_DIMS):
            print(f" {d: >3d}  {coeffs[i]: >+8.5f}")
        print(f">{SUBSPACE_FRACTION_DIMS[-1]: >3d}  {coeffs[len(SUBSPACE_FRACTION_DIMS)]: >+8.5f}")
        coeffs = coeffs[len(SUBSPACE_FRACTION_DIMS)+1:] # Skip subspace fraction coefficients

        # For each gamma, show the max_sim, mean features, and query features.
        # First, extract different categories of coefficients for clarity.
        n_f = 1 + len(RBF_GAMMAS)  # Number of full-space features
        coeff_full = coeffs[:n_f]
        coeff_res = coeffs[n_f:2*n_f]
        coeffs = coeffs[2*n_f:]

        # if self.explicit_negative_vectors is not None:
        #     coeff_neg = coeffs[:n_f]
        #     coeffs = coeffs[n_f:]
        # else:
        #     coeff_neg = np.full_like(coeff_full, np.nan)

        coeff_query = (
            np.full_like(coeff_res, np.nan)
            if self.query_vectors is None else
            coeffs[:n_f]
        )

        # Print features in order from largest to smallest scales
        print('gamma  coeff_full  coeff_res   coeff_query')
        print('-----  ----------  ----------  ------------')
        for i, gamma in enumerate(RBF_GAMMAS):
            print(
                f'{np.log(gamma): >+5.2f}  '
               +f'{coeff_full[i+1]: >+10.5f}  '
               +f'{coeff_res[i+1]: >+10.5f}  '
            #    +f'{coeff_neg[i+1]: >+10.5f}  '
               +f'{coeff_query[i+1]: >+10.5f}'
            )
        # Nearest-neighbor features (gamma -> inf)
        print(
            f' inf   '
           +f'{coeff_full[0]: >+10.5f}  '
           +f'{coeff_res[0]: >+10.5f}  '
        #    +f'{coeff_neg[0]: >+10.5f}  '
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
    
    def _calc_center_features(self, vectors: np.ndarray) -> np.ndarray:
        """
        Calculate cosine distance from the positive and negative centers as features.

        Parameters
        ----------
        vectors : np.ndarray, shape (N, D)
            Original (unscaled) embedding vectors to calculate features for.
        
        Returns
        -------
        tuple of np.ndarray
            Tuple containing two arrays of shape (N, 1): (pos_center_features, neg_center_features).
              pos_center_features: cosine distance from the positive center.
              neg_center_features: cosine distance from the negative center.
        """
        pos_center_features = cosine_dist_from_center(self.pos_center, vectors)
        neg_center_features = cosine_dist_from_center(self.neg_center, vectors)
        return (pos_center_features, neg_center_features)
    
    def _calc_subspace_fraction_features(self, vectors: np.ndarray) -> np.ndarray:
        """
        Calculate subspace fraction features for the given vectors.

        Parameters
        ----------
        vectors : np.ndarray, shape (N, D)
            Original (unscaled) embedding vectors to calculate features for.
        
        Returns
        -------
        np.ndarray, shape (N, len(SUBSPACE_FRACTION_DIMS))
            Subspace fraction features for each vector and each specified number of dimensions.
        """
        return subspace_fraction_features(self.P, vectors-self.pos_center)
    
    def _calc_explicit_negative_features(self, vectors: np.ndarray) -> np.ndarray | None:
        """
        Calculate RBF features with respect to the explicit negative vectors.

        Parameters
        ----------
        vectors : np.ndarray, shape (N, D)
            Original (unscaled) embedding vectors to calculate features for.
        
        Returns
        -------
        np.ndarray, shape (N, n_explicit_negatives * len(RBF_GAMMAS))
            RBF features for each vector with respect to each explicit negative vector and each gamma.
            Returns None if no explicit negative vectors are stored in this model.
        """
        if self.explicit_negative_vectors is None:
            return None
        
        features = rbf_scoring(
            RBF_GAMMAS,
            self.explicit_negative_vectors,
            vectors,
            metric='cosine'
        )

        return features
    
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
        # Calculate center features
        features = self._calc_center_features(vectors)

        # Calculate fraction of vector in high-variance subspace
        features = features + (self._calc_subspace_fraction_features(vectors),)

        # Calculate l2 distance features vs. positive vectors
        l2_features, _ = calculate_rbf_features(
            self.scale_vectors(self.positive_vectors),
            v_eval=self.scale_vectors(vectors),
            gammas=RBF_GAMMAS,
            P_residual=self.P_residual
        )
        features = features + (l2_features,)

        # # Calculate features with respect to explicit negative vectors, if available
        # explicit_neg_features = self._calc_explicit_negative_features(vectors)
        # if explicit_neg_features is not None:
        #     features = features + (explicit_neg_features,)

        # If user query vectors are available, calculate features
        if self.query_vectors is not None:
            query_features = self._calc_query_features(vectors, query_vectors_for_papers)
            features = features + (query_features,)
        
        # Concatenate features along the feature dimension
        features = np.concatenate(features, axis=1)
        
        # Scale features using the training data statistics
        features = self.scale_features(features)

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
        # Calculate center features
        features = self._calc_center_features(self.positive_vectors)

        # Calculate fraction of vector in high-variance subspace
        features = features + (self._calc_subspace_fraction_features(self.positive_vectors),)

        # Calculate l2 distance features vs. positive vectors, avoiding self-comparisons
        l2_features, _ = calculate_rbf_features(
            self.scale_vectors(self.positive_vectors),
            v_eval=None, # Avoid self-comparisons by passing None
            gammas=RBF_GAMMAS,
            P_residual=self.P_residual
        )
        features = features + (l2_features,)

        # # Calculate features with respect to explicit negative vectors, if available
        # explicit_neg_features = self._calc_explicit_negative_features(self.positive_vectors)
        # if explicit_neg_features is not None:
        #     features = features + (explicit_neg_features,)

        # If user query vectors are available, calculate features.
        # Use stored positive_query_vectors (search-space) when available.
        if self.query_vectors is not None:
            query_features = self._calc_query_features(
                self.positive_vectors, self.positive_query_vectors
            )
            features = features + (query_features,)

        # Concatenate features along the feature dimension
        features = np.concatenate(features, axis=1)
        
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
            "highvar_projection_matrix": self.P.tolist(),
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
            "query_terms": self.query_terms,
            "pos_center": self.pos_center.tolist(),
            "neg_center": self.neg_center.tolist(),
            "explicit_negative_vectors": (
                self.explicit_negative_vectors.tolist()
                if self.explicit_negative_vectors is not None else None
            ),
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
            highvar_projection_matrix=np.array(data["highvar_projection_matrix"]),
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
            query_terms=data.get("query_terms", None),
            pos_center=np.array(data["pos_center"]),
            neg_center=np.array(data["neg_center"]),
            explicit_negative_vectors=(
                np.array(data["explicit_negative_vectors"])
                if data.get("explicit_negative_vectors") is not None else None
            ),
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
    the embedding dimensions, RBF gammas, subspace fraction dims,
    RBF PCA components, or scoring version changes, allowing callers
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
        + json.dumps(sorted(SUBSPACE_FRACTION_DIMS))
        + str(RBF_PCA_COMPONENTS)
        + SCORING_VERSION
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]

