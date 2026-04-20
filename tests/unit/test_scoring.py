"""
Unit tests for arxiv_lib/scoring.py.

All tests are pure numpy/sklearn — no network calls, no DB access.
"""

import json

import numpy as np
import pytest

from arxiv_lib.config import (
    RBF_GAMMAS,
    RECOMMENDATION_EMBEDDING_DIM,
)
from arxiv_lib.scoring import (
    ScoringModel,
    calculate_projection_matrices,
    rbf_scoring,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_D = RECOMMENDATION_EMBEDDING_DIM   # 128
_RNG = np.random.default_rng(seed=0)


def _make_separable_data(n: int = 20, d: int = _D) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (positive_vectors, negative_vectors) that are linearly separable.

    Positives cluster tightly around +e_0; negatives around -e_0.
    """
    rng = np.random.default_rng(seed=1)
    center_pos = np.zeros(d); center_pos[0] =  3.0
    center_neg = np.zeros(d); center_neg[0] = -3.0
    pos = center_pos + rng.normal(scale=0.1, size=(n, d))
    neg = center_neg + rng.normal(scale=0.1, size=(n, d))
    return pos.astype(np.float32), neg.astype(np.float32)


def _train_model(n: int = 20) -> ScoringModel:
    pos, neg = _make_separable_data(n)
    return ScoringModel.from_training_data(pos, neg)


# ---------------------------------------------------------------------------
# rbf_scoring
# ---------------------------------------------------------------------------


class TestRbfScoring:
    def test_output_shape_with_search_vectors(self):
        pos = _RNG.standard_normal((10, 8)).astype(np.float32)
        search = _RNG.standard_normal((5, 8)).astype(np.float32)
        out = rbf_scoring(RBF_GAMMAS, pos, search)
        assert out.shape == (5, len(RBF_GAMMAS) + 1)

    def test_self_scoring_shape(self):
        """With search_vectors=None, rows scored against themselves (diagonal replaced)."""
        # Use orthogonal basis vectors so no two rows are equal — max-sim < 0 guaranteed.
        d = 10
        pos = np.eye(d, dtype=np.float32)   # mutually orthogonal unit vectors
        out = rbf_scoring(RBF_GAMMAS, pos, search_vectors=None)
        assert out.shape == (d, len(RBF_GAMMAS) + 1)
        # max-sim column (col 0) is -gamma_0 * squared_distance; all distances > 0
        assert np.all(out[:, 0] < 0), "All max-sim values should be < 0 for distinct vectors"


# ---------------------------------------------------------------------------
# calculate_projection_matrices
# ---------------------------------------------------------------------------


class TestCalculateProjectionMatrices:
    def test_shapes(self):
        X = _RNG.standard_normal((20, 8)).astype(np.float32)
        P, P_res = calculate_projection_matrices(X, n_components=3)
        assert P.shape == (8, 3)
        assert P_res.shape == (8, 5)

    def test_orthogonality(self):
        X = _RNG.standard_normal((20, 8)).astype(np.float32)
        P, P_res = calculate_projection_matrices(X, n_components=3)
        cross = P.T @ P_res   # should be ~0
        np.testing.assert_allclose(cross, np.zeros((3, 5)), atol=1e-5)


# ---------------------------------------------------------------------------
# ScoringModel train → score
# ---------------------------------------------------------------------------


class TestScoringModel:
    def test_score_embeddings_shape_and_range(self):
        """score_embeddings returns shape (N,) of finite log-probabilities ≤ 0."""
        model = _train_model()
        eval_vecs = _RNG.standard_normal((10, _D)).astype(np.float32)
        scores = model.score_embeddings(eval_vecs)
        assert scores.shape == (10,)
        assert np.all(np.isfinite(scores))
        # log-probabilities from predict_log_proba are always ≤ 0
        assert np.all(scores <= 1e-6), f"Some scores exceed 0: {scores[scores > 1e-6]}"

    def test_positives_score_higher_than_negatives(self):
        """Held-out positives should have a higher median score than held-out negatives."""
        # Train on 200, hold out 30.  With enough training data the jitter applied
        # to subspace-fraction features during fit() does not prevent clean separation.
        pos, neg = _make_separable_data(n=230)
        model = ScoringModel.from_training_data(pos[:200], neg[:200])

        score_pos = np.median(model.score_embeddings(pos[200:]))
        score_neg = np.median(model.score_embeddings(neg[200:]))
        assert score_pos > score_neg, (
            f"Expected held-out positives to score higher: pos={score_pos:.4f} neg={score_neg:.4f}"
        )


# ---------------------------------------------------------------------------
# ScoringModel serialization / deserialization
# ---------------------------------------------------------------------------


class TestScoringModelSerialization:
    def test_serialize_deserialize_roundtrip_scores_match(self):
        """Deserialized model produces numerically identical scores."""
        model = _train_model()
        eval_vecs = _RNG.standard_normal((15, _D)).astype(np.float32)

        scores_before = model.score_embeddings(eval_vecs)
        restored = ScoringModel.deserialize(model.serialize())
        scores_after = restored.score_embeddings(eval_vecs)

        # float32 arithmetic limits precision to ~1e-4
        np.testing.assert_allclose(scores_before, scores_after, atol=1e-4)

    def test_serialize_output_is_json_serializable(self):
        """serialize() must produce a dict that round-trips through json.dumps."""
        model = _train_model()
        blob = model.serialize()
        # Should not raise
        json_str = json.dumps(blob)
        assert isinstance(json_str, str)

    def test_deserialize_restores_metadata_fields(self):
        """positive_ids survive the round-trip unchanged."""
        pos, neg = _make_separable_data()
        ids = ["2309.06676", "2401.00001"]
        model = ScoringModel.from_training_data(pos, neg, positive_ids=ids)
        restored = ScoringModel.deserialize(model.serialize())
        assert restored.positive_ids == ids

    def test_deserialize_none_optional_fields_stay_none(self):
        """Optional fields absent from training remain None after deserialization."""
        # Train without query_vectors, positive_query_vectors, explicit_negative_vectors
        model = _train_model()
        assert model.query_vectors is None
        assert model.positive_query_vectors is None
        assert model.explicit_negative_vectors is None

        restored = ScoringModel.deserialize(model.serialize())
        assert restored.query_vectors is None
        assert restored.positive_query_vectors is None
        assert restored.explicit_negative_vectors is None
