"""
Tests for POST /api/search
"""

from unittest.mock import patch

import pytest

from arxiv_lib.search import SearchEmbeddingError

_FAKE_RESULTS = [
    {"arxiv_id": "2309.06676", "title": "Test Paper", "score": 0.95},
]


class TestSearch:
    def test_search_success(self, client):
        """A valid search query should return the list of results produced by
        search_papers, with the correct arxiv_ids and HTTP 200."""
        with patch("web.routers.search.search_papers", return_value=_FAKE_RESULTS):
            r = client.post("/api/search", json={"query": "black holes"})
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 1
        assert data[0]["arxiv_id"] == "2309.06676"

    def test_search_embedding_error_503(self, client):
        """If the embedding service is unavailable (SearchEmbeddingError) the
        endpoint should return 503 Service Unavailable rather than propagating
        the exception."""
        with patch(
            "web.routers.search.search_papers",
            side_effect=SearchEmbeddingError("API down"),
        ):
            r = client.post("/api/search", json={"query": "black holes"})
        assert r.status_code == 503

    def test_unauthenticated_401(self, raw_client):
        """Search requests with no authentication cookie should be rejected
        with 401 Unauthorized."""
        r = raw_client.post("/api/search", json={"query": "black holes"})
        assert r.status_code == 401
