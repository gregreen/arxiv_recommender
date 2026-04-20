"""
Tests for GET /api/recommendations?window=...
"""

from unittest.mock import patch

import pytest

from arxiv_lib.recommend import NotEnoughDataError

_FAKE_RESULT = {
    "arxiv_id": "2309.06676",
    "title": "Test Paper",
    "score": -0.5,
    "rank": 1,
    "generated_at": "2026-04-20 00:00:00",
}


class TestRecommendations:
    def test_invalid_window_422(self, client):
        """A window value that is not one of the accepted choices (day, week,
        month) should be rejected with 422 Unprocessable Content."""
        r = client.get("/api/recommendations?window=forever")
        assert r.status_code == 422

    def test_onboarding_when_not_enough_data(self, client):
        """When get_recommendations raises NotEnoughDataError the endpoint should
        fall back to onboarding papers, return onboarding=True, and include a
        non-empty guidance message."""
        with (
            patch("web.routers.recommendations.get_recommendations", side_effect=NotEnoughDataError),
            patch("web.routers.recommendations.get_onboarding_papers", return_value=[]),
        ):
            r = client.get("/api/recommendations?window=week")
        assert r.status_code == 200
        data = r.json()
        assert data["onboarding"] is True
        assert data["count"] == 0
        assert data["message"] is not None

    def test_returns_recommendations(self, client):
        """When recommendations are available the endpoint should return
        onboarding=False, the correct count, and the full results list."""
        with patch(
            "web.routers.recommendations.get_recommendations",
            return_value=[_FAKE_RESULT],
        ):
            r = client.get("/api/recommendations?window=week")
        assert r.status_code == 200
        data = r.json()
        assert data["onboarding"] is False
        assert data["count"] == 1
        assert data["results"][0]["arxiv_id"] == "2309.06676"
