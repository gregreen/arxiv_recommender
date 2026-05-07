"""
Tests for GET /api/explore
"""

import pytest


def _insert_lowres_proj_row(db, arxiv_id: str, x: float, y: float,
                            computed_at: str = "2026-04-30T03:00:00") -> None:
    db.execute(
        "INSERT INTO paper_lowres_proj (arxiv_id, x, y, computed_at) VALUES (?, ?, ?, ?)",
        (arxiv_id, x, y, computed_at),
    )
    db.commit()


def _insert_paper(db, arxiv_id: str, published_date: str = "2026-04-30") -> None:
    db.execute(
        "INSERT OR IGNORE INTO papers (arxiv_id, title, published_date) VALUES (?, ?, ?)",
        (arxiv_id, f"Title of {arxiv_id}", published_date),
    )
    db.commit()


def _like_paper(db, user_id: int, arxiv_id: str) -> None:
    db.execute(
        "INSERT OR IGNORE INTO user_papers (user_id, arxiv_id, liked) VALUES (?, ?, 1)",
        (user_id, arxiv_id),
    )
    db.commit()


def _insert_recommendation(
    db,
    user_id: int,
    arxiv_id: str,
    time_window: str,
    score: float,
    rank: int = 1,
    model_hash: str = "test-model",
) -> None:
    db.execute(
        """
        INSERT OR REPLACE INTO recommendations
            (user_id, arxiv_id, time_window, score, rank, model_hash, generated_at)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
        """,
        (user_id, arxiv_id, time_window, score, rank, model_hash),
    )
    db.commit()


class TestExplore:
    def test_no_projection_returns_unavailable(self, client):
        """If paper_lowres_proj is empty, lowres_proj_available should be False."""
        r = client.get("/api/explore?window=week")
        assert r.status_code == 200
        data = r.json()
        assert data["lowres_proj_available"] is False
        assert data["papers"] == []
        assert data["liked_overlay"] == []
        assert data["lowres_proj_computed_at"] is None

    def test_returns_papers_in_window(self, client, web_db):
        """Papers with projection coords in the selected window should be returned."""
        _insert_paper(web_db, "2404.00001", published_date="2024-04-01")
        _insert_paper(web_db, "2604.99999", published_date="2026-04-30")
        _insert_lowres_proj_row(web_db, "2404.00001", 0.1, 0.2)
        _insert_lowres_proj_row(web_db, "2604.99999", 0.5, 0.6)

        r = client.get("/api/explore?window=week")
        assert r.status_code == 200
        data = r.json()
        assert data["lowres_proj_available"] is True
        arxiv_ids = {p["arxiv_id"] for p in data["papers"]}
        # Recent paper should be in the week window; old one should not
        assert "2604.99999" in arxiv_ids
        assert "2404.00001" not in arxiv_ids
        # Each paper has required fields
        paper = next(p for p in data["papers"] if p["arxiv_id"] == "2604.99999")
        assert "x" in paper and "y" in paper and "title" in paper

    def test_liked_overlay_included(self, client, web_db):
        """Liked papers should appear in liked_overlay regardless of window."""
        _insert_paper(web_db, "2404.00001", published_date="2024-04-01")
        _insert_lowres_proj_row(web_db, "2404.00001", 0.3, 0.7)
        _like_paper(web_db, user_id=1, arxiv_id="2404.00001")

        r = client.get("/api/explore?window=day")
        assert r.status_code == 200
        data = r.json()
        # Old paper not in day window papers but in liked_overlay
        assert not any(p["arxiv_id"] == "2404.00001" for p in data["papers"])
        assert any(p["arxiv_id"] == "2404.00001" for p in data["liked_overlay"])

    def test_liked_overlay_includes_score_when_available(self, client, web_db):
        """Liked overlay points should include recommendation score when present."""
        _insert_paper(web_db, "2404.00001", published_date="2024-04-01")
        _insert_lowres_proj_row(web_db, "2404.00001", 0.3, 0.7)
        _like_paper(web_db, user_id=1, arxiv_id="2404.00001")
        _insert_recommendation(web_db, user_id=1, arxiv_id="2404.00001", time_window="day", score=-0.42)

        r = client.get("/api/explore?window=day")
        assert r.status_code == 200
        data = r.json()

        overlay = next(p for p in data["liked_overlay"] if p["arxiv_id"] == "2404.00001")
        assert overlay["score"] == pytest.approx(-0.42)

    def test_invalid_window_422(self, client):
        """An unknown window value should return 422."""
        r = client.get("/api/explore?window=year")
        assert r.status_code == 422

    def test_unauthenticated_401(self, raw_client):
        """Unauthenticated requests should be rejected with 401."""
        r = raw_client.get("/api/explore?window=week")
        assert r.status_code == 401
