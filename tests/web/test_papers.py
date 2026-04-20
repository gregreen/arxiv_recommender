"""
Tests for GET /api/papers/{arxiv_id}
"""

import json
import os

import pytest

from arxiv_lib.config import SUMMARY_CACHE_DIR


def _insert_paper(db, arxiv_id="2309.06676"):
    db.execute(
        "INSERT INTO papers (arxiv_id, title, abstract, authors, published_date, categories) "
        "VALUES (?, 'Test Title', 'Test abstract.', ?, '2023-09-12', ?)",
        (arxiv_id, json.dumps(["Author One"]), json.dumps(["astro-ph.CO"])),
    )
    db.commit()


class TestGetPaper:
    def test_returns_paper_without_summary(self, client, web_db):
        """A paper with no summary file on disk should return all metadata fields
        with summary=null and a correctly formed arxiv.org URL."""
        _insert_paper(web_db)
        r = client.get("/api/papers/2309.06676")
        assert r.status_code == 200
        data = r.json()
        assert data["arxiv_id"] == "2309.06676"
        assert data["title"] == "Test Title"
        assert data["authors"] == ["Author One"]
        assert data["summary"] is None
        assert data["url"] == "https://arxiv.org/abs/2309.06676"

    def test_returns_paper_with_summary(self, client, web_db):
        """When a summary .txt file exists in the summary cache directory the
        endpoint should read it and return its content in the summary field."""
        _insert_paper(web_db)
        summary_dir = SUMMARY_CACHE_DIR()  # also creates the directory
        with open(os.path.join(summary_dir, "2309.06676.txt"), "w") as f:
            f.write("My summary text")
        r = client.get("/api/papers/2309.06676")
        assert r.status_code == 200
        assert r.json()["summary"] == "My summary text"

    def test_404_when_not_found(self, client):
        """Requesting a paper whose arxiv_id does not exist in the database
        should return 404 Not Found."""
        r = client.get("/api/papers/9999.99999")
        assert r.status_code == 404

    def test_401_when_unauthenticated(self, raw_client):
        """A request with no authentication cookie should be rejected with
        401 Unauthorized before any DB lookup is performed."""
        r = raw_client.get("/api/papers/2309.06676")
        assert r.status_code == 401
