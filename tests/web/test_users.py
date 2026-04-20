"""
Tests for /api/users/me/* endpoints:

  GET    /api/users/me/papers
  POST   /api/users/me/papers
  POST   /api/users/me/papers/import/ads
  PATCH  /api/users/me/papers/{arxiv_id}
  DELETE /api/users/me/papers/{arxiv_id}
  GET    /api/users/me/categories
  PUT    /api/users/me/categories
"""

import json

import pytest

_USER_ID = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _insert_paper(db, arxiv_id="2309.06676"):
    db.execute(
        "INSERT INTO papers (arxiv_id, title, abstract, authors, published_date, categories) "
        "VALUES (?, 'Test Title', 'Abstract.', ?, '2023-09-12', ?)",
        (arxiv_id, json.dumps(["Author One"]), json.dumps(["astro-ph.CO"])),
    )
    db.commit()


def _add_to_library(db, arxiv_id="2309.06676", liked=1):
    db.execute(
        "INSERT INTO user_papers (user_id, arxiv_id, liked) VALUES (?, ?, ?)",
        (_USER_ID, arxiv_id, liked),
    )
    db.commit()


def _add_import_log_rows(db, count: int, arxiv_id_prefix: str = "2309.{:05d}"):
    """Insert `count` user_import_log rows with recent timestamps."""
    db.executemany(
        "INSERT INTO user_import_log (user_id, arxiv_id) VALUES (?, ?)",
        [(_USER_ID, arxiv_id_prefix.format(i)) for i in range(1, count + 1)],
    )
    db.commit()


# ---------------------------------------------------------------------------
# GET /api/users/me/papers
# ---------------------------------------------------------------------------

class TestListPapers:
    def test_empty_list(self, client):
        """A new user with no papers in their library should get an empty list."""
        r = client.get("/api/users/me/papers")
        assert r.status_code == 200
        assert r.json() == []

    def test_returns_liked_papers(self, client, web_db):
        """Papers added to the library with liked=1 should appear in the list
        with all metadata fields and the correct liked value."""
        _insert_paper(web_db)
        _add_to_library(web_db, liked=1)
        r = client.get("/api/users/me/papers")
        assert r.status_code == 200
        items = r.json()
        assert len(items) == 1
        assert items[0]["arxiv_id"] == "2309.06676"
        assert items[0]["liked"] == 1


# ---------------------------------------------------------------------------
# POST /api/users/me/papers
# ---------------------------------------------------------------------------

class TestAddPaper:
    def test_adds_new_paper_and_enqueues_task(self, client, web_db):
        """Adding a paper not yet in the database should create a user_papers row,
        a user_import_log row, and enqueue a fetch_meta task so the ingest
        pipeline will populate metadata and embeddings."""
        r = client.post("/api/users/me/papers", json={"arxiv_id": "2309.06676"})
        assert r.status_code == 201
        assert r.json() == {"arxiv_id": "2309.06676", "liked": 1}
        # Import log row created
        count = web_db.execute(
            "SELECT COUNT(*) FROM user_import_log WHERE user_id = ? AND arxiv_id = ?",
            (_USER_ID, "2309.06676"),
        ).fetchone()[0]
        assert count == 1
        # fetch_meta task enqueued (paper not yet in papers table)
        task = web_db.execute(
            "SELECT type FROM task_queue WHERE json_extract(payload, '$.arxiv_id') = ?",
            ("2309.06676",),
        ).fetchone()
        assert task is not None
        assert task["type"] == "fetch_meta"

    def test_duplicate_is_idempotent(self, client, web_db):
        """Re-adding a paper that is already in the user's library should
        return 201 without creating a second user_import_log entry, so the
        rate-limit counter is not incremented for duplicate submissions."""
        _add_to_library(web_db, liked=1)
        r = client.post("/api/users/me/papers", json={"arxiv_id": "2309.06676"})
        assert r.status_code == 201
        # No new import_log row for an already-present paper
        count = web_db.execute(
            "SELECT COUNT(*) FROM user_import_log WHERE user_id = ?", (_USER_ID,)
        ).fetchone()[0]
        assert count == 0

    def test_rate_limited_when_daily_limit_reached(self, client, web_db):
        """Once the user has reached the Tier A daily import limit (16 imports
        in the rolling 24-hour window) any further add attempt should be
        rejected with 429 Too Many Requests."""
        # Tier A daily limit = 16; insert 16 recent import_log rows
        _add_import_log_rows(web_db, count=16)
        r = client.post("/api/users/me/papers", json={"arxiv_id": "2309.99999"})
        assert r.status_code == 429

    def test_invalid_id_422(self, client):
        """A string that does not match the new-style (YYMM.NNNNN) or old-style
        (category/NNNNNNN) arXiv ID format should be rejected with 422."""
        r = client.post("/api/users/me/papers", json={"arxiv_id": "not-an-id"})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# PATCH /api/users/me/papers/{arxiv_id}
# ---------------------------------------------------------------------------

class TestPatchPaper:
    def test_updates_liked_flag(self, client, web_db):
        """PATCHing a paper already in the library should update the liked field
        both in the response body and in the underlying database row."""
        _add_to_library(web_db, liked=1)
        r = client.patch("/api/users/me/papers/2309.06676", json={"liked": -1})
        assert r.status_code == 200
        assert r.json()["liked"] == -1
        row = web_db.execute(
            "SELECT liked FROM user_papers WHERE user_id = ? AND arxiv_id = ?",
            (_USER_ID, "2309.06676"),
        ).fetchone()
        assert row["liked"] == -1


# ---------------------------------------------------------------------------
# DELETE /api/users/me/papers/{arxiv_id}
# ---------------------------------------------------------------------------

class TestDeletePaper:
    def test_delete_existing(self, client, web_db):
        """DELETEing a paper that is in the user's library should return 204
        and remove the row from user_papers."""
        _add_to_library(web_db)
        r = client.delete("/api/users/me/papers/2309.06676")
        assert r.status_code == 204
        row = web_db.execute(
            "SELECT 1 FROM user_papers WHERE user_id = ? AND arxiv_id = ?",
            (_USER_ID, "2309.06676"),
        ).fetchone()
        assert row is None

    def test_delete_not_found_404(self, client):
        """Attempting to DELETE a paper that is not in the user's library
        should return 404 Not Found."""
        r = client.delete("/api/users/me/papers/2309.06676")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/users/me/papers/import/ads
# ---------------------------------------------------------------------------

class TestImportAds:
    def test_basic_import(self, client, web_db):
        """Submitting ADS export text containing three arXiv IDs should import
        all three, returning imported=3 with skipped and rate_limited both 0."""
        ads_text = "arXiv:2309.00001\narXiv:2309.00002\narXiv:2309.00003"
        r = client.post("/api/users/me/papers/import/ads", json={"text": ads_text})
        assert r.status_code == 200
        data = r.json()
        assert data["imported"] == 3
        assert data["skipped"] == 0
        assert data["rate_limited"] == 0

    def test_too_many_ids_422(self, client):
        """A single import payload containing more than 64 unique arXiv IDs
        (the ADS import limit) should be rejected with 422 before any DB writes."""
        # _ADS_IMPORT_LIMIT = 64; 65 unique IDs exceeds it
        ads_text = " ".join(f"arXiv:2309.{i:05d}" for i in range(1, 66))
        r = client.post("/api/users/me/papers/import/ads", json={"text": ads_text})
        assert r.status_code == 422

    def test_partial_import_due_to_rate_limit(self, client, web_db):
        """When the daily Tier A quota is already exhausted, all IDs in the ADS
        import should be counted as rate_limited rather than imported."""
        # Fill the daily Tier A quota (16) with recent imports
        _add_import_log_rows(web_db, count=16, arxiv_id_prefix="9999.{:05d}")
        ads_text = "arXiv:2309.00010\narXiv:2309.00011\narXiv:2309.00012"
        r = client.post("/api/users/me/papers/import/ads", json={"text": ads_text})
        assert r.status_code == 200
        data = r.json()
        assert data["imported"] == 0
        assert data["rate_limited"] == 3


# ---------------------------------------------------------------------------
# GET /api/users/me/categories
# ---------------------------------------------------------------------------

class TestCategories:
    def test_get_empty(self, client):
        """A user who has not subscribed to any categories should receive
        an empty categories list."""
        r = client.get("/api/users/me/categories")
        assert r.status_code == 200
        assert r.json() == {"categories": []}

    def test_set_valid(self, client):
        """PUTting a list of valid arXiv category codes should replace the
        user's subscriptions and return the categories sorted alphabetically."""
        r = client.put("/api/users/me/categories", json={"categories": ["astro-ph.CO", "cs.LG"]})
        assert r.status_code == 200
        assert r.json() == {"categories": ["astro-ph.CO", "cs.LG"]}

    def test_set_invalid_422(self, client):
        """PUTting a category string that does not appear in ARXIV_CATEGORIES
        should be rejected with 422 Unprocessable Content."""
        r = client.put("/api/users/me/categories", json={"categories": ["not.real"]})
        assert r.status_code == 422
