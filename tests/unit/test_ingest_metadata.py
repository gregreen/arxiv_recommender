"""Unit tests for write_to_arxiv_metadata_cache / load_from_arxiv_metadata_cache.

These functions are pure SQLite operations (no network) so no HTTP mocking is needed.
The `app_db_con` fixture (from conftest.py) redirects config.DATA_DIR to a
temporary directory and initialises a fresh app.db schema before each test.
"""

from arxiv_lib.ingest import load_from_arxiv_metadata_cache, write_to_arxiv_metadata_cache


def test_write_and_read_single_paper(app_db_con):
    """Writing one paper's metadata and reading it back returns correct fields."""
    paper = {
        "2309.06676": {
            "title": "Attention Is All You Need",
            "authors": ["Ashish Vaswani", "Noam Shazeer"],
            "abstract": "We propose a new network architecture, the Transformer.",
        }
    }
    write_to_arxiv_metadata_cache(paper)

    result = load_from_arxiv_metadata_cache(["2309.06676"])

    assert "2309.06676" in result
    assert result["2309.06676"]["title"] == "Attention Is All You Need"
    assert result["2309.06676"]["authors"] == ["Ashish Vaswani", "Noam Shazeer"]
    assert "Transformer" in result["2309.06676"]["abstract"]


def test_load_missing_ids_omitted(app_db_con):
    """An ID that was never inserted is silently omitted from the result."""
    result = load_from_arxiv_metadata_cache(["9999.99999"])
    assert result == {}


def test_write_batch_and_read_back(app_db_con):
    """Writing a batch of papers and reading them all back returns all entries."""
    batch = {
        "2401.00001": {"title": "Paper A", "authors": ["Alice"], "abstract": "About A."},
        "2401.00002": {"title": "Paper B", "authors": ["Bob", "Carol"], "abstract": "About B."},
        "2401.00003": {"title": "Paper C", "authors": [], "abstract": "About C."},
    }
    write_to_arxiv_metadata_cache(batch)

    result = load_from_arxiv_metadata_cache(list(batch.keys()))

    assert set(result.keys()) == set(batch.keys())
    assert result["2401.00001"]["title"] == "Paper A"
    assert result["2401.00002"]["authors"] == ["Bob", "Carol"]
    assert result["2401.00003"]["abstract"] == "About C."


def test_write_upserts_on_duplicate(app_db_con):
    """Writing the same ID twice keeps the most recent data (INSERT OR REPLACE)."""
    arxiv_id = "2309.06676"
    write_to_arxiv_metadata_cache({
        arxiv_id: {"title": "Old Title", "authors": ["Alice"], "abstract": "Old abstract."}
    })
    write_to_arxiv_metadata_cache({
        arxiv_id: {"title": "New Title", "authors": ["Bob"], "abstract": "New abstract."}
    })

    result = load_from_arxiv_metadata_cache([arxiv_id])

    assert result[arxiv_id]["title"] == "New Title"
    assert result[arxiv_id]["authors"] == ["Bob"]
    assert result[arxiv_id]["abstract"] == "New abstract."
