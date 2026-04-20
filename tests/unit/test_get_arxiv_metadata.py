"""Unit tests for get_arxiv_metadata (orchestration layer).

Tests the coordination logic: DB cache check → S2 fetch → Atom fallback → cache write.
Sub-function parsing is covered at a lower level in test_ingest_fetch*.py.

fetch_arxiv_metadata_s2 and fetch_arxiv_metadata are patched directly so no
network calls are made and tenacity retry loops do not fire.

NOTE on old-style IDs (e.g. "hep-th/9901001"):
  sanitize_old_style_arxiv_id (slash → underscore) is ONLY used for file-system
  operations (source/summary cache).  It is not called inside get_arxiv_metadata,
  load_from_arxiv_metadata_cache, or write_to_arxiv_metadata_cache.  Therefore
  the slash is preserved throughout: stored in the DB, sent to S2 as
  "ARXIV:hep-th/9901001", and returned as the result-dict key.

NOTE on test 3 (rate-limit fallback):
  get_arxiv_metadata catches any Exception from the S2 call and falls back to
  Atom.  We raise RuntimeError directly rather than a tenacity-wrapped HTTPError
  to avoid triggering the 7-attempt retry loop (which would make the test slow).
  RuntimeError("Semantic Scholar API request failed: ...") is exactly what
  fetch_arxiv_metadata_s2 raises after exhausting retries in production.
"""

from unittest.mock import patch

from arxiv_lib.ingest import get_arxiv_metadata, write_to_arxiv_metadata_cache


# ---------------------------------------------------------------------------
# Anonymised metadata payloads (mimic real parsed API responses)
# ---------------------------------------------------------------------------

_PAPER_A = {
    "title": "Magnetic reconnection rates in low-density stellar coronae",
    "authors": ["Jane Smith", "Robert Chen"],
    "abstract": (
        "We investigate magnetic reconnection in the outer layers of stellar "
        "coronae using a semi-analytic framework."
    ),
    "published_date": "2016-01-07",
    "categories": ["astro-ph.SR"],
}

_PAPER_B = {
    "title": "Temporal bias and cultural memory in generative sequence models",
    "authors": ["Maria Torres", "David Kim"],
    "abstract": (
        "Generative sequence models encode the biases present at the time "
        "of data collection."
    ),
    "published_date": "2017-01-14",
    "categories": ["cs.LG"],
}

_PAPER_C = {
    "title": "Probing small-scale anisotropy in stochastic backgrounds",
    "authors": ["Alice Nguyen"],
    "abstract": "A stochastic background study using novel correlation estimators.",
    "published_date": "2016-01-21",
    "categories": ["astro-ph.CO"],
}

# Pre-2007 slash-style ID — slash is preserved throughout the metadata path.
_OLD_STYLE_ID = "hep-th/9901001"
_PAPER_OLD = {
    "title": "Variational inference for deep latent variable models",
    "authors": ["Thomas Ward"],
    "abstract": "We derive a variational lower bound for a family of models.",
    "published_date": None,
    "categories": ["hep-th"],
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@patch("arxiv_lib.ingest.fetch_arxiv_metadata")
@patch("arxiv_lib.ingest.fetch_arxiv_metadata_s2")
def test_all_found_in_s2(mock_s2, mock_atom, app_db_con):
    """S2 returns all requested papers; Atom API is never called."""
    mock_s2.return_value = {
        "1601.00001": _PAPER_A,
        "1701.00002": _PAPER_B,
    }

    result = get_arxiv_metadata(["1601.00001", "1701.00002"])

    assert set(result.keys()) == {"1601.00001", "1701.00002"}
    assert result["1601.00001"]["title"] == _PAPER_A["title"]
    assert result["1601.00001"]["authors"] == _PAPER_A["authors"]
    assert result["1701.00002"]["title"] == _PAPER_B["title"]
    mock_atom.assert_not_called()


@patch("arxiv_lib.ingest.fetch_arxiv_metadata")
@patch("arxiv_lib.ingest.fetch_arxiv_metadata_s2")
def test_partial_s2_atom_fallback_and_missing(mock_s2, mock_atom, app_db_con):
    """S2 finds 2 of 4; Atom supplies 1 more (old-style ID); 1 absent from both."""
    mock_s2.return_value = {
        "1601.00001": _PAPER_A,
        "1701.00002": _PAPER_B,
    }
    # Atom knows the old-style ID but not the truly unknown paper
    mock_atom.return_value = {_OLD_STYLE_ID: _PAPER_OLD}

    result = get_arxiv_metadata(
        ["1601.00001", "1701.00002", _OLD_STYLE_ID, "9999.00000"]
    )

    # Three papers found; the unknown one is absent
    assert set(result.keys()) == {"1601.00001", "1701.00002", _OLD_STYLE_ID}

    # Atom called with exactly the two IDs that S2 missed
    (atom_ids,), _ = mock_atom.call_args
    assert set(atom_ids) == {_OLD_STYLE_ID, "9999.00000"}

    # Old-style key retains the slash (no underscore conversion in metadata path)
    assert result[_OLD_STYLE_ID]["title"] == _PAPER_OLD["title"]

    # S2 was sent the slash-form ID, not an underscore-converted one
    (s2_ids,), _ = mock_s2.call_args
    assert _OLD_STYLE_ID in s2_ids


@patch("arxiv_lib.ingest.fetch_arxiv_metadata")
@patch("arxiv_lib.ingest.fetch_arxiv_metadata_s2")
def test_s2_rate_limited_falls_back_to_atom(mock_s2, mock_atom, app_db_con):
    """When S2 raises (exhausted retries on 429), Atom is used for the full batch."""
    mock_s2.side_effect = RuntimeError(
        "Semantic Scholar API request failed: 429 Too Many Requests"
    )
    mock_atom.return_value = {
        "1601.00001": _PAPER_A,
        "1601.00003": _PAPER_C,
    }

    result = get_arxiv_metadata(["1601.00001", "1601.00003"])

    assert set(result.keys()) == {"1601.00001", "1601.00003"}
    assert result["1601.00001"]["title"] == _PAPER_A["title"]
    assert result["1601.00003"]["title"] == _PAPER_C["title"]

    # Atom called with the full batch
    (atom_ids,), _ = mock_atom.call_args
    assert set(atom_ids) == {"1601.00001", "1601.00003"}


@patch("arxiv_lib.ingest.fetch_arxiv_metadata")
@patch("arxiv_lib.ingest.fetch_arxiv_metadata_s2")
def test_cache_hit_bypasses_apis(mock_s2, mock_atom, app_db_con):
    """A paper already in the DB is returned from cache; only the new one hits the APIs."""
    write_to_arxiv_metadata_cache({"1601.00001": _PAPER_A})

    mock_s2.return_value = {"1701.00002": _PAPER_B}

    result = get_arxiv_metadata(["1601.00001", "1701.00002"])

    assert set(result.keys()) == {"1601.00001", "1701.00002"}
    assert result["1601.00001"]["title"] == _PAPER_A["title"]
    assert result["1701.00002"]["title"] == _PAPER_B["title"]

    # S2 asked for only the uncached paper (get_arxiv_metadata sorts IDs before fetching)
    (s2_ids,), _ = mock_s2.call_args
    assert s2_ids == ["1701.00002"]
    mock_atom.assert_not_called()
