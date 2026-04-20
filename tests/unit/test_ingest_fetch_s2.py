"""Unit tests for fetch_arxiv_metadata_s2 (Semantic Scholar batch API parser).

Canned JSON fixtures are derived from the documented Semantic Scholar batch
API response format (https://api.semanticscholar.org/api-docs/) and are
consistent with real API responses.  All identifiable data (IDs, authors,
titles, abstracts) has been anonymised.

HTTP is mocked at arxiv_lib.ingest.requests.post so no network calls are made.
"""

from unittest.mock import MagicMock, patch

from arxiv_lib.ingest import fetch_arxiv_metadata_s2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(data) -> MagicMock:
    """Return a mock requests.Response whose .json() returns data."""
    mock = MagicMock()
    mock.json.return_value = data
    mock.raise_for_status.return_value = None
    return mock


# ---------------------------------------------------------------------------
# Canned JSON fixtures
# ---------------------------------------------------------------------------

# Two-paper batch — mirrors the real S2 batch API array response.
# Both entries have all fields populated.
_TWO_PAPER_S2 = [
    {
        "paperId": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
        "title": "Magnetic reconnection rates in low-density stellar coronae",
        "authors": [
            {"authorId": "111111", "name": "Jane Smith"},
            {"authorId": "222222", "name": "Robert Chen"},
        ],
        "abstract": (
            "We investigate magnetic reconnection in the outer layers of stellar "
            "coronae using a semi-analytic framework. Our model predicts energy "
            "release rates consistent with observed X-ray flare statistics across "
            "a sample of forty nearby main-sequence stars."
        ),
        "publicationDate": "2016-01-07",
        "year": 2016,
    },
    {
        "paperId": "b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3",
        "title": "Temporal bias and cultural memory in generative sequence models",
        "authors": [
            {"authorId": "333333", "name": "Maria Torres"},
            {"authorId": "444444", "name": "David Kim"},
            {"authorId": "555555", "name": "Elena Fischer"},
        ],
        "abstract": (
            "Generative sequence models learn distributions from historical corpora "
            "and therefore encode the biases present at the time of data collection. "
            "We formalise this temporal bias and propose an evaluation framework for "
            "measuring its magnitude in language and vision models."
        ),
        "publicationDate": "2017-01-14",
        "year": 2017,
    },
]

# One valid paper followed by null — S2 returns null when it does not know
# the paper.
_NULL_ENTRY_S2 = [
    {
        "paperId": "c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4",
        "title": "Probing small-scale anisotropy in stochastic backgrounds",
        "authors": [
            {"authorId": "666666", "name": "Alice Nguyen"},
        ],
        "abstract": "A stochastic background study using novel correlation estimators.",
        "publicationDate": "2016-01-21",
        "year": 2016,
    },
    None,
]

# Paper where S2 does not have a specific publication date but does have a year.
# This exercises the year-fallback branch in the parser.
_YEAR_FALLBACK_S2 = [
    {
        "paperId": "d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5",
        "title": "Variational inference for deep latent variable models",
        "authors": [
            {"authorId": "777777", "name": "Thomas Ward"},
            {"authorId": "888888", "name": "Priya Patel"},
        ],
        "abstract": (
            "We derive a variational lower bound for a family of deep latent "
            "variable models and show that the resulting estimator is unbiased "
            "and low-variance under standard regularity conditions."
        ),
        "publicationDate": None,
        "year": 2017,
    },
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@patch("arxiv_lib.ingest.requests.post")
def test_fetch_arxiv_metadata_s2_parses_fields(mock_post):
    """Parses a two-entry S2 batch response: keys, title, authors, abstract, date."""
    mock_post.return_value = _mock_response(_TWO_PAPER_S2)

    result = fetch_arxiv_metadata_s2(["1601.00001", "1701.00002"])

    assert set(result.keys()) == {"1601.00001", "1701.00002"}

    a = result["1601.00001"]
    assert a["title"] == "Magnetic reconnection rates in low-density stellar coronae"
    assert a["authors"] == ["Jane Smith", "Robert Chen"]
    assert "X-ray flare" in a["abstract"]
    assert a["published_date"] == "2016-01-07"

    b = result["1701.00002"]
    assert b["title"] == "Temporal bias and cultural memory in generative sequence models"
    assert b["authors"] == ["Maria Torres", "David Kim", "Elena Fischer"]
    assert b["published_date"] == "2017-01-14"


@patch("arxiv_lib.ingest.requests.post")
def test_fetch_arxiv_metadata_s2_null_entry_skipped(mock_post):
    """A null entry in the S2 response is silently skipped."""
    mock_post.return_value = _mock_response(_NULL_ENTRY_S2)

    result = fetch_arxiv_metadata_s2(["1601.00003", "1601.00004"])

    # Known paper is present
    assert "1601.00003" in result
    assert result["1601.00003"]["title"] == "Probing small-scale anisotropy in stochastic backgrounds"

    # Unknown paper (null entry) must not appear in the result at all
    assert "1601.00004" not in result


@patch("arxiv_lib.ingest.requests.post")
def test_fetch_arxiv_metadata_s2_year_fallback(mock_post):
    """When publicationDate is null, the year integer is used as published_date."""
    mock_post.return_value = _mock_response(_YEAR_FALLBACK_S2)

    result = fetch_arxiv_metadata_s2(["1701.00005"])

    assert result["1701.00005"]["published_date"] == "2017"
