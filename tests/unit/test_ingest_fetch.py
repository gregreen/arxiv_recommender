"""Unit tests for fetch_arxiv_metadata (Atom API parser).

Canned XML is derived from a real two-paper API response to
https://export.arxiv.org/api/query?id_list=2309.06676,2309.12345&max_results=2
fetched on 2026-04-20.  All identifiable data (IDs, authors, titles,
abstracts, dates) has been anonymised.

HTTP is mocked at arxiv_lib.ingest.requests.get so no network calls are made.
"""

from unittest.mock import MagicMock, patch

from arxiv_lib.ingest import fetch_arxiv_metadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(xml_text: str) -> MagicMock:
    """Return a mock requests.Response whose .text is xml_text."""
    mock = MagicMock()
    mock.text = xml_text
    mock.raise_for_status.return_value = None
    return mock


# ---------------------------------------------------------------------------
# Canned XML fixtures
# ---------------------------------------------------------------------------

# Two-paper response — structure mirrors the real arXiv Atom API output.
# Paper A: astrophysics paper, 2 authors, 2 categories, version suffix v2.
# Paper B: CS paper, 3 authors, 2 categories, version suffix v1.
_TWO_PAPER_ATOM = """\
<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
      xmlns:arxiv="http://arxiv.org/schemas/atom"
      xmlns="http://www.w3.org/2005/Atom">
  <id>https://arxiv.org/api/fakequery</id>
  <title>arXiv Query: id_list=1601.00001,1701.00002</title>
  <updated>2016-02-01T00:00:00Z</updated>
  <opensearch:totalResults>2</opensearch:totalResults>
  <opensearch:startIndex>0</opensearch:startIndex>
  <opensearch:itemsPerPage>2</opensearch:itemsPerPage>

  <entry>
    <id>http://arxiv.org/abs/1601.00001v2</id>
    <title>Magnetic reconnection rates in low-density stellar coronae</title>
    <updated>2016-03-10T08:00:00Z</updated>
    <summary>
      We investigate magnetic reconnection in the outer layers of stellar
      coronae using a semi-analytic framework. Our model predicts energy
      release rates consistent with observed X-ray flare statistics across
      a sample of forty nearby main-sequence stars.
    </summary>
    <published>2016-01-07T12:00:00Z</published>
    <category term="astro-ph.SR" scheme="http://arxiv.org/schemas/atom"/>
    <category term="gr-qc" scheme="http://arxiv.org/schemas/atom"/>
    <arxiv:primary_category term="astro-ph.SR"/>
    <author><name>Jane Smith</name></author>
    <author><name>Robert Chen</name></author>
  </entry>

  <entry>
    <id>http://arxiv.org/abs/1701.00002v1</id>
    <title>Temporal bias and cultural memory in generative sequence models</title>
    <updated>2017-01-14T09:30:00Z</updated>
    <summary>
      Generative sequence models learn distributions from historical corpora
      and therefore encode the biases present at the time of data collection.
      We formalise this temporal bias and propose an evaluation framework for
      measuring its magnitude in language and vision models.
    </summary>
    <published>2017-01-14T09:30:00Z</published>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.HC" scheme="http://arxiv.org/schemas/atom"/>
    <arxiv:primary_category term="cs.LG"/>
    <author><name>Maria Torres</name></author>
    <author><name>David Kim</name></author>
    <author><name>Elena Fischer</name></author>
  </entry>
</feed>
"""

# Single-paper response where title and summary contain the embedded newlines
# and multi-space runs that the real API delivers (observed in the live response).
_WHITESPACE_ATOM = """\
<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
      xmlns:arxiv="http://arxiv.org/schemas/atom"
      xmlns="http://www.w3.org/2005/Atom">
  <id>https://arxiv.org/api/fakequery2</id>
  <title>arXiv Query: id_list=1601.00003</title>
  <updated>2016-02-01T00:00:00Z</updated>
  <opensearch:totalResults>1</opensearch:totalResults>
  <opensearch:startIndex>0</opensearch:startIndex>
  <opensearch:itemsPerPage>1</opensearch:itemsPerPage>

  <entry>
    <id>http://arxiv.org/abs/1601.00003v1</id>
    <title>Probing   small-scale
  anisotropy</title>
    <updated>2016-01-21T15:00:00Z</updated>
    <summary>
  A stochastic  background   study.
</summary>
    <published>2016-01-21T15:00:00Z</published>
    <category term="astro-ph.CO" scheme="http://arxiv.org/schemas/atom"/>
    <arxiv:primary_category term="astro-ph.CO"/>
    <author><name>Alice Nguyen</name></author>
  </entry>
</feed>
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@patch("arxiv_lib.ingest.requests.get")
def test_fetch_arxiv_metadata_two_papers(mock_get):
    """Parses a two-entry Atom response: keys, authors, categories, published_date."""
    mock_get.return_value = _mock_response(_TWO_PAPER_ATOM)

    result = fetch_arxiv_metadata(["1601.00001", "1701.00002"])

    # Both IDs present with version suffix stripped
    assert set(result.keys()) == {"1601.00001", "1701.00002"}

    a = result["1601.00001"]
    assert a["title"] == "Magnetic reconnection rates in low-density stellar coronae"
    assert a["authors"] == ["Jane Smith", "Robert Chen"]
    assert "X-ray flare" in a["abstract"]
    assert a["published_date"] == "2016-01-07T12:00:00Z"
    assert a["categories"] == ["astro-ph.SR", "gr-qc"]

    b = result["1701.00002"]
    assert b["title"] == "Temporal bias and cultural memory in generative sequence models"
    assert b["authors"] == ["Maria Torres", "David Kim", "Elena Fischer"]
    assert b["published_date"] == "2017-01-14T09:30:00Z"
    assert b["categories"] == ["cs.LG", "cs.HC"]


@patch("arxiv_lib.ingest.requests.get")
def test_fetch_arxiv_metadata_whitespace_normalised(mock_get):
    """Embedded newlines and multi-space runs in title/abstract are collapsed."""
    mock_get.return_value = _mock_response(_WHITESPACE_ATOM)

    result = fetch_arxiv_metadata(["1601.00003"])

    assert result["1601.00003"]["title"] == "Probing small-scale anisotropy"
    assert result["1601.00003"]["abstract"] == "A stochastic background study."
