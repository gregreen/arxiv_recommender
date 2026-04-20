"""Unit tests for fetch_oaipmh_metadata (OAI-PMH daily ingest parser).

Canned XML fixtures are derived from real OAI-PMH responses to:
  https://oaipmh.arxiv.org/oai?verb=ListRecords&from=2026-04-10&until=2026-04-10
      &metadataPrefix=arXiv&set=cs:cs:SC
  https://oaipmh.arxiv.org/oai?verb=ListRecords&from=2026-04-14&until=2026-04-14
      &metadataPrefix=arXiv&set=cs:cs:SC
fetched on 2026-04-20.  All identifiable data (IDs, authors, titles, abstracts)
has been anonymised.

HTTP is mocked at arxiv_lib.ingest.requests.get so no network calls are made.

NOTE ON published_date:
  The parser converts the query date (midnight US Eastern) to UTC.
  For 2026-04-10, Eastern Daylight Time is UTC-4, so midnight EDT = 04:00 UTC
  → published_date == "2026-04-10T04:00:00Z".
"""

from unittest.mock import MagicMock, call, patch

from arxiv_lib.ingest import fetch_oaipmh_metadata

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OAI_NS = "http://www.openarchives.org/OAI/2.0/"


def _mock_xml_response(xml_text: str) -> MagicMock:
    """Return a mock requests.Response whose .content is xml_text encoded."""
    mock = MagicMock()
    mock.content = xml_text.encode()
    mock.raise_for_status.return_value = None
    return mock


# ---------------------------------------------------------------------------
# Canned XML fixtures
# ---------------------------------------------------------------------------

# One-page response with two new-submission records.
# Paper A: 2 authors (keyname + forenames).
# Paper B: 1 author with a name suffix.
# Both have <created> within 7 days of the query date 2026-04-10.
_TWO_RECORD_OAI = """\
<?xml version='1.0' encoding='UTF-8'?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <responseDate>2026-04-20T04:19:47Z</responseDate>
  <request verb="ListRecords" metadataPrefix="arXiv" from="2026-04-10"
           until="2026-04-10" set="cs:cs:LG">http://oaipmh.arxiv.org/oai</request>
  <ListRecords>
    <record>
      <header>
        <identifier>oai:arXiv.org:2604.00001</identifier>
        <datestamp>2026-04-10</datestamp>
        <setSpec>cs:cs:LG</setSpec>
      </header>
      <metadata>
        <arXiv xmlns="http://arxiv.org/OAI/arXiv/">
          <id>2604.00001</id>
          <created>2026-04-09</created>
          <title>Magnetic reconnection rates in low-density stellar coronae</title>
          <authors>
            <author>
              <keyname>Smith</keyname>
              <forenames>Jane</forenames>
            </author>
            <author>
              <keyname>Chen</keyname>
              <forenames>Robert</forenames>
            </author>
          </authors>
          <categories>cs.LG cs.MS</categories>
          <abstract>We investigate magnetic reconnection in the outer layers of
stellar coronae using a semi-analytic framework.</abstract>
        </arXiv>
      </metadata>
    </record>
    <record>
      <header>
        <identifier>oai:arXiv.org:2604.00002</identifier>
        <datestamp>2026-04-10</datestamp>
        <setSpec>cs:cs:LG</setSpec>
      </header>
      <metadata>
        <arXiv xmlns="http://arxiv.org/OAI/arXiv/">
          <id>2604.00002</id>
          <created>2026-04-09</created>
          <title>Temporal bias and cultural memory in generative sequence models</title>
          <authors>
            <author>
              <keyname>Ward</keyname>
              <forenames>Thomas</forenames>
              <suffix>Jr</suffix>
            </author>
          </authors>
          <categories>cs.LG</categories>
          <abstract>Generative sequence models learn distributions from historical
corpora and therefore encode the biases present at the time of data collection.</abstract>
        </arXiv>
      </metadata>
    </record>
  </ListRecords>
</OAI-PMH>
"""

# Response where the single record has <created> 9 days before the query date
# (2026-04-01 vs. query date 2026-04-10).  The parser's 7-day window should
# reject it as a replacement/update, returning an empty result.
_OLD_RECORD_OAI = """\
<?xml version='1.0' encoding='UTF-8'?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <responseDate>2026-04-20T04:20:00Z</responseDate>
  <request verb="ListRecords" metadataPrefix="arXiv" from="2026-04-10"
           until="2026-04-10" set="cs:cs:LG">http://oaipmh.arxiv.org/oai</request>
  <ListRecords>
    <record>
      <header>
        <identifier>oai:arXiv.org:2601.99999</identifier>
        <datestamp>2026-04-10</datestamp>
        <setSpec>cs:cs:LG</setSpec>
      </header>
      <metadata>
        <arXiv xmlns="http://arxiv.org/OAI/arXiv/">
          <id>2601.99999</id>
          <created>2026-04-01</created>
          <title>An old paper whose metadata was just updated</title>
          <authors>
            <author>
              <keyname>Older</keyname>
              <forenames>Author</forenames>
            </author>
          </authors>
          <categories>cs.LG</categories>
          <abstract>This paper was actually submitted weeks ago.</abstract>
        </arXiv>
      </metadata>
    </record>
  </ListRecords>
</OAI-PMH>
"""

# OAI-PMH error response: noRecordsMatch (weekend, holiday, or future date).
_NO_RECORDS_OAI = """\
<?xml version='1.0' encoding='UTF-8'?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <responseDate>2026-04-20T04:19:06Z</responseDate>
  <request verb="ListRecords" metadataPrefix="arXiv" from="2026-04-15"
           until="2026-04-15" set="cs:cs:LG">http://oaipmh.arxiv.org/oai</request>
  <error code="noRecordsMatch">The combination of the values of the from, until,
set and metadataPrefix arguments results in an empty list.</error>
</OAI-PMH>
"""

# Page 1 of a two-page response.  Contains paper A and a resumptionToken.
_PAGE_1_OAI = """\
<?xml version='1.0' encoding='UTF-8'?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <responseDate>2026-04-20T04:21:00Z</responseDate>
  <request verb="ListRecords" metadataPrefix="arXiv" from="2026-04-10"
           until="2026-04-10" set="cs:cs:LG">http://oaipmh.arxiv.org/oai</request>
  <ListRecords>
    <record>
      <header>
        <identifier>oai:arXiv.org:2604.00001</identifier>
        <datestamp>2026-04-10</datestamp>
        <setSpec>cs:cs:LG</setSpec>
      </header>
      <metadata>
        <arXiv xmlns="http://arxiv.org/OAI/arXiv/">
          <id>2604.00001</id>
          <created>2026-04-09</created>
          <title>Magnetic reconnection rates in low-density stellar coronae</title>
          <authors>
            <author>
              <keyname>Smith</keyname>
              <forenames>Jane</forenames>
            </author>
          </authors>
          <categories>cs.LG</categories>
          <abstract>Abstract for the first paper.</abstract>
        </arXiv>
      </metadata>
    </record>
    <resumptionToken>tok-abc123</resumptionToken>
  </ListRecords>
</OAI-PMH>
"""

# Page 2: paper B, no further resumption token.
_PAGE_2_OAI = """\
<?xml version='1.0' encoding='UTF-8'?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <responseDate>2026-04-20T04:21:05Z</responseDate>
  <request verb="ListRecords" resumptionToken="tok-abc123">http://oaipmh.arxiv.org/oai</request>
  <ListRecords>
    <record>
      <header>
        <identifier>oai:arXiv.org:2604.00002</identifier>
        <datestamp>2026-04-10</datestamp>
        <setSpec>cs:cs:LG</setSpec>
      </header>
      <metadata>
        <arXiv xmlns="http://arxiv.org/OAI/arXiv/">
          <id>2604.00002</id>
          <created>2026-04-09</created>
          <title>Temporal bias and cultural memory in generative sequence models</title>
          <authors>
            <author>
              <keyname>Torres</keyname>
              <forenames>Maria</forenames>
            </author>
          </authors>
          <categories>cs.LG</categories>
          <abstract>Abstract for the second paper.</abstract>
        </arXiv>
      </metadata>
    </record>
    <resumptionToken/>
  </ListRecords>
</OAI-PMH>
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@patch("arxiv_lib.ingest.requests.get")
def test_fetch_oaipmh_parses_records(mock_get):
    """Parses a two-record OAI-PMH page: IDs, authors (with suffix), categories, published_date."""
    mock_get.return_value = _mock_xml_response(_TWO_RECORD_OAI)

    result = fetch_oaipmh_metadata("2026-04-10", "cs.LG")

    assert set(result.keys()) == {"2604.00001", "2604.00002"}

    a = result["2604.00001"]
    assert a["title"] == "Magnetic reconnection rates in low-density stellar coronae"
    assert a["authors"] == ["Jane Smith", "Robert Chen"]
    assert a["categories"] == ["cs.LG", "cs.MS"]
    # Midnight US/Eastern on 2026-04-10 (EDT = UTC-4) → 04:00 UTC
    assert a["published_date"] == "2026-04-10T04:00:00Z"

    b = result["2604.00002"]
    # Suffix ("Jr") must be appended after the name
    assert b["authors"] == ["Thomas Ward Jr"]
    assert b["categories"] == ["cs.LG"]
    assert b["published_date"] == "2026-04-10T04:00:00Z"


@patch("arxiv_lib.ingest.requests.get")
def test_fetch_oaipmh_skips_old_created_dates(mock_get):
    """Records whose <created> date is outside the 7-day window are filtered out."""
    mock_get.return_value = _mock_xml_response(_OLD_RECORD_OAI)

    # created=2026-04-01 is 9 days before query date 2026-04-10 → rejected
    result = fetch_oaipmh_metadata("2026-04-10", "cs.LG")

    assert result == {}


@patch("arxiv_lib.ingest.requests.get")
def test_fetch_oaipmh_no_records_match(mock_get):
    """A noRecordsMatch OAI-PMH error returns {} without raising."""
    mock_get.return_value = _mock_xml_response(_NO_RECORDS_OAI)

    result = fetch_oaipmh_metadata("2026-04-15", "cs.LG")

    assert result == {}
    assert mock_get.call_count == 1


@patch("arxiv_lib.ingest.requests.get")
def test_fetch_oaipmh_follows_resumption_token(mock_get):
    """Pagination via resumptionToken: fetches page 2 and merges both pages."""
    mock_get.side_effect = [
        _mock_xml_response(_PAGE_1_OAI),
        _mock_xml_response(_PAGE_2_OAI),
    ]

    result = fetch_oaipmh_metadata("2026-04-10", "cs.LG")

    # Both pages' papers must appear in the merged result
    assert set(result.keys()) == {"2604.00001", "2604.00002"}

    # The second HTTP call must use the resumption token (no other params)
    assert mock_get.call_count == 2
    _, second_kwargs = mock_get.call_args_list[1]
    assert second_kwargs["params"] == {
        "verb": "ListRecords",
        "resumptionToken": "tok-abc123",
    }
