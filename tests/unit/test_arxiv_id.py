"""
Unit tests for arxiv_lib/arxiv_id.py — validate_arxiv_id().

All tests inject a fixed reference date (2026-04-21) so they remain
deterministic regardless of when they are run.
"""

import pytest
from datetime import date

from arxiv_lib.arxiv_id import validate_arxiv_id

# Fixed reference date injected into all calls that exercise date-based checks.
_TODAY = date(2026, 4, 21)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok(raw: str, expected: str | None = None) -> None:
    """Assert that validate_arxiv_id accepts raw and returns expected (or raw if None)."""
    result = validate_arxiv_id(raw, now=_TODAY)
    assert result == (expected if expected is not None else raw)


def _bad(raw: str) -> None:
    """Assert that validate_arxiv_id raises ValueError for raw."""
    with pytest.raises(ValueError):
        validate_arxiv_id(raw, now=_TODAY)


# ---------------------------------------------------------------------------
# New-style IDs
# ---------------------------------------------------------------------------


class TestNewStyle:
    def test_valid_typical(self):
        _ok("2309.06676")

    def test_valid_first_new_style_id(self):
        _ok("0704.0001")

    def test_valid_last_4digit_month(self):
        # 1503 is the last month with 4-digit sequences
        _ok("1503.9999")

    def test_valid_first_5digit_month(self):
        # 1504 is the first month requiring 5-digit sequences
        _ok("1504.00001")

    def test_valid_5digit_recent(self):
        _ok("2601.12345")

    def test_valid_leading_zero_seq(self):
        # Leading zeros are fine as long as the number is > 0
        _ok("2309.00001")

    def test_invalid_month_zero(self):
        _bad("2300.06676")

    def test_invalid_month_13(self):
        _bad("2313.06676")

    def test_invalid_pre_new_style_era(self):
        # 0612 < 0704 — before new-style started
        _bad("0612.1234")

    def test_invalid_exact_boundary_below(self):
        # 0703 is one month before new-style started
        _bad("0703.0001")

    def test_valid_exact_boundary(self):
        # 0704 is exactly the first valid new-style YYMM
        _ok("0704.0001")

    def test_invalid_future(self):
        # Far future — well beyond cutoff
        _bad("9912.00001")

    def test_invalid_next_month(self):
        # One month ahead (2026-05), beyond the 2-day grace window in April
        _bad("2605.00001")

    def test_valid_grace_window(self):
        # 2026-04-19: today is Apr 21, so YYMM 2604 is this month — always valid
        _ok("2604.00001")

    def test_invalid_4digit_seq_after_1504(self):
        # 1504 requires 5 digits; 4-digit sequence is wrong
        _bad("1504.0001")

    def test_invalid_5digit_seq_before_1504(self):
        # 1503 requires 4 digits; 5-digit sequence is wrong
        _bad("1503.00001")

    def test_invalid_all_zero_seq_4digit(self):
        _bad("0704.0000")

    def test_invalid_all_zero_seq_5digit(self):
        _bad("2309.00000")

    def test_invalid_original_bug(self):
        # The paper that triggered this fix: 01 is before 2007, month 23 is
        # impossible, and 5-digit seq is wrong for the (hypothetical) YYMM 0123.
        _bad("0123.45678")

    def test_strips_whitespace(self):
        _ok("  2309.06676  ", expected="2309.06676")

    def test_strips_version_suffix(self):
        _ok("2309.06676v2", expected="2309.06676")

    def test_strips_version_and_whitespace(self):
        _ok("  2309.06676v10  ", expected="2309.06676")


# ---------------------------------------------------------------------------
# Old-style IDs
# ---------------------------------------------------------------------------


class TestOldStyle:
    def test_valid_hep_th(self):
        _ok("hep-th/9802150")

    def test_valid_with_subcategory(self):
        _ok("math.AG/0512103")

    def test_valid_astro_ph(self):
        _ok("astro-ph/0601001")

    def test_valid_cs_with_subcategory(self):
        _ok("cs.LG/0703001")

    def test_valid_earliest_plausible(self):
        # 9108 — August 1991, the month arXiv launched
        _ok("hep-th/9108001")

    def test_invalid_before_arxiv_launch(self):
        # arXiv launched August 1991; July and earlier are impossible
        _bad("hep-th/9107001")

    def test_invalid_january_1991(self):
        _bad("hep-th/9101001")

    def test_valid_last_valid_month(self):
        # 0703 — March 2007, last month of old-style
        _ok("hep-th/0703001")

    def test_invalid_unknown_archive(self):
        _bad("xyz/0512103")

    def test_invalid_month_zero(self):
        _bad("hep-th/9800150")

    def test_invalid_month_13(self):
        _bad("hep-th/9813150")

    def test_invalid_in_new_style_era(self):
        # 0704 is when new-style started — old-style IDs can't exist then
        _bad("hep-th/0704001")

    def test_invalid_year_range_mid(self):
        # yy=50 is neither >= 91 (1900s) nor <= 7 (2000s)
        _bad("hep-th/5001001")

    def test_invalid_year_range_high_2000s(self):
        # yy=10 is in the 2010s — beyond the old-style era
        _bad("hep-th/1001001")

    def test_invalid_all_zero_seq(self):
        _bad("hep-th/9801000")

    def test_strips_version_suffix(self):
        _ok("hep-th/9802150v1", expected="hep-th/9802150")

    def test_strips_whitespace(self):
        _ok("  hep-th/9802150  ", expected="hep-th/9802150")

    def test_subcategory_preserved_in_output(self):
        result = validate_arxiv_id("math.AG/0512103", now=_TODAY)
        assert result == "math.AG/0512103"


# ---------------------------------------------------------------------------
# Completely invalid formats
# ---------------------------------------------------------------------------


class TestInvalidFormat:
    def test_empty_string(self):
        _bad("")

    def test_only_whitespace(self):
        _bad("   ")

    def test_no_dot(self):
        _bad("23090001")

    def test_no_slash_old_style(self):
        _bad("hepth9802150")

    def test_letters_in_seq(self):
        _bad("2309.abc01")

    def test_too_many_digits(self):
        _bad("2309.123456")
