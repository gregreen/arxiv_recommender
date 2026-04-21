"""
arXiv ID parsing and validation.

Supports both identifier styles:

  New-style (April 2007 onwards):  YYMM.NNNN   (4-digit seq, 0704–1503)
                                   YYMM.NNNNN  (5-digit seq, 1504–present)

  Old-style (before April 2007):   archive[.XX]/YYMMNNN
      e.g. hep-th/9802150, math.AG/0512103

Public API
----------
  validate_arxiv_id(raw, *, now=None) -> str
      Strips whitespace and version suffix, validates semantically,
      returns the clean canonical ID, or raises ValueError with a
      human-readable reason.
"""

import re
from datetime import date, timedelta

# ---------------------------------------------------------------------------
# Old-style archive names
# ---------------------------------------------------------------------------

# Archives that existed in the YYMMNNN era (before April 2007).
# The *top-level* name is the part before the optional ".XX" subcategory.
_OLD_STYLE_ARCHIVES: frozenset[str] = frozenset(
    {
        # Still-active top-level archives
        "astro-ph",
        "cond-mat",
        "cs",
        "gr-qc",
        "hep-ex",
        "hep-lat",
        "hep-ph",
        "hep-th",
        "math",
        "math-ph",
        "nlin",
        "nucl-ex",
        "nucl-th",
        "physics",
        "q-bio",
        "q-fin",
        "quant-ph",
        "stat",
        # Legacy single-topic archives (mostly folded into math/physics/cs)
        "acc-phys",
        "adap-org",
        "alg-geom",
        "ao-sci",
        "atom-ph",
        "bayes-an",
        "chao-dyn",
        "chem-ph",
        "comp-gas",
        "dg-ga",
        "funct-an",
        "mtrl-th",
        "patt-sol",
        "plasm-ph",
        "solv-int",
        "supr-con",
    }
)

# ---------------------------------------------------------------------------
# Compiled regexes
# ---------------------------------------------------------------------------

# New-style: YYMM.NNNN or YYMM.NNNNN, optional version suffix
_NEW_RE = re.compile(r"^(\d{2})(\d{2})\.(\d{4,5})(v\d+)?$")

# Old-style: archive[.XX]/YYMMNNN, optional version suffix
_OLD_RE = re.compile(r"^([a-z][a-z-]*)(\.[A-Z]{2})?/(\d{2})(\d{2})(\d{3})(v\d+)?$")


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def validate_arxiv_id(raw: str, *, now: date | None = None) -> str:
    """
    Validate a raw arXiv ID string (possibly with whitespace or version suffix).

    Parameters
    ----------
    raw : str
        User-supplied arXiv ID, e.g. "2309.06676", "2309.06676v2",
        "hep-th/9802150", "  0704.0001v1  ".
    now : date, optional
        Reference date for the "not in the future" check.  Defaults to
        ``date.today()``.  Inject a fixed date in tests.

    Returns
    -------
    str
        Clean canonical ID with whitespace and version suffix removed.

    Raises
    ------
    ValueError
        If the ID is malformed or semantically impossible (bad month, wrong
        digit count, date out of range, unknown archive, etc.).
    """
    if now is None:
        now = date.today()

    # Allow 2-day grace window to handle timezone skew at month boundaries.
    cutoff = now + timedelta(days=2)

    stripped = raw.strip()

    # --- Try new-style first ---
    m = _NEW_RE.match(stripped)
    if m:
        yy_s, mm_s, seq_s, _ver = m.groups()
        yy, mm = int(yy_s), int(mm_s)
        yymm = yy * 100 + mm  # integer YYMM — used for era/digit-count thresholds only

        if not (1 <= mm <= 12):
            raise ValueError(
                f"Invalid arXiv ID {raw!r}: month {mm:02d} is not in 01–12."
            )
        if yymm < 704:
            raise ValueError(
                f"Invalid arXiv ID {raw!r}: new-style IDs start at 0704 (April 2007); "
                f"got {yy_s}{mm_s}."
            )
        # New-style IDs are unambiguously 21st-century (started 2007), so reconstruct
        # a full date and compare directly — no integer encoding needed.
        if date(2000 + yy, mm, 1) > cutoff:
            raise ValueError(
                f"Invalid arXiv ID {raw!r}: date {yy_s}{mm_s} is in the future."
            )

        # Digit-count rule: 4 digits before 1504, 5 digits from 1504 onward.
        expected_digits = 4 if yymm < 1504 else 5
        if len(seq_s) != expected_digits:
            raise ValueError(
                f"Invalid arXiv ID {raw!r}: sequence number should have "
                f"{expected_digits} digit(s) for {yy_s}{mm_s}, got {len(seq_s)}."
            )
        if int(seq_s) == 0:
            raise ValueError(
                f"Invalid arXiv ID {raw!r}: sequence number must not be all zeros."
            )

        # Return without version suffix.
        return f"{yy_s}{mm_s}.{seq_s}"

    # --- Try old-style ---
    m = _OLD_RE.match(stripped)
    if m:
        archive, _subcat, yy_s, mm_s, seq_s, _ver = m.groups()
        yy, mm = int(yy_s), int(mm_s)
        yymm = yy * 100 + mm

        if archive not in _OLD_STYLE_ARCHIVES:
            raise ValueError(
                f"Invalid arXiv ID {raw!r}: unknown old-style archive {archive!r}."
            )
        if not (1 <= mm <= 12):
            raise ValueError(
                f"Invalid arXiv ID {raw!r}: month {mm:02d} is not in 01–12."
            )

        # Old-style era: 9101–9912 (1991–1999) and 0001–0703 (2000–March 2007).
        # yy >= 91 → 1900s; yy <= 7 → 2000s (yy == 8..90 is impossible in era).
        in_1900s = yy >= 91
        in_2000s = yy <= 7  # 00–07
        if not (in_1900s or in_2000s):
            raise ValueError(
                f"Invalid arXiv ID {raw!r}: old-style IDs span 1991–2007; "
                f"year component {yy_s!r} is out of range."
            )
        if in_1900s and yy == 91 and mm < 8:
            raise ValueError(
                f"Invalid arXiv ID {raw!r}: arXiv launched in August 1991; "
                f"got {yy_s}{mm_s}."
            )
        if in_2000s and yymm > 703:
            raise ValueError(
                f"Invalid arXiv ID {raw!r}: old-style IDs ended at 0703 (March 2007); "
                f"got {yy_s}{mm_s}."
            )
        if int(seq_s) == 0:
            raise ValueError(
                f"Invalid arXiv ID {raw!r}: sequence number must not be all zeros."
            )

        subcat = m.group(2) or ""
        return f"{archive}{subcat}/{yy_s}{mm_s}{seq_s}"

    raise ValueError(
        f"Invalid arXiv ID format: {raw!r}. "
        "Expected YYMM.NNNN[N] (new-style) or archive[.XX]/YYMMNNN (old-style)."
    )
