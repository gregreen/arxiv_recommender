"""
Ingest pipeline: fetch metadata, download LaTeX, summarise, embed.

All functions that touch external APIs (arXiv, Semantic Scholar, HuggingFace)
or the local file caches live here.  Import this module instead of using the
monolithic experiments/arxiv_embedding.py in production code.
"""

import glob
import os
import re
import json
import sqlite3
import logging
import urllib.request

import requests
import xml.etree.ElementTree as ET
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    wait_random,
    retry_if_exception,
    before_sleep_log,
)

from huggingface_hub import InferenceClient
from arxiv_to_prompt import process_latex_source, count_tokens

import numpy as np

from arxiv_lib.config import (
    EMBEDDING_CACHE_DB,
    EMBEDDING_CACHE_FILE,
    APP_DB_PATH,
    METADATA_CACHE_DIR,
    SOURCE_CACHE_DIR,
    SUMMARY_CACHE_DIR,
    TOKENS_FILE,
    USER_AGENT,
    ARXIV_CATEGORIES,
    SUMMARY_PROVIDER,
    SUMMARY_MODEL,
    SUMMARY_MAX_TOKENS,
    EMBEDDING_PROVIDER,
    EMBEDDING_MODEL,
    EMBEDDING_MAX_TOKENS,
)


# ---------------------------------------------------------------------------
# Embedding DB
# ---------------------------------------------------------------------------

_embedding_db_initialized = False


def _init_embedding_db() -> None:
    """Create the embeddings table if needed, enable WAL, and auto-migrate from legacy .npz."""
    global _embedding_db_initialized
    if _embedding_db_initialized:
        return
    con = sqlite3.connect(EMBEDDING_CACHE_DB)
    try:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute(
            "CREATE TABLE IF NOT EXISTS embeddings "
            "(arxiv_id TEXT PRIMARY KEY, vector BLOB NOT NULL)"
        )
        con.commit()
        # One-time migration from legacy .npz file
        if (os.path.exists(EMBEDDING_CACHE_FILE) and
                con.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 0):
            print(f"Migrating {EMBEDDING_CACHE_FILE} -> {EMBEDDING_CACHE_DB} ...")
            data = np.load(EMBEDDING_CACHE_FILE, allow_pickle=False)
            rows = [(k, data[k].astype(np.float32).tobytes()) for k in data.files]
            con.executemany("INSERT OR IGNORE INTO embeddings VALUES (?, ?)", rows)
            con.commit()
            print(f"  Migrated {len(rows)} embeddings.")
    finally:
        con.close()
    _embedding_db_initialized = True


def load_embedding_cache() -> dict[str, np.ndarray]:
    """Load all embeddings from the cache DB. Returns a dict mapping arXiv ID to embedding vector."""
    _init_embedding_db()
    with sqlite3.connect(EMBEDDING_CACHE_DB) as con:
        rows = con.execute("SELECT arxiv_id, vector FROM embeddings").fetchall()
    return {aid: np.frombuffer(blob, dtype=np.float32) for aid, blob in rows}


def save_embedding_cache(cache: dict[str, np.ndarray]) -> None:
    """Bulk-upsert a dict of embeddings into the cache DB."""
    _init_embedding_db()
    rows = [(aid, vec.astype(np.float32).tobytes()) for aid, vec in cache.items()]
    with sqlite3.connect(EMBEDDING_CACHE_DB) as con:
        con.executemany("INSERT OR REPLACE INTO embeddings VALUES (?, ?)", rows)


# ---------------------------------------------------------------------------
# Metadata cache (monthly JSON bundles)
# ---------------------------------------------------------------------------

def load_from_arxiv_metadata_cache(arxiv_ids: list[str]) -> dict:
    """
    Returns metadata for the given arXiv IDs from the ``papers`` table in app.db.

    IDs not present in the database are omitted from the result.
    """
    if len(arxiv_ids) == 0:
        return {}

    from arxiv_lib.appdb import get_connection
    with get_connection(APP_DB_PATH) as con:
        placeholders = ",".join("?" * len(arxiv_ids))
        rows = con.execute(
            f"SELECT arxiv_id, title, abstract, authors "
            f"FROM papers WHERE arxiv_id IN ({placeholders})",
            list(arxiv_ids),
        ).fetchall()

    return {
        row["arxiv_id"]: {
            "title":    row["title"]    or "",
            "abstract": row["abstract"] or "",
            "authors":  json.loads(row["authors"]) if row["authors"] else [],
        }
        for row in rows
    }


def write_to_arxiv_metadata_cache(metadata_dict: dict[str, dict]) -> None:
    """
    Upsert metadata for the given arXiv IDs into the ``papers`` table in app.db.

    Only the metadata columns (title, abstract, authors) are written;
    published_date and categories are left NULL if not present in the input.
    Uses INSERT OR REPLACE so re-fetching a paper updates its stored metadata.

    The input dict should map arXiv ID to a dict with at least the keys
    ``title``, ``authors`` (list), and ``abstract``.
    """
    if not metadata_dict:
        return
    from arxiv_lib.appdb import get_connection
    rows = [
        (
            aid,
            meta.get("title"),
            meta.get("abstract"),
            json.dumps(meta["authors"]) if "authors" in meta else None,
            meta.get("published_date"),
            json.dumps(meta["categories"]) if "categories" in meta else None,
        )
        for aid, meta in metadata_dict.items()
    ]
    with get_connection(APP_DB_PATH) as con:
        con.executemany(
            """
            INSERT OR REPLACE INTO papers
                (arxiv_id, title, abstract, authors, published_date, categories)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def get_arxiv_metadata(arxiv_ids: list[str],
                       s2_token: str | None = None,
                       batch_size: int = 256) -> dict[str, dict]:
    """
    Returns cached metadata for the given arXiv IDs, fetching and caching any
    that are not already stored.

    Metadata is persisted in the ``papers`` table in app.db.

    Parameters
    ----------
    arxiv_ids : list[str]
        e.g. ["2309.06676", "2309.12345"].
    s2_token : str | None
        Semantic Scholar API token, if available.
    batch_size : int
        Maximum number of papers to fetch in each API call.

    Returns
    -------
    dict[str, dict]
        Maps each arXiv ID to its {"title", "authors", "abstract"} dict.
        IDs that could not be fetched are omitted.
    """
    result = load_from_arxiv_metadata_cache(arxiv_ids)
    arxiv_ids = [aid for aid in arxiv_ids if aid not in result]

    if not arxiv_ids:
        return result

    # Chronological order optimises cache write locality (papers grouped by month)
    arxiv_ids = sorted(arxiv_ids)

    for i0 in range(0, len(arxiv_ids), batch_size):
        batch_ids = arxiv_ids[i0:i0 + batch_size]

        print(f"Fetching metadata for {len(batch_ids)} papers.")
        try:
            fetched = fetch_arxiv_metadata_s2(batch_ids, s2_api_key=s2_token)
        except Exception as e:
            print(f"  S2 fetch failed ({e}); falling back to arXiv Atom API.")
            fetched = {}

        missed = [aid for aid in batch_ids if aid not in fetched]
        if missed:
            print(f"  Fetching {len(missed)} papers not found in S2 from arXiv Atom API.")
            try:
                fetched.update(fetch_arxiv_metadata(missed))
            except Exception as e:
                print(f"  Atom fallback failed ({e}); skipping {len(missed)} unfound papers.")

        write_to_arxiv_metadata_cache(fetched)
        result.update(fetched)

    return result


# ---------------------------------------------------------------------------
# Metadata fetchers
# ---------------------------------------------------------------------------

def fetch_arxiv_metadata(arxiv_ids: list[str]) -> dict[str, dict]:
    """
    Returns metadata for each arXiv ID via the official arXiv Atom API (batched).

    Returned dict keys per paper: title, authors, abstract, published_date, categories.

    Raises RuntimeError on unrecoverable HTTP failure.
    """
    _NS = "http://www.w3.org/2005/Atom"
    _ARXIV_NS = "http://arxiv.org/schemas/atom"
    _logger = logging.getLogger(__name__)

    def _is_retryable(exc: BaseException) -> bool:
        return (
            isinstance(exc, requests.HTTPError)
            and exc.response.status_code in (429, 500, 502, 503, 504)
        )

    @retry(
        stop=stop_after_attempt(7),
        wait=wait_exponential(multiplier=2, min=4, max=64) + wait_random(0, 1),
        retry=retry_if_exception(_is_retryable),
        before_sleep=before_sleep_log(_logger, logging.WARNING),
        reraise=True,
    )
    def _get() -> requests.Response:
        url = (
            "https://export.arxiv.org/api/query"
            f"?id_list={','.join(arxiv_ids)}&max_results={len(arxiv_ids)}"
        )
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        r.raise_for_status()
        return r

    try:
        resp = _get()
    except requests.HTTPError as e:
        raise RuntimeError(f"arXiv API request failed: {e}")

    root = ET.fromstring(resp.text)
    results = {}
    for entry in root.findall(f"{{{_NS}}}entry"):
        id_el = entry.find(f"{{{_NS}}}id")
        if id_el is None or not id_el.text:
            continue
        aid = re.sub(r"v\d+$", "", id_el.text.strip().split("/abs/")[-1])

        title_el     = entry.find(f"{{{_NS}}}title")
        summary_el   = entry.find(f"{{{_NS}}}summary")
        author_els   = entry.findall(f"{{{_NS}}}author")
        published_el = entry.find(f"{{{_NS}}}published")
        category_els = entry.findall(f"{{{_NS}}}category")

        title    = " ".join((title_el.text   or "").split()) if title_el   is not None else ""
        abstract = " ".join((summary_el.text or "").split()) if summary_el is not None else ""
        authors  = []
        for a in author_els:
            name_el = a.find(f"{{{_NS}}}name")
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())

        # published: full ISO 8601 timestamp, e.g. "2024-10-01T14:12:45Z"
        published_date = None
        if published_el is not None and published_el.text:
            published_date = published_el.text.strip()

        # categories: term attribute on each <category> element
        categories = [
            el.get("term", "") for el in category_els
            if el.get("term", "")
        ]

        results[aid] = {
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "published_date": published_date,
            "categories": categories,
        }

    return results


def fetch_arxiv_metadata_html(arxiv_id: str) -> dict:
    """
    Scrapes title, authors, and abstract from the arXiv abstract page HTML.

    Fallback for cases where the Atom API is unavailable or returns no result.

    Raises RuntimeError if the page cannot be fetched or parsed.
    """
    html_url = f"https://arxiv.org/abs/{arxiv_id}"
    resp = requests.get(html_url, headers={"User-Agent": USER_AGENT}, timeout=30)
    try:
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"arXiv abstract page request failed for {arxiv_id}: {e}")

    html = resp.text

    title_match = re.search(
        r'<h1[^>]*class="[^"]*title[^"]*"[^>]*>(?:Title:)?\s*(.*?)</h1>',
        html, re.DOTALL | re.IGNORECASE
    )
    title = re.sub(r"<[^>]+>", "", title_match.group(1)).strip() if title_match else ""

    author_matches = re.findall(
        r'<a[^>]+href="/search/[^"]*"[^>]*>(.*?)</a>',
        html, re.DOTALL
    )
    authors = [re.sub(r"<[^>]+>", "", a).strip() for a in author_matches if a.strip()]

    abstract_match = re.search(
        r'<blockquote[^>]*class="[^"]*abstract[^"]*"[^>]*>(?:Abstract:)?\s*(.*?)</blockquote>',
        html, re.DOTALL | re.IGNORECASE
    )
    abstract = re.sub(r"<[^>]+>", "", abstract_match.group(1)).strip() if abstract_match else ""
    abstract = " ".join(abstract.split())

    if not title:
        raise RuntimeError(f"Could not retrieve metadata for arXiv ID {arxiv_id!r}")

    return {"title": title, "authors": authors, "abstract": abstract}


def fetch_arxiv_metadata_s2(
    arxiv_ids: list[str],
    s2_api_key: str | None = None,
) -> dict[str, dict]:
    """
    Returns title, authors, abstract, and published_date for each arXiv ID via
    the Semantic Scholar Graph batch API.  `published_date` is YYYY-MM-DD when
    available from S2, or a 4-digit year string as fallback.  May be None for
    very old papers.

    Rate limits:
      - Without key : ~1 req/sec, 5,000 req/day
      - With free key: same per-second, higher daily quota (~10,000+/day)

    Raises RuntimeError on unrecoverable HTTP failure.
    """
    S2_BATCH_URL  = "https://api.semanticscholar.org/graph/v1/paper/batch"
    S2_FIELDS     = "title,authors,abstract,externalIds,publicationDate,year"
    S2_BATCH_SIZE = 500

    headers: dict[str, str] = {"User-Agent": USER_AGENT}
    if s2_api_key:
        headers["x-api-key"] = s2_api_key

    results: dict[str, dict] = {}
    _logger = logging.getLogger(__name__)

    def _is_retryable(exc: BaseException) -> bool:
        if isinstance(exc, requests.exceptions.ConnectionError):
            return True
        return (
            isinstance(exc, requests.HTTPError)
            and exc.response.status_code in (429, 500, 502, 503, 504)
        )

    @retry(
        stop=stop_after_attempt(7),
        wait=wait_exponential(multiplier=2, min=2, max=64) + wait_random(0, 1),
        retry=retry_if_exception(_is_retryable),
        before_sleep=before_sleep_log(_logger, logging.WARNING),
        reraise=True,
    )
    def _post_chunk(s2_ids: list[str]) -> requests.Response:
        resp = requests.post(
            S2_BATCH_URL,
            params={"fields": S2_FIELDS},
            headers=headers,
            json={"ids": s2_ids},
            timeout=30,
        )
        resp.raise_for_status()
        return resp

    for start in range(0, len(arxiv_ids), S2_BATCH_SIZE):
        chunk  = arxiv_ids[start: start + S2_BATCH_SIZE]
        s2_ids = [f"arXiv:{aid}" for aid in chunk]

        try:
            resp = _post_chunk(s2_ids)
        except requests.HTTPError as e:
            raise RuntimeError(f"Semantic Scholar API request failed: {e}")

        for arxiv_id, paper in zip(chunk, resp.json()):
            if paper is None:
                continue
            title    = " ".join((paper.get("title")    or "").split())
            abstract = " ".join((paper.get("abstract") or "").split())
            authors  = [a["name"] for a in (paper.get("authors") or [])]
            # publicationDate is YYYY-MM-DD when available; fall back to year-only string
            pub_date: str | None = paper.get("publicationDate") or None
            if pub_date is None and paper.get("year"):
                pub_date = str(paper["year"])
            results[arxiv_id] = {
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "published_date": pub_date,
            }

    return results


# ---------------------------------------------------------------------------
# arXiv category validation and mailing list
# ---------------------------------------------------------------------------

def raise_on_arxiv_category(category: str) -> None:
    """Raise ValueError if the category is not in the configured ARXIV_CATEGORIES set."""
    if category not in ARXIV_CATEGORIES:
        raise ValueError(
            f"Unsupported arXiv category: {category!r}. "
            f"Add it to ARXIV_CATEGORIES in arxiv_lib/config.py to enable it."
        )


def fetch_latest_mailing_ids(category: str) -> list[str]:
    """
    Fetch arXiv IDs for all new papers in the specified category from the most
    recent mailing by scraping the arXiv new-submissions listing page.

    Returns a deduplicated list of arXiv IDs in announcement order.
    """
    raise_on_arxiv_category(category)

    url  = f"https://arxiv.org/list/{category}/new"
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    try:
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch latest mailing for {category}: {e}")

    ids = re.findall(r'arXiv:(\d{4}\.\d{4,5})', resp.text)
    return list(dict.fromkeys(ids))  # deduplicate, preserve order


# ---------------------------------------------------------------------------
# LaTeX source cache
# ---------------------------------------------------------------------------

def sanitize_old_style_arxiv_id(arxiv_id: str) -> str:
    """Convert old-style arXiv IDs with slashes to the underscore format used in filenames."""
    return arxiv_id.replace(r'/', '_')


def load_cached_arxiv_source(arxiv_id: str) -> str | None:
    """Returns the cached LaTeX source for the given arXiv ID, or None if not cached."""
    # Old-style arXiv IDs with slashes are stored with slashes replaced by underscores
    arxiv_id_clean = sanitize_old_style_arxiv_id(arxiv_id)
    cache_fname = os.path.join(SOURCE_CACHE_DIR, f"{arxiv_id_clean}.tex")
    if os.path.exists(cache_fname):
        with open(cache_fname, "r", encoding="utf-8") as f:
            return f.read()
    return None

def get_arxiv_source(arxiv_id: str) -> str:
    """
    Returns the processed LaTeX source for a given arXiv ID, fetching and
    caching it if not already present.
    """
    latex = load_cached_arxiv_source(arxiv_id)
    if latex is not None:
        return latex

    latex = process_latex_source(arxiv_id, keep_comments=False)
    if latex is None:
        latex = "No LaTeX source available for this paper."
        print(f"Warning: Unable to fetch/read {arxiv_id}!")

    # arXiv-to-prompt renames old-style IDs like hep-th/9901001 to hep-th_9901001
    arxiv_id_clean = sanitize_old_style_arxiv_id(arxiv_id)
    cache_fname = os.path.join(SOURCE_CACHE_DIR, f"{arxiv_id_clean}.tex")
    with open(cache_fname, "w", encoding="utf-8") as f:
        f.write(latex)

    return latex


def compress_latex_whitespace(latex: str) -> str:
    """
    Reduces excessive whitespace in a LaTeX source string:
    - strips trailing whitespace from each line
    - collapses runs of spaces/tabs within a line to a single space
    - removes blank lines entirely
    """
    lines = latex.splitlines()
    lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in lines]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def report_compression_stats(max_tokens: int = EMBEDDING_MAX_TOKENS) -> None:
    """
    Runs compress_latex_whitespace on every cached .tex file in SOURCE_CACHE_DIR
    and prints per-file and aggregate compression statistics.
    """
    tex_files = sorted(glob.glob(os.path.join(SOURCE_CACHE_DIR, "*.tex")))
    if not tex_files:
        print("No cached .tex files found.")
        return

    total_orig_chars  = 0
    total_comp_chars  = 0
    total_orig_tokens = 0
    total_comp_tokens = 0
    still_truncated   = 0

    print(f"{'File':<25} {'Orig chars':>12} {'Comp chars':>12} {'Ratio':>7} "
          f"{'Orig tok':>10} {'Comp tok':>10} {'Trunc?':>7}")
    print("-" * 85)

    for path in tex_files:
        with open(path, "r", encoding="utf-8") as f:
            original = f.read()

        compressed  = compress_latex_whitespace(original)
        orig_chars  = len(original)
        comp_chars  = len(compressed)
        ratio       = comp_chars / orig_chars if orig_chars else 1.0
        orig_tokens = count_tokens(original)
        comp_tokens = count_tokens(compressed)
        truncated   = comp_tokens > max_tokens
        if truncated:
            still_truncated += 1

        fname = os.path.basename(path)
        print(f"{fname:<25} {orig_chars:>12,} {comp_chars:>12,} {ratio:>7.1%} "
              f"{orig_tokens:>10,} {comp_tokens:>10,} {'YES' if truncated else 'no':>7}")

        total_orig_chars  += orig_chars
        total_comp_chars  += comp_chars
        total_orig_tokens += orig_tokens
        total_comp_tokens += comp_tokens

    n             = len(tex_files)
    overall_ratio = total_comp_chars / total_orig_chars if total_orig_chars else 1.0
    print("-" * 85)
    print(f"\nFiles processed : {n}")
    print(f"Total chars     : {total_orig_chars:,} → {total_comp_chars:,} "
          f"(saved {total_orig_chars - total_comp_chars:,}, "
          f"{1 - overall_ratio:.1%} reduction)")
    print(f"Total tokens    : {total_orig_tokens:,} → {total_comp_tokens:,} "
          f"(saved {total_orig_tokens - total_comp_tokens:,})")
    print(f"Still truncated : {still_truncated}/{n} files exceed "
          f"{max_tokens:,} tokens after compression")


# ---------------------------------------------------------------------------
# LLM summarisation
# ---------------------------------------------------------------------------

def summarize_arxiv_paper(
    arxiv_id: str,
    hf_token: str,
    provider: str = SUMMARY_PROVIDER,
    model: str = SUMMARY_MODEL,
    max_tokens: int = SUMMARY_MAX_TOKENS,
) -> str:
    """
    Produces a structured summary of an arXiv paper in a single LLM call.

    Feeds the full LaTeX source together with the paper's title, author list,
    and abstract.  Returns seven labelled sections:
      Keywords / Scientific Questions / Data / Methods / Results /
      Conclusions / Key takeaway

    Results are cached in SUMMARY_CACHE_DIR.

    Parameters
    ----------
    arxiv_id : str
    hf_token : str
        HuggingFace API token.
    provider : str
        InferenceClient provider (default from config: SUMMARY_PROVIDER).
    model : str
        Chat-completion model (default from config: SUMMARY_MODEL).
    max_tokens : int
        LaTeX token budget before truncation (default from config: SUMMARY_MAX_TOKENS).
        Context limit note: Novita + Qwen3-Next-80B-A3B-Thinking fails above ~110K tokens.
        96*1024 leaves ~3K headroom for metadata and ~16K for output.

    Returns
    -------
    str
        Structured summary, also persisted to cache.
    """
    os.makedirs(SUMMARY_CACHE_DIR, exist_ok=True)
    arxiv_id_clean = sanitize_old_style_arxiv_id(arxiv_id)
    cache_file = os.path.join(SUMMARY_CACHE_DIR, f"{arxiv_id_clean}.txt")
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return f.read()

    # --- Fetch paper data -------------------------------------------------
    metadata = get_arxiv_metadata([arxiv_id])[arxiv_id]
    title    = metadata["title"]
    authors  = (
        ", ".join(metadata["authors"][:24]) + " et al."
        if len(metadata["authors"]) > 32
        else ", ".join(metadata["authors"])
    )
    abstract = metadata["abstract"]

    raw_latex = get_arxiv_source(arxiv_id)
    raw_latex = compress_latex_whitespace(raw_latex)

    n_tok = count_tokens(raw_latex)
    print(f'Estimated number of tokens: {n_tok}')
    if n_tok > max_tokens:
        chars_per_token = len(raw_latex) / max(n_tok, 1)
        chars_to_keep   = int(max_tokens * chars_per_token)
        raw_latex = raw_latex[:chars_to_keep] + "\n\n[... source truncated ...]"
        print(f"  LaTeX source truncated from ~{n_tok} to ~{max_tokens} tokens.")

    # --- Build prompt -----------------------------------------------------
    # TODO: make field-agnostic (currently says "astrophysics researcher")
    system_prompt = (
        "You are an expert astrophysics researcher. "
        "Read the provided paper and produce a concise structured summary. "
        "Your response must contain exactly these seven labelled sections, "
        "each on its own line, with no extra commentary before or after:\n\n"
        "Keywords: <five comma-separated keywords>\n\n"
        "Scientific Questions: <one paragraph>\n\n"
        "Data: <one paragraph>\n\n"
        "Methods: <one paragraph>\n\n"
        "Results: <one paragraph>\n\n"
        "Conclusions: <one paragraph>\n\n"
        "Key takeaway: <one paragraph>\n\n"
        "Each paragraph should be 5-8 concise sentences. "
        "Preserve specific numerical results, dataset names, and technique names. "
        "For Keywords, choose five short descriptive terms that best characterise "
        "the paper's topic and methods. Do not include any numeric codes in the keywords. "
        "For the key takeaway, provide what you think is the most important and novel "
        "insight of the paper. Limit the key takeaway to 1-2 sentences."
    )

    user_message = (
        f"Title: {title}\n"
        f"Authors: {authors}\n"
        f"Abstract: {abstract}\n\n"
        f"LaTeX Source:\n{raw_latex}"
    )

    # --- Call the LLM -----------------------------------------------------
    client = InferenceClient(provider=provider, api_key=hf_token)
    print(f"  Requesting summary for {arxiv_id} via {provider} / {model} ...")
    try:
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            model=model,
            max_tokens=16384,
        )
        raw_response = response.choices[0].message.content.strip()
    except Exception as e:
        print("Full prompt content was:")
        print("--- System Prompt ---")
        print(system_prompt)
        print("--- User Message ---")
        prompt_lines = user_message.splitlines()
        if len(prompt_lines) > 128:
            print("\n".join(prompt_lines[:64]) + "\n...\n" + "\n".join(prompt_lines[-64:]))
        else:
            print(user_message)
        raise RuntimeError(f"Summary API call failed for {arxiv_id}: {e}")

    # Strip chain-of-thought from thinking models (Qwen3, DeepSeek-R1, etc.)
    # Known closing markers — add more here as needed.
    COT_END_MARKERS = ["</think>"]
    summary = raw_response
    for marker in COT_END_MARKERS:
        idx = summary.rfind(marker)
        if idx != -1:
            summary = summary[idx + len(marker):]
            break
    summary = summary.strip()

    with open(cache_file, "w", encoding="utf-8") as f:
        f.write(summary)

    return summary


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------

def gen_arxiv_embedding(
    arxiv_id: str,
    tokens: dict[str, str],
    summary_provider: str = SUMMARY_PROVIDER,
    summary_model: str = SUMMARY_MODEL,
    embedding_provider: str = EMBEDDING_PROVIDER,
    embedding_model: str = EMBEDDING_MODEL,
    max_tokens: int = EMBEDDING_MAX_TOKENS,
) -> np.ndarray:
    """
    Generates an embedding vector for a given arXiv paper.

    Pipeline: fetch metadata → fetch/generate structured summary →
    build prompt → call embedding API.

    The full 4096-dim vector is returned; truncate to EMBEDDING_DIM at
    scoring time (do not store truncated vectors).

    # TODO: make embedding prompt field-agnostic (currently says "astrophysics paper")

    Parameters
    ----------
    arxiv_id : str
        arXiv ID of the paper to embed (e.g. "2309.06676").
    tokens : dict[str, str]
        Dict of API tokens, must include 'huggingface' and optionally 'semantic_scholar'.
    summary_provider : str
        InferenceClient provider for the summary step (default from config: SUMMARY_PROVIDER).
    summary_model : str
        Chat-completion model for the summary step (default from config: SUMMARY_MODEL).
    embedding_provider : str
        InferenceClient provider for the embedding step (default from config: EMBEDDING_PROVIDER).
    embedding_model : str
        Model for the embedding step (default from config: EMBEDDING_MODEL).
    max_tokens : int
        Maximum number of tokens for the combined metadata + summary input to the embedding model.
         (default from config: EMBEDDING_MAX_TOKENS). Context limit note: Qwen3-Next-80B-A3B-Thinking
         fails above ~110K tokens; leaving headroom for metadata and output, we set the default to 96K tokens.
    
    Returns
    -------
    np.ndarray
        The embedding vector for the given arXiv ID.
    """
    print("")
    print(f"Fetching metadata for {arxiv_id}...")
    metadata = get_arxiv_metadata(
        [arxiv_id],
        s2_token=tokens.get('semantic_scholar'),
    )[arxiv_id]

    print(f"Fetching structured summary for {arxiv_id}...")
    summary = summarize_arxiv_paper(
        arxiv_id, tokens['huggingface'],
        provider=summary_provider,
        model=summary_model,
    )

    # TODO (later): make field-agnostic (no mention of astrophysics)
    prompt = (
        "Instruct: "
        "Given an astrophysics paper title, author list, abstract, "
        "and a structured summary of the paper, "
        "retrieve other papers that are most similar, "
        "based on a combination of the research topic, "
        "scientific questions answered, "
        "and observational methods, analysis techniques, "
        "and datasets used.\n"
        "Query:\n"
        "Title: " + metadata["title"] + "\n"
        "Authors: " + ", ".join(metadata["authors"]) + "\n"
        "Abstract: " + metadata["abstract"] + "\n"
        "Structured Summary:\n\n"
    )
    full_input = prompt + summary

    n_tokens = count_tokens(full_input)
    if n_tokens > max_tokens:
        chars_per_token = len(full_input) / n_tokens
        chars_to_keep   = int(max_tokens * chars_per_token)
        print(f"Truncating from ~{n_tokens} to ~{max_tokens} tokens.")
        full_input = full_input[:chars_to_keep] + "\n\n[Truncated due to token limit]"

    print("\nLLM input:")
    print("\n".join(full_input.splitlines()[:20]) + "\n...\n")
    print(f"Total input length: {len(full_input)} characters.")

    print("Requesting embedding via Hugging Face InferenceClient...")
    client = InferenceClient(provider=embedding_provider, api_key=tokens['huggingface'])
    try:
        result = client.feature_extraction(full_input, model=embedding_model)
    except Exception as e:
        raise RuntimeError(f"API Error during feature extraction: {e}")

    return np.asarray(result).flatten()


def fetch_arxiv_embedding(arxiv_id: str, tokens: dict[str, str]) -> np.ndarray:
    """
    Returns the embedding vector for a given arXiv ID, using the local cache
    when available and generating a new one otherwise.

    Parameters
    ----------
    arxiv_id : str
        arXiv ID of the paper to embed (e.g. "2309.06676").
    tokens : dict[str, str]
        Dict of API tokens, must include 'huggingface' and optionally 'semantic_scholar'.
    
    Returns
    -------
    np.ndarray
        The embedding vector for the given arXiv ID.
    """
    _init_embedding_db()
    with sqlite3.connect(EMBEDDING_CACHE_DB) as con:
        row = con.execute(
            "SELECT vector FROM embeddings WHERE arxiv_id = ?", (arxiv_id,)
        ).fetchone()
        if row is not None:
            return np.frombuffer(row[0], dtype=np.float32)

        vector = gen_arxiv_embedding(arxiv_id, tokens)
        con.execute(
            "INSERT OR REPLACE INTO embeddings VALUES (?, ?)",
            (arxiv_id, vector.astype(np.float32).tobytes()),
        )
    return vector


def embed_arxiv_ids(
    arxiv_ids: list[str],
    tokens: dict[str, str],
    **kwargs,
) -> dict[str, np.ndarray]:
    """
    Fetch and embed a list of arXiv IDs.

    Returns a dict mapping arXiv ID -> embedding vector for every paper that
    was successfully embedded (including cache hits).

    Parameters
    ----------
    arxiv_ids : list[str]
        List of arXiv IDs to embed.
    tokens : dict[str, str]
        Dict of API tokens, must include 'huggingface' and optionally 'semantic_scholar'.
    **kwargs : passed to gen_arxiv_embedding for each ID (e.g. summary/embedding
         provider/model, max_tokens). See gen_arxiv_embedding for details.

    Returns
    -------
    dict[str, np.ndarray]
        Maps each arXiv ID to its embedding vector. IDs that could not be embedded
        due to errors are omitted.
    """
    get_arxiv_metadata(arxiv_ids, s2_token=tokens.get('semantic_scholar'))

    from tqdm.auto import tqdm
    vectors = {}
    for aid in tqdm(arxiv_ids):
        print(f"Processing {aid}...")
        vectors[aid] = fetch_arxiv_embedding(aid, tokens, **kwargs)
    return vectors


def embed_latest_mailing(category: str, tokens: dict[str, str]) -> dict[str, np.ndarray]:
    """
    Fetch and embed all papers from the most recent arXiv mailing for a category.

    Returns a dict mapping arXiv ID -> embedding vector.

    Parameters
    ----------
    category : str
        arXiv category (e.g. "astro-ph.GA"). Must be in the configured ARXIV_CATEGORIES set.
    tokens : dict[str, str]
        Dict of API tokens, must include 'huggingface' and optionally 'semantic_scholar'.
    
    Returns
    -------
    dict[str, np.ndarray]
         Maps each arXiv ID from the latest mailing to its embedding vector.
    """
    raise_on_arxiv_category(category)
    ids = fetch_latest_mailing_ids(category)
    print(f"Found {len(ids)} papers in the latest {category} mailing.")
    return embed_arxiv_ids(ids, tokens)


# ---------------------------------------------------------------------------
# Token loading
# ---------------------------------------------------------------------------

def load_tokens() -> dict[str, str]:
    """Load API tokens from TOKENS_FILE and return as a dict."""
    try:
        with open(TOKENS_FILE, "r") as f:
            tokens = json.load(f)
    except FileNotFoundError:
        print(f"Warning: {TOKENS_FILE} not found. API calls will fail without valid tokens.")
        tokens = {}

    for key in ("huggingface", "semantic_scholar"):
        if key not in tokens:
            print(f"Warning: '{key}' token not found in {TOKENS_FILE}.")

    return tokens
