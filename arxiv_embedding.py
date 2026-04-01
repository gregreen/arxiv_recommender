#/usr/bin/env python3

import glob
import os
import re
import json

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
import logging

from huggingface_hub import InferenceClient
from arxiv_to_prompt import process_latex_source, count_tokens

import numpy as np


EMBEDDING_CACHE_FILE = "embeddings_cache.npz" # {aid: embedding} dict will be stored here
SOURCE_CACHE_DIR = "arxiv_source_cache" # {aid}.tex files will be stored here
METADATA_CACHE_DIR = "arxiv_metadata_cache" # {month}.json files will be stored here
SUMMARY_CACHE_DIR = "arxiv_summary_cache" # {aid}.txt LLM summary files will be stored here
USER_AGENT = "arxiv-recommender/1.0"


def consolidate_metadata_cache() -> None:
    """
    Consolidates legacy per-paper JSON files (e.g. "2411.16590.json") in
    SOURCE_CACHE_DIR into monthly bundle files (e.g. "2411.json").

    The original per-paper files are left in place.
    """
    # 1. Collect all per-paper JSON files and group by month
    by_month: dict[str, list[str]] = {}
    for paper_file in (glob.glob(os.path.join(SOURCE_CACHE_DIR, "????.?????.json")) +
                       glob.glob(os.path.join(SOURCE_CACHE_DIR, "????.????.json"))):
        filename = os.path.basename(paper_file)
        month = filename.split(".")[0]
        by_month.setdefault(month, []).append(paper_file)

    # 2. Process month by month, writing each monthly cache file once
    for month, paper_files in by_month.items():
        monthly_file = os.path.join(SOURCE_CACHE_DIR, f"{month}.json")

        if os.path.exists(monthly_file):
            with open(monthly_file, "r", encoding="utf-8") as f:
                monthly_data = json.load(f)
        else:
            monthly_data = {}

        for paper_file in paper_files:
            with open(paper_file, "r", encoding="utf-8") as f:
                paper_data = json.load(f)
            arxiv_id = os.path.basename(paper_file)[:-5]  # strip ".json"
            if "title" in paper_data:
                monthly_data[arxiv_id] = paper_data
            else:
                monthly_data.update(paper_data)

        with open(monthly_file, "w", encoding="utf-8") as f:
            json.dump(monthly_data, f, ensure_ascii=False, indent=2)


def load_from_arxiv_metadata_cache(arxiv_ids: list[str]) -> dict | None:
    """
    Loads metadata for the given arXiv IDs from the monthly cache files in
    METADATA_CACHE_DIR.

    Returns a dict mapping arXiv ID to metadata dict for all IDs that were
    found in the cache.  IDs that are not found are omitted from the result.
    """
    result = {}

    # Group by month
    by_month: dict[str, list[str]] = {}
    for aid in arxiv_ids:
        month = aid.split(".")[0]
        by_month.setdefault(month, []).append(aid)
    
    for month, ids in by_month.items():
        cache_file = os.path.join(METADATA_CACHE_DIR, f"{month}.json")
        if not os.path.exists(cache_file):
            continue
        with open(cache_file, "r", encoding="utf-8") as f:
            month_cache = json.load(f)
        for aid in ids:
            if aid in month_cache:
                result[aid] = month_cache[aid]
    
    return result


def write_to_arxiv_metadata_cache(metadata_dict: dict[str, dict]) -> None:
    """
    Writes the given metadata dict to the monthly cache files in
    METADATA_CACHE_DIR.

    The input dict should map arXiv ID to metadata dict.  Each ID will be
    written to the appropriate monthly cache file based on its prefix (e.g.
    "2309" from "2309.06676").  Existing cache files will be updated with new
    entries, and new files will be created as needed.
    """
    # Group arXiv IDs by month
    by_month: dict[str, dict[str, dict]] = {}
    for aid, meta in metadata_dict.items():
        month = aid.split(".")[0]
        by_month.setdefault(month, {})[aid] = meta

    # Each month corresponds to one cache file
    for month, entries in by_month.items():
        cache_file = os.path.join(METADATA_CACHE_DIR, f"{month}.json")
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                month_cache = json.load(f)
        else:
            month_cache = {}
        
        month_cache.update(entries)

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(month_cache, f, ensure_ascii=False, indent=2)
    

def get_arxiv_metadata(arxiv_ids: list[str],
                       s2_token: str | None = None,
                       batch_size: int = 256) -> dict[str, dict]:
    """
    Returns cached metadata for the given arXiv IDs, fetching and caching any
    that are not already stored.

    Metadata is persisted in per-month JSON files (e.g. "2309.json") inside
    SOURCE_CACHE_DIR, bundling all papers from that month together.

    Parameters:
    - arxiv_ids: List of arXiv identifiers, e.g. ["2309.06676", "2309.12345"].
    - s2_token: Semantic Scholar API token, if available.
    - batch_size: Maximum number of papers to fetch in each batch.

    Returns:
    - A dict mapping each arXiv ID to its {"title", "authors", "abstract"} dict.
      IDs that could not be fetched are omitted.
    """
    
    # Load cached metadata for all requested IDs
    result = load_from_arxiv_metadata_cache(arxiv_ids)

    # Remove cached IDs from the list of IDs to fetch
    arxiv_ids = [aid for aid in arxiv_ids if aid not in result]

    if not arxiv_ids:
        return result

    # Order chronologically (as easy as sorting the IDs). This optimizes
    # cache efficiency, since papers are cached by month.
    arxiv_ids = sorted(arxiv_ids)

    # Batch according to maximum batch size
    for i0 in range(0, len(arxiv_ids), batch_size):
        batch_ids = arxiv_ids[i0:i0+batch_size]

        # Fetch metadata for this batch from Semantic Scholar
        print(f"Fetching metadata for {len(batch_ids)} papers.")
        try:
            fetched = fetch_arxiv_metadata_s2(batch_ids, s2_api_key=s2_token)
        except RuntimeError as e:
            print(f"  S2 fetch failed ({e}); falling back to arXiv Atom API for all {len(batch_ids)} papers.")
            fetched = {}

        # Fall back to arXiv Atom API for any IDs that S2 missed or didn't index yet
        missed = [aid for aid in batch_ids if aid not in fetched]
        if missed:
            print(f"  Fetching {len(missed)} papers not found in S2 from arXiv Atom API.")
            fetched.update(fetch_arxiv_metadata(missed))

        # Update the cache
        write_to_arxiv_metadata_cache(fetched)

        # Add fetched metadata to the cached dict for final output
        result.update(fetched)

    return result


def fetch_arxiv_metadata(arxiv_ids: list[str]) -> dict[str, dict]:
    """
    Returns the title, authors, and abstract for each arXiv ID in the list.

    Uses a single batched query to the official arXiv Atom API.

    Parameters:
    - arxiv_ids: List of arXiv identifiers, e.g. ["2101.00001", "2101.00002"].

    Returns:
    - A dict mapping each arXiv ID to a dict with keys "title" (str),
      "authors" (list[str]), "abstract" (str). IDs not found in the API
      response are omitted.

    Raises:
    - RuntimeError: If the API request fails.
    """
    _NS = "http://www.w3.org/2005/Atom"
    _arxiv_logger = logging.getLogger(__name__)

    def _is_retryable(exc: BaseException) -> bool:
        return (
            isinstance(exc, requests.HTTPError)
            and exc.response.status_code in (429, 500, 502, 503, 504)
        )

    @retry(
        stop=stop_after_attempt(7),
        wait=wait_exponential(multiplier=2, min=4, max=64) + wait_random(0, 1),
        retry=retry_if_exception(_is_retryable),
        before_sleep=before_sleep_log(_arxiv_logger, logging.WARNING),
        reraise=True,
    )
    def _get() -> requests.Response:
        url = (
            f"https://export.arxiv.org/api/query"
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
        # The <id> element is a URL like https://arxiv.org/abs/2101.00001v2
        aid = re.sub(r"v\d+$", "", id_el.text.strip().split("/abs/")[-1])

        title_el = entry.find(f"{{{_NS}}}title")
        summary_el = entry.find(f"{{{_NS}}}summary")
        author_els = entry.findall(f"{{{_NS}}}author")

        title = " ".join((title_el.text or "").split()) if title_el is not None else ""
        abstract = " ".join((summary_el.text or "").split()) if summary_el is not None else ""
        authors = []
        for a in author_els:
            name_el = a.find(f"{{{_NS}}}name")
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())

        results[aid] = {"title": title, "authors": authors, "abstract": abstract}

    return results


def fetch_arxiv_metadata_html(arxiv_id: str) -> dict:
    """
    Scrapes the title, authors, and abstract from the arXiv abstract page HTML.

    This is a fallback for cases where the Atom API is unavailable or does not
    return a result for a given ID.

    Parameters:
    - arxiv_id: arXiv identifier, e.g. "2101.00001".

    Returns:
    - A dict with keys "title" (str), "authors" (list[str]), "abstract" (str).

    Raises:
    - RuntimeError: If the page cannot be fetched or parsed.
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
    Returns title, authors, and abstract for each arXiv ID using the
    Semantic Scholar Graph batch API.

    Unlike the official arXiv Atom API (1 request every 3 seconds), Semantic
    Scholar accepts up to 500 IDs per POST request, so an entire mailing can
    be resolved in a single call.  Rate limits:
      - Without key : ~1 req/sec, 5 000 req/day
      - With free key: same per-second, higher daily quota (~10 000+/day)
    Free API keys: https://www.semanticscholar.org/product/api

    Parameters
    ----------
    arxiv_ids : list[str]
        arXiv identifiers, e.g. ["2309.06676", "2309.12345"].
    s2_api_key : str | None
        Optional Semantic Scholar API key.  Pass None to use the
        unauthenticated tier.

    Returns
    -------
    dict[str, dict]
        Maps each arXiv ID that was found to a dict with keys
        "title" (str), "authors" (list[str]), "abstract" (str).
        IDs not found in Semantic Scholar are omitted.

    Raises
    ------
    RuntimeError
        If the HTTP request fails.
    """
    S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
    S2_FIELDS    = "title,authors,abstract,externalIds"
    S2_BATCH_SIZE = 500   # hard limit imposed by the API

    headers: dict[str, str] = {"User-Agent": USER_AGENT}
    if s2_api_key:
        headers["x-api-key"] = s2_api_key

    results: dict[str, dict] = {}

    _s2_logger = logging.getLogger(__name__)

    def _is_retryable(exc: BaseException) -> bool:
        return (
            isinstance(exc, requests.HTTPError)
            and exc.response.status_code in (429, 500, 502, 503, 504)
        )

    @retry(
        stop=stop_after_attempt(7),
        wait=wait_exponential(multiplier=2, min=2, max=64) + wait_random(0, 1),
        retry=retry_if_exception(_is_retryable),
        before_sleep=before_sleep_log(_s2_logger, logging.WARNING),
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

    # Process in chunks of S2_BATCH_SIZE
    for start in range(0, len(arxiv_ids), S2_BATCH_SIZE):
        chunk = arxiv_ids[start : start + S2_BATCH_SIZE]
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
            results[arxiv_id] = {
                "title":    title,
                "authors":  authors,
                "abstract": abstract,
            }

    return results


def load_embedding_cache() -> dict[str, np.ndarray]:
    """Load the embeddings cache from disk. Returns a dict mapping arXiv ID to embedding vector."""
    if not os.path.exists(EMBEDDING_CACHE_FILE):
        return {}
    data = np.load(EMBEDDING_CACHE_FILE, allow_pickle=False)
    return {key: data[key] for key in data.files}


def save_embedding_cache(cache: dict[str, np.ndarray]) -> None:
    """Persist the embeddings cache to disk."""
    np.savez(EMBEDDING_CACHE_FILE, **cache)


def raise_on_arxiv_category(category: str) -> bool:
    """Check if the provided category is valid."""
    categories = {"astro-ph", "hep-th", "cs.AI", "math.PR"}
    if category not in categories:
        raise ValueError(f"Unsupported category: {category}. Supported categories are: "
                         "astro-ph, hep-th, cs.AI, math.PR.")


def fetch_latest_mailing_ids(category: str) -> list[str]:
    """
    Fetch arXiv IDs for all new papers in the specified category in the most recent mailing
    by scraping the arXiv new-submissions listing page.
    Returns a deduplicated list of arXiv IDs in announcement order.
    """
    # Validate category before making any network requests
    raise_on_arxiv_category(category)

    # Fetch the mailing page for the specified category
    url = f"https://arxiv.org/list/{category}/new"
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)

    # Raise an exception if the request failed (e.g. network error, 404, etc.)
    try:
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch latest mailing for {category}: {e}")
    
    # Extract the arXiv IDs using regex.
    ids = re.findall(r'arXiv:(\d{4}\.\d{4,5})', resp.text)
    ids = list(dict.fromkeys(ids)) # Deduplicate while preserving order

    return ids


def fetch_arxiv_embedding(arxiv_id: str, tokens: dict[str, str]) -> np.ndarray:
    """
    Fetches the embedding vector for a given arXiv ID. Will look in the
    local cache first, and if not found, will generate a new embedding.

    Parameters:
    - arxiv_id: The arXiv ID of the paper to embed (e.g "2101.00001").
    - tokens: A dictionary containing API tokens for different services.

    Returns:
    - A numpy array containing the embedding vector for the specified arXiv ID.

    Raises:
    - RuntimeError: If there is an error fetching or processing the paper, or if
        the embedding API call fails.
    """
    # If already in cache, return cached embedding
    cache = load_embedding_cache()
    if arxiv_id in cache:
        return cache[arxiv_id]

    # If not in cache, fetch and embed, then update cache
    vector = gen_arxiv_embedding(arxiv_id, tokens)

    # Update cache and persist to disk
    cache[arxiv_id] = vector
    save_embedding_cache(cache)
    
    return vector


def embed_arxiv_ids(arxiv_ids: list[str], tokens: dict[str, str], **kwargs) -> dict[str, np.ndarray]:
    """
    Fetch and embed a list of arXiv IDs.

    Returns a dict mapping arXiv ID -> embedding vector for every paper that
    was successfully embedded (including cache hits).
    """
    # To avoid overloading the arXiv API, request metadata in batches first
    # (get_arxiv_metadata fetches missing entries and persists them to cache)
    get_arxiv_metadata(arxiv_ids, s2_token=tokens.get('semantic_scholar'))

    from tqdm.auto import tqdm
    vectors = {}
    for aid in tqdm(arxiv_ids):
        print(f"Processing {aid}...")
        vectors[aid] = fetch_arxiv_embedding(aid, tokens, **kwargs)
    return vectors


def embed_latest_mailing(category: str, tokens: dict[str, str]) -> dict[str, np.ndarray]:
    """
    Fetch and embed all papers from the specified category in the most recent arXiv mailing.

    Returns a dict mapping arXiv ID -> embedding vector for every paper in
    this mailing that was successfully embedded (including cache hits).
    """
    raise_on_arxiv_category(category)
    ids = fetch_latest_mailing_ids(category)
    print(f"Found {len(ids)} papers in the latest {category} mailing.")
    return embed_arxiv_ids(ids, tokens)


def get_arxiv_source(arxiv_id: str) -> str:
    """Fetches the processed LaTeX source for a given arXiv ID."""
    # Check cache first
    cache_file = os.path.join(SOURCE_CACHE_DIR, f"{arxiv_id}.tex")
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return f.read()
    
    # If not cached, fetch from arXiv and cache it
    latex = process_latex_source(arxiv_id, keep_comments=False)

    if latex is None:
        latex = "No LaTeX source available for this paper."
        print(f"Warning: Unable to fetch/read {arxiv_id}!")

    # Cache the fetched LaTeX
    with open(cache_file, "w", encoding="utf-8") as f:
        f.write(latex)
    
    return latex


def compress_latex_whitespace(latex: str) -> str:
    """
    Reduces excessive whitespace in a LaTeX source string without removing
    meaningful structure.

    Specifically:
    - Strips trailing whitespace from each line
    - Collapses runs of more than one blank line into a single blank line
    - Collapses runs of multiple spaces/tabs within a line into a single space
    """
    lines = latex.splitlines()

    # Strip leading/trailing whitespace and collapse internal runs of spaces/tabs
    lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in lines]

    # Remove empty lines
    lines = [line for line in lines if line]

    # Put back together
    compressed = "\n".join(lines)

    # # Collapse runs of consecutive blank lines into one
    # compressed = re.sub(r'\n{3,}', '\n\n', '\n'.join(lines))

    # lines = []

    return compressed


def summarize_arxiv_paper(
    arxiv_id: str,
    hf_token: str,
    provider: str = "novita", #"hyperbolic",
    model: str = "Qwen/Qwen3-Next-80B-A3B-Thinking",
    max_tokens: int = 96 * 1024,
) -> str:
    """
    Produces a structured, high-SNR summary of an arXiv paper in a single LLM
    call.

    The full LaTeX source is fed together with the paper's title, author list,
    and abstract.  The model is asked to return:
      - 5 relevant keywords
      - One short paragraph each for: scientific questions, data, methods,
        results, and conclusions.

    Results are cached in SUMMARY_CACHE_DIR so repeated calls are cheap.

    Parameters
    ----------
    arxiv_id : str
        arXiv identifier, e.g. "2309.06676".
    hf_token : str
        Hugging Face API token.
    provider : str
        InferenceClient provider (default "novita").
    model : str
        Chat-completion model (default "Qwen/Qwen3-Next-80B-A3B-Thinking").
    max_tokens : int
        Maximum tokens of LaTeX source to feed to the model; content beyond
        this limit is truncated before the API call.

    Returns
    -------
    str
        Structured summary as returned by the LLM, also persisted to cache.
    """
    os.makedirs(SUMMARY_CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(SUMMARY_CACHE_DIR, f"{arxiv_id}.txt")
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return f.read()

    # --- Fetch paper data -------------------------------------------------
    metadata = get_arxiv_metadata([arxiv_id])[arxiv_id]
    title    = metadata["title"]
    if len(metadata["authors"]) > 32:
        authors  = ", ".join(metadata["authors"][:24]) + " et al."
    else:
        authors  = ", ".join(metadata["authors"])
    abstract = metadata["abstract"]

    raw_latex = get_arxiv_source(arxiv_id)
    raw_latex = compress_latex_whitespace(raw_latex)

    # Truncate LaTeX source if it exceeds the token budget
    n_tok = count_tokens(raw_latex)
    print(f'Estimated number of tokens: {n_tok}')
    if n_tok > max_tokens:
        chars_per_token = len(raw_latex) / max(n_tok, 1)
        chars_to_keep   = int(max_tokens * chars_per_token)
        raw_latex = raw_latex[:chars_to_keep] + "\n\n[... source truncated ...]"
        print(f"  LaTeX source truncated from ~{n_tok} to ~{max_tokens} tokens.")

    # --- Build prompt -----------------------------------------------------
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
        "Preserve specific numerical results, dataset names, and "
        "technique names. "
        "For Keywords, choose five short descriptive terms that best "
        "characterise the paper's topic and methods. Do not include any "
        "numeric codes in the keywords. "
        "For the key takeaway, provide what you think is the most "
        "important and novel insight of the paper. Limit the key "
        "takeaway to 1-2 sentences."
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
        print('--- System Prompt ---')
        print(system_prompt)
        print('--- User Message ---')
        # If prompt is >128 lines, print only the first and last 128 lines to avoid overwhelming the logs
        prompt_lines = user_message.splitlines()
        if len(prompt_lines) > 128:
            print("\n".join(prompt_lines[:64]) + "\n...\n" + "\n".join(prompt_lines[-64:]))
        else:
            print(user_message)
        raise RuntimeError(f"Summary API call failed for {arxiv_id}: {e}")

    # Thinking models emit a chain-of-thought before the final answer.
    # The CoT may be wrapped in <think>...</think> tags, but some providers
    # omit the opening tag and only emit the closing marker. Strip everything
    # up to and including the last known termination marker.
    # Known closing markers (add more here as needed):
    #   </think>   — standard Qwen3 / DeepSeek-R1
    COT_END_MARKERS = ["</think>"]
    summary = raw_response
    for marker in COT_END_MARKERS:
        idx = summary.rfind(marker)
        if idx != -1:
            summary = summary[idx + len(marker):]
            break
    summary = summary.strip()

    # Persist to cache
    with open(cache_file, "w", encoding="utf-8") as f:
        f.write(summary)

    return summary


def gen_arxiv_embedding(arxiv_id: str, tokens: dict[str, str],
                        summary_provider: str = "novita", #"hyperbolic",
                        summary_model: str = "Qwen/Qwen3-Next-80B-A3B-Thinking",
                        embedding_provider: str = "scaleway",
                        embedding_model: str = "Qwen/Qwen3-Embedding-8B",
                        max_tokens: int = 24*1024) -> list[float]:
    """
    Downloads an arXiv paper, flattens the LaTeX, prepends a task prompt,
    and returns its embedding vector using the Hugging Face InferenceClient.
    """
    
    # Debugging
    print("")

    print(f"Fetching metadata for {arxiv_id}...")
    metadata = get_arxiv_metadata(
        [arxiv_id],
        s2_token=tokens.get('semantic_scholar')
    )[arxiv_id]

    print(f"Fetching structured summary for {arxiv_id}...")
    summary = summarize_arxiv_paper(
        arxiv_id, tokens['huggingface'],
        provider=summary_provider,
        model=summary_model
    )

    # print(f"Fetching processed LaTeX source for {arxiv_id}...")
    # raw_latex = get_arxiv_source(arxiv_id)
    # raw_latex = compress_latex_whitespace(raw_latex)
    
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

    # prompt = (
    #     "Instruct: "
    #     "Given an astrophysics paper title, author list, and abstract, "
    #     "retrieve other papers that cover similar research topics, "
    #     "answer similar scientific questions, "
    #     "and use similar observational methods, analysis techniques, "
    #     "and datasets.\n"
    #     "Query:\n"
    #     "Title: " + metadata["title"] + "\n"
    #     "Authors: " + ", ".join(metadata["authors"]) + "\n"
    #     "Abstract: " + metadata["abstract"] + "\n"
    # )
    # full_input = prompt

    # Estimate number of tokens and truncate if necessary
    n_tokens = count_tokens(full_input)
    if n_tokens > max_tokens:
        # Estimate number of characters per token
        chars_per_token = len(full_input) / n_tokens
        # Calculate number of characters to keep
        chars_to_keep = int(max_tokens * chars_per_token)
        print(f"Truncating from ~{n_tokens} to ~{max_tokens} tokens.")
        full_input = full_input[:chars_to_keep]
        full_input += "\n\n[Truncated due to token limit]"

    # Debugging
    print("\nLLM input:")
    print("\n".join(full_input.splitlines()[:20]) + "\n...\n")
    
    print(f"Total input length: {len(full_input)} characters.")

    print("Requesting embedding via Hugging Face InferenceClient...")
    
    # Initialize the client
    client = InferenceClient(
        provider=embedding_provider,
        api_key=tokens['huggingface']
    )
    
    try:
        # The client handles the API routing automatically
        result = client.feature_extraction(
            full_input,
            model=embedding_model,
        )
    except Exception as e:
        raise RuntimeError(f"API Error during feature extraction: {e}")

    # Return embedding as flat numpy array
    return np.asarray(result).flatten()


def arxiv_papers_umap(vectors_dict):
    """
    Projects arXiv paper embeddings into 2D using UMAP, and visualizes
    the results.
    
    Args:
        vectors_dict (dict): {arxiv_id: np.array(2048,)}
        
    Returns:
        dict: {arxiv_id: (x, y)} where (x, y) are the 2D coordinates of
        the embedding.
    """
    import umap
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt

    # Extract IDs and stack vectors into a matrix (N, D)
    arxiv_ids = list(vectors_dict.keys())
    X = np.array([vectors_dict[aid] for aid in arxiv_ids])

    # Truncate to 64 dimensions (Matryoshka) + L2 norm
    X = X[:, :64]
    # X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-10
    
    # Standardize features before UMAP
    X_scaled = StandardScaler().fit_transform(X)
    
    # UMAP projection to 2D
    u_reducer = umap.UMAP(
        n_neighbors=15,
        n_components=2,
        min_dist=0.0,
        metric='euclidean',
        random_state=42
    )
    u_embedded = u_reducer.fit_transform(X_scaled)

    fig,ax = plt.subplots(figsize=(6,6))
    ax.scatter(u_embedded[:,0], u_embedded[:,1], s=10, alpha=0.7)
    ax.set_title("UMAP Projection of arXiv Paper Embeddings")

    # Load metadata for hover labels
    metadata = get_arxiv_metadata(arxiv_ids)
    for i, aid in enumerate(arxiv_ids):
        title = metadata[aid]["title"] if aid in metadata else "Unknown Title"
        # Matplotlib treats $...$ as mathtext; escape special chars so titles
        # containing $, _, ^, { or } are rendered as plain text.
        safe_title = title.replace("$", r"\$").replace("_", r"\_").replace("^", r"\^")
        ax.annotate(safe_title, (u_embedded[i,0], u_embedded[i,1]), fontsize=6, alpha=0.6)
    
    plt.show()
    
    # Map back to dict format
    return {aid: tuple(u_embedded[i]) for i, aid in enumerate(arxiv_ids)}


def train_logistic_model(v_positive, v_negative, test_size=0.2, random_state=42):
    """
    Trains a logistic regression model to distinguish between two sets of vectors.

    Args:
        v_positive (np.ndarray): Array of shape (N_pos, D) containing positive examples.
        v_negative (np.ndarray): Array of shape (N_neg, D) containing negative examples.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        model: The trained logistic regression model.
        X_test: Test feature vectors.
        y_test: Test labels.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    # Create labels
    y_positive = np.ones(len(v_positive))
    y_negative = np.zeros(len(v_negative))

    # Combine data
    X = np.vstack((v_positive, v_negative))
    y = np.concatenate((y_positive, y_negative))

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train logistic regression model
    model = LogisticRegression(
        random_state=random_state,
        class_weight='balanced',
        C=0.2,
        # penalty='elasticnet',
        # l1_ratio=0.5,
        # solver='saga'
    )
    model.fit(X_train, y_train)

    return model, X_test, y_test


def cluster_arxiv_papers(vectors_dict):
    """
    Clusters arXiv paper embeddings using a PCA + UMAP + HDBSCAN pipeline.
    
    Args:
        vectors_dict (dict): {arxiv_id: np.array(2048,)}
        
    Returns:
        list: A list of dictionaries, where each dict represents one cluster 
              and contains the original {arxiv_id: vector} pairs.
    """

    import umap
    import hdbscan
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # 1. Extract IDs and stack vectors into a matrix (N, D)
    arxiv_ids = list(vectors_dict.keys())
    X = np.array([vectors_dict[aid] for aid in arxiv_ids])
    
    # 2. Pre-processing: Standardize features
    # This is critical if the embedding dimensions have varying scales
    X_scaled = StandardScaler().fit_transform(X)
    
    # 3. Dimensionality Reduction Step 1: PCA
    # Reduce 2048-D to 16-D to remove noise and speed up UMAP
    n_pca = min(len(arxiv_ids) - 1, 16)
    pca_reduced = PCA(n_components=n_pca, random_state=42).fit_transform(X_scaled)
    
    # 4. Dimensionality Reduction Step 2: UMAP
    # Project onto a 5D manifold to preserve local/global structure
    # For N=150, we keep n_neighbors relatively small
    u_reducer = umap.UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )
    u_embedded = u_reducer.fit_transform(pca_reduced)
    
    # 5. Clustering: HDBSCAN
    # min_cluster_size=5 is a reasonable heuristic for 150 papers
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    labels = clusterer.fit_predict(u_embedded)
    
    # 6. Reconstruct the output: List of dictionaries
    # labels == -1 indicates noise/outliers in HDBSCAN
    unique_labels = set(labels)
    clusters_list = []
    
    for label in sorted(unique_labels):
        cluster_dict = {
            arxiv_ids[i]: vectors_dict[arxiv_ids[i]]
            for i, l in enumerate(labels) if l == label
        }
        clusters_list.append(cluster_dict)
        
    return clusters_list


def report_compression_stats(max_tokens: int = 24 * 1024) -> None:
    """
    Runs compress_latex_whitespace on every cached .tex file in SOURCE_CACHE_DIR
    and reports per-file and aggregate compression statistics.

    For each file it reports:
      - original and compressed character counts
      - compression ratio
      - estimated token counts before and after (using count_tokens)
      - whether the compressed version still exceeds max_tokens
    """
    tex_files = sorted(glob.glob(os.path.join(SOURCE_CACHE_DIR, "*.tex")))
    if not tex_files:
        print("No cached .tex files found.")
        return

    total_orig_chars = 0
    total_comp_chars = 0
    total_orig_tokens = 0
    total_comp_tokens = 0
    still_truncated = 0

    print(f"{'File':<25} {'Orig chars':>12} {'Comp chars':>12} {'Ratio':>7} "
          f"{'Orig tok':>10} {'Comp tok':>10} {'Trunc?':>7}")
    print("-" * 85)

    for path in tex_files:
        with open(path, "r", encoding="utf-8") as f:
            original = f.read()

        compressed = compress_latex_whitespace(original)

        orig_chars = len(original)
        comp_chars = len(compressed)
        ratio = comp_chars / orig_chars if orig_chars else 1.0

        orig_tokens = count_tokens(original)
        comp_tokens = count_tokens(compressed)
        truncated = comp_tokens > max_tokens

        if truncated:
            still_truncated += 1

        fname = os.path.basename(path)
        print(f"{fname:<25} {orig_chars:>12,} {comp_chars:>12,} {ratio:>7.1%} "
              f"{orig_tokens:>10,} {comp_tokens:>10,} {'YES' if truncated else 'no':>7}")

        total_orig_chars += orig_chars
        total_comp_chars += comp_chars
        total_orig_tokens += orig_tokens
        total_comp_tokens += comp_tokens

    n = len(tex_files)
    overall_ratio = total_comp_chars / total_orig_chars if total_orig_chars else 1.0
    saved_chars = total_orig_chars - total_comp_chars
    saved_tokens = total_orig_tokens - total_comp_tokens

    print("-" * 85)
    print(f"\nFiles processed : {n}")
    print(f"Total chars     : {total_orig_chars:,} → {total_comp_chars:,} "
          f"(saved {saved_chars:,}, {1 - overall_ratio:.1%} reduction)")
    print(f"Total tokens    : {total_orig_tokens:,} → {total_comp_tokens:,} "
          f"(saved {saved_tokens:,})")
    print(f"Still truncated : {still_truncated}/{n} files exceed {max_tokens:,} tokens after compression")


def load_tokens() -> dict[str, str]:
    """Load API tokens from local files and return as a dict."""
    try:
        with open("tokens.json", "r") as f:
            tokens = json.load(f)
    except FileNotFoundError:
        print("Warning: tokens.json not found. API calls will fail without valid tokens.")
        tokens = {}
    
    # Warn if any expected tokens are missing
    expected_keys = ["huggingface", "semantic_scholar"]
    for key in expected_keys:
        if key not in tokens:
            print(f"Warning: '{key}' token not found in tokens.json. "
                   "API calls requiring this token will fail.")
    
    return tokens


def rbf_scoring(gamma: np.ndarray,
                positive_vectors: np.ndarray,
                search_vectors: np.ndarray = None) -> np.ndarray:
    """
    Computes RBF scores for search vectors based on their similarity to positive vectors.

    Args:
        gamma (np.ndarray): Parameter for the RBF kernel. Higher values make
                            the kernel more sensitive to distance. If 'scale',
        positive_vectors (np.ndarray): Array of shape (N_pos, D) containing
                                       positive examples.
        search_vectors (np.ndarray): Array of shape (N_search, D) containing
                                     vectors to score.

    Returns:
        np.ndarray: Array of shape (n_search, n_gamma) containing the RBF scores for
                    each combination of search vector and gamma.
    """
    # Assume features have already been standardized.
    # Set gamma to 1 / n_features
    gamma_0 = 1.0 / positive_vectors.shape[1]

    # Implement RBF kernel manually, using the formula:
    # K(x, y) = exp(-gamma * ||x - y||^2)
    # We will calculate the log of the kernel.

    # Compute the Euclidean distance between all pairs of search and positive vectors
    from scipy.spatial.distance import cdist
    if search_vectors is None:
        # If no search vectors provided, compare positive vectors to themselves
        scaled_dist2_matrix = -gamma_0 * cdist(
            positive_vectors,
            positive_vectors,
            'sqeuclidean'
        )
        # Set diagonal to median to avoid self-similarity dominating the scores
        diag_indices = np.diag_indices_from(scaled_dist2_matrix)
        median_value = np.median(scaled_dist2_matrix, axis=1)
        scaled_dist2_matrix[diag_indices] = median_value
        print(np.min(scaled_dist2_matrix), np.max(scaled_dist2_matrix), np.median(scaled_dist2_matrix))
    else:
        scaled_dist2_matrix = -gamma_0 * cdist(
            search_vectors,
            positive_vectors,
            'sqeuclidean'
        )

    # Create return array
    kernel_scores = np.zeros((scaled_dist2_matrix.shape[0], len(gamma)+1))

    # Baseline score (max similarity to any positive vector)
    kernel_scores[:,0] = np.max(scaled_dist2_matrix, axis=1)

    # For each gamma, compute the mean score across all positive vectors
    from scipy.special import logsumexp
    for i,g in enumerate(gamma):
        kernel_scores[:,i+1] = logsumexp(g * scaled_dist2_matrix, axis=1)

    return kernel_scores


def logistic_regression_example():
    with open("my_papers.txt", "r") as f:
        my_papers = [
            line.strip()[6:] for line in f
            if line.strip().startswith("arXiv:")
        ]
    
    # Load paper embeddings and metadata from disk
    embeddings = load_embedding_cache()
    metadata = load_from_arxiv_metadata_cache(list(embeddings.keys()))

    # Get all embedding vectors
    arxiv_ids = np.array(list(embeddings.keys()))

    vectors = np.array([embeddings[aid][:64] for aid in arxiv_ids])  # Use first 64 dimensions (Matryoshka)
    print(f"Embedding matrix shape: {vectors.shape}")

    # # Standardize
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # vectors = scaler.fit_transform(vectors)

    # Train logistic model
    idx_pos = np.array([aid in my_papers for aid in arxiv_ids], dtype=bool)
    v_pos = vectors[idx_pos]
    v_neg = vectors[~idx_pos]
    model, X_test, y_test = train_logistic_model(v_pos, v_neg)
    print(f"Logistic regression test accuracy: {model.score(X_test, y_test):.2%}")

    lnp = model.predict_log_proba(vectors)

    # lnp = model.predict_log_proba(v_pos)
    print("\nLog-probabilities for my papers:")
    for i in np.where(idx_pos)[0]:
        aid = arxiv_ids[i]
        print(f" * ({lnp[i,1]: >5.2f}) https://arxiv.org/abs/{aid}: {metadata[aid]['title']}")

    # Use logistic model to find papers most similar to my papers
    lnp_neg = lnp[~idx_pos]
    idx = np.argsort(lnp_neg[:,1])[::-1]  # Sort by log-probability of being in the positive class
    sorted_ids = arxiv_ids[~idx_pos][idx]  # Get corresponding arXiv IDs for the sorted negative examples

    # lnp = model.predict_log_proba(v_neg)
    # idx = np.argsort(lnp[:,1])[::-1]  # Sort by log-probability of being in the positive class
    # sorted_ids = arxiv_ids[~idx_pos][idx]  # Get corresponding arXiv IDs for the sorted negative examples

    print("\nTop 10 papers most similar to my papers:")
    for i,aid in enumerate(sorted_ids[:10]):
        print(f" * ({lnp_neg[idx[i]][1]: >5.2f}) https://arxiv.org/abs/{aid}: {metadata[aid]['title']}")

    print("\n10 least similar papers to my own:")
    for i,aid in enumerate(sorted_ids[::-1][:10]):
        print(f" * ({lnp_neg[idx[::-1][i]][1]: >5.2f}) https://arxiv.org/abs/{aid}: {metadata[aid]['title']}")


def calculate_projection_matrices(vectors, n_components):
    """
    Calculates the projection matrix for projecting onto the top n_components
    principal components of the given vectors. Uses SVD to perform PCA. Returns
    two projection matrices: one for the top n_components components, and
    one for the remaining components (the "residual" subspace).

    Args:
        vectors (np.ndarray): Array of shape (N, D) containing the original vectors.
        n_components (int): The number of principal components to project onto.
    
    Returns:
        np.ndarray: Array of shape (D, n_components) containing the projection matrix.
        np.ndarray: Array of shape (D, D - n_components) containing the residual projection matrix.
    """
    # Center the data
    X_centered = vectors - np.mean(vectors, axis=0)

    # Compute SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Get the top n_components eigenvectors (principal components)
    projection_matrix = Vt[:n_components].T  # Shape (D, n_components)

    # Get the remaining components for the residual subspace
    residual_projection_matrix = Vt[n_components:].T  # Shape (D, D - n_components)

    return projection_matrix, residual_projection_matrix


def project_to_subspace(vectors, projection_matrix):
    """
    Projects the given vectors onto a subspace defined by the projection matrix.
    Additionally returns the complement of the projection (the "residual" vectors),
    in their own subspace.

    Args:
        vectors (np.ndarray): Array of shape (N, D) containing the original vectors.
        projection_matrix (np.ndarray): Array of shape (D, d) where d < D, defining
                                        the subspace to project onto.

    Returns:
        np.ndarray: Array of shape (N, d) containing the projected vectors.
    """
    return vectors @ projection_matrix


def rbf_example():
    with open("my_papers.txt", "r") as f:
        my_papers = [
            line.strip()[6:] for line in f
            if line.strip().startswith("arXiv:")
        ]
    
    # Load paper embeddings and metadata from disk
    embeddings = load_embedding_cache()
    metadata = load_from_arxiv_metadata_cache(list(embeddings.keys()))

    # Get all embedding vectors
    arxiv_ids = np.array(list(embeddings.keys()))
    vectors = np.array([embeddings[aid][:64] for aid in arxiv_ids]) # Use first 64 dimensions (Matryoshka)

    # Standardize vectors before RBF scoring
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    vectors = scaler.fit_transform(vectors)

    # Select positive and search vectors
    idx_pos = np.array([aid in my_papers for aid in arxiv_ids], dtype=bool)
    v_pos = vectors[idx_pos]
    v_search = vectors[~idx_pos]

    # RBF distance scales chosen logarithmically
    gammas = np.logspace(-8, 8, num=12, base=2)

    # Score search vectors against positive vectors
    scores_search = rbf_scoring(gammas, v_pos, v_search)
    # For the positive vectors, we have to exclude self-similarity.
    scores_pos = rbf_scoring(gammas, v_pos)

    scores = np.zeros((len(scores_search)+len(scores_pos), scores_search.shape[1]))
    scores[idx_pos] = scores_pos
    scores[~idx_pos] = scores_search

    # Standardize scores for each gamma column
    scaler = StandardScaler()
    scores = scaler.fit_transform(scores)

    print(scores.shape)

    # Learn a logistic regression on the RBF scores to predict which papers are in my set
    model, X_test, y_test = train_logistic_model(
        scores[idx_pos],
        scores[~idx_pos],
        test_size=0.2,
        random_state=42
    )

    # Calculate scores using logistic model
    lnp = model.predict_log_proba(scores)

    # Feature importance: coef_[0, i] is the weight on the i-th gamma column.
    # Positive weight → higher RBF score at that gamma → more like a "my paper".
    print("\nLogistic model feature importances (by gamma):")
    coeffs = model.coef_[0]
    print(f"  gamma=nearest  coeff={coeffs[0]:+.4f}")
    for gamma, w in zip(gammas, coeffs[1:]):
        print(f"  gamma={gamma: >7.4f}  coeff={w:+.4f}")

    print("\nLog-probabilities for my papers:")
    for i in np.where(idx_pos)[0]:
        aid = arxiv_ids[i]
        print(f" * ({lnp[i,1]: >5.2f}) https://arxiv.org/abs/{aid}: {metadata[aid]['title']}")

    # Use logistic model to find papers most similar to my papers
    lnp_neg = lnp[~idx_pos]
    idx = np.argsort(lnp_neg[:,1])[::-1]  # Sort by log-probability of being in the positive class
    sorted_ids = arxiv_ids[~idx_pos][idx]  # Get corresponding arXiv IDs for the sorted negative examples

    # lnp = model.predict_log_proba(v_neg)
    # idx = np.argsort(lnp[:,1])[::-1]  # Sort by log-probability of being in the positive class
    # sorted_ids = arxiv_ids[~idx_pos][idx]  # Get corresponding arXiv IDs for the sorted negative examples

    print("\nTop 10 papers most similar to my papers:")
    for i,aid in enumerate(sorted_ids[:10]):
        print(f" * ({lnp_neg[idx[i]][1]: >5.2f}) https://arxiv.org/abs/{aid}: {metadata[aid]['title']}")

    print("\n10 least similar papers to my own:")
    for i,aid in enumerate(sorted_ids[::-1][:10]):
        print(f" * ({lnp_neg[idx[::-1][i]][1]: >5.2f}) https://arxiv.org/abs/{aid}: {metadata[aid]['title']}")

    # idx = np.argsort(scores)[::-1]  # Sort by score descending
    # sorted_ids = arxiv_ids[~idx_pos][idx]  # Get corresponding arXiv IDs for the sorted search examples

    # print("\nTop 10 papers most similar to my papers (RBF scoring):")
    # for i,aid in enumerate(sorted_ids[:10]):
    #     print(f" * ({scores[idx[i]]: >5.2f}) https://arxiv.org/abs/{aid}: {metadata[aid]['title']}")
    
    # print("\n10 least similar papers to my own (RBF scoring):")
    # for i,aid in enumerate(sorted_ids[::-1][:10]):
    #     print(f" * ({scores[idx[::-1][i]]: >5.2f}) https://arxiv.org/abs/{aid}: {metadata[aid]['title']}")
    
    # pctiles = [0, 1, 16, 25, 50, 75, 84, 99, 100]
    # score_pct = np.percentile(scores, pctiles)
    # print("\nRBF score percentiles:")
    # for p, s in zip(pctiles, score_pct):
    #     print(rf"  {p: >3.0f}% : {s:.5g}")


def svm_example():
    from sklearn import svm

    with open("my_papers.txt", "r") as f:
        my_papers = [
            line.strip()[6:] for line in f
            if line.strip().startswith("arXiv:")
        ]
    
    # Load paper embeddings and metadata from disk
    embeddings = load_embedding_cache()
    metadata = load_from_arxiv_metadata_cache(list(embeddings.keys()))

    # Get all embedding vectors
    arxiv_ids = np.array(list(embeddings.keys()))
    vectors = np.array([embeddings[aid][:64] for aid in arxiv_ids]) # Use first 64 dimensions (Matryoshka)

    # Standardize vectors before RBF scoring
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    vectors = scaler.fit_transform(vectors)

    # Select positive and search vectors
    idx_pos = np.array([aid in my_papers for aid in arxiv_ids], dtype=bool)
    v_pos = vectors[idx_pos]
    v_search = vectors[~idx_pos]

    # Train SVM with RBF kernel
    X = np.vstack((v_pos, v_search))
    y = np.concatenate((np.ones(len(v_pos)), np.zeros(len(v_search))))
    model = svm.SVC(kernel='rbf', gamma='scale', class_weight='balanced', probability=True, random_state=42)
    model.fit(X, y)

    # Get decision function scores for all papers
    scores = model.decision_function(vectors)
    # Get predicted probabilities for the positive class
    probs = model.predict_proba(vectors)[:, 1]

    print("\nSVM decision function scores for my papers:")
    for i in np.where(idx_pos)[0]:
        aid = arxiv_ids[i]
        print(f" * (score={scores[i]: >5.2f}, prob={probs[i]:.2%}) https://arxiv.org/abs/{aid}: {metadata[aid]['title']}")
    
    # Use SVM scores to find papers most similar to my papers
    idx = np.argsort(scores[~idx_pos])[::-1]  # Sort search vectors by score descending
    sorted_ids = arxiv_ids[~idx_pos][idx]
    print("\nTop 10 papers most similar to my papers (SVM RBF scores):")
    for i,aid in enumerate(sorted_ids[:10]):
        print(f" * (score={scores[~idx_pos][idx[i]]: >5.2f}, prob={probs[~idx_pos][idx[i]]:.2%}) https://arxiv.org/abs/{aid}: {metadata[aid]['title']}")
    
    print("\n10 least similar papers to my own (SVM RBF scores):")
    for i,aid in enumerate(sorted_ids[::-1][:10]):
        print(f" * (score={scores[~idx_pos][idx[::-1][i]]: >5.2f}, prob={probs[~idx_pos][idx[::-1][i]]:.2%}) https://arxiv.org/abs/{aid}: {metadata[aid]['title']}")


def rbf_svd_example():
    with open("weicheng_papers.txt", "r") as f:
        my_papers = [
            line.strip()[6:] for line in f
            if line.strip().startswith("arXiv:")
        ]
    
    # Load paper embeddings and metadata from disk
    embeddings = load_embedding_cache()
    metadata = load_from_arxiv_metadata_cache(list(embeddings.keys()))

    # Get all embedding vectors
    arxiv_ids = np.array(list(embeddings.keys()))
    vectors = np.array([embeddings[aid][:64] for aid in arxiv_ids]) # Use first 64 dimensions (Matryoshka)

    # Standardize vectors before RBF scoring
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    vectors = scaler.fit_transform(vectors)

    # Select positive and search vectors
    idx_pos = np.array([aid in my_papers for aid in arxiv_ids], dtype=bool)
    v_pos = vectors[idx_pos]
    v_search = vectors[~idx_pos]

    # RBF distance scales chosen logarithmically
    gammas = np.logspace(-6, 6, num=6, base=2)

    # Score search vectors against positive vectors
    scores_search = rbf_scoring(gammas, v_pos, v_search)
    # For the positive vectors, we have to exclude self-similarity.
    scores_pos = rbf_scoring(gammas, v_pos)

    # Get projection matrix from SVD of the positive vectors
    P, P_residual = calculate_projection_matrices(v_pos, n_components=4)

    # Project the vectors onto the subspace defined by the projection matrix
    v_search_proj = project_to_subspace(v_search, P)
    v_pos_proj = project_to_subspace(v_pos, P)

    # Score the projected vectors using RBF scoring
    scores_search_proj = rbf_scoring(gammas, v_pos_proj, v_search_proj)
    scores_pos_proj = rbf_scoring(gammas, v_pos_proj)

    # Scores for the orthogonal complements
    v_search_residual = project_to_subspace(v_search, P_residual)
    v_pos_residual = project_to_subspace(v_pos, P_residual)
    scores_search_residual = rbf_scoring(gammas, v_pos_residual, v_search_residual)
    scores_pos_residual = rbf_scoring(gammas, v_pos_residual)

    # Combine all the scores into a single array for logistic regression
    features_pos = [
        scores_pos,
        # scores_pos_proj,
        scores_pos_residual
    ]
    features_search = [
        scores_search,
        # scores_search_proj,
        scores_search_residual
    ]
    n_features = sum(f.shape[1] for f in features_search)
    n_samples = scores_search.shape[0] + scores_pos.shape[0]
    scores = np.zeros((n_samples, n_features))
    idx = 0
    for f0,f1 in zip(features_pos, features_search):
        scores[idx_pos, idx:idx+f0.shape[1]] = f0
        scores[~idx_pos, idx:idx+f1.shape[1]] = f1
        idx += f0.shape[1]
    
    # Standardize scores for each feature column
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scores = scaler.fit_transform(scores)

    print(scores.shape)

    # Learn a logistic regression on the RBF scores to predict which papers are in my set
    model, X_test, y_test = train_logistic_model(
        scores[idx_pos],
        scores[~idx_pos],
        test_size=0.2,
        random_state=42
    )

    # Calculate scores using logistic model
    lnp = model.predict_log_proba(scores)

    # Feature importance: coef_[0, i] is the weight on the i-th gamma column.
    # Positive weight → higher RBF score at that gamma → more like a "my paper".
    print("\nLogistic model feature importances (by gamma):")
    coeffs = model.coef_[0]
    # Split coeffs into groups of length gamma.shape[0]+1
    coeffs_groups = [coeffs[i:i+len(gammas)+1] for i in range(0, len(coeffs), len(gammas)+1)]
    coeffs = np.array(coeffs_groups).T  # Shape (n_features_per_group, n_groups)
    print(f"  gamma=nearest  coeff={coeffs[0]}")
    for gamma, w in zip(gammas, coeffs[1:]):
        print(f"  gamma={gamma: >7.4f}  coeff={w}")

    print("\nLog-probabilities for my papers:")
    for i in np.where(idx_pos)[0]:
        aid = arxiv_ids[i]
        print(f" * ({lnp[i,1]: >5.2f}) https://arxiv.org/abs/{aid}: {metadata[aid]['title']}")

    # Use logistic model to find papers most similar to my papers
    lnp_neg = lnp[~idx_pos]
    idx = np.argsort(lnp_neg[:,1])[::-1]  # Sort by log-probability of being in the positive class
    sorted_ids = arxiv_ids[~idx_pos][idx]  # Get corresponding arXiv IDs for the sorted negative examples

    print("\nTop 10 papers most similar to my papers:")
    for i,aid in enumerate(sorted_ids[:10]):
        print(f" * ({lnp_neg[idx[i]][1]: >5.2f}) https://arxiv.org/abs/{aid}: {metadata[aid]['title']}")

    print("\n10 least similar papers to my own:")
    for i,aid in enumerate(sorted_ids[::-1][:10]):
        print(f" * ({lnp_neg[idx[::-1][i]][1]: >5.2f}) https://arxiv.org/abs/{aid}: {metadata[aid]['title']}")
    
    return {aid: lnp[i,1] for i, aid in enumerate(arxiv_ids)}




def main():
    tokens = load_tokens()
    # embeddings = embed_latest_mailing("astro-ph", tokens)
    # fetch_arxiv_embedding("2603.28400", tokens)
    embeddings = load_embedding_cache()

    # rbf_example()
    # svm_example()
    lnp = rbf_svd_example()
    print(json.dumps(lnp, indent=2))
    print(lnp['2603.28400'])
    
    return 0


if __name__ == "__main__":
    main()