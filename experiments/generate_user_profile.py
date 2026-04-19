#!/usr/bin/env python3
"""
Generate a dense research profile for a user based on their liked papers.

Reads structured summaries from the production summary cache and prompts the
summary LLM to build a six-bullet research profile.  The production cache is
never written to.

Usage:
    python experiments/generate_user_profile.py <email>
    python experiments/generate_user_profile.py <email> --config /path/to/alt_llm_config.json
"""

import argparse
import json
import os
import re
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from openai import OpenAI

from arxiv_lib import config as _config
from arxiv_lib.appdb import get_connection
from arxiv_lib.ingest import sanitize_old_style_arxiv_id

_EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))

_SYSTEM_PROMPT = (
    "You are an expert astrophysicist. "
    "Construct a research profile from the provided paper summaries. "
    "Follow the output format exactly, using only the six specified bullet points."
)

_USER_PROMPT_HEADER = """\
An astrophysicist has highly rated the following papers. Do not write a narrative \
summary. Instead, construct a dense, highly specific research profile using only \
these six bullet points. List only proper nouns, specific mechanisms, and technical terms.

* Categorization(s): <e.g., theorist, observer, instrumentation, simulation, data analysis>
* Scientific field(s): <e.g., cosmology, extragalactic astronomy, interstellar medium>
* Target Objects: <e.g., high-redshift quasars, molecular clouds>
* Physical Processes: <e.g., ram pressure stripping, dust coagulation>
* Methodologies: <e.g., N-body simulations, integral field spectroscopy>
* Data & Instruments: <e.g., JWST NIRSpec, Gaia DR3, VLBI, N-body simulations>

CRITICAL RULES:

1. Focus on the Primary: Extract only the primary focus of the research. You must ignore objects, datasets, or fields that are mentioned merely as background context or foreground contaminants (e.g., if a paper studies Milky Way dust to correct quasar observations, do not list quasars).

2. Limit Quantity: Limit each bullet point to the 3 to 5 most dominant, recurring terms across the provided papers.

3. Deduplicate: Group incremental versions of datasets or codes together (e.g., use "Gaia" instead of listing DR2, eDR3, and DR3).
"""


def _sanitize_email_for_filename(email: str) -> str:
    """Replace characters unsafe in filenames with underscores."""
    return re.sub(r"[^\w\-]", "_", email)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a research profile for a user from their liked papers."
    )
    parser.add_argument("email", help="Email address of the user.")
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="Path to an alternate llm_config.json.  Only the 'summary' section is used.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Build effective LLM config
    # ------------------------------------------------------------------
    effective_summary_cfg: dict = dict(_config.LLM_CONFIG.get("summary", {}))

    if args.config:
        alt_path = os.path.abspath(args.config)
        with open(alt_path, "r", encoding="utf-8") as f:
            alt_cfg = json.load(f)
        if "summary" not in alt_cfg:
            parser.error(f"Alternate config {alt_path!r} has no 'summary' section.")
        effective_summary_cfg.update(alt_cfg["summary"])
        print(f"Using alternate LLM config from: {alt_path}")

    model    = effective_summary_cfg.get("model", "")
    max_tok  = effective_summary_cfg.get("max_input_tokens", 98304)
    if "api_key" in effective_summary_cfg:
        api_key = effective_summary_cfg["api_key"]
    else:
        api_key = _config.API_KEYS.get(
            effective_summary_cfg.get("api_key_name", "summary_api_key"), ""
        )
    base_url = effective_summary_cfg.get("base_url", "https://router.huggingface.co/v1")
    cot_tags = effective_summary_cfg.get("cot_closing_tags", [])

    print(f"Model    : {model}")
    print(f"Base URL : {base_url}")
    print()

    # ------------------------------------------------------------------
    # Look up user and liked papers
    # ------------------------------------------------------------------
    con = get_connection()
    user_row = con.execute(
        "SELECT id, email FROM users WHERE email = ?", (args.email,)
    ).fetchone()
    if user_row is None:
        print(f"Error: no user with email {args.email!r} found.", file=sys.stderr)
        sys.exit(1)

    liked_rows = con.execute(
        "SELECT arxiv_id FROM user_papers WHERE user_id = ? AND liked = 1 ORDER BY added_at",
        (user_row["id"],),
    ).fetchall()
    con.close()

    if not liked_rows:
        print(f"No liked papers found for {args.email!r}.", file=sys.stderr)
        sys.exit(1)

    arxiv_ids = [row["arxiv_id"] for row in liked_rows]
    print(f"Found {len(arxiv_ids)} liked paper(s) for {args.email!r}.")

    # ------------------------------------------------------------------
    # Load summaries from cache
    # ------------------------------------------------------------------
    summaries: list[tuple[str, str]] = []  # (arxiv_id, summary_text)
    for arxiv_id in arxiv_ids:
        cache_file = os.path.join(
            _config.SUMMARY_CACHE_DIR(),
            sanitize_old_style_arxiv_id(arxiv_id) + ".txt",
        )
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                summaries.append((arxiv_id, f.read().strip()))
        else:
            print(f"  Warning: no cached summary for {arxiv_id} — skipping.")

    if not summaries:
        print(
            "Error: no cached summaries found for any liked papers.\n"
            "Run summarize_paper.py for each paper first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loaded {len(summaries)} summary/summaries.")

    # ------------------------------------------------------------------
    # Build prompt
    # ------------------------------------------------------------------
    paper_blocks = "\n\n".join(
        f"Paper {i + 1} ({arxiv_id}):\n{summary}"
        for i, (arxiv_id, summary) in enumerate(summaries)
    )
    user_message = _USER_PROMPT_HEADER + "\n" + paper_blocks

    print(user_message)

    # ------------------------------------------------------------------
    # Call the LLM
    # ------------------------------------------------------------------
    print(f"\nRequesting profile via {base_url} / {model} ...")
    client = OpenAI(base_url=base_url, api_key=api_key)
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            model=model,
            max_tokens=2048,
        )
        raw_response = response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"Profile API call failed for {args.email!r}: {e}")

    # Strip chain-of-thought if closing tags are configured
    profile = raw_response
    if cot_tags:
        best = max(
            (profile.rfind(tag) + len(tag) for tag in cot_tags if profile.rfind(tag) != -1),
            default=0,
        )
        profile = profile[best:].strip()

    # ------------------------------------------------------------------
    # Save and print
    # ------------------------------------------------------------------
    out_name = f"user_profile_{_sanitize_email_for_filename(args.email)}.txt"
    out_path = os.path.join(_EXPERIMENTS_DIR, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(profile)

    print()
    print("=" * 72)
    print(profile)
    print("=" * 72)
    print(f"\nProfile written to: {out_path}")


if __name__ == "__main__":
    main()
