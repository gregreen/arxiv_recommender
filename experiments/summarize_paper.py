#!/usr/bin/env python3
"""
Generate a summary of an arXiv paper using a custom (or default) LLM config,
without touching the production summary cache.

Usage:
    python experiments/summarize_paper.py <arxiv_id>
    python experiments/summarize_paper.py <arxiv_id> --config /path/to/alt_llm_config.json

The summary is written to <arxiv_id>.txt in the current working directory and
printed to stdout.  The production summary cache is never read from or written to.
Metadata and LaTeX source caches are reused as normal.
"""

import argparse
import json
import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from openai import OpenAI
from arxiv_to_prompt import count_tokens

from arxiv_lib import config as _config
from arxiv_lib.ingest import (
    compress_latex_whitespace,
    get_arxiv_metadata,
    get_arxiv_source,
    sanitize_old_style_arxiv_id,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarise an arXiv paper without touching the production summary cache."
    )
    parser.add_argument("arxiv_id", help="arXiv ID (e.g. 2309.06676)")
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="Path to an alternate llm_config.json. Only the 'summary' section is used.",
    )
    parser.add_argument(
        "--system-prompt",
        metavar="PATH",
        help="Path to a .txt file whose contents replace the default system prompt.",
    )
    args = parser.parse_args()

    arxiv_id: str = args.arxiv_id

    # ------------------------------------------------------------------
    # Build effective LLM config (start from project default, then overlay)
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

    model     = effective_summary_cfg.get("model", "")
    max_tok   = effective_summary_cfg.get("max_input_tokens", 98304)
    # "api_key" in the config takes precedence over the named key in api_keys.json
    if "api_key" in effective_summary_cfg:
        api_key = effective_summary_cfg["api_key"]
    else:
        api_key = _config.API_KEYS.get(effective_summary_cfg.get("api_key_name", "summary_api_key"), "")
    base_url  = effective_summary_cfg.get("base_url", "https://router.huggingface.co/v1")
    cot_tags  = effective_summary_cfg.get("cot_closing_tags", [])
    completion_kwargs = effective_summary_cfg.get("kwargs", {})

    print(f"Model    : {model}")
    print(f"Base URL : {base_url}")
    print()

    # ------------------------------------------------------------------
    # Fetch metadata (reuses app.db cache; fetches from S2/Atom if missing)
    # ------------------------------------------------------------------
    print(f"Fetching metadata for {arxiv_id}...")
    meta_map = get_arxiv_metadata([arxiv_id], s2_token=_config.API_KEYS.get("semantic_scholar"))
    if arxiv_id not in meta_map:
        print(f"Error: could not retrieve metadata for {arxiv_id!r}.", file=sys.stderr)
        sys.exit(1)

    metadata = meta_map[arxiv_id]
    title    = metadata["title"]
    authors_list = metadata["authors"]
    authors  = (
        ", ".join(authors_list[:24]) + " et al."
        if len(authors_list) > 32
        else ", ".join(authors_list)
    )
    abstract = metadata["abstract"]

    # ------------------------------------------------------------------
    # Fetch LaTeX source (reuses source cache; downloads if missing)
    # ------------------------------------------------------------------
    print(f"Fetching LaTeX source for {arxiv_id}...")
    raw_latex = get_arxiv_source(arxiv_id)
    raw_latex = compress_latex_whitespace(raw_latex)

    n_tok = count_tokens(raw_latex)
    print(f"Estimated LaTeX tokens: {n_tok:,}")
    if n_tok > max_tok:
        chars_per_token = len(raw_latex) / max(n_tok, 1)
        chars_to_keep   = int(max_tok * chars_per_token)
        raw_latex = raw_latex[:chars_to_keep] + "\n\n[... source truncated ...]"
        print(f"  LaTeX source truncated from ~{n_tok:,} to ~{max_tok:,} tokens.")

    # ------------------------------------------------------------------
    # Build prompt
    # ------------------------------------------------------------------
    if args.system_prompt:
        sp_path = os.path.abspath(args.system_prompt)
        with open(sp_path, "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
        print(f"System prompt loaded from: {sp_path}")
    else:
        system_prompt = _config.SUMMARIZE_SYSTEM_PROMPT

    user_message = (
        f"Title: {title}\n"
        f"Authors: {authors}\n"
        f"Abstract: {abstract}\n\n"
        f"LaTeX Source:\n{raw_latex}"
    )

    # ------------------------------------------------------------------
    # Call the LLM
    # ------------------------------------------------------------------
    print(f"\nRequesting summary via {base_url} / {model} ...")
    client = OpenAI(base_url=base_url, api_key=api_key)
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            model=model,
            max_tokens=16384,
            **completion_kwargs
        )
        content = response.choices[0].message.content.strip()
        reasoning = getattr(response.choices[0].message, "reasoning", "").strip()
        # raw_response = response.choices[0].message.content.strip()
        print(response.model_dump_json(indent=2))
        # print(json.dumps(response.json(), indent=2))  # print full response for debugging
    except Exception as e:
        raise RuntimeError(f"Summary API call failed for {arxiv_id}: {e}")

    # Strip chain-of-thought if closing tags are configured
    if cot_tags:
        best = max(
            (content.rfind(tag) + len(tag) for tag in cot_tags if content.rfind(tag) != -1),
            default=0,
        )
        content = content[best:].strip()

    # ------------------------------------------------------------------
    # Write output to CWD and print
    # ------------------------------------------------------------------
    out_name = sanitize_old_style_arxiv_id(arxiv_id) + ".txt"
    out_path = os.path.join(os.getcwd(), out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(reasoning + '\n\n' + ('=' * 40) + '\n\n' + content)

    print()
    print("=" * 72)
    print(content)
    print("=" * 72)
    print(f"\nSummary written to: {out_path}")


if __name__ == "__main__":
    main()
