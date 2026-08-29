#!/usr/bin/env python3
"""
LLM summary critic: generate a summary, then have a second LLM critique it.

This script uses two models, each specified by an llm_config_*.json file:

  * the *summarizer* summarizes a paper (drawn randomly from a user-provided
    list of arXiv IDs, or from the most recent arXiv mailing for a category),
    using the project's standard summarization prompt;
  * the *critic* reads the summary prompt, the paper, the summary, and the
    summarizer's reasoning trace, grades the summary on a rubric (accuracy,
    completeness, formatting, style, reasoning), and proposes concrete
    changes to the summary prompt.

The paper info, the summary prompt, the summary, the criticism, and the
proposed prompt changes are written to a single, clearly-delimited file:

    experiments/summary_criticism/critic_{arxiv_id}_{YYYY-MM-DD}_{6char}.txt

Each section is delimited by a "===== SECTION =====" header so the files are
easy to parse later.

This script is intentionally standalone (lives under experiments/). It reuses
the project's metadata + LaTeX source caches and the standard summarize prompt,
but never touches the production summary cache and never modifies any
web/production code.
"""

import argparse
import json
import os
import random
import secrets
import string
import sys
import time
from datetime import datetime

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from openai import OpenAI
from arxiv_to_prompt import count_tokens

from arxiv_lib import config as _config
from arxiv_lib.arxiv_id import validate_arxiv_id
from arxiv_lib.ingest import (
    compress_latex_whitespace,
    fetch_latest_mailing_ids,
    get_arxiv_metadata,
    get_arxiv_source,
)

_EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_OUTPUT_DIR = os.path.join(_EXPERIMENTS_DIR, "summary_criticism")

# Delimiters used in the output file. Keep in sync with the docstring.
_SECTION_SEP = "=" * 72


def _section(title: str) -> str:
    return f"{_SECTION_SEP}\n===== {title} =====\n{_SECTION_SEP}"


# ---------------------------------------------------------------------------
# Critic prompt
# ---------------------------------------------------------------------------

CRITIC_SYSTEM_PROMPT = """\
You are an expert critic of LLM-generated arXiv paper summaries. You will be given:
  1. The system prompt ("summary prompt") that was used to instruct a
     summarizer LLM.
  2. A paper (title, authors, abstract, and LaTeX source).
  3. The summary of that paper produced by the summarizer LLM.
  4. The reasoning trace (chain of thought) of the summarizer LLM, if one
     was captured.

Your job is to grade the summary on the following rubric, then propose
concrete changes to the summary prompt.

Rubric:
1. Accuracy: How accurately did the summarizer summarize the contents of the
   paper? Are there any mistakes? Check every factual claim in the summary
   against the paper.
2. Completeness: Did the summarizer miss any major elements of the paper that
   should have been included in the summary, given the length constraints in
   the prompt?
3. Formatting: Did the summarizer obey all of the formatting rules in the
   prompt (required section headings, length limits, KaTeX, no Markdown, etc.)?
4. Style: Are there any issues of style, tone or voice in the summary that are NOT captured
   by the prompt, but which should be added to the prompt?
5. Reasoning: Did the summarizer LLM get confused or stuck in a loop? Look
   for repeated attempts, self-corrections that go nowhere, contradictions,
   or signs it misunderstood the paper or the prompt. What concrete changes
   to the prompt would clear up the summarizer's confusion?

Respond in EXACTLY the following format, with these section headers:

## ACCURACY
<your assessment, including a list of any specific factual errors>

## COMPLETENESS
<your assessment, including a list of any major omissions>

## FORMATTING
<your assessment, including any violations of the prompt's formatting rules>

## STYLE
<your assessment of stylistic issues not covered by the prompt>

## REASONING
<your assessment of the summarizer's reasoning trace: did it get confused,
misunderstand the paper or the prompt, or get stuck in a loop? Quote the
relevant parts of the trace as evidence.>

## PROPOSED_PROMPT_CHANGES
<a numbered list of concrete, actionable changes to the summary prompt that
would fix the issues you identified. Quote the exact prompt text to be changed
or added where possible. If the prompt is already good, say so explicitly.>

Be specific and quote evidence from the paper and the summary. Do not rewrite
the summary itself; your output is about the prompt, not the summary.
"""

PROMPT_UPDATER_SYSTEM_PROMPT = """\
You are an expert prompt engineer for an LLM that summarizes arXiv astrophysics
papers. You will be given:
  1. The current summary prompt (a system prompt instructing a summarizer LLM
     to produce concise, structured paper summaries).
  2. A collection of proposed changes to that prompt, extracted from critique
     sessions in which a critic LLM graded summaries produced by the prompt.

Your job is to produce an updated version of the summary prompt.

Important guidelines:
  * Do NOT blindly implement every suggestion. The proposals may conflict with
    each other, may be based on a single unrepresentative paper, or may add
    rules that bloat the prompt and dilute the important ones. Use your
    judgment about which changes are worth making.
  * Keep in mind the overall goals of the prompt: effective arXiv paper
    summaries that are accurate, informative, short, and fulfill various
    formatting rules (required section headings, length limits, KaTeX-compatible
    LaTeX, no Markdown).
  * Preserve the overall structure and voice of the prompt. Make targeted
    edits rather than rewriting from scratch.
  * If a proposal is redundant with existing text, skip it. If two proposals
    conflict, pick the better one or synthesize.
  * If the current prompt is sufficient, you are allowed to leave it unchanged.

Respond in EXACTLY the following format, with these section headers:

## UPDATED_PROMPT
<the complete updated summary prompt, ready to use as-is>

## EXPLANATION
<a clear explanation of the changes you made and why: which proposals you
implemented, which you skipped and why, and any changes you made that were
not proposed but that you judged beneficial. If you made no changes, say so
explicitly and explain why.>
"""


# ---------------------------------------------------------------------------
# Config / input loading
# ---------------------------------------------------------------------------

def load_model_config(path: str, temperature: float | None) -> dict:
    """Load and validate one llm_config-like JSON file (uses 'summary' section)."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "summary" not in cfg:
        raise SystemExit(f"Config {path!r} has no 'summary' section.")
    s = cfg["summary"]

    api_key = s.get("api_key")
    if not api_key:
        raise SystemExit(
            f"Config {path!r} is missing an inline 'summary.api_key'. "
            "Add the key directly to the config file."
        )

    kwargs = dict(s.get("kwargs", {}) or {})
    if temperature is not None:
        kwargs["temperature"] = temperature

    return {
        "name": s.get("name") or s.get("model") or os.path.basename(path),
        "model": s.get("model", ""),
        "base_url": s.get("base_url", "https://router.huggingface.co/v1"),
        "api_key": api_key,
        "max_input_tokens": s.get("max_input_tokens", 98304),
        "cot_closing_tags": s.get("cot_closing_tags", []),
        "kwargs": kwargs,
    }


def _normalize_arxiv_id(raw: str) -> str:
    """Strip an optional 'arXiv:' prefix and validate the ID.

    Returns the canonical ID (version suffix and surrounding whitespace
    removed). Raises ValueError for malformed IDs.
    """
    cleaned = raw.strip()
    if cleaned.lower().startswith("arxiv:"):
        cleaned = cleaned[len("arxiv:"):].strip()
    return validate_arxiv_id(cleaned)


def load_papers(path: str) -> list[str]:
    """Read one arXiv ID per line; blank lines and '#' comments are ignored."""
    papers: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                papers.append(_normalize_arxiv_id(line))
            except ValueError as e:
                raise SystemExit(f"Invalid arXiv ID in {path!r}: {line!r} ({e})")
    if not papers:
        raise SystemExit(f"No arXiv IDs found in {path!r}.")
    return papers


def pick_paper(args, rng: random.Random) -> str:
    """Pick a random paper: from --papers if given, else the latest mailing."""
    if args.papers:
        papers = load_papers(args.papers)
        return rng.choice(papers)
    print(f"Fetching latest arXiv mailing for category {args.category!r} ...")
    ids = fetch_latest_mailing_ids(args.category)
    if not ids:
        raise SystemExit(
            f"No IDs found in the latest arXiv mailing for {args.category!r}."
        )
    print(f"Got {len(ids)} IDs from the latest mailing.")
    return rng.choice(ids)


# ---------------------------------------------------------------------------
# LLM calls (reuses the project prompt + caches; never the summary cache)
# ---------------------------------------------------------------------------

def _extract_reasoning(message) -> str:
    """Extract a chain-of-thought trace from the API message, if present."""
    for attr in ("reasoning", "reasoning_content"):
        val = getattr(message, attr, None)
        if val is None:
            continue
        if isinstance(val, str):
            if val.strip():
                return val.strip()
        elif isinstance(val, list):
            parts = [getattr(item, "text", "") for item in val]
            parts = [p for p in parts if isinstance(p, str) and p.strip()]
            if parts:
                return "\n".join(parts).strip()
    return ""


def _strip_cot(content: str, cot_tags: list[str]) -> tuple[str, str]:
    """Split raw content into (final_text, reasoning) using cot_closing_tags."""
    reasoning = ""
    final = content
    if cot_tags:
        best = max(
            (final.rfind(tag) + len(tag)
             for tag in cot_tags
             if final.rfind(tag) != -1),
            default=0,
        )
        if best:
            reasoning = final[:best].strip()
            final = final[best:].strip()
    return final, reasoning


def _truncate_latex(raw_latex: str, max_tok: int) -> tuple[str, bool, int]:
    """Truncate LaTeX source to max_tok tokens. Returns (latex, truncated, n_tok)."""
    n_tok = count_tokens(raw_latex)
    if n_tok <= max_tok:
        return raw_latex, False, n_tok
    chars_per_token = len(raw_latex) / max(n_tok, 1)
    chars_to_keep = int(max_tok * chars_per_token)
    return (
        raw_latex[:chars_to_keep] + "\n\n[... source truncated ...]",
        True,
        n_tok,
    )


def _call_llm(model_cfg: dict, system_prompt: str, user_message: str,
              label: str) -> dict:
    """Make one chat-completion call; returns content/reasoning/usage info."""
    client = OpenAI(base_url=model_cfg["base_url"], api_key=model_cfg["api_key"])
    try:
        t0 = time.time()
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            model=model_cfg["model"],
            max_tokens=16384,
            **model_cfg["kwargs"],
        )
        elapsed = time.time() - t0
    except Exception as e:
        raise RuntimeError(f"{label} API call failed: {e}")

    message = response.choices[0].message
    content = message.content
    if content is None:
        raise RuntimeError(
            f"{label} returned null content "
            f"(finish_reason={response.choices[0].finish_reason!r})"
        )
    content = content.strip()

    reasoning = _extract_reasoning(message)
    final, cot_reasoning = _strip_cot(content, model_cfg["cot_closing_tags"])
    if cot_reasoning and not reasoning:
        reasoning = cot_reasoning

    return {
        "text": final,
        "reasoning": reasoning,
        "response_json": response.model_dump_json(indent=2),
        "elapsed": elapsed,
    }


def generate_summary(arxiv_id: str, model_cfg: dict, system_prompt: str) -> dict:
    """Generate one summary for *arxiv_id* using *model_cfg*.

    Mirrors experiments/summarize_paper.py: reuse metadata + LaTeX source
    caches, build the same prompt, call the LLM, and split reasoning from the
    final summary. No summary caching — a fresh call every time.
    """
    meta_map = get_arxiv_metadata(
        [arxiv_id], s2_token=_config.API_KEYS.get("semantic_scholar")
    )
    if arxiv_id not in meta_map:
        raise RuntimeError(f"Could not retrieve metadata for {arxiv_id!r}.")
    metadata = meta_map[arxiv_id]

    title = metadata["title"]
    authors_list = metadata["authors"]
    authors = (
        ", ".join(authors_list[:24]) + " et al."
        if len(authors_list) > 32
        else ", ".join(authors_list)
    )
    abstract = metadata["abstract"]

    raw_latex = compress_latex_whitespace(get_arxiv_source(arxiv_id))
    latex, truncated, n_tok = _truncate_latex(raw_latex, model_cfg["max_input_tokens"])

    user_message = (
        f"Title: {title}\n"
        f"Authors: {authors}\n"
        f"Abstract: {abstract}\n\n"
        f"LaTeX Source:\n{latex}"
    )
    res = _call_llm(model_cfg, system_prompt, user_message, "Summarizer")
    res.update(
        title=title,
        authors=authors,
        abstract=abstract,
        truncated=truncated,
        n_input_tokens=n_tok,
    )
    return res


def run_critic(summary_prompt: str, paper: dict, summary: str,
               summarizer_reasoning: str, model_cfg: dict) -> dict:
    """Ask the critic to grade *summary* against the rubric and propose
    changes to *summary_prompt*. The summarizer's reasoning trace is included
    in its own clearly-separated section for the Reasoning rubric item."""
    latex, truncated, n_tok = _truncate_latex(
        paper["latex"], model_cfg["max_input_tokens"]
    )
    required = "\n".join(f"- {h}" for h in _config.SUMMARY_REQUIRED_HEADINGS)
    user_message = (
        f"# SUMMARY PROMPT\n"
        f"{summary_prompt}\n\n"
        f"# REQUIRED SUMMARY SECTION HEADINGS (from the project config)\n"
        f"{required}\n\n"
        f"# PAPER\n"
        f"Title: {paper['title']}\n"
        f"Authors: {paper['authors']}\n"
        f"Abstract: {paper['abstract']}\n\n"
        f"LaTeX Source:\n{latex}\n\n"
        f"# SUMMARY PRODUCED BY THE SUMMARIZER\n"
        f"{summary}\n\n"
        f"# SUMMARIZER REASONING TRACE\n"
        f"{summarizer_reasoning or '(no reasoning trace was captured)'}\n"
    )
    res = _call_llm(model_cfg, CRITIC_SYSTEM_PROMPT, user_message, "Critic")
    res["truncated"] = truncated
    res["n_input_tokens"] = n_tok
    return res


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _random_suffix(n: int = 6) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(n))


def write_criticism_file(output_dir: str, arxiv_id: str, paper: dict,
                         summary_prompt: str, summary_res: dict,
                         critic_res: dict) -> str:
    """Write the paper info, summary, criticism, and proposed changes to one
    clearly-delimited file. Returns the output path."""
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"critic_{arxiv_id}_{date_str}_{_random_suffix()}.txt"
    path = os.path.join(output_dir, filename)

    def model_header(role: str, model_cfg: dict, res: dict) -> str:
        return (
            f"model: {model_cfg['name']}\n"
            f"provider model: {model_cfg['model']}\n"
            f"base_url: {model_cfg['base_url']}\n"
            f"elapsed: {res['elapsed']:.2f}s  "
            f"estimated input tokens: {res['n_input_tokens']:,}"
            f"{'  (source truncated)' if res['truncated'] else ''}\n"
        )

    lines = [
        _section("PAPER INFO"),
        f"arxiv_id: {arxiv_id}",
        f"title: {paper['title']}",
        f"authors: {paper['authors']}",
        f"abstract: {paper['abstract']}",
        f"date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        _section("SUMMARY PROMPT"),
        summary_prompt,
        "",
        _section("SUMMARY"),
        model_header("summarizer", paper["summarizer_cfg"], summary_res),
        summary_res["text"],
        "",
        _section("CRITICISM"),
        model_header("critic", paper["critic_cfg"], critic_res),
        critic_res["text"],
        "",
        _section("CRITIC REASONING"),
        critic_res["reasoning"] or "(none captured)",
        "",
        _section("SUMMARIZER REASONING"),
        summary_res["reasoning"] or "(none captured)",
        "",
        _section("RAW RESPONSE — SUMMARIZER"),
        summary_res["response_json"],
        "",
        _section("RAW RESPONSE — CRITIC"),
        critic_res["response_json"],
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# --summarize: aggregate criticisms and update the summary prompt
# ---------------------------------------------------------------------------

def extract_proposed_changes(path: str) -> str | None:
    """Extract the proposed prompt changes from a criticism file.

    The proposals live inside the CRITICISM section under the critic's
    markdown header "## PROPOSED_PROMPT_CHANGES"; the section ends at the
    next "===== " file-section header (or EOF). Returns the section text,
    or None if the file has no such section.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    marker = "## PROPOSED_PROMPT_CHANGES"
    start = content.find(marker)
    if start == -1:
        return None
    start += len(marker)
    # Stop at the next "===== " file-section header (e.g. CRITIC REASONING).
    next_match = content.find("\n===== ", start)
    section = content[start:next_match if next_match != -1 else None]
    return section.strip() or None


def load_proposed_changes(criticism_dir: str) -> list[dict]:
    """Load the proposed-changes section from every criticism file in *dir*.

    Only files named critic_*.txt are considered, so the summary_*.txt
    session files and prompt_*.txt updated-prompt files in the same directory
    are never ingested. Returns a list of {"file": basename, "changes":
    section_text} dicts, sorted by filename. Files without a
    PROPOSED_PROMPT_CHANGES section are skipped.
    """
    entries: list[dict] = []
    for name in sorted(os.listdir(criticism_dir)):
        if not name.startswith("critic_") or not name.endswith(".txt"):
            continue
        path = os.path.join(criticism_dir, name)
        if not os.path.isfile(path):
            continue
        changes = extract_proposed_changes(path)
        if changes:
            entries.append({"file": name, "changes": changes})
    return entries


def run_prompt_update(summary_prompt: str, proposals: list[dict],
                      model_cfg: dict) -> dict:
    """Ask the critic LLM to update the summary prompt given the proposals."""
    parts = [
        f"# CURRENT SUMMARY PROMPT\n{summary_prompt}\n\n"
        f"# PROPOSED CHANGES ({len(proposals)} critique sessions)\n"
    ]
    for entry in proposals:
        parts.append(
            f"## From critique file: {entry['file']}\n{entry['changes']}\n"
        )
    res = _call_llm(
        model_cfg, PROMPT_UPDATER_SYSTEM_PROMPT, "\n".join(parts), "Prompt updater"
    )
    res["n_input_tokens"] = count_tokens("\n".join(parts))
    res["truncated"] = False
    return res


def write_prompt_update_file(output_dir: str, summary_prompt: str,
                             proposals: list[dict], update_res: dict,
                             updated_prompt: str, explanation: str) -> str:
    """Write the prompt update session to one clearly-delimited file, and the
    updated prompt itself to a standalone, ready-to-use prompt file.

    The session file is named summary_{date}_{suffix}.txt; the standalone
    updated prompt is named prompt_{date}_{suffix}.txt (same suffix) so it can
    be passed directly to --system-prompt in later runs. Returns
    (session_path, prompt_path)."""
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    suffix = _random_suffix()
    path = os.path.join(output_dir, f"summary_{date_str}_{suffix}.txt")
    prompt_path = os.path.join(output_dir, f"prompt_{date_str}_{suffix}.txt")

    lines = [
        _section("PROMPT UPDATE INFO"),
        f"date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"model: {update_res['model_name']}",
        f"provider model: {update_res['model']}",
        f"base_url: {update_res['base_url']}",
        f"elapsed: {update_res['elapsed']:.2f}s  "
        f"estimated input tokens: {update_res['n_input_tokens']:,}",
        f"criticism files used ({len(proposals)}):",
        *(f"  - {e['file']}" for e in proposals),
        "",
        _section("CURRENT SUMMARY PROMPT"),
        summary_prompt,
        "",
        _section("PROPOSED CHANGES"),
        *(f"--- From: {e['file']} ---\n{e['changes']}\n" for e in proposals),
        _section("UPDATED PROMPT"),
        updated_prompt,
        "",
        _section("EXPLANATION"),
        explanation,
        "",
        _section("REASONING"),
        update_res["reasoning"] or "(none captured)",
        "",
        _section("RAW RESPONSE"),
        update_res["response_json"],
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # Standalone updated prompt, ready to be passed to --system-prompt.
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(updated_prompt.rstrip() + "\n")

    return path, prompt_path


def _split_update_response(text: str) -> tuple[str, str]:
    """Split the updater's response into (updated_prompt, explanation).

    Falls back to treating the whole response as the updated prompt if the
    expected section headers are missing.
    """
    marker = "## EXPLANATION"
    idx = text.find(marker)
    if idx == -1:
        return text.strip(), "(no EXPLANATION section found in response)"
    updated = text[:idx].strip()
    # Drop the "## UPDATED_PROMPT" header itself if present.
    header = "## UPDATED_PROMPT"
    if updated.startswith(header):
        updated = updated[len(header):].strip()
    explanation = text[idx + len(marker):].strip()
    return updated, explanation


def cmd_summarize(args) -> None:
    critic = load_model_config(args.critic_config, args.temperature)
    criticism_dir = os.path.abspath(args.output_dir)
    if not os.path.isdir(criticism_dir):
        raise SystemExit(f"Criticism directory {criticism_dir!r} not found.")

    proposals = load_proposed_changes(criticism_dir)
    if not proposals:
        raise SystemExit(
            f"No criticism files with a 'PROPOSED_PROMPT_CHANGES' section "
            f"found in {criticism_dir!r}."
        )
    print(f"Loaded proposed changes from {len(proposals)} criticism files.")

    if args.system_prompt:
        with open(args.system_prompt, "r", encoding="utf-8") as f:
            summary_prompt = f.read().strip()
    else:
        summary_prompt = _config.SUMMARIZE_SYSTEM_PROMPT

    print("Updating summary prompt ...")
    update_res = run_prompt_update(summary_prompt, proposals, critic)
    print(f"Prompt update finished in {update_res['elapsed']:.2f}s.")

    update_res["model_name"] = critic["name"]
    update_res["model"] = critic["model"]
    update_res["base_url"] = critic["base_url"]
    updated_prompt, explanation = _split_update_response(update_res["text"])

    path, prompt_path = write_prompt_update_file(
        criticism_dir, summary_prompt, proposals, update_res,
        updated_prompt, explanation,
    )
    print(f"Wrote: {path}")
    print(f"Wrote standalone updated prompt: {prompt_path}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_run(args) -> None:
    summarizer = load_model_config(args.summarizer_config, args.temperature)
    critic = load_model_config(args.critic_config, args.temperature)
    if summarizer["name"] == critic["name"]:
        print(
            "WARNING: Summarizer and critic configs have the same 'name'; give each "
            "config a distinct 'name' field."
        )

    if args.system_prompt:
        with open(args.system_prompt, "r", encoding="utf-8") as f:
            summary_prompt = f.read().strip()
    else:
        summary_prompt = _config.SUMMARIZE_SYSTEM_PROMPT

    output_dir = os.path.abspath(args.output_dir)
    rng = random.Random(args.seed)

    for i in range(args.count):
        print("\n" + "=" * 72)
        print(f"Critique {i + 1}/{args.count}")
        print("=" * 72)

        try:
            arxiv_id = pick_paper(args, rng)
            print(f"Paper: {arxiv_id}")

            print("Generating summary ...")
            summary_res = generate_summary(arxiv_id, summarizer, summary_prompt)
            print(f"Summary generated in {summary_res['elapsed']:.2f}s.")

            paper = {
                "title": summary_res["title"],
                "authors": summary_res["authors"],
                "abstract": summary_res["abstract"],
                "latex": compress_latex_whitespace(get_arxiv_source(arxiv_id)),
                "summarizer_cfg": summarizer,
                "critic_cfg": critic,
            }

            print("Running critic ...")
            critic_res = run_critic(
                summary_prompt,
                paper,
                summary_res["text"],
                summary_res["reasoning"],
                critic,
            )
            print(f"Critic finished in {critic_res['elapsed']:.2f}s.")
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            print("Skipping this critique (no file written).\n")
            continue

        path = write_criticism_file(
            output_dir, arxiv_id, paper, summary_prompt, summary_res, critic_res
        )
        print(f"Wrote: {path}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a paper summary with one LLM and critique it with "
            "another, proposing changes to the summary prompt."
        ),
        epilog=(
            "Examples:\n"
            "  %(prog)s --summarizer-config llm_config_qwen3.5-35b-a3b.json "
            "--critic-config llm_config_deepseek4pro.json --papers ids.txt\n"
            "  %(prog)s --summarizer-config s.json --critic-config c.json "
            "--category astro-ph.CO --count 3"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Instead of running new critiques, load all stored criticisms "
             "from --output-dir, extract their proposed prompt changes, and "
             "ask the critic LLM to produce an updated summary prompt.",
    )
    parser.add_argument(
        "--summarizer-config",
        required=True,
        metavar="CONFIG",
        help="llm_config.json-like file for the summarizer model.",
    )
    parser.add_argument(
        "--critic-config",
        required=True,
        metavar="CONFIG",
        help="llm_config.json-like file for the critic model.",
    )
    parser.add_argument(
        "--papers",
        metavar="FILE",
        help="Optional text file with one arXiv ID per line. If omitted, a "
             "paper is drawn from the most recent arXiv mailing (--category).",
    )
    parser.add_argument(
        "--category",
        default="astro-ph.CO",
        help="arXiv category used to fetch the latest mailing when --papers "
             "is not given (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        default=_DEFAULT_OUTPUT_DIR,
        help="Directory for criticism files (default: %(default)s).",
    )
    parser.add_argument(
        "--system-prompt",
        metavar="FILE",
        help="Optional .txt file to replace the default summarization prompt.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of critiques to run in this invocation (default: 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible paper selection.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional sampling temperature, merged into both models' kwargs.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    try:
        if not args.summarize:
            if args.count < 1:
                raise SystemExit("--count must be at least 1.")
            if args.papers is None and not args.category:
                raise SystemExit("--category is required when --papers is not given.")
            cmd_run(args)
        else:
            cmd_run(args)

        if args.summarize:
            cmd_summarize(args)
        
    except KeyboardInterrupt:
        print("\nInterrupted — files written so far are saved.")


if __name__ == "__main__":
    main()
