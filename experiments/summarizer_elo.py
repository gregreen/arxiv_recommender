#!/usr/bin/env python3
"""
Blind head-to-head comparison of LLM paper summaries, with Elo ratings.

This script runs one blind A/B matchup at a time: it randomly picks a paper
from a provided list and a random pair of models, generates a fresh summary
from each (no caching, so temperature can vary the output between repeats),
presents the two summaries side by side (labelled only "A" and "B"), and asks
the human judge which one is better.

The verdict is appended to a results CSV (result is +1 / 0 / -1 relative to
model_1, exactly like chess scoring), and the full summaries together with the
chain-of-thought (reasoning) traces are written to a per-match text file.

The same matchup (same models + same paper) may be run any number of times;
every run creates a new, uniquely named match file and CSV row.

Usage:
    # Run a single blind matchup (a random paper and a random pair of models):
    python experiments/summarizer_elo.py \
        --configs llm_config_qwen3.5-35b-a3b.json llm_config_deepseek4pro.json \
        --papers experiments/my_papers.txt

    # Run 5 matchups in one invocation:
    python experiments/summarizer_elo.py --count 5 \
        --configs c1.json c2.json c3.json --papers ids.txt

    # Compute Elo ratings from the accumulated results:
    python experiments/summarizer_elo.py --elo [--results path/to/results.csv]

    # Compute order-invariant Bradley-Terry (BayesElo) MAP ratings:
    python experiments/summarizer_elo.py --elo --method bradley-terry

    # Compute order-insensitive classic Elo (32 shuffled epochs, decaying K):
    python experiments/summarizer_elo.py --elo --method randomized-elo

Config files are llm_config.json-like: only the "summary" section is used, and
each file must contain an inline "api_key" in that section, e.g.
    {"summary": {"name": "model-alias", "model": "provider/model",
                 "base_url": "https://.../v1", "api_key": "sk-...",
                 "max_input_tokens": 98304,
                 "cot_closing_tags": ["</think>", "</reasoning>"],
                 "kwargs": {"extra_body": {"reasoning": {"enabled": true}}}}}

This script is intentionally standalone (lives under experiments/). It reuses
the project's metadata + LaTeX source caches, but never touches the production
summary cache and never modifies any web/production code.
"""

import argparse
import csv
import json
import os
import random
import sys
import time
import uuid
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
    get_arxiv_metadata,
    get_arxiv_source,
)

_EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
_MATCHES_DIR = os.path.join(_EXPERIMENTS_DIR, "summarizer_matches")
_DEFAULT_RESULTS = os.path.join(_MATCHES_DIR, "results.csv")

_RESULTS_FIELDNAMES = [
    "match_id",
    "arxiv_id",
    "model_1",
    "model_2",
    "shown_a",
    "shown_b",
    "result",
]

# Result is +1/0/-1 relative to model_1; score (S) used by the Elo update.
_SCORE_FOR_RESULT = {1: 1.0, 0: 0.5, -1: 0.0}


# ---------------------------------------------------------------------------
# Config / input loading
# ---------------------------------------------------------------------------

def load_models(config_paths: list[str], temperature: float | None) -> list[dict]:
    """Load and validate one llm_config-like JSON file per model."""
    models: list[dict] = []
    for path in config_paths:
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

        models.append(
            {
                "name": s.get("name") or s.get("model") or os.path.basename(path),
                "model": s.get("model", ""),
                "base_url": s.get("base_url", "https://router.huggingface.co/v1"),
                "api_key": api_key,
                "max_input_tokens": s.get("max_input_tokens", 98304),
                "cot_closing_tags": s.get("cot_closing_tags", []),
                "kwargs": kwargs,
            }
        )

    names = [m["name"] for m in models]
    if len(set(names)) != len(names):
        raise SystemExit(
            "Duplicate model names detected; add a distinct 'name' field to "
            "each config's 'summary' section."
        )
    if len(models) < 2:
        raise SystemExit("Need at least two models via --configs.")
    return models


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
    """Read one arXiv ID per line; blank lines and '#' comments are ignored.

    Each ID may optionally carry an 'arXiv:' prefix; it is stripped and the ID
    is validated via arxiv_lib.arxiv_id.validate_arxiv_id.
    """
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


# ---------------------------------------------------------------------------
# Summarization (reuses the project prompt + caches; never the summary cache)
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


def generate_summary(arxiv_id: str, model_cfg: dict, system_prompt: str) -> dict:
    """Generate one summary (+ reasoning trace) for *arxiv_id* using *model_cfg*.

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

    raw_latex = get_arxiv_source(arxiv_id)
    raw_latex = compress_latex_whitespace(raw_latex)

    max_tok = model_cfg["max_input_tokens"]
    n_tok = count_tokens(raw_latex)
    truncated = False
    if n_tok > max_tok:
        chars_per_token = len(raw_latex) / max(n_tok, 1)
        chars_to_keep = int(max_tok * chars_per_token)
        raw_latex = raw_latex[:chars_to_keep] + "\n\n[... source truncated ...]"
        truncated = True

    user_message = (
        f"Title: {title}\n"
        f"Authors: {authors}\n"
        f"Abstract: {abstract}\n\n"
        f"LaTeX Source:\n{raw_latex}"
    )

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
        raise RuntimeError(f"Summary API call failed for {arxiv_id}: {e}")

    message = response.choices[0].message
    content = message.content
    if content is None:
        raise ValueError(
            f"LLM returned null content "
            f"(finish_reason={response.choices[0].finish_reason!r})"
        )
    content = content.strip()

    reasoning = _extract_reasoning(message)
    cot_tags = model_cfg["cot_closing_tags"]
    summary = content
    if cot_tags:
        best = max(
            (summary.rfind(tag) + len(tag)
             for tag in cot_tags
             if summary.rfind(tag) != -1),
            default=0,
        )
        if best and not reasoning:
            reasoning = summary[:best].strip()
        summary = summary[best:].strip()

    return {
        "summary": summary,
        "reasoning": reasoning,
        "content": content,
        "response_json": response.model_dump_json(indent=2),
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "truncated": truncated,
        "n_input_tokens": n_tok,
        "elapsed": elapsed,
    }


# ---------------------------------------------------------------------------
# Presentation + recording
# ---------------------------------------------------------------------------

def prompt_judgement():
    """Ask the human judge; returns 1 (A wins), 0 (draw), -1 (B wins),
    'skip', or 'quit'."""
    while True:
        ans = input(
            "Which summary is better? [a] A wins  [b] B wins  [d] draw  "
            "[s] skip  [q] quit: "
        ).strip().lower()
        if ans in ("a",):
            return 1
        if ans in ("b",):
            return -1
        if ans in ("d",):
            return 0
        if ans in ("s",):
            return "skip"
        if ans in ("q",):
            return "quit"
        print("Invalid choice. Please enter a, b, d, s, or q.")


def append_result(results_path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    new_file = not os.path.exists(results_path)
    with open(results_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_RESULTS_FIELDNAMES)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def _format_model_block(label: str, model_cfg: dict, res: dict) -> str:
    separator = "=" * 72
    thin = "-" * 72
    reasoning = res["reasoning"] or "(none captured)"
    return (
        f"{separator}\n"
        f"MODEL {label}: {model_cfg['name']}\n"
        f"  provider model: {model_cfg['model']}\n"
        f"  base_url: {model_cfg['base_url']}\n"
        f"  elapsed: {res['elapsed']:.2f}s  "
        f"estimated input tokens: {res['n_input_tokens']:,}"
        f"{'  (source truncated)' if res['truncated'] else ''}\n"
        f"{thin}\n"
        f"REASONING (CoT)\n"
        f"{thin}\n"
        f"{reasoning}\n"
        f"{thin}\n"
        f"SUMMARY\n"
        f"{thin}\n"
        f"{res['summary']}\n"
    )


def write_match_file(
    matches_dir: str,
    match_id: str,
    arxiv_id: str,
    model_1: dict,
    model_2: dict,
    shown_a: dict,
    shown_b: dict,
    res_by_name: dict,
    result: int,
    displayed_result: str,
) -> str:
    os.makedirs(matches_dir, exist_ok=True)
    path = os.path.join(matches_dir, f"match_{match_id}.txt")

    separator = "=" * 72
    paper = res_by_name[model_1["name"]]
    lines = [
        f"MATCH_ID: {match_id}",
        f"ARXIV_ID: {arxiv_id}",
        f"TITLE: {paper['title']}",
        f"AUTHORS: {paper['authors']}",
        f"ABSTRACT: {paper['abstract']}",
        f"MODEL_1: {model_1['name']}",
        f"MODEL_2: {model_2['name']}",
        f"RESULT (for model_1): {result:+d}",
        f"DISPLAYED RESULT: {displayed_result}",
        f"SHOWN ORDER: A = {shown_a['name']}  |  B = {shown_b['name']}",
        "",
        _format_model_block("A", shown_a, res_by_name[shown_a["name"]]),
        _format_model_block("B", shown_b, res_by_name[shown_b["name"]]),
        f"{separator}\nRAW RESPONSE — MODEL A ({shown_a['name']})\n{separator}",
        res_by_name[shown_a["name"]]["response_json"],
        f"{separator}\nRAW RESPONSE — MODEL B ({shown_b['name']})\n{separator}",
        res_by_name[shown_b["name"]]["response_json"],
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def _make_match_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{uuid.uuid4().hex[:6]}"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_run(args) -> None:
    models = load_models(args.configs, temperature=args.temperature)
    papers = load_papers(args.papers)
    rng = random.Random(args.seed)

    if args.system_prompt:
        with open(args.system_prompt, "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
    else:
        system_prompt = _config.SUMMARIZE_SYSTEM_PROMPT

    results_path = os.path.abspath(args.results)
    matches_dir = os.path.abspath(args.matches_dir)

    for i in range(args.count):
        arxiv_id = rng.choice(papers)
        a, b = rng.sample(models, 2)
        model_1, model_2 = sorted((a, b), key=lambda m: m["name"])

        print("\n" + "=" * 72)
        print(f"Match {i + 1}/{args.count}: paper {arxiv_id}")
        print("=" * 72)

        res_by_name: dict[str, dict] = {}
        for idx, m in enumerate((model_1, model_2), start=1):
            print(f"Generating summary {idx}/2 ...")
            try:
                res_by_name[m["name"]] = generate_summary(arxiv_id, m, system_prompt)
            except Exception as e:
                print(f"ERROR: {e}", file=sys.stderr)
                print("Skipping this matchup (no result recorded).\n")
                break
        else:
            # Both summaries succeeded.
            if rng.random() < 0.5:
                shown_a, shown_b = model_1, model_2
            else:
                shown_a, shown_b = model_2, model_1

            paper = res_by_name[shown_a["name"]]

            print("\n" + "=" * 72)
            print("PAPER")
            print("=" * 72)
            print(f"Title:    {paper['title']}")
            print(f"Authors:  {paper['authors']}")
            print(f"Abstract: {paper['abstract']}")

            print("\n" + "=" * 72)
            print("SUMMARY A")
            print("=" * 72)
            print(res_by_name[shown_a["name"]]["summary"])
            print("\n" + "=" * 72)
            print("SUMMARY B")
            print("=" * 72)
            print(res_by_name[shown_b["name"]]["summary"])
            print()

            choice = prompt_judgement()
            if choice == "skip":
                print("Skipped — no result recorded.\n")
                continue
            if choice == "quit":
                print("Quit.\n")
                break

            # choice is +1/0/-1 relative to the DISPLAYED order (A, B).
            # Map it to +1/0/-1 relative to model_1.
            result = choice if shown_a["name"] == model_1["name"] else -choice
            displayed_result = {1: "A wins", 0: "draw", -1: "B wins"}[choice]

            match_id = _make_match_id()
            row = {
                "match_id": match_id,
                "arxiv_id": arxiv_id,
                "model_1": model_1["name"],
                "model_2": model_2["name"],
                "shown_a": shown_a["name"],
                "shown_b": shown_b["name"],
                "result": result,
            }
            append_result(results_path, row)
            path = write_match_file(
                matches_dir,
                match_id,
                arxiv_id,
                model_1,
                model_2,
                shown_a,
                shown_b,
                res_by_name,
                result,
                displayed_result,
            )
            print(f"Recorded: {displayed_result} (match_id {match_id}).")
            print(f"Match file: {path}\n")


def read_results(results_path: str) -> list[dict]:
    """Read and validate match rows from the results CSV.

    Returns rows as {"model_1", "model_2", "result"} with result in {-1, 0, 1}
    (+1 / 0 / -1 relative to model_1). Invalid rows and self-matches are
    skipped.
    """
    rows: list[dict] = []
    with open(results_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m1 = row.get("model_1")
            m2 = row.get("model_2")
            try:
                result = int(row["result"])
            except (TypeError, ValueError):
                continue
            if result not in _SCORE_FOR_RESULT or not m1 or not m2 or m1 == m2:
                continue
            rows.append({"model_1": m1, "model_2": m2, "result": result})
    return rows


def _compute_stats(rows: list[dict], names) -> dict[str, dict]:
    """Per-model win/draw/loss counts. Small loop; not a compute hotspot."""
    stats = {n: {"games": 0, "wins": 0, "draws": 0, "losses": 0} for n in names}
    for row in rows:
        m1, m2, result = row["model_1"], row["model_2"], row["result"]
        stats[m1]["games"] += 1
        stats[m2]["games"] += 1
        if result == 1:
            stats[m1]["wins"] += 1
            stats[m2]["losses"] += 1
        elif result == -1:
            stats[m2]["wins"] += 1
            stats[m1]["losses"] += 1
        else:
            stats[m1]["draws"] += 1
            stats[m2]["draws"] += 1
    return stats


def _print_elo_table(args, rows: list[dict]) -> None:
    """Sequential fixed-K Elo update, in CSV file order (order-sensitive)."""
    k = args.k
    names = sorted({r["model_1"] for r in rows} | {r["model_2"] for r in rows})
    ratings = {name: args.start_rating for name in names}

    for row in rows:
        m1, m2, result = row["model_1"], row["model_2"], row["result"]
        r1, r2 = ratings[m1], ratings[m2]
        e1 = 1.0 / (1.0 + 10 ** ((r2 - r1) / 400.0))
        s1 = _SCORE_FOR_RESULT[result]
        ratings[m1] = r1 + k * (s1 - e1)
        ratings[m2] = r2 + k * ((1.0 - s1) - (1.0 - e1))

    stats = _compute_stats(rows, names)
    print(
        f"Elo ratings from {os.path.abspath(args.results)} "
        f"(K={k:g}, start={args.start_rating:g})\n"
    )
    print(f"{'Model':<40} {'Rating':>8} {'Games':>6} {'W':>4} {'D':>4} {'L':>4}")
    print("-" * 72)
    for name in sorted(names, key=lambda n: ratings[n], reverse=True):
        s = stats[name]
        print(
            f"{name:<40} {ratings[name]:>8.1f} {s['games']:>6} "
            f"{s['wins']:>4} {s['draws']:>4} {s['losses']:>4}"
        )


def _fit_bradley_terry(rows, sigma, base):
    """Batch MAP fit of the draw-margin Bradley-Terry (BayesElo) model.

    Latent strengths theta_i with logistic link and draw margin kappa:
        P(i wins) = sigmoid(d - kappa)
        P(draw)   = sigmoid(d + kappa) - sigmoid(d - kappa)
        P(j wins) = sigmoid(-d - kappa)      where d = theta_i - theta_j
    plus a Gaussian prior N(0, sigma^2) on theta.

    This is a single-batch, match-order-invariant fit. JAX is imported lazily
    so the run path and the plain-Elo path stay lightweight. The negative
    log-posterior, gradient, and Hessian are computed with JAX autodiff on a
    fully vectorized (no Python loop over games) objective, and optimization
    uses JAX's own BFGS (jax.scipy.optimize.minimize).
    """
    if sigma <= 0:
        raise SystemExit("--sigma must be > 0.")
    # Avoid noisy TPU/GPU backend probes: this is a tiny CPU-sized optimization.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    try:
        import jax
        import jax.numpy as jnp
        from jax.scipy.optimize import minimize as _jax_minimize
    except ImportError as e:
        raise SystemExit(
            "The bradley-terry method requires jax. "
            f"Missing module: {e.name}. Install with: pip install jax"
        )
    jax.config.update("jax_enable_x64", True)

    names = sorted({r["model_1"] for r in rows} | {r["model_2"] for r in rows})
    if len(names) < 2:
        raise SystemExit("Bradley-Terry fit needs at least two distinct models.")
    name_to_idx = {n: i for i, n in enumerate(names)}

    i_idx = jnp.asarray([name_to_idx[r["model_1"]] for r in rows], dtype=jnp.int32)
    j_idx = jnp.asarray([name_to_idx[r["model_2"]] for r in rows], dtype=jnp.int32)
    # outcome: 0 = model_1 wins, 1 = draw, 2 = model_2 wins
    outcome = jnp.asarray(
        [{1: 0, 0: 1, -1: 2}[r["result"]] for r in rows], dtype=jnp.int32
    )

    log_sigmoid = lambda x: -jnp.logaddexp(0.0, -x)

    def logdiffexp(a, b):
        # log(exp(a) - exp(b)) assuming a >= b.
        return a + jnp.log1p(-jnp.exp(b - a))

    def neg_log_posterior(x):
        theta = x[:-1]
        kappa = jnp.logaddexp(0.0, x[-1])  # softplus keeps kappa > 0
        d = theta[i_idx] - theta[j_idx]
        logp_win1 = log_sigmoid(d - kappa)
        logp_win2 = log_sigmoid(-d - kappa)
        logp_draw = logdiffexp(log_sigmoid(d + kappa), log_sigmoid(d - kappa))
        logp = jnp.where(
            outcome == 0,
            logp_win1,
            jnp.where(outcome == 1, logp_draw, logp_win2),
        )
        prior = 0.5 * jnp.sum(theta * theta) / (sigma * sigma)
        return -jnp.sum(logp) + prior

    m = len(names)
    x0 = jnp.zeros(m + 1)
    x0 = x0.at[-1].set(float(jnp.log(jnp.expm1(1.0))))  # kappa0 = 1

    res = _jax_minimize(neg_log_posterior, x0, method="BFGS")
    x_opt = res.x

    warnings = []
    status = getattr(res, "status", 0)
    if status not in (0, None):
        warnings.append(f"optimizer status={status}: {getattr(res, 'message', '')}")

    theta = x_opt[:-1]
    kappa = float(jnp.logaddexp(0.0, x_opt[-1]))

    # Exact Hessian via JAX autodiff; invert for the Laplace covariance.
    cov = jnp.linalg.inv(jax.hessian(neg_log_posterior)(x_opt))
    se_theta = jnp.sqrt(jnp.diag(cov)[:-1])

    scale = float(400.0 / jnp.log(10.0))
    ratings = {names[i]: base + scale * float(theta[i]) for i in range(m)}
    ses = {names[i]: scale * float(se_theta[i]) for i in range(m)}

    return names, ratings, ses, kappa, _compute_stats(rows, names), warnings


def _print_bradley_terry_table(args, rows: list[dict]) -> None:
    names, ratings, ses, kappa, stats, warnings = _fit_bradley_terry(
        rows, sigma=args.sigma, base=args.start_rating
    )
    print(
        f"Bradley-Terry (BayesElo) MAP ratings from {os.path.abspath(args.results)}\n"
        f"(sigma={args.sigma:g}, base={args.start_rating:g}, "
        f"draw margin kappa={kappa:.4f})\n"
    )
    if warnings:
        print("Warnings: " + "; ".join(warnings) + "\n")
    print(
        f"{'Model':<40} {'Rating':>8} {'±SE':>7} "
        f"{'Games':>6} {'W':>4} {'D':>4} {'L':>4}"
    )
    print("-" * 78)
    for name in sorted(names, key=lambda n: ratings[n], reverse=True):
        s = stats[name]
        print(
            f"{name:<40} {ratings[name]:>8.1f} {ses[name]:>7.1f} "
            f"{s['games']:>6} {s['wins']:>4} {s['draws']:>4} {s['losses']:>4}"
        )


def _randomized_elo_ratings(rows, k0, k_min, epochs, start_rating, seed):
    """Classic Elo over *epochs* shuffled passes with a slowly decaying K.

    Each pass is a fresh random permutation of the matches, and the update
    constant follows a Robbins-Monro-style schedule
        K(t) = max(k_min, k0 / (1 + t * c)),
    with c chosen so that K(total_steps) == k_min. Many shuffled passes with a
    decreasing K approximate the Bradley-Terry maximum-likelihood solution via
    stochastic gradient descent, making the result nearly independent of the
    original match order.
    """
    if k0 <= 0:
        raise SystemExit("--k must be > 0 for randomized-elo.")
    if k_min <= 0:
        raise SystemExit("--k-min must be > 0.")
    if epochs < 1:
        raise SystemExit("--epochs must be >= 1.")

    names = sorted({r["model_1"] for r in rows} | {r["model_2"] for r in rows})
    ratings = {name: start_rating for name in names}
    rng = random.Random(seed)

    total_steps = epochs * len(rows)
    c = (k0 / k_min - 1.0) / total_steps

    t = 0
    for _ in range(epochs):
        for row in rng.sample(rows, len(rows)):
            m1, m2, result = row["model_1"], row["model_2"], row["result"]
            r1, r2 = ratings[m1], ratings[m2]
            e1 = 1.0 / (1.0 + 10 ** ((r2 - r1) / 400.0))
            s1 = _SCORE_FOR_RESULT[result]
            k = max(k_min, k0 / (1.0 + t * c))
            ratings[m1] = r1 + k * (s1 - e1)
            ratings[m2] = r2 + k * ((1.0 - s1) - (1.0 - e1))
            t += 1

    return names, ratings, _compute_stats(rows, names)


def _print_randomized_elo_table(args, rows: list[dict]) -> None:
    names, ratings, stats = _randomized_elo_ratings(
        rows,
        k0=args.k,
        k_min=args.k_min,
        epochs=args.epochs,
        start_rating=args.start_rating,
        seed=args.seed,
    )
    print(
        f"Randomized Elo ratings from {os.path.abspath(args.results)} "
        f"(K0={args.k:g}, K_min={args.k_min:g}, epochs={args.epochs}, "
        f"start={args.start_rating:g})\n"
    )
    print(f"{'Model':<40} {'Rating':>8} {'Games':>6} {'W':>4} {'D':>4} {'L':>4}")
    print("-" * 72)
    for name in sorted(names, key=lambda n: ratings[n], reverse=True):
        s = stats[name]
        print(
            f"{name:<40} {ratings[name]:>8.1f} {s['games']:>6} "
            f"{s['wins']:>4} {s['draws']:>4} {s['losses']:>4}"
        )


def _print_head_to_head_matrix(rows: list[dict]) -> None:
    """Print a white-vs-black head-to-head score matrix.

    Row = white, column = black. Each off-diagonal cell shows the white
    model's total score (W=+1, D=0, L=-1) over the number of games between the
    pair, e.g. "+1.0 / 4". The diagonal is "--", as are pairs that never met.
    The matrix is identical regardless of the rating method.
    """
    names = sorted({r["model_1"] for r in rows} | {r["model_2"] for r in rows})
    scores = {(a, b): 0.0 for a in names for b in names}
    counts = {(a, b): 0 for a in names for b in names}

    for row in rows:
        m1, m2, result = row["model_1"], row["model_2"], row["result"]
        scores[(m1, m2)] += result
        counts[(m1, m2)] += 1
        scores[(m2, m1)] += -result
        counts[(m2, m1)] += 1

    name_w = max(len(n) for n in names)
    cell_w = max(name_w, 10)

    print(
        "\nHead-to-head matrix "
        "(row = white, column = black; W=+1, D=0, L=-1):\n"
    )
    print(" " * name_w + "".join(f"{n:>{cell_w}}" for n in names))
    for a in names:
        cells = []
        for b in names:
            if a == b or counts[(a, b)] == 0:
                cells.append(f"{'--':>{cell_w}}")
            else:
                text = f"{scores[(a, b)]:+.1f} / {counts[(a, b)]}"
                cells.append(f"{text:>{cell_w}}")
        print(f"{a:<{name_w}}" + "".join(cells))


def cmd_elo(args) -> None:
    results_path = os.path.abspath(args.results)
    if not os.path.exists(results_path):
        raise SystemExit(f"Results file {results_path!r} not found.")

    rows = read_results(results_path)
    if not rows:
        raise SystemExit(f"No valid rows found in {results_path!r}.")

    if args.method == "elo":
        _print_elo_table(args, rows)
    elif args.method == "bradley-terry":
        _print_bradley_terry_table(args, rows)
    else:
        _print_randomized_elo_table(args, rows)

    _print_head_to_head_matrix(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Blind head-to-head comparison of LLM paper summaries with Elo ratings."
        ),
        epilog=(
            "Examples:\n"
            "  %(prog)s --configs c1.json c2.json --papers ids.txt\n"
            "  %(prog)s --count 5 --configs c1.json c2.json c3.json --papers ids.txt\n"
            "  %(prog)s --elo [--results summarizer_matches/results.csv]\n"
            "  %(prog)s --elo --method bradley-terry [--sigma 2.0]\n"
            "  %(prog)s --elo --method randomized-elo [--epochs 32 --k-min 1.0]"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--elo",
        action="store_true",
        help="Compute Elo ratings from --results instead of running matches.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        metavar="CONFIG",
        help="llm_config.json-like files (one per model). Required unless --elo.",
    )
    parser.add_argument(
        "--papers",
        metavar="FILE",
        help="Text file with one arXiv ID per line. Required unless --elo.",
    )
    parser.add_argument(
        "--results",
        default=_DEFAULT_RESULTS,
        help="Results CSV path (default: %(default)s).",
    )
    parser.add_argument(
        "--matches-dir",
        default=_MATCHES_DIR,
        help="Directory for match files (default: %(default)s).",
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
        help="Number of matchups to run in this invocation (default: 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible paper/pair selection and "
             "randomized-elo shuffling.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional sampling temperature, merged into each model's kwargs.",
    )
    parser.add_argument(
        "--k",
        type=float,
        default=32.0,
        help="Elo K-factor (default: 32); initial K for randomized-elo.",
    )
    parser.add_argument(
        "--start-rating",
        type=float,
        default=1500.0,
        help="Starting Elo rating for all models (default: 1500).",
    )
    parser.add_argument(
        "--method",
        choices=["elo", "bradley-terry", "randomized-elo"],
        default="elo",
        help="Rating method used with --elo (default: elo). "
             "'bradley-terry' is a batch, match-order-invariant MAP fit; "
             "'randomized-elo' runs classic Elo over shuffled epochs with a "
             "decaying K.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=2.0,
        help="Gaussian prior std (natural-log scale) for --method bradley-terry "
             "(default: 2.0).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=32,
        help="Number of shuffled passes for --method randomized-elo (default: 32).",
    )
    parser.add_argument(
        "--k-min",
        type=float,
        default=1.0,
        help="Lower bound for the decaying K in --method randomized-elo "
             "(default: 1.0).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.elo:
        cmd_elo(args)
        return

    if not args.configs or not args.papers:
        parser.error("--configs and --papers are required (unless using --elo).")
    if args.count < 1:
        parser.error("--count must be at least 1.")

    try:
        cmd_run(args)
    except KeyboardInterrupt:
        print("\nInterrupted — results recorded so far are saved.")


if __name__ == "__main__":
    main()
