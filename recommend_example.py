#!/usr/bin/env python3
"""
Demonstration of the recommendation pipeline via arxiv_lib/recommend.py.

Connects to app.db, trains (or loads from cache) a model for the given user,
and displays ranked recommendations for each time window.

Usage (from project root):
    python recommend_example.py [--user USERNAME] [--top N] [--window day|week|month]
                                [--show-coefficients]
"""

import argparse
import sys

from arxiv_lib.appdb import get_connection
from arxiv_lib.config import RBF_GAMMAS
from arxiv_lib.recommend import (
    NotEnoughDataError,
    get_or_train_model,
    get_recommendations,
    recommendations_are_stale,
)
from arxiv_lib.scoring import compute_model_hash


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_user(con, email: str | None) -> tuple[int, str]:
    if email:
        row = con.execute(
            "SELECT id, email FROM users WHERE email = ?", (email,)
        ).fetchone()
        if not row:
            print(f"Error: no user with email {email!r} found in app.db.")
            sys.exit(1)
    else:
        row = con.execute("SELECT id, email FROM users LIMIT 1").fetchone()
        if not row:
            print("Error: no users found in app.db.")
            sys.exit(1)
    return row[0], row[1]


def print_model_info(con, user_id: int, model_hash: str) -> None:
    cached = con.execute(
        "SELECT n_liked, n_disliked, trained_at FROM user_models"
        " WHERE user_id = ? AND model_hash = ?",
        (user_id, model_hash),
    ).fetchone()
    if cached:
        print(f"  Trained at      : {cached[2]}")
        print(f"  Liked papers    : {cached[0]}")
        print(f"  Disliked papers : {cached[1]}")
    print(f"  Model hash      : {model_hash}")


def print_coeff_table(model) -> None:
    coefficients = model.logistic_model.coef_.flatten()
    print(f"\n  {'gamma':>10}  {'coeff_full':>+12}  {'coeff_resid':>+12}")
    print(f"  {'-'*40}")
    print(f"  {'nearest':>10}  {coefficients[0]:>+12.5f}  {coefficients[len(RBF_GAMMAS)+1]:>+12.5f}")
    for i, gamma in enumerate(RBF_GAMMAS):
        print(f"  {gamma:>10.5f}  {coefficients[i+1]:>+12.5f}  {coefficients[i+len(RBF_GAMMAS)+2]:>+12.5f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run the recommendation pipeline against app.db.")
    parser.add_argument("--user",   default=None, help="User email (defaults to first user in db).")
    parser.add_argument("--top",    type=int, default=10, help="Recommendations to show per window.")
    parser.add_argument("--window", default=None, choices=["day", "week", "month"],
                        help="Show only one time window (default: all three).")
    parser.add_argument("--show-coefficients", action="store_true",
                        help="Print RBF kernel coefficients of the trained model.")
    args = parser.parse_args()

    con = get_connection()
    user_id, email = resolve_user(con, args.user)
    print(f"User: {email!r}  (id={user_id})")

    # ------------------------------------------------------------------ #
    # Compute hash and report cache status before training
    # ------------------------------------------------------------------ #
    rows = con.execute(
        "SELECT arxiv_id, liked FROM user_papers WHERE user_id = ? AND liked != 0",
        (user_id,),
    ).fetchall()
    liked_ids    = [r[0] for r in rows if r[1] == 1]
    disliked_ids = [r[0] for r in rows if r[1] == -1]
    model_hash   = compute_model_hash(liked_ids, disliked_ids)

    print(f"\nTraining data:")
    print(f"  Liked papers    : {len(liked_ids)}")
    print(f"  Disliked papers : {len(disliked_ids)}")
    print(f"  Cache is stale  : {recommendations_are_stale(con, user_id, model_hash)}")

    # ------------------------------------------------------------------ #
    # Train or load model
    # ------------------------------------------------------------------ #
    try:
        model, model_hash = get_or_train_model(con, user_id)
        con.commit()
    except NotEnoughDataError as e:
        print(f"\nNot enough data to train a model: {e}")
        con.close()
        sys.exit(1)

    print(f"\nModel:")
    print_model_info(con, user_id, model_hash)

    if args.show_coefficients:
        print("\nModel coefficients:")
        print_coeff_table(model)

    # ------------------------------------------------------------------ #
    # Recommendations per time window
    # ------------------------------------------------------------------ #
    windows = [args.window] if args.window else ["day", "week", "month"]

    for window in windows:
        try:
            recs = get_recommendations(con, user_id, window)
        except NotEnoughDataError as e:
            print(f"\n[{window}] Not enough data: {e}")
            continue

        unseen = [r for r in recs if not r["liked"]]
        liked  = [r for r in recs if r["liked"] == 1]

        print(f"\n{'='*70}")
        print(f"Window: {window!r}  —  {len(recs)} papers  "
              f"({len(liked)} liked, {len(unseen)} unseen)")
        print(f"{'='*70}")

        if not unseen:
            print("  (no unseen papers in this window)")
        else:
            top_n = unseen[: args.top]
            print(f"\n  Top {len(top_n)} unseen:")
            for r in top_n:
                authors = r["authors"]
                author_str = (authors[0] + (" et al." if len(authors) > 1 else "")) if authors else "—"
                print(f"  [{r['rank']:4d}] {r['score']:+.3f}  {r['published_date']}  "
                      f"https://arxiv.org/abs/{r['arxiv_id']}")
                print(f"          {r['title']}")
                print(f"          {author_str}")

            if len(unseen) > args.top:
                print(f"\n  Bottom {min(args.top, len(unseen))} unseen:")
                for r in unseen[-args.top:]:
                    print(f"  [{r['rank']:4d}] {r['score']:+.3f}  {r['published_date']}  "
                          f"https://arxiv.org/abs/{r['arxiv_id']}")
                    print(f"          {r['title']}")

        if liked:
            print(f"\n  Liked-paper scores (sanity check):")
            for r in liked[: args.top]:
                print(f"  [{r['rank']:4d}] {r['score']:+.3f}  {r['arxiv_id']}  "
                      f"{r['title'][:55]}")

    con.close()


if __name__ == "__main__":
    main()
