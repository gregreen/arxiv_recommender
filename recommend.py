#!/usr/bin/env python3
"""
Quick recommendation check.

Usage (from project root):
    python recommend.py [--top N] [--liked experiments/my_papers.txt]
"""

import argparse
import sys

import numpy as np
from sklearn.preprocessing import StandardScaler

from arxiv_lib.config import EMBEDDING_DIM, RBF_GAMMAS, RBF_PCA_COMPONENTS
from arxiv_lib.ingest import load_embedding_cache, load_from_arxiv_metadata_cache
from arxiv_lib.scoring import ScoringModel


def load_paper_ids(path: str) -> list[str]:
    import re
    ids = []
    with open(path) as f:
        for line in f:
            line = line.strip().lower()
            if line.startswith("#") or not line:
                continue
            new_id = line
            if new_id.startswith("arxiv:"):
                new_id = new_id[6:]
            # Verify that the ID looks like a valid arXiv ID using regex
            if not re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", new_id):
                print(f"Warning: Skipping invalid arXiv ID '{new_id}' in {path}")
                continue
            ids.append(new_id)
            
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--liked", default="experiments/my_papers.txt")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    # Load liked paper IDs and embedding cache
    liked_ids = load_paper_ids(args.liked)
    print(f"Liked papers : {len(liked_ids)}")

    embeddings = load_embedding_cache()
    print(f"Embedding DB : {len(embeddings)} vectors")

    # Load embeddings for liked papers, and identify any missing ones
    liked_with_embeddings = [aid for aid in liked_ids if aid in embeddings]
    missing = [aid for aid in liked_ids if aid not in embeddings]
    print(f"Liked with embeddings: {len(liked_with_embeddings)}")
    if missing:
        print(f"  (missing embeddings for: {missing})")

    if len(liked_with_embeddings) < 2:
        print("Need at least 2 liked papers with embeddings to score.")
        sys.exit(1)

    all_ids = np.array(list(embeddings.keys()))
    arxiv_ids = np.array(all_ids)
    vectors = np.array(
        [embeddings[aid][:EMBEDDING_DIM] for aid in arxiv_ids], dtype=np.float32
    )

    liked_set = set(liked_ids)
    idx_pos = np.array([aid in liked_set for aid in arxiv_ids], dtype=bool)

    # Train scoring model and apply to all papers
    model = ScoringModel.from_training_data(vectors[idx_pos], vectors[~idx_pos])
    scores_neg = model.score_embeddings(vectors[~idx_pos])
    scores_pos = model.score_positive_embeddings()

    # Convert scores to dictionary for easy lookup
    scores = dict(zip(all_ids[idx_pos], scores_pos))
    scores.update(dict(zip(all_ids[~idx_pos], scores_neg)))

    # Load titles for display
    metadata = load_from_arxiv_metadata_cache(all_ids)

    # Sort all papers by score
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    unseen = [(aid, s) for aid, s in ranked if aid not in liked_set]

    print(f"\n{'='*70}")
    print(f"Top {args.top} recommendations (unseen papers)")
    print(f"{'='*70}")
    for aid, score in unseen[: args.top]:
        title = metadata.get(aid, {}).get("title", "(no title cached)")
        print(f"  {score:+.2f}  https://arxiv.org/abs/{aid}")
        print(f"         {title}")
    
    print(f"\n{'='*70}")
    print(f"Bottom {args.top} recommendations (unseen papers)")
    print(f"{'='*70}")
    for aid, score in unseen[-args.top:]:
        title = metadata.get(aid, {}).get("title", "(no title cached)")
        print(f"  {score:+.2f}  https://arxiv.org/abs/{aid}")
        print(f"         {title}")

    print(f"\n{'='*70}")
    print("Scores for your liked papers (sanity check)")
    print(f"{'='*70}")
    for aid, score in ranked:
        if aid in liked_set:
            title = metadata.get(aid, {}).get("title", "(no title cached)")
            print(f"  {score:+.2f}  https://arxiv.org/abs/{aid}")
            print(f"         {title}")
    
    # Get serialized user model and print its coefficients (for debugging)
    # serialized_model = model.serialize()
    # coefficients = np.array(serialized_model['logistic_model']['coef']).flatten()
    coefficients = model.logistic_model.coef_.flatten()
    print('gamma     coeff     coeff_resid')
    print('---------------------------------')
    print(f'nearest   {coefficients[0]: >+8.5f}  {coefficients[len(RBF_GAMMAS)+1]: >+8.5f}')
    for i,gamma in enumerate(RBF_GAMMAS):
        coeff_full = coefficients[i+1]
        coeff_resid = coefficients[i+len(RBF_GAMMAS)+2]
        print(f"{gamma: >8.5f}  {coeff_full: >+8.5f}  {coeff_resid: >+8.5f}")


if __name__ == "__main__":
    main()
