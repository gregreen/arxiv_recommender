#/usr/bin/env python3
"""
Experiment / exploration script.

All ingest and scoring logic has been moved to the arxiv_lib package.
This file re-exports those symbols for backwards compatibility and contains
the standalone *_example() functions used for exploratory analysis.
"""

import sys
import os

# Allow running directly from the experiments/ directory without installing the package.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import glob
import numpy as np

# -- Re-export everything from the library so existing call sites still work --
from arxiv_lib.config import (
    EMBEDDING_CACHE_DB,
    EMBEDDING_CACHE_FILE,
    SOURCE_CACHE_DIR,
    METADATA_CACHE_DIR,
    SUMMARY_CACHE_DIR,
    TOKENS_FILE,
    USER_AGENT,
    ARXIV_CATEGORIES,
    EMBEDDING_DIM,
    SCORING_VERSION,
    RBF_GAMMAS,
    RBF_PCA_COMPONENTS,
    SUMMARY_PROVIDER,
    SUMMARY_MODEL,
    SUMMARY_MAX_TOKENS,
    EMBEDDING_PROVIDER,
    EMBEDDING_MODEL,
    EMBEDDING_MAX_TOKENS,
)
from arxiv_lib.ingest import (
    _init_embedding_db,
    load_embedding_cache,
    save_embedding_cache,
    load_from_arxiv_metadata_cache,
    write_to_arxiv_metadata_cache,
    get_arxiv_metadata,
    fetch_arxiv_metadata,
    fetch_arxiv_metadata_html,
    fetch_arxiv_metadata_s2,
    raise_on_arxiv_category,
    fetch_latest_mailing_ids,
    fetch_arxiv_embedding,
    embed_arxiv_ids,
    embed_latest_mailing,
    get_arxiv_source,
    compress_latex_whitespace,
    summarize_arxiv_paper,
    gen_arxiv_embedding,
    load_tokens,
    report_compression_stats,
)
from arxiv_lib.scoring import (
    rbf_scoring,
    calculate_projection_matrices,
    project_to_subspace,
    train_logistic_model,
    build_rbf_features,
    fit_scoring_model,
    apply_scoring_model,
    score_papers_for_user,
)


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
    with open("my_papers.txt", "r") as f:
        my_papers = [
            line.strip()[6:] for line in f
            if line.strip().startswith("arXiv:")
        ]
    
    # Make sure my papers are embedded
    tokens = load_tokens()
    embed_arxiv_ids(my_papers, tokens)
    
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
    # tokens = load_tokens()
    # embeddings = embed_latest_mailing("astro-ph", tokens)
    # fetch_arxiv_embedding("2603.28400", tokens)
    # embeddings = load_embedding_cache()

    # rbf_example()
    # svm_example()
    lnp = rbf_svd_example()
    # print(json.dumps(lnp, indent=2))
    # print(lnp['2603.28400'])
    
    return 0


if __name__ == "__main__":
    main()