arXiv recommender
=================

This repository contains code for embedding arXiv papers and using those embeddings to make recommendations. The main file is `arxiv_embedding.py`, which contains functions for loading paper metadata, embedding papers, and making recommendations.

How it works
------------

The recommender works by converting each paper into a vector embedding, based on its contents. Then, given a set of "my papers" (papers that the user is interested in), it computes the similarity between those papers and all other papers in the embedding space, and returns the most similar papers as recommendations.

More details about the embedding process:
- The arXiv metadata (title, author list and abstract) and the raw LaTeX source of the paper are fed into an LLM (currently Qwen3-Next-80B-A3B-Thinking) to produce a 7-paragraph structured summary of the paper (Keywords, Scientific Questions, Data, Methods, Results, Conclusions, Key Takeaway).
- The structured summary is fed into an LLM (currently Qwen3-Embedding-8B) to produce a 1024-dimensional embedding vector for the paper. These embeddings are truncated to 64 dimensions in practice, since they use a Matryoshka embedding format in which the first 64 dimensions capture most of the information.

More details about the recommendation process:
- The user provides a list of "my papers" (papers that they are interested in).
- The system loads a second set of "candidate papers" (papers that are being considered for recommendation).
- The distance matrix between "my papers" and the "candidate papers" is computed. For each candidate paper, the minimum distance to one of "my papers" is used as a feature. Additionally, the mean radial basis function (RBF) values of the distances to "my papers" for several different gamma values (representing different distance scales) are used as additional features. The distance scales are logarithmically spaced. The same features are also computed in a subspace in which "my papers" have low variance, in order to measure how far out-of-distribution each candidate paper is relative to "my papers."
- The features are fed into a simple logistic regression model that tries to predict whether a paper belongs to "my papers" or not. The model thus produces a measure of how similar each candidate paper is to "my papers," which is used to rank the candidate papers for recommendation.

The recommendation process is fast. It is the embedding step that takes time, since it requires two LLM calls per paper. At present, the LLM calls are made remotely via the Hugging Face API.

How to get started
------------------

1. Export a list of papers you like to a text file (the more, the better). It should have one arXiv ID per line, and each line should start with "arXiv:". For example:
```
arXiv:1234.56789
arXiv:9876.54321
```
2. Create Hugging Face and Semantic Scholar accounts and get API keys for both. Put these tokens in a .json file with the following format:
```
{
    "huggingface": "your_huggingface_token_here",
    "semantic_scholar": "your_semanticscholar_token_here"
}
```
3. Run the following code snippet to embed your papers and get recommendations:
```python
from arxiv_embedding import load_tokens, embed_latest_mailing, rbf_svd_example
tokens = load_tokens()
embeddings = embed_latest_mailing("astro-ph", tokens)
rbf_svd_example()
```

This will embed the latest mailing of astro-ph papers, and then run the RBF SVD example, which uses the embeddings to make recommendations based on the papers you provided in step 1. Warning: embedding an entire mailing may cost you a $1-2 in Hugging Face inference API calls.