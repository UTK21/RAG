"""
bm25_index.py
=============
Sparse (keyword) retrieval using BM25.

Why a SECOND retriever?
-----------------------
Our dense retriever (FAISS + sentence-transformers) is great at MEANING but
weak on EXACT TOKENS. Try searching for `Vaswani`, `GPT-3.5`, `AX-9281` or
any other rare identifier — the embedding model squishes those into similar
vectors as their context words and you lose them.

BM25 is the opposite: pure keyword matching that REWARDS rare exact tokens
and is blind to meaning. Running BOTH in parallel and merging the results
("hybrid search") is the standard production fix.

What BM25 actually does
-----------------------
For each word in the query, BM25 scores a document on three factors:

    1. TF  (term frequency) — how many times the word appears in the doc.
           Crucially, it SATURATES — going from 1 to 2 mentions matters
           a lot; going from 50 to 51 barely moves the score. This stops
           keyword-stuffing from winning.

    2. IDF (inverse document frequency) — rare words across the whole
           corpus count more than common ones. "Transformer" is more
           informative than "the".

    3. LENGTH NORMALIZATION — penalize very long documents that match by
           accident (longer text → more chances to contain any word).

Final score = sum of these per-word contributions. No neural network, no
embeddings — just word counts and some clever math. Fast, deterministic,
nothing to download.

The library
-----------
`rank_bm25.BM25Okapi` — pure-Python implementation of the Okapi BM25
variant (the standard one). Tiny dependency, no compiled code.

Tokenization
------------
BM25 works on TOKENS, so we have to split text into words ourselves. We
use a simple regex `\\w+` and lowercase everything. This is crude but
predictable; production systems use stemming (`runs` → `run`) or a real
NLP tokenizer for better recall. Easy to swap in later.
"""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from pdf_loader import Chunk


# Word-token regex: keeps unicode letters/digits, drops punctuation. We
# lowercase before splitting so "Transformer" and "transformer" collapse
# to the same token — otherwise BM25 treats them as different words.
_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase + split on word boundaries. Used for BOTH chunks and queries."""
    return _TOKEN_RE.findall(text.lower())


def build_bm25(chunks: list[Chunk]) -> BM25Okapi:
    """
    Build the BM25 index over all chunks.

    Internally `BM25Okapi` precomputes term frequencies, document lengths,
    and the average doc length — that's all the data it needs at query time.
    Fast: a few thousand chunks index in well under a second.
    """
    tokenized_corpus = [tokenize(c.text) for c in chunks]
    return BM25Okapi(tokenized_corpus)


def bm25_search(
    bm25: BM25Okapi,
    chunks: list[Chunk],
    query: str,
    k: int,
) -> list[Chunk]:
    """
    Return the top-k chunks by BM25 score for the query.

    `bm25.get_scores` returns ONE score per document in corpus order (same
    order we built the index with). We argsort descending and slice the
    top k, then map indices back to Chunk objects.
    """
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)

    # np.argsort sorts ASCENDING by default. `[::-1]` flips it to descending,
    # then we take the first `k` indices.
    top_indices = np.argsort(scores)[::-1][:k]

    return [chunks[int(i)] for i in top_indices]
