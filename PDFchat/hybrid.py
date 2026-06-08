"""
hybrid.py
=========
Merge results from multiple retrievers using Reciprocal Rank Fusion (RRF).

The problem RRF solves
----------------------
We now have two retrievers producing two ranked lists:

    Dense (FAISS):   [chunk_7, chunk_3, chunk_12, ...]  scores in [-1, 1]
    Sparse (BM25):   [chunk_3, chunk_19, chunk_7, ...]  scores in [0, ∞)

We want ONE merged list. Two naive approaches that DON'T work:

    (a) "Just add the scores"
        Broken — the scores live on different scales. BM25 of 14.2 vs
        cosine of 0.81 are not comparable. You'd need to normalize first
        (min-max? z-score?), and the right normalization depends on the
        corpus, the query, the model... endless tuning.

    (b) "Just take the union"
        Loses the agreement signal. A chunk that BOTH retrievers think is
        relevant should rank higher than one only ONE retriever picked.

Reciprocal Rank Fusion (the trick)
----------------------------------
Throw away the scores entirely. Use only the RANKS:

    fused_score(doc) = Σ over retrievers   1 / (k_rrf + rank_in_that_retriever)

Properties that make this work in practice:
  * Scale-free. Doesn't matter whether one retriever returns cosine or BM25
    or made-up numbers — only the rank order is used.
  * Agreement is rewarded automatically. A doc ranked #1 in both lists
    accumulates 1/61 + 1/61 ≈ 0.033, beating a #1-only doc at 1/61.
  * The `k_rrf` constant (typically 60) dampens the gap between rank 1 and
    rank 2 so lower-ranked items still matter. With k=0 the #1 doc would
    dominate everything.

It's a one-liner from a 2009 paper (Cormack et al.) that quietly became the
default fusion method in Elasticsearch, Vespa, Weaviate, etc. No tuning per
corpus.

A small implementation detail
-----------------------------
We use `id(chunk)` as the dictionary key to identify "this exact Chunk
object". This works because all retrievers return references to chunks from
the same master list — same chunk object, same id. If you ever start
copying chunks around, switch to a stable hashable key like (page, text).
"""

from __future__ import annotations

from pdf_loader import Chunk


def reciprocal_rank_fusion(
    rank_lists: list[list[Chunk]],
    k_rrf: int = 60,
) -> list[Chunk]:
    """
    Merge multiple ranked candidate lists into one fused ranking.

    Each list should be in best-to-worst order. Returns a single list of
    chunks sorted by total RRF score (best first), deduplicated.
    """
    scores: dict[int, float] = {}   # id(chunk) -> accumulated RRF score
    objs: dict[int, Chunk] = {}     # id(chunk) -> chunk reference

    for ranked in rank_lists:
        # `start=1` because the formula uses 1-based ranks (rank 0 would
        # blow up to 1/60 vs 1/61 — small effect but the paper specifies 1).
        for rank, chunk in enumerate(ranked, start=1):
            key = id(chunk)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k_rrf + rank)
            objs[key] = chunk

    # Sort chunk references by their accumulated score, best first.
    return sorted(objs.values(), key=lambda c: scores[id(c)], reverse=True)
