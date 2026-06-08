"""
reranker.py
===========
Second-stage retrieval: cross-encoder re-ranking.

The two-stage retrieval pattern
-------------------------------
Modern RAG systems almost universally use two stages:

    Stage 1 — RECALL     (bi-encoder + vector DB)
        Goal: make sure the right chunk is SOMEWHERE in the top 20-50.
        Strength: fast. Can handle millions of chunks.
        Weakness: rough scoring — relevance ranking is noisy.

    Stage 2 — PRECISION  (cross-encoder re-ranker)
        Goal: pick the 3-5 BEST chunks out of those candidates.
        Strength: very accurate relevance scoring.
        Weakness: slow — must score (query, chunk) pairs one by one.

Bi-encoder vs cross-encoder (the key distinction in modern retrieval)
---------------------------------------------------------------------

BI-ENCODER (what FAISS + sentence-transformers uses):

        query  ─► encoder ─► vec_q ─┐
                                    ├─► cosine similarity ─► score
        chunk  ─► encoder ─► vec_c ─┘

  The query and chunk are encoded SEPARATELY. They never "see" each other
  inside the model. We can precompute all chunk vectors once and reuse them
  forever — that's why this scales to millions of chunks.

CROSS-ENCODER (this module):

        [CLS] query [SEP] chunk [SEP]  ─► encoder ─► score

  The query and chunk are concatenated into ONE input. Every transformer
  attention head reads both at once, so the model can directly compare
  "this word in the query" against "that word in the chunk". Much more
  accurate, but you can't precompute anything — every (query, chunk) pair
  has to be re-encoded from scratch.

Hence: bi-encoder narrows millions → 20, cross-encoder narrows 20 → 4. Best
of both worlds.

The model we use
----------------
`BAAI/bge-reranker-base` — a free, ~280MB cross-encoder from the BGE family,
trained specifically for re-ranking. It runs locally on CPU; the first run
downloads the weights to ~/.cache/huggingface.

`CrossEncoder` is sentence-transformers' wrapper class for cross-encoder
models — different class than `SentenceTransformer` (which is for bi-encoders).
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from pdf_loader import Chunk


def load_reranker(model_name: str) -> CrossEncoder:
    """Load the cross-encoder. Cached after first download."""
    return CrossEncoder(model_name)


def rerank(
    reranker: CrossEncoder,
    query: str,
    candidates: list[Chunk],
    top_k: int,
) -> list[Chunk]:
    """
    Re-score candidate chunks against the query and return the top `top_k`.

    `reranker.predict` expects a list of [query, chunk_text] pairs and
    returns one relevance score per pair. Higher = more relevant.

    We then sort by score (desc) and slice the top_k.
    """
    if not candidates:
        return []

    # Build all (query, chunk) pairs. The cross-encoder will look at each
    # pair holistically.
    pairs = [[query, c.text] for c in candidates]

    scores = reranker.predict(pairs)

    # zip + sort by score, keep top_k. `reverse=True` because higher = better.
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_k]]
