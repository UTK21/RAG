"""Cross-encoder re-ranking."""
from __future__ import annotations

from sentence_transformers import CrossEncoder

from pdfchat.loader import Chunk


def load_reranker(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name)


def rerank(
    reranker: CrossEncoder,
    query: str,
    candidates: list[Chunk],
    top_k: int,
) -> list[Chunk]:
    if not candidates:
        return []
    pairs = [[query, c.text] for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_k]]
