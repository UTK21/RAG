"""Reciprocal Rank Fusion across N ranked lists."""
from __future__ import annotations

from pdfchat.loader import Chunk


def reciprocal_rank_fusion(
    rank_lists: list[list[Chunk]], k_rrf: int = 60
) -> list[Chunk]:
    scores: dict[int, float] = {}
    objs: dict[int, Chunk] = {}
    for ranked in rank_lists:
        for rank, chunk in enumerate(ranked, start=1):
            key = id(chunk)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k_rrf + rank)
            objs[key] = chunk
    return sorted(objs.values(), key=lambda c: scores[id(c)], reverse=True)
