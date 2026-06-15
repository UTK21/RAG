"""Sparse retrieval with BM25."""
from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from pdfchat.loader import Chunk

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def build_bm25(chunks: list[Chunk]) -> BM25Okapi:
    return BM25Okapi([tokenize(c.text) for c in chunks])


def bm25_search(
    bm25: BM25Okapi, chunks: list[Chunk], query: str, k: int
) -> list[Chunk]:
    tokens = tokenize(query)
    if not tokens:
        return []
    scores = bm25.get_scores(tokens)
    top = np.argsort(scores)[::-1][:k]
    return [chunks[int(i)] for i in top]
