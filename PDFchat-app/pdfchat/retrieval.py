"""Dense top-k retrieval over the FAISS index."""
from __future__ import annotations

import faiss
from sentence_transformers import SentenceTransformer

from pdfchat.loader import Chunk


def retrieve(
    query: str,
    embedder: SentenceTransformer,
    index: faiss.IndexFlatIP,
    chunks: list[Chunk],
    k: int,
) -> list[Chunk]:
    q = embedder.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")
    _, idx = index.search(q, k)
    return [chunks[i] for i in idx[0] if i != -1]


def retrieve_with_vector(
    vec, index: faiss.IndexFlatIP, chunks: list[Chunk], k: int
) -> list[Chunk]:
    """Dense search using a pre-built vector (used by HyDE)."""
    _, idx = index.search(vec, k)
    return [chunks[i] for i in idx[0] if i != -1]
