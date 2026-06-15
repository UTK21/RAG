"""Bi-encoder embedder + FAISS dense index."""
from __future__ import annotations

import faiss
from sentence_transformers import SentenceTransformer

from pdfchat.loader import Chunk


def load_embedder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def build_index(chunks: list[Chunk], embedder: SentenceTransformer) -> faiss.IndexFlatIP:
    vecs = embedder.encode(
        [c.text for c in chunks],
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")
    # Normalized vectors + inner product = cosine similarity.
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index
