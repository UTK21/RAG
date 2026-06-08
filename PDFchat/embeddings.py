"""
embeddings.py
=============
Pipeline stage 2: turn chunks into vectors and store them for fast similarity
search.

What is an embedding?
---------------------
An embedding is a fixed-length list of numbers (e.g. 384 floats) that captures
the *meaning* of a piece of text. Two texts about the same topic land near
each other in this vector space; unrelated texts land far apart. This is what
makes "semantic search" — search by meaning, not just keywords — possible.

We use sentence-transformers (a HuggingFace library):
  * `all-MiniLM-L6-v2` is small (~80 MB), CPU-friendly, and good enough for
    most English Q&A.
  * Runs entirely locally — no API calls, no cost.

Why FAISS?
----------
FAISS (Facebook AI Similarity Search) is a vector database. Given a query
vector, it finds the K nearest stored vectors quickly. For our use case
(probably hundreds to thousands of chunks) `IndexFlatIP` is exact and fast
enough — no approximations needed.

The cosine-via-inner-product trick
----------------------------------
Cosine similarity is the standard "are these two meanings close?" metric. But
computing cosine for every chunk on every query is wasteful. The trick:

    If you NORMALIZE every vector to unit length, then
    dot product BECOMES cosine similarity.

So we:
  (1) ask sentence-transformers to normalize on output  (`normalize_embeddings=True`)
  (2) use FAISS's `IndexFlatIP` (IP = inner product = dot product)
  (3) normalize the query the same way at retrieval time

The result is exact cosine similarity at maximum speed.
"""

from __future__ import annotations

import faiss
from sentence_transformers import SentenceTransformer

from pdf_loader import Chunk


def load_embedder(model_name: str) -> SentenceTransformer:
    """
    Download (first time) and load the sentence-transformers model.

    The model gets cached under ~/.cache/huggingface so subsequent runs are
    instant. First run downloads ~80MB for MiniLM.
    """
    return SentenceTransformer(model_name)


def build_index(chunks: list[Chunk], embedder: SentenceTransformer) -> faiss.IndexFlatIP:
    """
    Embed every chunk and stuff the vectors into a FAISS index.

    Returns the populated index. The CALLER must keep `chunks` around in the
    same order — FAISS only stores vectors, not the original text. We map
    back to text using positional indices in retriever.py.
    """
    vecs = embedder.encode(
        [c.text for c in chunks],
        normalize_embeddings=True,   # unit-length so IP == cosine similarity
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")              # FAISS requires float32, not float64

    # vecs.shape[1] = embedding dimension (384 for MiniLM).
    # "Flat" means brute-force exhaustive search — exact, no approximation.
    # For millions of vectors you'd switch to IVF or HNSW indexes.
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index
