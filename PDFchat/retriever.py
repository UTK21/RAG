"""
retriever.py
============
Pipeline stage 3: given a user question, find the top-k most relevant chunks.

How it works
------------
  1. Embed the QUESTION using the SAME embedder that embedded the chunks.
     (Critical: query and chunks must live in the same vector space, so
     they must come from the same model.)
  2. Ask FAISS for the K vectors closest to the question vector.
  3. Map those vector positions back to the original `Chunk` objects.

Why top-k > 1?
--------------
A single chunk often doesn't contain the full answer — pulling 3-5 lets the
LLM synthesize across them. But cranking k too high wastes tokens and can
mislead the model with off-topic chunks. k=4 is a sane default for ~800-word
chunks.

Why no re-ranking here?
-----------------------
A common upgrade in real systems is to retrieve many candidates (e.g. k=20)
and then RE-RANK them with a small cross-encoder model before sending the top
few to the LLM. Cross-encoders are slower but more accurate at scoring
"does this chunk actually answer this question?". We skip that for simplicity.
"""

from __future__ import annotations

import faiss
from sentence_transformers import SentenceTransformer

from pdf_loader import Chunk


def retrieve(
    query: str,
    embedder: SentenceTransformer,
    index: faiss.IndexFlatIP,
    chunks: list[Chunk],
    k: int,
) -> list[Chunk]:
    """Return the top-k chunks most semantically similar to `query`."""
    q = embedder.encode(
        [query],
        normalize_embeddings=True,  # MUST match how the chunks were embedded
        convert_to_numpy=True,
    ).astype("float32")

    # `index.search` returns (similarities, indices). We don't need scores
    # here — just the positions in our `chunks` list.
    _, idx = index.search(q, k)

    # `-1` means "fewer than k vectors exist in the index" — skip those.
    return [chunks[i] for i in idx[0] if i != -1]
