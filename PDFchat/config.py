"""
config.py
=========
All configuration in one place.

Why a separate module?
----------------------
Hard-coding model names, API keys, or magic numbers across many files makes
them painful to change. We centralize them here so the rest of the codebase
just imports `settings` and reads attributes from it.

We also load the `.env` file here exactly once, at import time. Any module
that imports `settings` is guaranteed the env vars are already loaded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load variables from a local `.env` file into os.environ. Safe to call even
# if the file doesn't exist — it just does nothing.
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Immutable bag of configuration values.

    `frozen=True` means once we create the settings object you can't reassign
    its fields by mistake — a small safety net.
    """

    # --- LLM (Groq) ---------------------------------------------------------
    groq_api_key: str | None
    groq_model: str

    # --- Embeddings ---------------------------------------------------------
    # Any sentence-transformers model name from HuggingFace works here.
    # MiniLM is small (~80MB), fast on CPU, and good enough for general English.
    embed_model: str

    # --- Chunking -----------------------------------------------------------
    # Words per chunk. ~800 words ≈ ~1 page of dense text.
    chunk_size: int
    # How many words adjacent chunks share. Prevents splitting a key sentence
    # cleanly in half across two chunks.
    chunk_overlap: int

    # --- Retrieval (now hybrid: dense + sparse, then fuse, then re-rank) ----
    # Stage 1 (recall): how many chunks EACH retriever returns. Dense and
    # sparse each cast a wide net of this size; RRF then fuses their lists.
    # We want the right chunk somewhere in the pool even if neither retriever
    # ranks it #1.
    retrieve_k: int
    # RRF dampening constant. The classic paper value is 60 — basically never
    # needs tuning.
    rrf_k: int
    # Stage 2 (precision): how many chunks survive the cross-encoder re-rank
    # and actually get sent to the LLM. Kept small to save tokens + reduce noise.
    top_k: int
    # Cross-encoder model used for re-ranking. Cross-encoders read query + chunk
    # TOGETHER, so they score relevance much more accurately than bi-encoders.
    rerank_model: str

    # --- Generation ---------------------------------------------------------
    # 0.0 = fully deterministic, 1.0 = creative. For factual Q&A we want low.
    temperature: float
    # How many past (user, assistant) turns to keep when rewriting follow-ups
    # and when feeding history to the answer model. Bigger = better memory but
    # more tokens billed and slower.
    history_turns: int


def load_settings() -> Settings:
    """Build a Settings instance from environment variables with safe defaults."""
    return Settings(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        embed_model=os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
        retrieve_k=int(os.getenv("RETRIEVE_K", "20")),
        rrf_k=int(os.getenv("RRF_K", "60")),
        top_k=int(os.getenv("TOP_K", "4")),
        rerank_model=os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base"),
        temperature=float(os.getenv("TEMPERATURE", "0.2")),
        history_turns=int(os.getenv("HISTORY_TURNS", "5")),
    )


# Built once at import time so every module sees the same instance.
settings = load_settings()
