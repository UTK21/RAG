"""Centralized settings loaded from .env."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    groq_api_key: str | None
    groq_model: str
    groq_rewrite_model: str

    embed_model: str
    rerank_model: str

    data_dir: str
    index_dir: str

    parent_size: int
    parent_overlap: int
    child_size: int
    child_overlap: int

    retrieve_k: int
    rrf_k: int
    top_k: int

    temperature: float
    history_turns: int

    use_hyde: bool
    hyde_temperature: float


def _bool(s: str) -> bool:
    return s.lower() in {"1", "true", "yes", "on"}


def load_settings(overrides: dict[str, str] | None = None) -> Settings:
    """Build a Settings. `overrides` lets callers (e.g. eval.py) flip flags
    without editing .env."""
    src = dict(os.environ)
    if overrides:
        src.update(overrides)

    return Settings(
        groq_api_key=src.get("GROQ_API_KEY"),
        groq_model=src.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        groq_rewrite_model=src.get("GROQ_REWRITE_MODEL", "llama-3.1-8b-instant"),
        embed_model=src.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        rerank_model=src.get("RERANK_MODEL", "BAAI/bge-reranker-base"),
        data_dir=src.get("DATA_DIR", "data"),
        index_dir=src.get("INDEX_DIR", "indexes"),
        parent_size=int(src.get("PARENT_SIZE", "1200")),
        parent_overlap=int(src.get("PARENT_OVERLAP", "200")),
        child_size=int(src.get("CHILD_SIZE", "240")),
        child_overlap=int(src.get("CHILD_OVERLAP", "40")),
        retrieve_k=int(src.get("RETRIEVE_K", "20")),
        rrf_k=int(src.get("RRF_K", "60")),
        top_k=int(src.get("TOP_K", "4")),
        temperature=float(src.get("TEMPERATURE", "0.2")),
        history_turns=int(src.get("HISTORY_TURNS", "5")),
        use_hyde=_bool(src.get("USE_HYDE", "false")),
        hyde_temperature=float(src.get("HYDE_TEMPERATURE", "0.5")),
    )


settings = load_settings()
