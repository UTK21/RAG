"""Orchestration: query (+ history) -> retrieve -> rerank -> small->big -> answer.

This is the shared engine. chat.py streams from it; eval.py calls it
non-streaming to compare against ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from groq import Groq

from pdfchat import bm25 as bm25_mod
from pdfchat import hybrid, llm, query_rewrite
from pdfchat import retrieval as retr
from pdfchat.config import Settings
from pdfchat.loader import Chunk, ParentChunk, children_to_parents
from pdfchat.rerank import rerank


@dataclass
class RetrievalTrace:
    """Returned alongside the answer so eval.py can score retrieval too."""

    standalone_query: str
    hyde_text: str | None
    top_children: list[Chunk]
    context_parents: list[ParentChunk]


class Pipeline:
    """Holds loaded models + indexes; answers questions."""

    def __init__(self, *, settings: Settings, loaded_index, embedder, reranker, client: Groq):
        self.s = settings
        self.idx = loaded_index
        self.embedder = embedder
        self.reranker = reranker
        self.client = client

    # ----- internal: retrieval pipeline -----------------------------------

    def _retrieve(self, standalone_query: str) -> tuple[list[Chunk], str | None]:
        # Always-on: dense + sparse on the standalone query.
        dense_hits = retr.retrieve(
            standalone_query, self.embedder, self.idx.dense_index, self.idx.children,
            k=self.s.retrieve_k,
        )
        sparse_hits = bm25_mod.bm25_search(
            self.idx.bm25, self.idx.children, standalone_query, k=self.s.retrieve_k,
        )
        rank_lists: list[list[Chunk]] = [dense_hits, sparse_hits]

        hyde_text: str | None = None
        if self.s.use_hyde:
            # Q + H safety pattern: HyDE search runs ALONGSIDE the original
            # query searches, never instead of. RRF + reranker filter out
            # HyDE noise if its candidates are off-topic.
            hyde_text = query_rewrite.hyde(
                self.client, self.s.groq_rewrite_model, standalone_query,
                temperature=self.s.hyde_temperature,
            )
            hyde_vec = self.embedder.encode(
                [hyde_text], normalize_embeddings=True, convert_to_numpy=True,
            ).astype("float32")
            hyde_hits = retr.retrieve_with_vector(
                hyde_vec, self.idx.dense_index, self.idx.children, k=self.s.retrieve_k,
            )
            rank_lists.append(hyde_hits)

        fused = hybrid.reciprocal_rank_fusion(rank_lists, k_rrf=self.s.rrf_k)

        # Cross-encoder uses the STANDALONE query (the user's intent), not
        # the HyDE text. Reranker is the safety net for HyDE's failure modes.
        top_children = rerank(
            self.reranker, standalone_query, fused, top_k=self.s.top_k,
        )
        return top_children, hyde_text

    # ----- public: one-shot, non-streaming (used by eval) -----------------

    def answer(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
    ) -> tuple[str, RetrievalTrace]:
        history = history or []
        recent = history[-self.s.history_turns :]

        standalone_query = query_rewrite.standalone(
            self.client, self.s.groq_rewrite_model, recent, query,
        )
        top_children, hyde_text = self._retrieve(standalone_query)
        parents = children_to_parents(top_children, self.idx.parents)

        reply = llm.answer(
            self.client,
            model=self.s.groq_model,
            query=query,
            context_chunks=parents,
            temperature=self.s.temperature,
            history=recent,
        )
        return reply, RetrievalTrace(
            standalone_query=standalone_query,
            hyde_text=hyde_text,
            top_children=top_children,
            context_parents=parents,
        )

    # ----- public: streaming (used by chat) -------------------------------

    def answer_stream(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
    ) -> tuple[Iterator[str], RetrievalTrace]:
        """Returns (token iterator, retrieval trace). Trace is ready before
        the first token, so callers can print citations / debug info first."""
        history = history or []
        recent = history[-self.s.history_turns :]

        standalone_query = query_rewrite.standalone(
            self.client, self.s.groq_rewrite_model, recent, query,
        )
        top_children, hyde_text = self._retrieve(standalone_query)
        parents = children_to_parents(top_children, self.idx.parents)

        token_iter = llm.answer_stream(
            self.client,
            model=self.s.groq_model,
            query=query,
            context_chunks=parents,
            temperature=self.s.temperature,
            history=recent,
        )
        trace = RetrievalTrace(
            standalone_query=standalone_query,
            hyde_text=hyde_text,
            top_children=top_children,
            context_parents=parents,
        )
        return token_iter, trace
