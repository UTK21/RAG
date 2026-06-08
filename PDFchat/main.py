"""
main.py
=======
Thin CLI that wires every module together.

Pipeline (with parent-child chunking + hybrid + memory + reranking):

        PDF ──► pdf_loader ──► PARENTS  (big context units for the LLM)
                              CHILDREN (small precise units for retrieval)
                                 │
                          ┌──────┴──────┐
                          ▼             ▼
                       FAISS         BM25            (built on CHILDREN)
                     (dense)        (sparse)
                          │             │
        ──────────────────────────────────────────  (per-question loop)
        user question
              │
              ▼
        query_rewriter (uses chat history) ─► standalone query
              │
              ├─► dense retrieval (children, k=20) ─┐
              │                                      ├─► RRF fusion
              └─► sparse retrieval (children, k=20) ─┘     │
                                                           ▼
                                       cross-encoder rerank (children)
                                                           │
                                                           ▼
                                          take top_k CHILDREN, then
                                          MAP CHILDREN → PARENTS (dedupe)
                                                           │
                                                           ▼
                                     LLM gets PARENT chunks for context
                                                           │
                                                           ▼
                                            grounded answer w/ citations

Usage:
    python main.py path/to/file.pdf
"""

from __future__ import annotations

import os
import sys

from bm25_index import bm25_search, build_bm25
from config import settings
from embeddings import build_index, load_embedder
from hybrid import reciprocal_rank_fusion
from llm import answer, make_client
from pdf_loader import children_to_parents, load_pdf
from query_rewriter import rewrite_query
from reranker import load_reranker, rerank
from retriever import retrieve


def main() -> int:
    # --- 0. Argument + environment checks ------------------------------------
    if len(sys.argv) != 2:
        print("Usage: python main.py <path-to-pdf>", file=sys.stderr)
        return 2

    pdf_path = sys.argv[1]
    if not os.path.isfile(pdf_path):
        print(f"File not found: {pdf_path}", file=sys.stderr)
        return 1

    if not settings.groq_api_key:
        print("GROQ_API_KEY missing. Copy .env.example to .env and set it.", file=sys.stderr)
        return 1

    # --- 1. Load + chunk the PDF (now TWO levels) ---------------------------
    # `parents` are the big context blocks we send to the LLM later.
    # `children` are the small precise blocks we actually embed & search.
    # Every child knows its parent_idx so we can hop from match → context.
    print(f"Loading PDF: {pdf_path}")
    parents, children = load_pdf(
        pdf_path,
        parent_size=settings.parent_size,
        parent_overlap=settings.parent_overlap,
        child_size=settings.child_size,
        child_overlap=settings.child_overlap,
    )
    if not children:
        print(
            "No extractable text found. The PDF may be scanned/image-only — "
            "you'd need an OCR step (e.g. pytesseract) to handle that.",
            file=sys.stderr,
        )
        return 1
    print(f"Loaded {len(parents)} parents, {len(children)} children.")

    # --- 2. Load models + build indexes -------------------------------------
    # Both indexes are built on CHILDREN. Parents never get embedded — they
    # only exist to be fetched once a child match is found.
    print(f"Loading embedder (bi-encoder): {settings.embed_model}")
    embedder = load_embedder(settings.embed_model)

    print(f"Loading re-ranker (cross-encoder): {settings.rerank_model}")
    reranker = load_reranker(settings.rerank_model)

    print("Building FAISS index (dense / semantic) over children...")
    dense_index = build_index(children, embedder)

    print("Building BM25 index (sparse / keyword) over children...")
    bm25 = build_bm25(children)

    # --- 3. Chat loop -------------------------------------------------------
    client = make_client(settings.groq_api_key)
    print(f"Ready. Model: {settings.groq_model}. Type 'exit' to quit.\n")

    history: list[dict[str, str]] = []

    while True:
        try:
            query = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not query:
            continue
        if query.lower() in {"exit", "quit", ":q"}:
            return 0

        recent_history = history[-settings.history_turns :]

        # --- Step A: rewrite the question into a standalone form -----------
        standalone_query = rewrite_query(
            client=client,
            model=settings.groq_model,
            history=recent_history,
            new_question=query,
        )
        if standalone_query != query:
            print(f"   (rewrote → {standalone_query!r})")

        # --- Step B: hybrid retrieval over CHILDREN ------------------------
        # We search children because they're small and precise. The right
        # paragraph might be 5 sentences long; we want to match on the
        # specific sentence, not on a noisy 1000-word window.
        dense_hits = retrieve(
            standalone_query, embedder, dense_index, children, k=settings.retrieve_k
        )
        sparse_hits = bm25_search(
            bm25, children, standalone_query, k=settings.retrieve_k
        )
        fused = reciprocal_rank_fusion(
            [dense_hits, sparse_hits], k_rrf=settings.rrf_k
        )

        # --- Step C: cross-encoder re-rank (still on children) -------------
        # The reranker scores (query, child) pairs. Children are the right
        # size for the cross-encoder — full paragraphs would dilute the
        # relevance signal.
        top_children = rerank(
            reranker=reranker,
            query=standalone_query,
            candidates=fused,
            top_k=settings.top_k,
        )

        # --- Step D: SMALL → BIG. Map child matches to their parents ------
        # The crucial parent-child step. The LLM now receives the bigger
        # parent block instead of the small child that matched. Same
        # citation page (children inherit it from their parent).
        # Dedupe: if 3 children point at the same parent, we send the
        # parent once, not three times.
        context_parents = children_to_parents(top_children, parents)

        # --- Step E: generate the grounded answer --------------------------
        reply = answer(
            client=client,
            model=settings.groq_model,
            query=query,
            context_chunks=context_parents,
            temperature=settings.temperature,
            history=recent_history,
        )
        print(f"\nbot> {reply}\n")

        # --- Step F: update conversation memory ----------------------------
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    raise SystemExit(main())
