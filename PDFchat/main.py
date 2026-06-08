"""
main.py
=======
Thin CLI that wires every module together.

Pipeline (now with hybrid search + conversational memory + re-ranking):

        PDF ──► pdf_loader ──► chunks
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
        embeddings          bm25_index          (kept in memory)
        (FAISS dense)       (BM25 sparse)
              └──────────────────┬──────────────────┘
                                 │  (built once on startup)
        ─────────────────────────────────────────────────────  (per-question)
        user question
              │
              ▼
        query_rewriter (uses chat history)  ──► standalone query
              │
              ├─► dense retrieval (FAISS, k=20)  ─┐
              │                                    ├─► RRF fusion (hybrid.py)
              └─► sparse retrieval (BM25, k=20)  ─┘        │
                                                           ▼
                                              reranker (cross-encoder, top 4)
                                                           │
                                                           ▼
                                          llm.answer (system + history + ctx)
                                                           │
                                                           ▼
                                          history.append(user + assistant)

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
from pdf_loader import load_pdf
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

    # --- 1. Load + chunk the PDF --------------------------------------------
    print(f"Loading PDF: {pdf_path}")
    chunks = load_pdf(
        pdf_path,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    if not chunks:
        print(
            "No extractable text found. The PDF may be scanned/image-only — "
            "you'd need an OCR step (e.g. pytesseract) to handle that.",
            file=sys.stderr,
        )
        return 1
    print(f"Loaded {len(chunks)} chunks.")

    # --- 2. Load models + build indexes -------------------------------------
    # We build BOTH a dense index (FAISS) and a sparse index (BM25). They're
    # independent — different views of the same chunks — and run in parallel
    # at query time.
    print(f"Loading embedder (bi-encoder): {settings.embed_model}")
    embedder = load_embedder(settings.embed_model)

    print(f"Loading re-ranker (cross-encoder): {settings.rerank_model}")
    reranker = load_reranker(settings.rerank_model)

    print("Building FAISS index (dense / semantic)...")
    dense_index = build_index(chunks, embedder)

    print("Building BM25 index (sparse / keyword)...")
    # BM25 needs no model download. It just precomputes term frequencies.
    bm25 = build_bm25(chunks)

    # --- 3. Chat loop -------------------------------------------------------
    client = make_client(settings.groq_api_key)
    print(f"Ready. Model: {settings.groq_model}. Type 'exit' to quit.\n")

    # Conversation history grows as the user chats. We trim to the last N
    # turns before sending — old context usually hurts more than it helps.
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
        # Turns "what about its limitations?" into "what are the limitations
        # of transformers?" using the prior chat as context. No-op on the
        # first message.
        standalone_query = rewrite_query(
            client=client,
            model=settings.groq_model,
            history=recent_history,
            new_question=query,
        )
        if standalone_query != query:
            print(f"   (rewrote → {standalone_query!r})")

        # --- Step B: hybrid retrieval --------------------------------------
        # Dense and sparse run independently. Each casts a wide net of
        # `retrieve_k` candidates. They surface DIFFERENT failure modes:
        #   - dense will catch "automobile" when query says "car"
        #   - sparse will catch "AX-9281" or "Vaswani" exactly
        dense_hits = retrieve(
            standalone_query,
            embedder,
            dense_index,
            chunks,
            k=settings.retrieve_k,
        )
        sparse_hits = bm25_search(
            bm25,
            chunks,
            standalone_query,
            k=settings.retrieve_k,
        )

        # Fuse the two rankings with Reciprocal Rank Fusion. A chunk that
        # both retrievers liked floats to the top because its RRF score
        # accumulates twice. Output is deduplicated.
        fused = reciprocal_rank_fusion(
            [dense_hits, sparse_hits],
            k_rrf=settings.rrf_k,
        )

        # --- Step C: re-rank with the cross-encoder ------------------------
        # The fused list may have up to 2*retrieve_k unique chunks. The
        # cross-encoder re-scores them by reading (query, chunk) pairs
        # together, then we keep the top `top_k` for the LLM.
        top_chunks = rerank(
            reranker=reranker,
            query=standalone_query,
            candidates=fused,
            top_k=settings.top_k,
        )

        # --- Step D: generate the grounded answer --------------------------
        # We pass the ORIGINAL query and recent history so the reply feels
        # like a continuation of the conversation. The rewrite only existed
        # to help retrieval, not to be visible to the user.
        reply = answer(
            client=client,
            model=settings.groq_model,
            query=query,
            context_chunks=top_chunks,
            temperature=settings.temperature,
            history=recent_history,
        )
        print(f"\nbot> {reply}\n")

        # --- Step E: update conversation memory ----------------------------
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    raise SystemExit(main())
