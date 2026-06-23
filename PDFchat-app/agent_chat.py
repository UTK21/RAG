"""Agentic chat REPL — the LLM decides how many times to search.

Differs from chat.py (single retrieve-then-generate pass) by letting the
model call search_corpus repeatedly with refined queries before answering.
Best demoed with a question that needs information from multiple places,
e.g. "Compare how the two cookbooks treat spice level across all dishes."

Usage:
    python agent_chat.py
"""
from __future__ import annotations

import sys
import time

from pdfchat import storage
from pdfchat.agent import run_agent
from pdfchat.config import settings
from pdfchat.embeddings import load_embedder
from pdfchat.llm import make_client
from pdfchat.pipeline import Pipeline
from pdfchat.rerank import load_reranker


def main() -> int:
    if not settings.groq_api_key:
        print("GROQ_API_KEY missing.", file=sys.stderr)
        return 1

    t0 = time.time()
    print(f"Loading indexes from: {settings.index_dir}")
    loaded = storage.load(settings.index_dir)

    is_fresh, msg = storage.check_fresh(
        loaded, settings.data_dir, storage.settings_fingerprint(settings)
    )
    if not is_fresh:
        print(f"Stale index: {msg}\nRe-run: python ingest.py", file=sys.stderr)
        return 1

    embedder = load_embedder(settings.embed_model)
    reranker = load_reranker(settings.rerank_model)
    client = make_client(settings.groq_api_key)
    pipe = Pipeline(
        settings=settings, loaded_index=loaded,
        embedder=embedder, reranker=reranker, client=client,
    )
    print(f"Ready in {time.time()-t0:.1f}s. Agentic mode — model may search "
          f"multiple times per question. Type 'exit' to quit.\n")

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

        reply, trace = run_agent(
            client, settings.groq_model, pipe, query, history=history[-settings.history_turns:]
        )

        print(f"   ({trace.iterations} iteration(s), {len(trace.tool_calls)} search(es))")
        for i, tc in enumerate(trace.tool_calls, start=1):
            print(f"     [{i}] searched: {tc['query']!r}")
        print(f"\nbot> {reply}\n")

        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    raise SystemExit(main())
