"""Agentic chat REPL, LangGraph version. Same behavior as agent_chat.py
(manual loop) — compare the two to see what the framework changed vs kept.

Usage:
    python agent_chat_langgraph.py
"""
from __future__ import annotations

import sys
import time

from langchain_core.messages import AIMessage, ToolMessage

from pdfchat import storage
from pdfchat.agent_langgraph import run_agent_langgraph
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
    print(f"Ready in {time.time()-t0:.1f}s. LangGraph agentic mode. "
          f"Type 'exit' to quit.\n")

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

        reply, all_messages = run_agent_langgraph(
            pipe, settings.groq_model, query, history=history[-settings.history_turns:]
        )

        # Pull out the tool calls/results for display, same as agent_chat.py.
        tool_calls = [m for m in all_messages if isinstance(m, ToolMessage)]
        think_steps = [m for m in all_messages if isinstance(m, AIMessage)]
        print(f"   ({len(think_steps)} think step(s), {len(tool_calls)} search(es))")
        for i, tm in enumerate(tool_calls, start=1):
            preview = (tm.content or "")[:80].replace("\n", " ")
            print(f"     [{i}] result: {preview}...")
        print(f"\nbot> {reply}\n")

        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    raise SystemExit(main())
