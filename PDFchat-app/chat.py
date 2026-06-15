"""Chat REPL. Loads persisted indexes, streams answers.

Usage:
    python chat.py
"""
from __future__ import annotations

import sys
import time

from pdfchat import storage
from pdfchat.config import settings
from pdfchat.embeddings import load_embedder
from pdfchat.llm import make_client
from pdfchat.pipeline import Pipeline
from pdfchat.rerank import load_reranker


def _format_citations(parents) -> str:
    seen = []
    for p in parents:
        tag = f"{p.doc_name} p.{p.page}"
        if tag not in seen:
            seen.append(tag)
    return ", ".join(seen)


def main() -> int:
    if not settings.groq_api_key:
        print("GROQ_API_KEY missing. Copy .env.example to .env and set it.", file=sys.stderr)
        return 1

    t0 = time.time()
    print(f"Loading indexes from: {settings.index_dir}")
    try:
        loaded = storage.load(settings.index_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    is_fresh, msg = storage.check_fresh(
        loaded, settings.data_dir, storage.settings_fingerprint(settings)
    )
    if not is_fresh:
        print(f"Stale index: {msg}", file=sys.stderr)
        print("Re-run: python ingest.py", file=sys.stderr)
        return 1

    print(f"  {len(loaded.parents)} parents, {len(loaded.children)} children")
    print(f"Loading embedder: {settings.embed_model}")
    embedder = load_embedder(settings.embed_model)
    print(f"Loading reranker: {settings.rerank_model}")
    reranker = load_reranker(settings.rerank_model)

    client = make_client(settings.groq_api_key)
    pipe = Pipeline(
        settings=settings,
        loaded_index=loaded,
        embedder=embedder,
        reranker=reranker,
        client=client,
    )
    print(f"Ready in {time.time()-t0:.1f}s. Model: {settings.groq_model}. "
          f"HyDE: {'on' if settings.use_hyde else 'off'}. Type 'exit' to quit.\n")

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

        token_iter, trace = pipe.answer_stream(query, history=history)

        if trace.standalone_query != query:
            print(f"   (rewrote → {trace.standalone_query!r})")
        print(f"   (sources: {_format_citations(trace.context_parents)})\n")

        print("bot> ", end="", flush=True)
        reply_parts: list[str] = []
        for tok in token_iter:
            print(tok, end="", flush=True)
            reply_parts.append(tok)
        print("\n")

        full_reply = "".join(reply_parts)
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": full_reply})


if __name__ == "__main__":
    raise SystemExit(main())
