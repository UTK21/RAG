"""Groq client + grounded prompt. Supports streaming and non-streaming."""
from __future__ import annotations

from typing import Iterator

from groq import Groq

from pdfchat.loader import ParentChunk

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about a collection of PDFs.\n"
    "Rules:\n"
    "  1. Use ONLY the provided context. If the answer is not present, say you don't know.\n"
    "  2. Cite every claim using the format (doc.pdf p. N). Be specific.\n"
    "  3. If sources DISAGREE, name each source and what it says — do not silently pick one.\n"
    "  4. Be concise."
)


def make_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


def _build_messages(
    query: str,
    context_chunks: list[ParentChunk],
    history: list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    context = "\n\n".join(
        f"[{c.doc_name} p. {c.page}]\n{c.text}" for c in context_chunks
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append(
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    )
    return messages


def answer(
    client: Groq,
    model: str,
    query: str,
    context_chunks: list[ParentChunk],
    temperature: float,
    history: list[dict[str, str]] | None = None,
) -> str:
    """Non-streaming. Returns the full reply. Used by eval.py."""
    resp = client.chat.completions.create(
        model=model,
        messages=_build_messages(query, context_chunks, history),
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def answer_stream(
    client: Groq,
    model: str,
    query: str,
    context_chunks: list[ParentChunk],
    temperature: float,
    history: list[dict[str, str]] | None = None,
) -> Iterator[str]:
    """Yields token deltas as they arrive. Used by chat.py."""
    stream = client.chat.completions.create(
        model=model,
        messages=_build_messages(query, context_chunks, history),
        temperature=temperature,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
