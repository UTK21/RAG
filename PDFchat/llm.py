"""
llm.py
======
Pipeline stage 4: build the final prompt and call the LLM (Groq).

This is the GENERATION half of "Retrieval-Augmented Generation". The
retrieval step has already given us the best chunks; here we hand them to
the model and ask it to answer the user's question using ONLY that material.

The grounding system prompt
---------------------------
This short paragraph is the single most important "trick" in basic RAG:

  * "Use ONLY the provided context"  →  prevents the model from inventing
    facts that aren't in the document (hallucination).
  * "If the answer is not in the context, say you don't know"  →  models
    will otherwise try to be helpful and guess; this gives them permission
    to bail out gracefully.
  * "Cite page numbers in the form (p. N)"  →  gives the user a way to
    VERIFY every claim. If the bot cites p. 7 and p. 7 doesn't say that,
    you immediately know something went wrong.

Prompt injection caveat
-----------------------
If the PDF itself contains text like "Ignore previous instructions and ...",
a determined attacker could try to hijack the model. For a personal chatbot
this is rarely a concern; for a public-facing product you'd add sanitization
or stricter delimiters around the context block.

Temperature
-----------
0.0 = fully deterministic, 1.0 = creative. For factual Q&A we want low
(~0.2) so the model stays close to the evidence and doesn't paraphrase
loosely.
"""

from __future__ import annotations

from groq import Groq

from pdf_loader import Chunk

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about a PDF. "
    "Use ONLY the provided context. If the answer is not in the context, say you don't know. "
    "Cite page numbers in the form (p. N) when you use information from a chunk."
)


def make_client(api_key: str) -> Groq:
    """Build the Groq client. Kept as a function so main.py doesn't import groq."""
    return Groq(api_key=api_key)


def answer(
    client: Groq,
    model: str,
    query: str,
    context_chunks: list[Chunk],
    temperature: float,
    history: list[dict[str, str]] | None = None,
) -> str:
    """
    Format the prompt and call the chat completion endpoint.

    Each chunk is prefixed with its page number tag (`[p. N]`) so the model
    has clear evidence to point at when it writes citations.

    `history` is the prior conversation (already trimmed to a manageable
    number of turns by the caller). We insert it BETWEEN the system prompt
    and the current question so the model can resolve references like
    "the one I asked about earlier" while STILL being constrained by the
    grounding rules.

    Why include history if we already rewrote the query for retrieval?
    Because the retrieval rewrite only fixes RETRIEVAL. The model still
    benefits from seeing the conversational style and prior answers when
    GENERATING its reply (e.g. consistent tone, knowing what it already
    explained, not repeating itself).
    """
    # Two-newline separation between chunks makes boundaries obvious to the
    # model. The page tag is on its own line so it's hard to miss.
    context = "\n\n".join(f"[p. {c.page}]\n{c.text}" for c in context_chunks)

    # Build the messages list. Order matters:
    #   1. system prompt (rules)
    #   2. prior turns (memory)
    #   3. current user turn (with fresh context)
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append(
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    )

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    # `choices[0]` because we only asked for one completion.
    return resp.choices[0].message.content or ""
