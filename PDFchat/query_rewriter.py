"""
query_rewriter.py
=================
Conversational memory for retrieval.

The problem
-----------
Imagine this dialog with our bot:

    you> What does the paper say about transformers?
    bot> They use self-attention and were introduced in Vaswani et al. 2017 (p. 3).
    you> What about its limitations?

If we embed "What about its limitations?" and search FAISS, we get garbage.
The query has no mention of "transformer" — the embedding model can't read
our minds. It will match any chunk discussing "limitations" of anything.

The trick: STANDALONE QUESTION REWRITING
----------------------------------------
Before retrieving, we ask the LLM:

    "Given this conversation + the new question, rewrite the new question
     into a standalone form that makes sense without the chat history."

Output for the example above:
    "What are the limitations of transformers?"

THAT goes to FAISS. Now retrieval works perfectly.

This pattern has a few names in the wild:
  * "history-aware retriever" (LangChain)
  * "query condensation" (LlamaIndex)
  * "contextual query rewriting" (papers)

All the same idea.

Two tiny tricks we use
----------------------
1. We tell the model to return the question UNCHANGED if it's already
   standalone. Avoids the model "improving" already-good queries and
   accidentally changing their meaning.
2. We only feed the last N turns of history, not all of it. Old context
   usually hurts more than it helps and burns tokens.
"""

from __future__ import annotations

from groq import Groq

# This system prompt is intentionally narrow. We don't want the LLM to answer
# the question or add anything — only to rewrite it. Models love to be
# helpful, so being firm here saves a lot of debugging later.
REWRITE_SYSTEM_PROMPT = (
    "You rewrite follow-up questions to be standalone.\n"
    "Given the chat history and a new user message, output ONLY the new "
    "message rewritten so it makes sense without the history.\n"
    "If the new message is already standalone, output it unchanged.\n"
    "Do NOT answer the question. Do NOT add explanations. Output only the "
    "rewritten question."
)


def rewrite_query(
    client: Groq,
    model: str,
    history: list[dict[str, str]],
    new_question: str,
) -> str:
    """
    Convert a possibly-context-dependent follow-up into a standalone query.

    `history` is a list of {"role": "user"|"assistant", "content": str} turns
    in chronological order. The caller is responsible for trimming it to the
    last N turns.

    If history is empty (it's the first message), skip the LLM call entirely
    — the question is already standalone and we save a round trip.
    """
    if not history:
        return new_question

    # Render history as plain text so the LLM sees it clearly. We could also
    # pass the history as proper chat messages, but giving the model an
    # explicit "Chat history:" block in a single user message makes the
    # rewriting task crisper.
    rendered = "\n".join(f"{turn['role']}: {turn['content']}" for turn in history)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Chat history:\n{rendered}\n\n"
                    f"New message: {new_question}\n\n"
                    f"Standalone version:"
                ),
            },
        ],
        # Temperature 0 — we want the rewrite to be deterministic and faithful.
        temperature=0.0,
    )

    rewritten = (resp.choices[0].message.content or "").strip()

    # Defensive fallback: if the model returned nothing or something weird,
    # use the original question. Better to retrieve with a slightly worse
    # query than to retrieve with an empty one.
    return rewritten or new_question
