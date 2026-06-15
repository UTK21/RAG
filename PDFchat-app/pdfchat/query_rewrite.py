"""Query rewriters: standalone (history-aware) and HyDE (hypothetical answer)."""
from __future__ import annotations

from groq import Groq

STANDALONE_PROMPT = (
    "You rewrite follow-up questions to be standalone.\n"
    "Given chat history and a new user message, output ONLY the message "
    "rewritten so it makes sense without the history.\n"
    "If already standalone, output unchanged. Do NOT answer. Output only the question."
)

HYDE_PROMPT = (
    "Write a short paragraph (3-5 sentences) that PLAUSIBLY answers the user's "
    "question. Write it as if it's an excerpt from a document or textbook. "
    "Stick close to the topic in the question — do not drift. "
    "Output only the paragraph, no preamble."
)


def standalone(
    client: Groq,
    model: str,
    history: list[dict[str, str]],
    new_question: str,
) -> str:
    if not history:
        return new_question

    rendered = "\n".join(f"{t['role']}: {t['content']}" for t in history)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": STANDALONE_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Chat history:\n{rendered}\n\n"
                    f"New message: {new_question}\n\n"
                    f"Standalone version:"
                ),
            },
        ],
        temperature=0.0,
    )
    return (resp.choices[0].message.content or "").strip() or new_question


def hyde(client: Groq, model: str, query: str, temperature: float = 0.5) -> str:
    """Generate a hypothetical answer to use as a search vector source.
    The text doesn't have to be correct — it just needs to look like a real
    answer chunk so its embedding lands near real answer chunks."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": HYDE_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()
