"""Query rewriters: standalone (history-aware) and HyDE (hypothetical answer)."""
from __future__ import annotations

from groq import Groq

# Few-shot prompt. Abstract rules ("do not answer") get ignored by small
# models. Concrete examples ("here is exactly the shape we want") work
# much more reliably. Bug history: Isses_faced/01-*.md.
STANDALONE_PROMPT = (
    "You rewrite follow-up questions to be standalone search queries.\n"
    "Given chat history and a new user message, output ONLY the message "
    "rewritten so it makes sense without the history.\n"
    "Do NOT answer. Do NOT explain. Output a QUESTION ending in '?'.\n\n"
    "Examples:\n\n"
    "History:\n"
    "  user: tell me about transformers\n"
    "  assistant: Transformers use self-attention (Vaswani 2017).\n"
    "New message: what about its limitations?\n"
    "Standalone version: What are the limitations of transformers?\n\n"
    "History:\n"
    "  user: Does carbonara use cream?\n"
    "  assistant: Sources disagree. Italian says no, American says yes.\n"
    "New message: what about for a beginner?\n"
    "Standalone version: Which version of carbonara is best for a beginner cook?\n\n"
    "History:\n"
    "  user: What is FAISS?\n"
    "  assistant: FAISS is a vector similarity search library.\n"
    "New message: how fast is it?\n"
    "Standalone version: How fast is FAISS at vector similarity search?\n\n"
    "If the new message is already standalone, output it unchanged."
)

# Heuristics for detecting a rewrite that "helpfully" answered the question
# instead of producing a question. If the rewrite trips these, we fall back
# to the original query — better to retrieve with a slightly worse query
# than to retrieve with the small model's hallucinated opinion.
_ANSWER_PREFIXES = (
    "i would", "i recommend", "i suggest",
    "the ", "a ", "an ",
    "according to", "based on",
    "for someone", "for a beginner", "for beginners",
    "yes,", "no,", "it depends",
)


def _looks_like_answer(rewritten: str) -> bool:
    s = rewritten.strip()
    if not s:
        return True
    if len(s.split()) > 30:
        return True
    if not s.rstrip(".!").endswith("?"):
        return True
    if s.lower().startswith(_ANSWER_PREFIXES):
        return True
    return False

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
    rewritten = (resp.choices[0].message.content or "").strip()

    # Defensive fallback. Two failure modes:
    #   1. Empty / garbage output.
    #   2. Small model "helpfully" answered instead of rephrasing — see
    #      Isses_faced/01-*.md for the bug this defends against.
    if not rewritten or _looks_like_answer(rewritten):
        return new_question

    return rewritten


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
