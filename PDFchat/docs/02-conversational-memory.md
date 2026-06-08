# 2. Conversational Memory — making follow-ups work

> **TL;DR:** Before retrieving, ask the LLM to rewrite the follow-up question into a **standalone** form using the chat history. Retrieve with the rewrite; answer with the original.

## The problem

Naive RAG breaks the moment a user asks a follow-up:

```
you> What does the paper say about transformers?
bot> They use self-attention, introduced in Vaswani et al. 2017 (p. 3).

you> What about its limitations?
bot> 🤷  (retrieves random "limitations" chunks from anywhere in the doc)
```

The query **"What about its limitations?"** has no mention of "transformer".
Embeddings don't read your mind — they just see the literal words.

## The fix in one picture

```
                  prior chat
                       │
                       │           "What about its limitations?"
                       ▼                       │
                ┌─────────────┐                │
                │     LLM     │ ◄──────────────┘
                │  (rewrite)  │
                └──────┬──────┘
                       ▼
        "What are the limitations of transformers?"
                       │
                       ▼
              ► retrieval works
```

We use the rewritten query **only for retrieval**.
The original question + history still go to the LLM for generation, so the
reply sounds natural and conversational.

## How it works

### 1. Trim history
- Keep only the last N messages (default 5).
- Old context hurts more than it helps and burns tokens.

### 2. Ask the LLM to rewrite (small, fast call)
- System prompt:
  > "You rewrite follow-up questions to be standalone.
  > Output ONLY the rewritten question. Do NOT answer."
- Temperature = 0 (deterministic, faithful).
- If the question is already standalone, the LLM returns it unchanged.
- If history is empty, **skip the call** — save a round trip.

### 3. Two queries, two purposes
- **Standalone query** → used for retrieval (dense + sparse + rerank).
- **Original query + history** → used for generation (so the reply is conversational).

## Code

```python
REWRITE_SYSTEM_PROMPT = (
    "You rewrite follow-up questions to be standalone.\n"
    "Given the chat history and a new user message, output ONLY the new "
    "message rewritten so it makes sense without the history.\n"
    "If the new message is already standalone, output it unchanged.\n"
    "Do NOT answer the question. Output only the rewritten question."
)

def rewrite_query(client, model, history, new_question):
    if not history:
        return new_question  # nothing to rewrite against

    rendered = "\n".join(f"{t['role']}: {t['content']}" for t in history)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content":
                f"Chat history:\n{rendered}\n\n"
                f"New message: {new_question}\n\n"
                f"Standalone version:"},
        ],
        temperature=0.0,
    )
    return (resp.choices[0].message.content or "").strip() or new_question
```

Then in the chat loop:

```python
standalone = rewrite_query(client, model, history[-5:], user_question)
hits = retrieve(standalone, ...)
reply = answer(client, model, query=user_question, context_chunks=hits, history=history[-5:])
history += [{"role": "user", "content": user_question},
            {"role": "assistant", "content": reply}]
```

**In this repo:** `query_rewriter.py`. Used by `main.py`.

## When to use it

- **Any multi-turn chatbot.** This is the absolute minimum for a conversational RAG.
- Customer support, doc Q&A assistants, research helpers — anything with follow-ups.

## Aliases (same idea, different names)

| Name | Where it comes from |
|---|---|
| Standalone-question rewriting | Common name |
| History-aware retriever | LangChain |
| Query condensation | LlamaIndex |
| Contextual query rewriting | Papers |

All describe the same trick.

## Caveats

| Caveat | Notes |
|---|---|
| Extra LLM call per turn | Cheap if you use a fast/cheap model for the rewrite. Skip on the first message (no history). |
| Bad rewrite → bad retrieval | Pin temperature=0 and tightly constrain the system prompt to "output the question only". Fallback: if rewrite is empty/weird, use the original. |
| History grows unbounded | Trim to last N messages. Bigger N = better memory but more tokens. |
| Sensitive info in history | The rewrite step sends history to the LLM. Don't dump secrets into chat. |

## Key concepts this teaches

- The query the user types is **not** always the query you should search with.
- Retrieval quality depends on query phrasing, not just the index.
- Separating **retrieval-time** representation from **generation-time** representation.
- LLMs as **text transformers**, not just text generators — using them to clean inputs is a powerful pattern.

→ Next: [Re-ranking](03-reranking.md)
