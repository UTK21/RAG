"""A hand-rolled agentic loop (the ReAct pattern), no framework.

Why build this by hand instead of reaching for LangGraph immediately?
Frameworks like LangGraph give you nodes/edges/state machinery for exactly
this loop. Building it manually once means you'll never wonder what
LangGraph is "hiding" from you later — you'll have already written the
hiding part yourself: the termination condition, the message-history
bookkeeping, the tool-dispatch, the iteration cap.

The ReAct pattern (Reason + Act), in one picture:

    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │   THINK   ──►   ACT   ──►   OBSERVE   ──┐                │
    │     ▲                                   │                │
    │     └───────────────────────────────────┘                │
    │                                                         │
    │   Repeat until the model says "I have enough            │
    │   information" (no more tool calls) OR we hit the       │
    │   iteration cap.                                        │
    │                                                         │
    └─────────────────────────────────────────────────────────┘

Concretely, each loop iteration:
  1. THINK   — send the conversation (system + history + tool results so
               far) to the LLM. It either calls a tool or writes a final
               answer.
  2. ACT     — if it called a tool, we actually execute that Python
               function (e.g. search_corpus).
  3. OBSERVE — the tool's return value gets appended to the conversation
               as a "tool" role message, and we loop back to THINK with
               that new information available.

Why this matters for RAG specifically: a single retrieve-then-generate
pass (everything we built before this file) can't answer questions that
need MULTIPLE searches. "Compare what both cookbooks say about spice
levels across all their recipes" benefits from searching once per
cookbook, or once per dish, then synthesizing — exactly what this loop
enables and a single-shot pipeline cannot.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from groq import Groq

from pdfchat.pipeline import Pipeline
from pdfchat.tools import SEARCH_CORPUS_TOOL, search_corpus

AGENT_SYSTEM_PROMPT = (
    "You are a research assistant answering questions about a collection of "
    "PDFs. You have a tool, search_corpus, to search them.\n\n"
    "Rules:\n"
    "  1. Use search_corpus as many times as needed. If a question has "
    "multiple parts (e.g. comparing two things), search for each part "
    "SEPARATELY with a focused query rather than one vague combined query.\n"
    "  2. Use ONLY information returned by search_corpus. If something "
    "wasn't found, say so — do not guess.\n"
    "  3. Cite every claim as (doc.pdf p. N).\n"
    "  4. Once you have enough information, STOP calling tools and write "
    "the final answer directly."
)

# Safety valve. Without a hard cap, a confused model can call tools forever
# — each iteration costs an LLM call (money + latency). 5 is generous for
# a small corpus; tune per use case.
MAX_ITERATIONS = 5


@dataclass
class AgentTrace:
    """Everything that happened during one agent run, for debugging/eval."""

    iterations: int = 0
    tool_calls: list[dict] = field(default_factory=list)  # {query, result_preview}
    final_answer: str = ""


def run_agent(
    client: Groq,
    model: str,
    pipe: Pipeline,
    user_question: str,
    history: list[dict[str, str]] | None = None,
) -> tuple[str, AgentTrace]:
    """Run the THINK -> ACT -> OBSERVE loop until the model stops calling
    tools or we hit MAX_ITERATIONS. Returns (final_answer, trace)."""

    messages: list[dict] = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_question})

    trace = AgentTrace()

    for _ in range(MAX_ITERATIONS):
        trace.iterations += 1

        # --- THINK ----------------------------------------------------
        # tool_choice="auto" lets the model decide each turn: call a tool,
        # or respond directly. This single parameter is the entire
        # "decide what to do next" mechanism — no extra prompting trick
        # needed, it's a first-class feature of the chat API.
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[SEARCH_CORPUS_TOOL],
            tool_choice="auto",
            temperature=0.0,
        )
        choice = resp.choices[0].message

        # No tool call => the model decided it has enough info. Done.
        if not choice.tool_calls:
            trace.final_answer = choice.content or ""
            return trace.final_answer, trace

        # --- ACT + OBSERVE ----------------------------------------------
        # Echo the assistant's tool-call message back into history (the
        # API requires this — it needs to see its own request alongside
        # the tool's response on the next turn).
        messages.append(
            {
                "role": "assistant",
                "content": choice.content,
                "tool_calls": [tc.model_dump() for tc in choice.tool_calls],
            }
        )

        for tool_call in choice.tool_calls:
            args = json.loads(tool_call.function.arguments)
            query = args.get("query", "")

            # ACT: actually execute the tool. This is the one line where
            # the "agent" stops being just an LLM and starts touching our
            # real retrieval code.
            result = search_corpus(pipe, query)

            trace.tool_calls.append({"query": query, "result_preview": result[:120]})

            # OBSERVE: feed the result back as a "tool" role message keyed
            # to the specific tool_call_id, so the model can match
            # request -> response if it issued multiple calls at once.
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

    # Hit MAX_ITERATIONS without the model settling on a final answer.
    # Force a stop: ask it to answer NOW with whatever it has gathered,
    # rather than silently failing or looping forever.
    messages.append(
        {
            "role": "user",
            "content": "You've used all available searches. Answer now with what you have.",
        }
    )
    resp = client.chat.completions.create(model=model, messages=messages, temperature=0.0)
    trace.final_answer = resp.choices[0].message.content or ""
    return trace.final_answer, trace
