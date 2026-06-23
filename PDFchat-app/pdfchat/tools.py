"""Tools the agent loop can call.

A "tool" here is just a Python function plus a JSON-schema description of
its signature, in the format the Groq/OpenAI-compatible chat API expects
under the `tools=` parameter. The LLM never calls these functions directly
— it emits a structured request ("call search_corpus with query=...") and
our loop is the one that actually executes the function and feeds the
result back.

Only one tool for now: search_corpus, which reuses the exact same
retrieve -> fuse -> rerank -> small-to-big pipeline you already built for
the regular chat. The agent's superpower isn't a smarter retriever — it's
being able to call this SAME retriever multiple times with different
queries before answering.
"""
from __future__ import annotations

from pdfchat.loader import children_to_parents
from pdfchat.pipeline import Pipeline

# JSON schema describing the tool to the LLM. This is the ENTIRE interface
# the model sees — it knows nothing about our Python code, only this
# description. Precision here directly affects how well the model uses
# the tool.
SEARCH_CORPUS_TOOL = {
    "type": "function",
    "function": {
        "name": "search_corpus",
        "description": (
            "Search the PDF knowledge base for chunks relevant to a query. "
            "Returns the most relevant passages with their source document "
            "and page number. Call this multiple times with different, "
            "more specific queries if one search isn't enough to answer "
            "the user's full question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A specific, standalone search query.",
                }
            },
            "required": ["query"],
        },
    },
}


def search_corpus(pipe: Pipeline, query: str) -> str:
    """Execute the search and format results as a string for the LLM.

    Reuses Pipeline._retrieve (dense + sparse + RRF + rerank) then maps to
    parents, exactly like the regular chat pipeline does. The only
    difference: we return formatted text instead of feeding straight into
    a generation prompt, because the AGENT decides what to do with it next.
    """
    top_children, _ = pipe._retrieve(query)
    parents = children_to_parents(top_children, pipe.idx.parents)

    if not parents:
        return "No relevant passages found for this query."

    blocks = [f"[{p.doc_name} p. {p.page}]\n{p.text}" for p in parents]
    return "\n\n".join(blocks)
