"""LangGraph port of pdfchat/agent.py — same ReAct loop, expressed as a graph.

Read pdfchat/agent.py first if you haven't. Every concept here maps to a
line in that file:

    manual loop                          LangGraph
    ──────────────────────────────────   ──────────────────────────────
    messages list + .append()            AgentState + add_messages reducer
    THINK (one LLM call)                 "think" node
    ACT + OBSERVE (inner for-loop)        ToolNode (prebuilt — does the
                                          json.loads + dispatch + OBSERVE
                                          append for you)
    if not choice.tool_calls: return     tools_condition (prebuilt routing
                                          function) + conditional edge
    for _ in range(MAX_ITERATIONS)        recursion_limit (set at invoke time)

This file is intentionally thin. Almost all of the "real work" (the tool,
the retrieval pipeline) is unchanged from agent.py — we're only
re-expressing the LOOP SHAPE using LangGraph's primitives.
"""
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from pdfchat.loader import children_to_parents
from pdfchat.pipeline import Pipeline

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


# ==============================================================================
# STATE — the typed equivalent of your `messages = [...]` list.
# ==============================================================================
# `Annotated[list, add_messages]` tells LangGraph: "when a node returns
# {"messages": [new_msg]}, APPEND new_msg to the existing list, don't
# replace it." That's the formalized version of every `.append()` call
# you wrote by hand in agent.py.
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def build_graph(pipe: Pipeline, model: str):
    """Construct and compile the graph. Called once per Pipeline (mirrors
    how agent.py's run_agent takes a `pipe` and reuses it across calls)."""

    # --------------------------------------------------------------------
    # THE TOOL — identical retrieval logic to pdfchat/tools.py::search_corpus.
    # The @tool decorator is LangChain's equivalent of the JSON schema we
    # hand-wrote in SEARCH_CORPUS_TOOL — it inspects the function signature
    # and docstring to build that schema automatically.
    # --------------------------------------------------------------------
    @tool
    def search_corpus(query: str) -> str:
        """Search the PDF knowledge base for chunks relevant to a query.
        Call this multiple times with different, more specific queries if
        one search isn't enough to answer the user's full question."""
        top_children, _ = pipe._retrieve(query)
        parents = children_to_parents(top_children, pipe.idx.parents)
        if not parents:
            return "No relevant passages found for this query."
        return "\n\n".join(f"[{p.doc_name} p. {p.page}]\n{p.text}" for p in parents)

    tools = [search_corpus]

    # llm.bind_tools(...) is the LangChain equivalent of passing
    # tools=[SEARCH_CORPUS_TOOL], tool_choice="auto" to client.chat.completions.create.
    llm = ChatGroq(model=model, temperature=0.0)
    llm_with_tools = llm.bind_tools(tools)

    # --------------------------------------------------------------------
    # THE "think" NODE — identical to the THINK step in agent.py's for-loop:
    #   resp = client.chat.completions.create(messages=messages, tools=[...])
    #   choice = resp.choices[0].message
    # Here: invoke the bound LLM on the current message list, return the
    # new AIMessage so add_messages appends it to state.
    # --------------------------------------------------------------------
    def think(state: AgentState) -> dict:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    # --------------------------------------------------------------------
    # THE "act" NODE — ToolNode is PREBUILT. It does exactly what your
    # inner for-loop did by hand:
    #   for tool_call in choice.tool_calls:
    #       args = json.loads(tool_call.function.arguments)
    #       result = search_corpus(pipe, args["query"])
    #       messages.append({"role": "tool", "tool_call_id": ..., "content": result})
    # ToolNode reads the last AIMessage's .tool_calls, runs each matching
    # tool by name, and appends ToolMessage results — same OBSERVE step,
    # zero lines of code from us.
    # --------------------------------------------------------------------
    tool_node = ToolNode(tools)

    # --------------------------------------------------------------------
    # GRAPH ASSEMBLY
    # --------------------------------------------------------------------
    graph = StateGraph(AgentState)
    graph.add_node("think", think)
    graph.add_node("act", tool_node)
    graph.set_entry_point("think")

    # tools_condition is PREBUILT — it's the formalized version of:
    #   if not choice.tool_calls: return choice.content   # -> END
    #   else: fall through to ACT                          # -> "act"
    graph.add_conditional_edges("think", tools_condition, {"tools": "act", "__end__": END})

    # The loop-back edge — your `for` loop's implicit "go around again."
    graph.add_edge("act", "think")

    return graph.compile()


def run_agent_langgraph(
    pipe: Pipeline,
    model: str,
    user_question: str,
    history: list[dict[str, str]] | None = None,
    max_iterations: int = 5,
) -> tuple[str, list]:
    """Same signature/contract as agent.py::run_agent, but backed by the
    compiled LangGraph above. Returns (final_answer, raw_message_list) so
    callers can inspect the full trace if needed."""
    graph = build_graph(pipe, model)

    messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)]
    if history:
        for turn in history:
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_question})

    # recursion_limit is LangGraph's framework-enforced version of your
    # `for _ in range(MAX_ITERATIONS)` safety cap. Each think<->act hop
    # counts as steps, so we scale it up a bit from a raw iteration count.
    result = graph.invoke(
        {"messages": messages},
        config={"recursion_limit": max_iterations * 2 + 2},
    )

    final = result["messages"][-1]
    final_text = final.content if isinstance(final, AIMessage) else str(final.content)
    return final_text, result["messages"]
