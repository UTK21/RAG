# 11. LangGraph — concepts, mapped directly to the manual loop we already built

> **TL;DR:** LangGraph gives the manual ReAct loop from `10-agentic-rag-manual-loop.md` a vocabulary, a few built-in nodes, and new capabilities (checkpointing, visualization, multi-agent composition) we didn't have. The control flow itself is unchanged — same THINK → ACT → OBSERVE cycle, same termination logic, just expressed as a graph instead of a `for` loop.

**Prerequisite:** read note 10 first. Every concept in this note is
explained *by reference* to code you already wrote in
`PDFchat-app/pdfchat/agent.py`. If that file isn't fresh in memory, this
note will feel like new vocabulary instead of a relabeling exercise.

---

## The four building blocks

```
   ┌─────────────────────────────────────────────────────────────┐
   │  1. STATE   — a typed dict that flows through the graph     │
   │  2. NODE    — a function that reads state, returns updates  │
   │  3. EDGE    — a connection: "after node A, go to node B"    │
   │  4. CONDITIONAL EDGE — "after node A, go to B or C based    │
   │              on a decision function"                        │
   └─────────────────────────────────────────────────────────────┘
```

That's the entire vocabulary. Everything else in LangGraph (checkpointing,
streaming, subgraphs, multi-agent) is built from these four ideas.

---

## 1. State — your `messages` list, formalized

In the manual loop:

```python
messages = [system_prompt, *history, user_question]
```

You mutated this one list with `.append()` throughout the loop — once
per assistant tool-call request, once per tool result. LangGraph makes
this explicit and typed:

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
```

`add_messages` is a **reducer**. It tells LangGraph: "when a node
returns `{"messages": [new_msg]}`, *append* `new_msg` to the existing
list — don't replace the whole list." That's the formalized version of
every `.append()` call you wrote by hand.

**Why a reducer instead of just mutating a list directly?** Because
LangGraph nodes are meant to be pure functions — read state in, return
*updates* out, no side effects. The reducer is what turns "I returned
this one new message" into "the conversation now has N+1 messages." It
also makes the state diffable/inspectable at every step, which is what
powers the visualization and checkpointing features.

---

## 2. Node — your loop body, split into named functions

Your single `for` loop body had two jobs:

```python
# THINK
resp = client.chat.completions.create(model=model, messages=messages, tools=[...])
choice = resp.choices[0].message

# ACT + OBSERVE
for tool_call in choice.tool_calls:
    ...
```

In LangGraph these become two separate node functions:

```python
def think(state: AgentState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}        # appended via add_messages

# ACT + OBSERVE — LangGraph ships this one PREBUILT, called ToolNode.
# It does exactly what your inner for-loop did by hand: read the last
# AIMessage's .tool_calls, run each matching tool by name, json-parse
# the arguments, append ToolMessage results.
tool_node = ToolNode(tools)
```

Each node is just: **state in, partial state out.** This is your loop
body, given a name and made return-based instead of mutation-based.

**The genuinely useful part here:** `ToolNode` is prebuilt. You don't
hand-write the `json.loads(tool_call.function.arguments)` +
dispatch-by-name + `tool_call_id` matching logic — LangGraph ships a
ready-made version because every agent needs this exact dispatcher.

---

## 3. Edge — "after THINK, go to ACT"

```python
graph.add_edge("act", "think")
```

This is your loop's *implicit* "go back to the top of the for-loop"
behavior — except now it's an explicit, drawable arrow instead of
Python control flow. The fact that your manual loop was a `for`
statement IS a graph with one cycle in it; LangGraph just makes that
cycle visible and inspectable.

---

## 4. Conditional edge — your one `if` statement, formalized

This is the most important mapping in the whole note. Your termination
check:

```python
if not choice.tool_calls:
    return choice.content   # done — no more looping
# else: fall through to ACT
```

becomes a **routing function** plus a conditional edge:

```python
from langgraph.prebuilt import tools_condition

graph.add_conditional_edges(
    "think",
    tools_condition,             # PREBUILT routing function
    {"tools": "act", "__end__": END},
)
```

`tools_condition` is LangGraph's prebuilt version of exactly your `if`
check: it looks at the last message, and if it has `tool_calls`, routes
to `"tools"` (your `"act"` node); otherwise routes to `END`. **One line
replaces one `if` statement** — same logic, framework-provided instead
of hand-written, because virtually every tool-using agent needs this
exact check.

---

## The graph shape, side by side with your loop

```
                    START
                      │
                      ▼
                  ┌───────┐
            ┌────►│ think │
            │     └───┬───┘
            │         │
            │   tools_condition()
            │         │
            │   ┌─────┴─────┐
            │   ▼           ▼
            │  act          END
            │   │
            └───┘
        (loop back to think)
```

```python
# your manual loop, annotated with the graph mapping
for _ in range(MAX_ITERATIONS):     # ──► recursion_limit (framework-enforced)
    # THINK                         # ──► the "think" node
    resp = client.chat.completions.create(...)
    choice = resp.choices[0].message

    if not choice.tool_calls:       # ──► tools_condition() + conditional edge to END
        return choice.content

    messages.append(...)            # ──► add_messages reducer (automatic)

    for tool_call in choice.tool_calls:   # ──► ToolNode (prebuilt "act" node)
        ...
        messages.append(...)        # ──► add_messages reducer (automatic)
    # (implicit) go back to top     # ──► edge: act -> think
```

**Identical control flow.** LangGraph didn't introduce a new idea here —
it gave your idea a vocabulary, two prebuilt nodes/functions, and a
framework-enforced safety cap (`recursion_limit`) replacing your
`MAX_ITERATIONS`.

---

## What LangGraph actually buys you over the manual version

| Your manual loop | LangGraph equivalent | What changed |
|---|---|---|
| `for _ in range(MAX_ITERATIONS)` | `recursion_limit` config at `.invoke()` time | Same safety valve, framework-enforced instead of a Python loop |
| `messages.append(...)` everywhere | `add_messages` reducer | Same effect, declared once instead of repeated at every call site |
| `if not choice.tool_calls: return` | `tools_condition` + conditional edge | Same logic, now drawable/inspectable as a graph |
| Manual `json.loads(...)` + dispatch-by-name | `ToolNode` (prebuilt) | You don't hand-write the dispatcher |
| Nothing | **Checkpointing** — pause, persist state, resume later, even across process restarts | New capability you didn't have |
| Nothing | **Streaming intermediate steps** — observe "thinking", "calling tool X" live, not just the final answer | New capability |
| Nothing | **Visualization** — `graph.get_graph().draw_mermaid()` produces an actual diagram of the graph | New capability |
| Nothing | **Multi-agent composition** — graphs can contain other graphs as nodes (subgraphs) | The on-ramp to multi-agent, covered in a future note |

Honest summary: for what you've built so far, LangGraph is mostly *the
same logic with better tooling*. The genuinely new capabilities
(checkpointing, multi-agent composition) are where it starts to earn
its keep — and that's exactly the direction the learning path is
heading next (MCP, multi-agent).

---

## The actual port — code

This is `PDFchat-app/pdfchat/agent_langgraph.py` in full, annotated.

### The tool — same retrieval logic, different schema mechanism

```python
from langchain_core.tools import tool

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
```

Compare to the manual version's hand-written JSON schema
(`pdfchat/tools.py::SEARCH_CORPUS_TOOL`). The `@tool` decorator inspects
the function's type hints and docstring and **builds that same JSON
schema automatically.** Same information, different authoring mechanism
— write a typed Python function with a good docstring instead of a
JSON dict by hand.

### Binding tools to the model

```python
llm = ChatGroq(model=model, temperature=0.0)
llm_with_tools = llm.bind_tools(tools)
```

This is the LangChain equivalent of passing `tools=[SEARCH_CORPUS_TOOL],
tool_choice="auto"` to `client.chat.completions.create(...)`. Same Groq
API underneath — `ChatGroq` is a thin wrapper.

### The graph assembly

```python
graph = StateGraph(AgentState)
graph.add_node("think", think)
graph.add_node("act", tool_node)
graph.set_entry_point("think")
graph.add_conditional_edges("think", tools_condition, {"tools": "act", "__end__": END})
graph.add_edge("act", "think")

compiled = graph.compile()
```

`.compile()` turns the graph definition into a runnable object — this is
the step where LangGraph validates the graph shape (no dangling nodes,
entry point set, etc.) and produces something you can `.invoke()`,
`.stream()`, or visualize.

### Running it

```python
result = compiled.invoke(
    {"messages": messages},
    config={"recursion_limit": max_iterations * 2 + 2},
)
final_answer = result["messages"][-1].content
```

`recursion_limit` counts **graph steps**, not full think-act cycles —
each hop between nodes is one step. We multiply by 2 (think + act per
cycle) plus a small buffer, to roughly match the manual loop's
`MAX_ITERATIONS` semantics.

---

## Proof: same question, same behavior, different machinery

We ran the identical multi-hop question through both implementations:

> "Compare how the Italian and American cookbooks each treat spice level
> and cream usage across their pasta dishes."

**Manual loop (`pdfchat/agent.py`):**
```
ITERATIONS: 2
TOOL CALLS: 4
  [1] query='Italian cookbooks spice level in pasta dishes'
  [2] query='American cookbooks spice level in pasta dishes'
  [3] query='Italian cookbooks cream usage in pasta dishes'
  [4] query='American cookbooks cream usage in pasta dishes'
```

**LangGraph port (`pdfchat/agent_langgraph.py`):**
```
THINK steps (AIMessages): 2
TOOL results: 4
  [1] '[italian_classics.pdf p. 1]\nItalian Classics Cookbook Spaghetti Carbonara...'
  [2] '[american_kitchen.pdf p. 1]\nThe American Kitchen — Pasta Edition Creamy Carbonara...'
  [3] '[italian_classics.pdf p. 1]\nItalian Classics Cookbook Spaghetti Carbonara...'
  [4] '[american_kitchen.pdf p. 1]\nThe American Kitchen — Pasta Edition Creamy Carbonara...'
```

**Same shape: 2 think-steps, 4 tool calls, synthesized answer citing both
PDFs on both axes.** The control flow is provably identical — only the
implementation vocabulary changed. This is the validation that the
"LangGraph is the same idea with better tooling" claim from the table
above isn't just an assertion — we measured it.

---

## Running it yourself

```bash
cd PDFchat-app
source .venv/bin/activate
pip install -r requirements.txt   # picks up langgraph, langchain-groq, langchain-core

python agent_chat_langgraph.py
```

Try the exact same questions you tried against `agent_chat.py` (the
manual version) and compare the iteration/search counts printed at each
turn. They should match closely — same model, same tool, same
underlying retrieval pipeline, same termination logic.

---

## What this sets up next

You now have the manual loop AND its LangGraph port — both genuinely
understood, not just one copy-pasted from docs. Per the original
roadmap from note 10, the next two steps:

1. **MCP (Model Context Protocol)** — wrap `search_corpus` as an MCP
   server instead of a Python function called directly (whether bound
   via the manual `SEARCH_CORPUS_TOOL` schema or LangChain's `@tool`
   decorator). The schema you already have is most of what MCP needs —
   it's a standardized transport for exactly this kind of tool
   definition. LangChain ships an MCP adapter
   (`langchain-mcp-adapters`) that turns MCP tools into LangChain tools
   automatically, which will read as familiar once you've seen `@tool`
   here.
2. **Multi-agent** — LangGraph graphs can be used as nodes inside other
   graphs (subgraphs). A second agent (e.g. a critic, or a
   web-search-fallback agent) becomes "two of these compiled graphs,
   composed" — not a new concept, a composition of this one.

→ Back to the [index](README.md).
