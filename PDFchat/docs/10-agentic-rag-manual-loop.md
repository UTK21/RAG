# 10. Agentic RAG — a hand-rolled ReAct loop (built before any framework)

> **TL;DR:** Instead of retrieve-once-then-generate, let the LLM decide whether it needs to search again, with a refined query, before answering. We build this manually first — no LangGraph, no LangChain — so the abstraction means something when we adopt a framework later.

---

## Why this is a different thing from everything before it

Every technique in notes 1–9 improved **one retrieval pass**: better
chunks, better ranking, better fusion, better rewriting. But they all
share a shape:

```
   question ──► ONE retrieval ──► generate ──► answer
```

Some questions can't be answered by one retrieval, no matter how good
it is. Example:

> "Compare how the Italian and American cookbooks each treat spice level
> and cream usage across their pasta dishes."

This question has **four sub-topics** (Italian-spice, American-spice,
Italian-cream, American-cream). A single fused query like "spice and
cream in Italian and American pasta" produces a vague embedding that
doesn't strongly match any of the four specific answers. Top-k retrieval
on that vague query is likely to surface 1-2 of the four facts and miss
the rest.

The fix isn't a better retriever. It's **letting the model search
multiple times**, each time with a focused, specific query.

```
   question ──► THINK: "I need 4 separate facts" ──► search ×4 ──► synthesize ──► answer
```

This is **agentic RAG**: the LLM controls *when* and *what* to retrieve,
not just consuming whatever a fixed pipeline handed it.

---

## The ReAct pattern

ReAct = **Rea**son + **Act**. The foundational loop behind nearly every
agent framework (LangGraph, AutoGen, CrewAI all implement variations of
this same idea).

```
   ┌─────────────────────────────────────────────────────────┐
   │                                                         │
   │   THINK   ──►   ACT   ──►   OBSERVE   ──┐                │
   │     ▲                                   │                │
   │     └───────────────────────────────────┘                │
   │                                                         │
   └─────────────────────────────────────────────────────────┘
```

- **THINK** — the LLM looks at the conversation so far (including any
  search results already gathered) and decides: call a tool, or answer.
- **ACT** — if it chose a tool, we (the code, not the LLM) actually
  execute that function.
- **OBSERVE** — the tool's output gets appended to the conversation, and
  we loop back to THINK with that new information available.

The loop terminates when the model stops requesting tools (it has
"decided" it knows enough), or when a hard iteration cap is hit (safety
valve against infinite loops).

---

## Why build this by hand instead of using LangGraph immediately

LangGraph would give you nodes, edges, and a `tools` integration for
free. So why not start there?

**Because the things LangGraph hides are exactly the things worth
understanding first:**

| What LangGraph would hide | What you learn by writing it yourself |
|---|---|
| Message-history bookkeeping (assistant tool-call message must be echoed back before the tool result) | The API's exact contract — miss this and the model gets confused about what it asked for |
| The termination condition | You decide: "no tool calls" = done. Forgetting this = infinite loop, real $ cost |
| The iteration cap | Without one, a confused model calls tools forever. You feel why this matters by almost hitting it yourself |
| Tool dispatch (matching a tool_call_id to the right Python function and its result) | You see exactly how the LLM's structured request becomes a real function call |

Once you've written this loop with your own hands, reading LangGraph's
docs later means "oh, `add_conditional_edge` is just my `if not
tool_calls: return` line, expressed as a graph." The framework becomes
legible instead of magic.

---

## The implementation

### The tool

A tool is two things: a JSON schema describing it to the LLM, and the
actual Python function that runs when called.

```python
# pdfchat/tools.py
SEARCH_CORPUS_TOOL = {
    "type": "function",
    "function": {
        "name": "search_corpus",
        "description": (
            "Search the PDF knowledge base for chunks relevant to a query. "
            "Call this multiple times with different, more specific "
            "queries if one search isn't enough."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", ...}},
            "required": ["query"],
        },
    },
}

def search_corpus(pipe: Pipeline, query: str) -> str:
    """Reuses the EXACT SAME retrieve -> fuse -> rerank -> small-to-big
    pipeline already built for regular chat. The agent's superpower
    isn't a smarter retriever — it's calling this retriever MULTIPLE
    TIMES with different queries before answering."""
    top_children, _ = pipe._retrieve(query)
    parents = children_to_parents(top_children, pipe.idx.parents)
    return "\n\n".join(f"[{p.doc_name} p. {p.page}]\n{p.text}" for p in parents)
```

The LLM never calls this Python function directly. It can only emit a
*request* to call it — our loop is the one that executes it.

### The loop

```python
# pdfchat/agent.py  (simplified)
messages = [system_prompt, *history, user_question]

for _ in range(MAX_ITERATIONS):
    # THINK
    resp = client.chat.completions.create(
        model=model, messages=messages,
        tools=[SEARCH_CORPUS_TOOL], tool_choice="auto",
    )
    choice = resp.choices[0].message

    if not choice.tool_calls:
        return choice.content  # model decided it has enough info — done

    messages.append({"role": "assistant", "tool_calls": choice.tool_calls, ...})

    for tool_call in choice.tool_calls:
        args = json.loads(tool_call.function.arguments)
        # ACT
        result = search_corpus(pipe, args["query"])
        # OBSERVE
        messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

# Hit the cap without a final answer — force one rather than looping forever.
```

Three things worth noticing:

1. **`tool_choice="auto"`** is the entire "decide what to do next"
   mechanism. No prompt trick needed — it's a first-class parameter of
   the chat API (works identically on OpenAI, Groq, Anthropic's tool
   use).
2. **The assistant's tool-call request must be echoed back** into the
   message list before the tool's result. The API uses this to match
   "here's what I asked for" with "here's what came back" — skip it and
   you get malformed-request errors or model confusion.
3. **`tool_call_id`** lets the model issue MULTIPLE tool calls in one
   turn (as it did in our test — 4 searches in one THINK step) and
   match each result back to the right request.

---

## What we actually saw when we ran it

Test question:

> "Compare how the Italian and American cookbooks each treat spice level
> and cream usage across their pasta dishes."

```
   ITERATIONS: 2
   TOOL CALLS: 4
     [1] query='Italian cookbooks spice level in pasta dishes'
     [2] query='American cookbooks spice level in pasta dishes'
     [3] query='Italian cookbooks cream usage in pasta dishes'
     [4] query='American cookbooks cream usage in pasta dishes'
```

The model **decomposed one vague question into four focused searches**
— one per (cookbook × topic) combination — all in a single THINK step
(parallel tool calling), then synthesized a comparison from all four
results in the second iteration.

```
                    "Compare spice + cream,
                     Italian vs American"
                            │
                            ▼
                      THINK (iter 1)
                            │
            ┌───────────────┼───────────────┬───────────────┐
            ▼               ▼               ▼               ▼
       search:          search:         search:         search:
       Italian-spice    American-spice  Italian-cream   American-cream
            │               │               │               │
            └───────────────┴───────┬───────┴───────────────┘
                                    ▼
                              THINK (iter 2)
                                    │
                                    ▼
                         synthesized comparison,
                         citing both PDFs on both axes
```

This is something the single-shot `chat.py` pipeline structurally cannot
do — it fuses everything into ONE retrieval call with ONE query
embedding, which can't strongly match four separate facts at once.

---

## Mental model: what changed vs everything before

```
   PASSIVE RETRIEVAL (notes 1-9)          AGENTIC RETRIEVAL (this note)
   ────────────────────────────          ──────────────────────────────
   Pipeline decides when to search        LLM decides when to search
   (always exactly once)                  (zero, one, or many times)

   Pipeline decides what to search        LLM decides what to search
   (one fused/rewritten query)            (can split into sub-queries)

   Fixed cost per question                Variable cost per question
   (1 retrieval, 1 generation)            (N retrievals, scales with
                                           question complexity)

   Good for: single-fact lookups          Good for: multi-part, multi-hop,
                                           comparison questions
```

Agentic RAG is strictly more capable but strictly more expensive and
slower per query. **Use it selectively** — route simple questions to the
cheap single-shot pipeline, complex ones to the agent. (This routing
decision is itself a Tier 2 technique — query routing, covered in
`08-tier-2-roadmap.md`.)

---

## Safety considerations (the part that actually matters in production)

| Risk | Mitigation in this implementation |
|---|---|
| Infinite loop | `MAX_ITERATIONS = 5` hard cap |
| Runaway cost | Each iteration = 1 LLM call. Cap directly bounds worst-case cost per question. |
| Model never settles | After hitting the cap, we force a final answer with whatever's gathered, instead of erroring or looping silently |
| Tool misuse (bad query) | The tool description explicitly tells the model to use specific, focused queries — prompt engineering is still load-bearing even with structured tool calling |

This is the smallest responsible version of an agent loop. Production
systems add: per-user rate limits, cost tracking per session, timeout
in addition to iteration cap, and (per Self-RAG / CRAG from the
roadmap) a critique step that checks whether the gathered information
actually supports an answer before generating one.

---

## What this sets up next

This file is the prerequisite for the next two items on the learning
path:

1. **LangGraph** — port this exact loop to nodes/edges/state. You'll
   recognize every piece: THINK is a node, the tool-or-answer branch is
   a conditional edge, MAX_ITERATIONS is a built-in recursion limit,
   the message list is the graph's state.
2. **MCP (Model Context Protocol)** — wrap `search_corpus` as an MCP
   server instead of a Python function called directly. The tool's JSON
   schema you already wrote is most of what MCP needs — it's a
   standardized transport for exactly this kind of tool definition.
3. **Multi-agent** — once one agent with one tool is solid, a second
   agent (e.g. a critic, or a web-search fallback) is "two of these
   loops, coordinating" — not a new concept, a composition of this one.

→ Back to the [index](README.md).
