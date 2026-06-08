# 5. Parent-Child Chunking — small for matching, big for context

> **TL;DR:** Embed **small** child chunks for precise retrieval, but send the **larger parent** they belong to into the LLM. Decouples retrieval granularity from generation granularity.

## The problem

In naive RAG, one chunk size has to do two conflicting jobs at once:

| Job | Wants chunks that are... | Why |
|---|---|---|
| Retrieval matching | **Small** | A vector should represent ONE specific idea. Big chunks "average" many topics into a mushy vector that matches nothing precisely. |
| LLM context | **Large** | The model needs surrounding sentences to actually answer. A 50-word chunk is often missing the very sentence that answers the question. |

Any single size compromises one. Parent-child fixes it by using **two**.

## The fix in one picture

```
   Document
      │
      ▼
   ┌────────────────────────────────────────────────┐
   │  PARENT  (~1200 words, paragraph-sized)        │
   │                                                │
   │   ┌──────────┐  ┌──────────┐  ┌──────────┐     │
   │   │  child 1 │  │  child 2 │  │  child 3 │     │ ◄── ~240-word
   │   └──────────┘  └──────────┘  └──────────┘     │     children
   │                                                │
   └────────────────────────────────────────────────┘

   Embed:        every CHILD     ──►   FAISS  +  BM25
                                            │
                                            ▼  child matches the query
   Send to LLM:  the PARENT that contains the matched child
```

The child does the precise *matching*. The parent provides the rich *context* the model needs to write a good answer.

## End-to-end flow in our codebase

```
                   PDF
                    │
                    ▼
              load_pdf() ─► parents (big)    children (small, know parent_idx)
                    │            │                  │
                    │            │                  ├──► dense index (FAISS)
                    │            │                  └──► sparse index (BM25)
                    │            │
   per question:    │            │
                    │            │
   rewrite query   ─┘            │
        │                        │
        ▼                        │
   dense + sparse retrieval ──► fused child candidates
                                 │
                                 ▼
                         cross-encoder rerank
                                 │
                                 ▼
                       top-k CHILDREN ─────► children_to_parents()
                                                 │ (dedupe)
                                                 ▼
                                          PARENTS to LLM
```

## Why "small for matching" works better

Take a 1000-word chunk that covers three topics:

```
   "...transformers use self-attention...  ❶
    ...recurrent networks have vanishing gradient...  ❷
    ...transformers' main limitations include high memory cost..."  ❸
```

When you embed the whole thing, its vector is an *average* of ❶ + ❷ + ❸. A query about "transformer limitations" will find this chunk, but ALSO match badly because half the chunk isn't about that.

Split it into three ~330-word children:

```
   child 1: "...transformers use self-attention..."         ◄ vector ≈ "attention"
   child 2: "...recurrent networks have vanishing..."       ◄ vector ≈ "RNN issues"
   child 3: "...transformers' main limitations include..."  ◄ vector ≈ "transformer limits"
```

Now a query about "transformer limitations" lands DIRECTLY on child 3. The vector is dense with the right meaning. **Retrieval becomes more precise as you shrink.**

## Why "big for context" works better

But child 3 alone might be 240 words of "Transformers' main limitations include their O(n²) attention cost; researchers have proposed..." — without the preceding setup ("recall that self-attention compares every token to every other token") the LLM might miss what "O(n²)" is referring to.

So instead of feeding the LLM child 3, we feed it **the whole parent that contains child 3**. Now the model has:
- The preceding paragraph explaining self-attention
- The matched paragraph about limitations
- The following paragraph about proposed fixes

It can write a much better answer.

## Code

**Data model (`pdf_loader.py`):**

```python
@dataclass
class Chunk:        # CHILD — embedded and searched
    text: str
    page: int
    parent_idx: int   # which parent it belongs to

@dataclass
class ParentChunk:  # PARENT — fetched after a match, sent to LLM
    text: str
    page: int
```

**Two-pass split (per page):**

```python
def load_pdf(path, parent_size, parent_overlap, child_size, child_overlap):
    parents, children = [], []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        # 1) Parent pass over the page
        for parent_text in _window(text.split(), parent_size, parent_overlap):
            parent_idx = len(parents)
            parents.append(ParentChunk(text=parent_text, page=page_num))

            # 2) Child pass over THIS parent's text
            for child_text in _window(parent_text.split(), child_size, child_overlap):
                children.append(
                    Chunk(text=child_text, page=page_num, parent_idx=parent_idx)
                )
    return parents, children
```

**Key design choice:** children are split from the *parent's* text, not the page's text. This guarantees no child crosses a parent boundary — when we map a child back to its parent, we always get a clean self-contained block.

**Map matches back to parents (`children_to_parents`):**

```python
def children_to_parents(matched_children, parents):
    seen, out = set(), []
    for c in matched_children:
        if c.parent_idx in seen:          # dedupe — same parent only once
            continue
        seen.add(c.parent_idx)
        out.append(parents[c.parent_idx])
    return out
```

**In the chat loop (`main.py`):**

```python
top_children   = rerank(reranker, query, fused_candidates, top_k=4)
context_parents = children_to_parents(top_children, parents)   # ◄ the magic step
reply = answer(client, model, query, context_chunks=context_parents, ...)
```

## When to use it

- **Dense, technical documents** — API docs, codebases, research papers, contracts. The answer is often in one sentence, but the model needs the surrounding paragraph to interpret it.
- **Q&A bots** where queries are short and pointed (good fit for small children) but answers benefit from broader context.
- **Long PDFs** with paragraph or section structure.

## When you can skip it

- Short, self-contained documents (FAQs, KB articles) — paragraph and chunk are basically the same.
- When the LLM's context window is very small and you can't afford to send big parents.

## Tuning knobs

| Setting | Default | Rule of thumb |
|---|---|---|
| `parent_size` | 1200 | Roughly one rich paragraph / one screen of text. |
| `parent_overlap` | 200 | Small overlap; parents are big enough that boundary sentences usually appear inside one or another. |
| `child_size` | 240 | A few sentences. Small enough for one focused idea. |
| `child_overlap` | 40 | Tiny — children are small, you don't want them to mostly duplicate. |

Ratio tip: **parents ≈ 4–6× children** is a common sweet spot. If parents get too big, the LLM context fills up too fast. If too small, you lose the whole point.

## Caveats & gotchas

| Caveat | Notes |
|---|---|
| **More chunks to embed** | Children outnumber parents 4–6×. Indexing takes proportionally longer. One-time cost. |
| **Dedupe matters** | Multiple children of the same parent often rank high together. Without dedupe you'd send the same parent multiple times — wasted tokens and zero new info. |
| **Page citations come from the parent** | Children inherit `page` from their parent — citations stay coherent. If a parent spans two pages (we split per-page so it can't, but in other designs it can), you'd have to track which page the *match* was on. |
| **Reranker still sees CHILDREN** | We rerank children, not parents, because the cross-encoder is more accurate on smaller, focused units. The parent only enters the picture after reranking. |
| **Doesn't fix bad recall** | If the answer-bearing child isn't retrieved at all, fetching its parent can't help. Pair with hybrid search (note #4) for solid recall. |

## Aliases & relatives

| Name | Notes |
|---|---|
| Parent-child retrieval | Common name. LangChain's `ParentDocumentRetriever`. |
| Small-to-big retrieval | LlamaIndex's term. Same idea. |
| Hierarchical chunking | Generalization: parent of parent of parent... a chunk tree. |
| Sentence-window retrieval | Variant where each "child" is one sentence; the "parent" is the few sentences around it. |
| Proposition indexing | Take it further: embed *single factual statements* extracted by an LLM, then send their source paragraph. Higher quality but expensive to build. |
| Late chunking | Don't chunk before embedding — embed the whole doc, then derive chunk vectors from spans. Newer research direction. |

## Key concepts this teaches

- **Retrieval granularity ≠ generation granularity.** The thing you search and the thing you feed to the LLM can (and often should) be different sizes.
- **Decoupling via metadata.** A simple `parent_idx` pointer is what makes the whole pattern work. This same trick — using metadata to link units of different sizes — shows up everywhere: parent docs, hierarchical RAG, citation graphs, GraphRAG.
- **Deduplication of retrieved units.** When you map small → big, multiple small hits can collapse into one big block. Always dedupe.
- **Why naive top-k is leaving quality on the table.** Once you've seen the small-to-big improvement, you start spotting other places where the wrong unit is being retrieved.

→ Back to the [index](README.md).
