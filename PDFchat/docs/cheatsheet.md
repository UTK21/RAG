# RAG Cheatsheet — Pipeline Walkthrough & Mental Models

A quick reference: how the whole pipeline fits together and the core
concepts that show up in almost every RAG paper or design doc.

---

## A) Parent-Child by example

Setup: the PDF is a **recipe book**. Page 4 covers spaghetti carbonara.

### Step 1 — One page becomes 1 parent and 3 children

```
PAGE 4 of the PDF
┌────────────────────────────────────────────────────────────┐
│                                                            │
│  PARENT #2  (~1200 words)                                  │
│  "Spaghetti Carbonara                                      │
│                                                            │
│   ── ingredients ──                                        │
│   Eggs, guanciale, pecorino, black pepper, spaghetti.      │  ◄── child 7
│                                                            │
│   ── method ──                                             │
│   Render the guanciale slowly until crisp. Whisk eggs      │  ◄── child 8
│   with pecorino and black pepper. Cook spaghetti…          │
│                                                            │
│   ── common mistakes ──                                    │
│   Don't add cream — it's not authentic. Don't scramble     │  ◄── child 9
│   the eggs; keep the pan off the heat when you add them."  │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

Each child knows it lives inside parent #2:

```
child 7  →  parent_idx = 2
child 8  →  parent_idx = 2
child 9  →  parent_idx = 2
```

### Step 2 — Embed only the children

Children go into FAISS + BM25. Parents sit in a list, waiting to be fetched.

### Step 3 — User asks: "Should I add cream to carbonara?"

Top reranked children:

```
#1  child 9   "Don't add cream — it's not authentic..."        (parent_idx=2)
#2  child 8   "Render the guanciale slowly..."                  (parent_idx=2)
#3  child 7   "Ingredients: eggs, guanciale..."                 (parent_idx=2)
#4  child 22  "Cream sauces often pair with pasta..."           (parent_idx=6)
```

Three of the four winners point to the same parent — normal when content
is concentrated in one paragraph.

### Step 4 — The small → big swap, with dedupe

Without dedupe, the LLM would receive parent 2 three times — wasted tokens,
zero new information:

```
❌ NAIVE:    to LLM = [parent 2, parent 2, parent 2, parent 6]
✅ DEDUPED:  to LLM = [parent 2, parent 6]
```

Order reflects which parent had the BEST-matching child. Parent 2 comes
first because child 9 (its best child) was rank #1.

### Step 5 — Rich context to the LLM

```python
def children_to_parents(matched_children, parents):
    seen, out = set(), []
    for child in matched_children:
        if child.parent_idx in seen:
            continue
        seen.add(child.parent_idx)
        out.append(parents[child.parent_idx])
    return out
```

A `set` to remember which parents we've grabbed, walking the ranked list in order.

---

## B) The complete pipeline — annotated by technique

```
   PDF
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: PARSE + DOUBLE CHUNK                                   │
│  load_pdf() returns parents (big) + children (small)            │
│  Each child stores parent_idx pointing at its parent.           │
│                                  ◄── Technique #5 Parent-Child  │
└─────────────────────────────────────────────────────────────────┘
    │
    ├── parents (in a list, not embedded)
    │
    ├── children ──────┬─────────────────────┐
    │                  ▼                     ▼
    │           ┌──────────────┐      ┌──────────────┐
    │           │  FAISS       │      │   BM25       │
    │           │  (dense)     │      │   (sparse)   │
    │           │  meaning     │      │   keywords   │
    │           └──────────────┘      └──────────────┘
    │                  ▲                     ▲
    │                  │ Technique #1        │ Technique #4
    │                  │ Naive RAG (dense)   │ Hybrid Search
    │
    ▼ per question ───────────────────────────────────────────────

   user question (e.g. "what about its limitations?")
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: REWRITE FOLLOW-UPS INTO STANDALONE QUERIES             │
│  query_rewriter() uses last N chat turns to fill in missing     │
│  context: "what about its limitations?" → "what are the         │
│  limitations of transformers?"                                  │
│                       ◄── Technique #2 Conversational Memory    │
└─────────────────────────────────────────────────────────────────┘
    │
    │  standalone_query
    │
    ├──────────────────────┐
    ▼                      ▼
┌─────────────┐    ┌─────────────┐
│ FAISS top-20│    │ BM25 top-20 │     ◄── Technique #1 + #4
│ children    │    │ children    │
└──────┬──────┘    └──────┬──────┘
       │                  │
       └────────┬─────────┘
                ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: RECIPROCAL RANK FUSION                                 │
│  Merge the two ranked lists using only RANKS (not scores).      │
│  Score per chunk = sum of 1/(60 + rank_in_list).                │
│  Chunks ranked high in BOTH lists float to the top.             │
│                                ◄── Technique #4 Hybrid Search   │
└─────────────────────────────────────────────────────────────────┘
                │
                ▼  ~30 unique candidate children
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: CROSS-ENCODER RE-RANK                                  │
│  Score each (query, child) pair TOGETHER with a cross-encoder.  │
│  Much more accurate than the bi-encoder, but slow — that's why  │
│  we only run it on ~30 candidates, not the whole index.         │
│                                     ◄── Technique #3 Re-ranking │
└─────────────────────────────────────────────────────────────────┘
                │
                ▼  top 4 children
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: SMALL → BIG MAP (with dedupe)                          │
│  children_to_parents() walks the top-4 list and collects each   │
│  child's parent_idx. Already-seen parents are skipped, so we    │
│  end up with the unique parents in best-first order.            │
│                                  ◄── Technique #5 Parent-Child  │
└─────────────────────────────────────────────────────────────────┘
                │
                ▼  parents (could be 1–4 of them)
┌─────────────────────────────────────────────────────────────────┐
│  STEP 6: GROUNDED GENERATION                                    │
│  Build the prompt:                                              │
│    system  = "Use ONLY the context. Cite (p. N)."               │
│    history = last N chat turns (natural conversational tone)    │
│    user    = "Context:\n<parents>\n\nQuestion: <original>"      │
│                                                                 │
│  We use the ORIGINAL question for generation, not the rewrite — │
│  the rewrite only helped retrieval. We pass history so replies  │
│  sound like a continuation.                                     │
│                                                                 │
│       ◄── Technique #1 grounding + Technique #2 history-aware   │
└─────────────────────────────────────────────────────────────────┘
                │
                ▼
        grounded answer w/ page citations
                │
                ▼
        append (user, assistant) to history     ◄── Technique #2
```

### Step-by-step technique map

| Step | Technique used |
|---|---|
| 1. Parse + double chunk | #5 Parent-Child |
| 2. Rewrite query | #2 Conversational Memory |
| 3a. Dense retrieval (FAISS) | #1 Naive RAG (the dense half) |
| 3b. Sparse retrieval (BM25) | #4 Hybrid Search |
| 3c. RRF fusion | #4 Hybrid Search |
| 4. Cross-encoder re-rank | #3 Re-ranking |
| 5. Small → big with dedupe | #5 Parent-Child |
| 6. Grounded generation | #1 + #2 |

Every question uses **all five** techniques.

---

## C) Mental models — explained in detail

### Chunking + overlap
Long documents need to be cut into smaller pieces because (a) embedding
models can only ingest a few hundred words at a time, and (b) sending a
whole document to the LLM is expensive and dilutes the signal.

**Why overlap.** Imagine an important sentence falling exactly on a chunk
boundary. Without overlap, half goes to chunk A and half to chunk B —
neither matches a query well. With overlap, the sentence appears whole in
at least one chunk.

**Where it shows up.** Every retrieval system has to make this choice.
Chunk-size tuning is one of the most common dials in production RAG.

### Embeddings = meaning vectors
An embedding turns text into a fixed-length list of numbers. The geometry
of these vectors captures meaning: "car" and "automobile" land near each
other; "car" and "banana" don't.

**Concrete intuition.** Imagine every word as a point in 384-dimensional
space: "cat" and "kitten" are neighbors, "cat" and "dog" are nearby (both
pets), "cat" and "skyscraper" are far apart. Generalize from words to
whole chunks.

**Where it shows up.** Search engines, recommender systems, deduplication,
clustering, classification, RAG, agents, multimodal models. Embeddings are
the universal currency of "are these two things similar?"

### Normalize → dot product = cosine
The "right" similarity for vectors is **cosine similarity** — the angle
between them. Computing cosine is slower than computing a dot product.

**The trick.** Scale every vector to length 1 → dot product equals cosine.
Free win. So we normalize at index time and use FAISS's `IndexFlatIP`
(inner product) for max speed with exact cosine semantics.

**Where it shows up.** Standard in every modern vector DB (FAISS, Qdrant,
Pinecone, Weaviate). So universal that most embedding libraries default to
outputting unit vectors.

### Grounding system prompt
Without instructions, LLMs *always try to be helpful* — even when they
don't know. They'll guess, invent, paraphrase loosely.

**The fix.** A short system prompt: "Use ONLY the provided context. If the
answer isn't there, say you don't know. Cite page numbers."

That's the highest-leverage anti-hallucination tool. The citation rule
also gives users a way to *audit* every claim.

**Where it shows up.** Every RAG system worth using has this. Most fancy
techniques are about feeding *better context* to a model that's already
correctly instructed.

### Query rewriting
The query the user *types* is often a bad query to *search* with.
Follow-ups are the obvious case ("what about page 3?"), but also: vague
phrasing, missing keywords, multiple intents in one sentence.

**Pattern.** Run the user's query through an LLM first to transform it.
Search with the transformed version. Generate with the original.

**Variations.** Multi-query, HyDE, decomposition, step-back, query
routing, agentic search. All the same core idea: LLMs as text
transformers, not just generators.

(Full deep-dive: [`06-query-rewriting.md`](06-query-rewriting.md).)

### Bi-encoder vs cross-encoder
Two ways to score "is this chunk relevant?":

- **Bi-encoder:** encode query and chunk *separately*, compare their
  vectors. Fast (encode chunks once, compare millions per query). But the
  model never sees them together.
- **Cross-encoder:** glue query and chunk into one input, run through the
  model together. Every attention head reads both, so scoring is much more
  accurate — but you have to re-run for every (query, chunk) pair.

**The trade-off.** Accuracy ↔ speed.

**The escape hatch.** Cascade them: bi-encoder for recall (millions → 20),
cross-encoder for precision (20 → 4).

**Where it shows up.** Same trade-off in search engine ranking, ad
systems, code completion, product recommendations.

### Recall → precision cascade
Don't make one model both fast *and* perfectly accurate. Stack them. The
first stage casts a wide net (high recall); later stages filter with
increasing precision.

**Library analogy.**
- Stage 1 (recall): grab any cart that *might* have the book.
- Stage 2 (precision): librarian skims and pulls the actual book.

Doing only stage 2 takes forever. Doing only stage 1 leaves you with 20
maybe-books.

**Where it shows up.** Google search, TikTok feed, ad auctions,
recommenders — every modern RAG stack.

### Dense vs sparse retrieval
Two ways to match queries against documents:

- **Dense** (embeddings) — captures *meaning*. "Vehicle" matches "car".
  But "Vaswani" (a rare proper noun) gets averaged into "academic words"
  and goes missing.
- **Sparse** (BM25) — pure word counting. "Vaswani" lights up instantly.
  But "automobile" and "car" are unrelated tokens to BM25.

They fail on opposite things. Running both and merging catches more.

**Where it shows up.** Elasticsearch, OpenSearch, Vespa, Weaviate — all
offer hybrid by default. Beats either alone in benchmarks.

### Rank-based fusion (RRF)
Multiple ranked lists from different scoring systems? You can't just add
the scores — the scales are different. RRF solves it by **throwing away
the scores** and using only the rank position.

```
score(doc) = Σ over each list of  1 / (60 + rank_in_that_list)
```

**Why universal.** Doesn't care what produced the rankings. Fuse dense +
sparse, two different embedders, or BM25 + metadata + LLM judge.

**Where it shows up.** Elasticsearch's `rrf` API, Vespa, Weaviate, every
modern search stack. The duct tape of fusion.

### Retrieval granularity ≠ generation granularity
The unit you *search* and the unit you *send to the LLM* don't have to be
the same.

**Why this is profound.** Most naive RAG treats them as the same — embed
a chunk, retrieve the chunk, send the chunk. That's one chunk size
compromising two opposite goals.

Once they can differ:
- **Parent-child** — embed small, send big.
- **Sentence-window** — embed a sentence, send its surroundings.
- **Proposition indexing** — embed atomic facts extracted by an LLM, send
  the source paragraph.
- **GraphRAG** — match against entity descriptions, send the relationships
  subgraph.
- **Summary indexing** — embed LLM-generated summaries, send the source.

Same pattern, different units. The single most powerful design lever in
modern RAG.

### Metadata pointers (`parent_idx`, `doc_name`, ...)
A tiny field on each chunk unlocks features way out of proportion to its
size:

- `parent_idx` → small→big retrieval
- `doc_name` → source attribution and citations
- `timestamp` → recency filtering / freshness
- `section_id` → hierarchical retrieval
- `tenant_id` → multi-tenant search
- `cited_by` → chunks become a graph

**Lesson.** When designing a chunk, ask "what tiny piece of metadata
might unlock a future feature?" Adding it later means re-embedding
everything.
