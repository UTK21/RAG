# 8. Tier 2 Roadmap — frontier techniques you can now actually measure

> **TL;DR:** Tier 1 turned the demo into a system. Tier 2 turns the system into something *competitive*. Every technique in this tier is a variation of patterns you've already seen — but each one targets a specific failure mode and unlocks a class of use cases that naive RAG can't touch.

---

## Where you are now

You've built and shipped the foundation:

```
   Tier 0 (study version, PDFchat/):
     5 techniques layered into one chatbot — naive RAG, conversational
     memory, re-ranking, hybrid search, parent-child chunking.

   Tier 1 (product version, PDFchat-app/):
     Multi-PDF + persistent indexes + streaming + HyDE + eval harness.
     Found and fixed two real bugs (rewriter, citation regex) using the
     harness.

   You are here ────────────────────────────────────────────► YOU
                                                              │
                                                              ▼
                                                          Tier 2
```

You have the two things that make Tier 2 worthwhile:

1. **An eval harness.** Without one, every Tier 2 technique you add is
   guesswork. With one, every change is a measured A/B.
2. **A clean architecture.** New retrievers slot into `pipeline.py::_retrieve`,
   new metadata fields slot into `loader.py::Chunk`, new metrics slot
   into `eval/metrics.py`. The shape is set.

---

## The mental model for Tier 2

Tier 0 and 1 were mostly about **retrieval quality** — getting the right
chunks to the LLM. Tier 2 splits into two distinct families:

```
   ┌────────────────────────────────────────────────────────────────┐
   │  FAMILY A: BETTER RETRIEVAL                                    │
   │                                                                │
   │  Make the chunks themselves smarter at index time, so the same │
   │  retrieval machinery finds better candidates.                  │
   │                                                                │
   │    → Contextual Retrieval                                      │
   │    → Fine-tuning the embedder                                  │
   │    → Multimodal RAG                                            │
   │    → GraphRAG                                                  │
   └────────────────────────────────────────────────────────────────┘

   ┌────────────────────────────────────────────────────────────────┐
   │  FAMILY B: SMARTER GENERATION LOOPS                            │
   │                                                                │
   │  Let the LLM control WHEN and WHAT to retrieve. Add feedback   │
   │  loops where the model critiques, refines, or escalates.       │
   │                                                                │
   │    → Self-RAG / CRAG                                           │
   │    → Agentic RAG                                               │
   │    → CAG (the "skip retrieval" option)                         │
   └────────────────────────────────────────────────────────────────┘
```

If Tier 0+1 was "build a better search engine for the LLM", Tier 2 is
"either make each chunk inherently smarter, or let the LLM drive the
search itself."

---

# The Tier 2 menu

| # | Technique | Family | Effort | Risk | Pays off most when... |
|---|---|---|---|---|---|
| **2.1** | Contextual Retrieval | A | Small | Low | Long docs with ambiguous sections |
| **2.2** | Self-RAG / CRAG | B | Medium | Medium | High-stakes answers, low tolerance for hallucination |
| **2.3** | Agentic RAG | B | Large | Medium | Complex / multi-hop questions, multi-source systems |
| **2.4** | Multimodal RAG | A | Medium | Medium | PDFs full of charts, tables, diagrams |
| **2.5** | GraphRAG | A | Large | High | Relationship questions over interconnected data |
| **2.6** | Fine-tuning the embedder | A | Large | Medium | Highly specialized domains (medical, legal, internal) |
| **2.7** | CAG (Cache-Augmented Generation) | B | Small | Low | Small, slow-changing corpora that fit in context |

> **Recommended order:** 2.1 → 2.2 → 2.3 → (then 2.4/2.5/2.6/2.7 based on use case).
> 2.1 is the highest impact for the lowest effort. 2.2 and 2.3 build the
> "self-reflecting / agentic" mindset that the rest of the field assumes.

---

## 2.1 — Contextual Retrieval (Anthropic, 2024)

> **TL;DR:** Before embedding each chunk, ask an LLM to write a 1–2 sentence summary of *where this chunk sits in the document*. Prepend that summary to the chunk's text. Re-embed. ~50% drop in retrieval failure rate on real docs (Anthropic's reported number).

### The problem it solves

Chunks taken out of their parent context can be ambiguous. Imagine a
paragraph in a financial filing that says: *"This figure includes a
one-time charge of $34M."*

What figure? Which company? Which year? The chunk alone doesn't say.
Embedding it produces a vector for "vague financial talk" — which won't
match queries like "what was Acme Corp's 2026 Q3 restructuring impact?"

### The trick

At index time, for each chunk, ask an LLM:

> "Here's the whole document. Here's a specific chunk from it. Write
> a 1–2 sentence note explaining how this chunk relates to the document
> as a whole."

Output: *"This chunk discusses Acme Corp's Q3 2026 restructuring,
specifically the $34M one-time charge mentioned in section 4.2."*

Prepend that summary to the chunk text *before* embedding. Now the
vector represents both the local content AND its global context.

```
   BEFORE: embed(chunk_text)
   AFTER:  embed(contextual_summary + chunk_text)
                ▲
                │
                LLM-generated, 1-2 sentences
                describing where this chunk fits
```

### Mental model

This is **index-time augmentation**. We've seen lots of query-time
augmentation (multi-query, HyDE, query rewriting). Contextual Retrieval
is the dual: augment the *chunks* instead of the *query*. Both are forms
of bridging the question/answer gap, but Contextual Retrieval does it
*once* at index time and benefits *every* future query.

### What's new conceptually

- **Pre-computation as a quality lever.** You spend more compute at index
  time so you spend less per query. With long-context models making this
  cheap (one call per chunk, batched), the math has shifted.
- **The metadata-pointer pattern, again.** `contextual_summary` is just
  another field on the chunk — same lever as `parent_idx` and `doc_name`,
  used differently.
- **Anthropic's full recipe** also combines this with hybrid (BM25 + dense)
  + reranking. You've already built those. Adding Contextual is the
  cherry on top of the cascade you have.

### When to use it

- Long technical docs with repetitive structure (10-Ks, RFPs, manuals).
- Domains where chunks lose meaning without surrounding context.
- Any RAG over docs > ~50 pages where naive chunking hurts.

### When to skip it

- Tiny corpora (the LLM cost per chunk dominates).
- Domains where chunks are already self-contained (FAQs, KB articles).

### Effort estimate

- New field on `Chunk` / `ParentChunk`: `contextual_summary: str`.
- New ingest-time step in `loader.py`: per-chunk LLM call.
- Adjust `embeddings.py` to prepend summary before embedding.
- Manifest fingerprint includes `use_contextual` flag.
- New eval `--config USE_CONTEXTUAL=true` for A/B.

Maybe 100–150 lines of code. Half a day with the eval harness in place.

### How to measure it

Run `python eval.py --config USE_CONTEXTUAL=true` and compare aggregate
`retrieval_recall` against baseline. Anthropic's claim is a ~50% drop in
"failed retrievals"; on your corpus the number will be different but
the *direction* tells you if it's worth keeping on.

---

## 2.2 — Self-RAG / CRAG (Corrective RAG)

> **TL;DR:** After retrieving and generating, ask the LLM: "Are these chunks actually relevant? Is my answer grounded?" If not — retry with different chunks, or escalate to a fallback (web search, refusal).

### The problem it solves

Even with hybrid + rerank + parent-child + contextual, retrieval
sometimes pulls *plausible but wrong* chunks. The LLM then dutifully
generates an answer from them — confidently wrong.

Worse, end-to-end metrics will pass it as a success if the *form* of the
answer looks right.

### The trick

Add a **self-critique loop**:

```
   retrieve → generate answer → ask LLM "is this grounded?"
                                       │
                          ┌────────────┼────────────┐
                          ▼            ▼            ▼
                       yes            no          partial
                          │            │            │
                          ▼            ▼            ▼
                     return     retry retrieval   escalate
                                 with refined     ("I don't know"
                                  query           OR web search)
```

Two named variants:

- **Self-RAG** (Asai et al. 2023) — model emits special "reflection
  tokens" indicating whether to retrieve, whether retrieved chunks are
  relevant, whether the answer is supported.
- **CRAG / Corrective RAG** (Yan et al. 2024) — simpler practical
  version. A separate small model classifies retrieved chunks as
  Correct / Ambiguous / Incorrect, and the system reacts accordingly.

### Mental model

Self-RAG is what happens when you stop trusting any single component
of the pipeline and add **mutual checking**. The retriever can be
wrong; the LLM should catch it. The LLM can be wrong; the critique
step should catch it.

This is the first step toward **agentic** behavior — the system makes
decisions about its own execution flow.

### What's new conceptually

- **The model as judge of its own retrieval.** A new use of LLMs:
  scoring the *input* to the LLM, not just the output.
- **Conditional pipelines.** Until now every query took the same path.
  Self-RAG branches: short path for confident queries, long path for
  uncertain ones.
- **Escape hatches.** Saying "I don't know" or "let me check the web"
  is an output, not a failure.

### When to use it

- Medical, legal, financial Q&A — domains where confident wrongness is
  dangerous.
- Customer-facing bots where hallucination = lost trust.
- Anywhere you'd rather get "I don't know" than a wrong answer.

### When to skip it

- Latency-critical chat (every Self-RAG check adds 1–2 LLM calls).
- Low-stakes / exploratory use cases.

### Effort estimate

- New module `pdfchat/critique.py` with relevance + grounding judges.
- Modify `pipeline.py` to branch on critique results.
- New metrics: `self_critique_accuracy`, `escalation_rate`.

200–300 lines of code. A full day.

### How to measure it

Add test cases where the answer is genuinely not in the corpus
(out-of-corpus tests). Without Self-RAG: bot invents an answer.
With Self-RAG: bot says "I don't know" or escalates. The new metric
`escalation_rate` should track gracefully — too low (bot still invents),
too high (bot is paranoid), or just right.

---

## 2.3 — Agentic RAG

> **TL;DR:** Don't make retrieval a fixed step. Give the LLM tools — "search_corpus", "search_web", "ask_user", "read_full_document" — and let it decide when and how to use them.

### The problem it solves

Many real questions need *more than one* retrieval. Examples:

- *"Compare the carbonara recipes across all my cookbooks."* — needs to
  retrieve, summarize, retrieve again with different queries, compare.
- *"What did the founder of Stripe's previous company do before that?"* —
  multi-hop. First retrieve to find the previous company. Then retrieve
  again to find what they did before.
- *"Find me information about X, but if it's not in the PDFs, check the
  web."* — conditional source switching.

A fixed single-shot pipeline can't do any of these.

### The trick

Treat the LLM as the **driver**, not a step. Give it tools:

```python
tools = [
    search_corpus(query),
    search_web(query),
    read_full_document(doc_name),
    ask_clarifying_question(question),
    summarize_chunks(chunks),
]
```

Then run a loop:

```
   while not done:
       thought = LLM.think(history)
       if thought.is_final_answer:
           return thought.answer
       tool, args = thought.choose_tool()
       result = call_tool(tool, args)
       history.append((thought, result))
```

The LLM decides when it has enough information. It can search multiple
times with refined queries, switch tools, ask the user for clarification,
or just answer.

### Mental model

We've used LLMs as **answer generators** and as **input transformers**
(query rewriting). Agentic RAG uses them as **executive controllers** —
deciding what action the system takes next.

This is the most powerful pattern in modern AI systems, and also the
most dangerous: a buggy agent can spend $50 on API calls in a loop and
still produce nonsense. Bounded loops, timeouts, max-iterations are
all critical.

### What's new conceptually

- **Tool calling / function calling.** The LLM emits structured output
  describing which tool to invoke. This is now built into all major
  model APIs (Groq supports it natively).
- **The thought-action-observation loop** (ReAct pattern). The
  foundational structure of every agent framework.
- **Termination conditions.** Unlike a pipeline, an agent has to decide
  when to stop. Forgetting this is the #1 way to set $$$ on fire.
- **Composability of skills.** Self-RAG (2.2) was a single critique
  step. Agentic RAG generalizes that to arbitrary multi-step planning.

### When to use it

- Multi-step questions that no single retrieval can answer.
- Multi-source systems (corpus + web + APIs + databases).
- Conversational copilots that need to take actions, not just answer.

### When to skip it

- Latency-sensitive (each loop iteration is an LLM call).
- High traffic — agentic loops are expensive at scale.
- Simple lookup-style Q&A.

### Effort estimate

- New module `pdfchat/agent.py` implementing the ReAct loop.
- Tool definitions for each capability.
- Termination logic + iteration cap.
- Substantial new prompting work.
- Eval needs entirely new metrics: `tool_use_correctness`,
  `iterations_per_query`, `cost_per_query`.

500+ lines, plus the eval extension. Multi-day project.

### How to measure it

Multi-hop dataset cases. Add questions like *"Which document discusses
both carbonara and arrabbiata?"* — single-shot fails, agentic succeeds
by issuing two queries. New metric: `multi_hop_success_rate`.

---

## 2.4 — Multimodal RAG

> **TL;DR:** Don't throw away images, charts, and tables. Use a vision-language model to embed visual content alongside text, so a query can match a chart in a PDF just as easily as a paragraph.

### The problem it solves

PDFs in the real world contain figures, plots, equations, scanned
tables. `pypdf` throws all of that away. Your bot can't answer "what
does Figure 4 show about latency?" because Figure 4 doesn't exist in
the index.

### The trick

At ingest time:
1. Extract images and tables as separate units alongside text.
2. Use a vision-language model (CLIP, ColPali, or a multimodal embedder)
   to produce a vector for each visual element.
3. Index everything in the same FAISS index (or sibling indexes that get
   fused at query time).
4. At generation time, send the matched images (or their descriptions)
   to a multimodal LLM (GPT-4o, Claude 3.5 Sonnet, Llama 3.2 Vision).

### Mental model

So far we've worked with one modality (text). Multimodal RAG generalizes
the same patterns — embed, store, retrieve, generate — to any modality
that has an embedder. The pipeline shape is identical; only the vectors
and the generator change.

### What's new conceptually

- **Cross-modal retrieval.** A text query landing near an image vector.
  Possible because CLIP-style models are trained to align text and
  images in a shared space.
- **The vision-language generator.** The LLM at the end has to accept
  images as input, not just text. Groq doesn't yet support image
  inputs at the time of writing — you'd need to switch generators.

### When to use it

- Slide decks, scientific papers, technical specs, financial filings —
  anywhere figures carry critical information.
- Image-heavy domains (medical imaging, satellite data, engineering
  drawings).

### When to skip it

- Pure text corpora (your test PDFs, news articles, books).
- When you can't afford the vision-LLM generation cost.

### Effort estimate

- New ingest path for image extraction (`pdfplumber`, `pdf2image`).
- New embedder (CLIP / ColPali).
- New generation path with a vision-capable LLM.
- Eval extensions for visual queries.

300–500 lines, plus model integrations. Medium project.

---

## 2.5 — GraphRAG (Microsoft, 2024)

> **TL;DR:** Extract entities and relationships from your corpus, build a knowledge graph, and retrieve **subgraphs** instead of chunks. Excellent for "connect-the-dots" questions like *"How is X related to Y?"*

### The problem it solves

Chunk-based retrieval can't answer *relational* questions well. *"Show
me everyone who collaborated with Vaswani."* Each collaborator might be
in a different paper. No single chunk contains all of them. Naive RAG
returns ONE chunk; the answer needs MANY, traversed through a
relationship graph.

### The trick

At ingest time:
1. Use an LLM to extract `(entity, relationship, entity)` triples from
   each chunk. ("Vaswani — coauthored — Shazeer".)
2. Build a knowledge graph from all triples.
3. Index both the graph and the original chunks.

At query time:
1. Match the query against the graph (find relevant entities).
2. Traverse the graph N hops outward to gather relevant subgraphs.
3. Send the subgraph + the source chunks to the LLM as context.

Microsoft's GraphRAG also adds a "community summarization" step that
clusters the graph and pre-summarizes each cluster.

### Mental model

This is **retrieval-at-the-level-of-structure** instead of retrieval-at-
the-level-of-text. Chunks are facts; graphs are *the facts plus their
relationships*. Some questions need the relationships.

### What's new conceptually

- **Entity extraction as a first-class ingest step.** Adds significant
  index-time cost and complexity, but unlocks query types that are
  otherwise impossible.
- **Hybrid graph + vector retrieval.** You search the graph for entity
  hits, then pull the corresponding chunks for grounded generation.
- **Pre-summarization at multiple resolutions** (chunk → entity →
  community → global). This is a new hierarchical pattern.

### When to use it

- Investigative research (who knows whom, who funded what).
- Intelligence / fraud / compliance (where relationships matter).
- Internal company knowledge bases where org structure / relationships
  carry meaning.

### When to skip it

- Most consumer Q&A applications.
- Small / loosely-connected corpora.
- When the cost of LLM-based entity extraction exceeds the value.

### Effort estimate

- Substantial. New entity extraction module, graph DB integration
  (NetworkX in-memory, Neo4j for scale), traversal logic, new prompting
  conventions.
- Best learned via Microsoft's open-source GraphRAG implementation
  before building your own.

Multi-day project minimum.

---

## 2.6 — Fine-tuning the embedder

> **TL;DR:** Train your bi-encoder on (query, chunk) pairs from *your* domain. Generic embedders (MiniLM, BGE) understand English; a fine-tuned one understands *your* documents.

### The problem it solves

Generic embedders are trained on internet English. For specialized
domains (medical, legal, internal company jargon, code), they don't
know which terms are synonyms in *your* context. "PCI" might mean
"percutaneous coronary intervention" (medical) or "Payment Card
Industry" (compliance) — the embedder has no idea which you mean.

### The trick

Collect ~1000+ pairs of (query, correct_chunk) from your data —
ideally with hard negatives (chunks that look similar but are wrong).
Train the embedder on a contrastive loss that pulls correct pairs
together and pushes negatives apart.

You don't need ML infrastructure — `sentence-transformers` has a
high-level training API. A few thousand pairs is enough for noticeable
improvement on most domains.

### Mental model

Everything else in this roadmap is about *using* an embedder smarter.
Fine-tuning is about *changing the embedder itself*. It's the heaviest
intervention; do it only after lighter techniques have plateaued.

### What's new conceptually

- **Training data curation as the bottleneck.** The model is the easy
  part — finding 1000+ high-quality query/chunk pairs is the hard part.
  Often done by LLM-assisted bootstrap (generate synthetic pairs, then
  filter by human review — exactly the `seed.py` pattern you've
  already used).
- **Cross-validation for embeddings.** Holdout pairs let you confirm
  the fine-tune actually helps and doesn't overfit.

### When to use it

- After contextual retrieval + hybrid + rerank stop moving the needle.
- Highly specialized domains where generic embedders consistently
  underperform.
- When you have enough corpus volume to make the training investment
  worthwhile.

### When to skip it

- General-purpose RAG (Wikipedia-flavored corpora). Generic embedders
  are good enough.
- Small projects where 1000+ training pairs is unrealistic.

### Effort estimate

- Data collection / generation: several days.
- Training: hours, on a single GPU.
- Integration: small (swap `EMBED_MODEL` in `.env`, re-ingest).
- Re-running the eval to confirm gains: easy.

Days of work, mostly on data, not code.

---

## 2.7 — Cache-Augmented Generation (CAG)

> **TL;DR:** If your whole corpus fits in the LLM's context window, skip retrieval entirely. Preload all documents into a cached prompt and let the LLM read them directly.

### The problem it solves

RAG is approximate by design — chunking discards structure, embeddings
discard precision, retrieval may miss the right chunk. For *small,
slow-changing* corpora, all that approximation is gratuitous.

Modern LLMs accept 200K+ token contexts. A book is ~80K tokens. A
small product manual might be 30K. **Why not just send the whole
thing?**

### The trick

Use prompt caching (supported by Anthropic, OpenAI, Groq for some
models) to load the entire corpus into the model's context *once*,
then reuse that cache across all queries. The per-query cost is just
the question itself.

```
   ┌─────────────────────────────────────────────────────┐
   │  CACHED (loaded once, reused):                      │
   │   - System prompt                                   │
   │   - Entire corpus (all PDFs, full text)             │
   └─────────────────────────────────────────────────────┘
                            │
                            ▼ (per question, just add:)
   ┌─────────────────────────────────────────────────────┐
   │   - User question                                   │
   └─────────────────────────────────────────────────────┘
                            │
                            ▼
                          answer
```

No chunking. No embedding. No FAISS. No reranking. No fusion. Just
"give the model everything and ask the question."

### Mental model

CAG is **the anti-RAG**. The premise of RAG is "we can't fit everything
into the context, so we must retrieve a subset." When that premise
isn't true, RAG is overhead. CAG is the corner case where the
constraint disappears.

### What's new conceptually

- **Prompt caching as architecture.** What used to be a cost
  optimization is now a structural decision.
- **The retrieval/context tradeoff has flipped** for some use cases.
  Long-context models have changed what "small" means.

### When to use it

- Single product manual, single policy doc, single short book.
- Compliance scenarios where you need *guaranteed* coverage (RAG
  might miss a relevant section; CAG cannot).
- Internal tools where the corpus is small and changes rarely.

### When to skip it

- Multi-document knowledge bases beyond ~100K tokens total.
- Frequently changing corpora (cache invalidation cost dominates).
- Cost-sensitive applications at scale.

### Effort estimate

- Almost trivial. Skip every Tier 0/1 component. Just read all PDFs
  into a single string and call the LLM with caching enabled.
- 30 lines of code.

The real work is *deciding* CAG is right for your use case.

---

## How to actually run this tier

Same loop you've already mastered:

```
   for each topic:
       1. read this roadmap section + the original paper if any
       2. implement
       3. write test cases that target the specific failure mode
       4. run baseline eval + new-technique eval
       5. compare metrics, decide if it's worth keeping on
       6. if yes, add to default config + commit
       7. if no, document the negative result (also valuable)
       8. write a docs/ note like the others
```

The negative results are *as valuable* as positive ones. They tell you
which fancy techniques don't apply to your corpus, saving you from
paying for them forever.

---

## Closing thought

Tier 2 looks intimidating from outside — papers, frameworks, vocabulary.
From inside, every technique is just a **variation** of things you've
already built:

- **Contextual Retrieval** is parent-child chunking + a per-chunk LLM
  call.
- **Self-RAG** is the system prompt for grounding, taken to its
  conclusion.
- **Agentic RAG** is query rewriting + retrieval, in a loop.
- **Multimodal RAG** is the same pipeline with different vectors.
- **GraphRAG** is metadata pointers, generalized to a graph.
- **Fine-tuning** is the embedder you already use, trained more.
- **CAG** is RAG with the retrieval step removed.

You have the mental tools. The roadmap exists. The eval harness is
ready. Pick a technique and run the loop.

→ Back to the [index](README.md).
