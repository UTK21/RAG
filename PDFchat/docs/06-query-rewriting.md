# 6. Query Rewriting — making the question search-friendly

> **TL;DR:** The query the user **types** is rarely the query you want to **search** with. Query rewriting is the family of techniques for transforming the input before retrieval — standalone form, longer form, multiple versions, a fake answer, sub-questions, or a step-back generalization — depending on which failure mode you're fixing.

## The core insight

A user's question and the chunk that answers it are usually written in **different styles**:

| User question (short, vague) | Answering chunk (long, explanatory) |
|---|---|
| "limitations?" | "Self-attention has O(n²) memory cost in sequence length, which becomes prohibitive for long contexts. Several approximations have been proposed..." |
| "how cheap?" | "Pricing starts at $0.05 per million input tokens, dropping to $0.03 for committed-use customers..." |
| "Vaswani's idea?" | "The 2017 paper 'Attention Is All You Need' introduced a model relying entirely on attention mechanisms..." |

The embedding model never knows the user meant something specific.
BM25 has nothing to keyword-match against in a one-word query.

**Query rewriting closes that gap by re-expressing the question in a form that's more like the text it should match.**

It's the same trick chess engines use: they don't think about positions, they think about positions *as the next move would see them*. We don't search with the question, we search with **what the question would look like once it's been said properly**.

---

## A quick taxonomy

```
                        Query Rewriting
                              │
   ┌──────────────────┬───────┴────────┬─────────────────────┐
   │                  │                │                     │
   ▼                  ▼                ▼                     ▼
TRANSFORM the      EXPAND into       DECOMPOSE into       ROUTE to the
single query       MULTIPLE queries   sub-queries          right source

• standalone       • multi-query     • decomposition      • query routing
  rewriting          expansion         (sub-questions)
• step-back        • HyDE
  prompting        • RAG-Fusion
```

| # | Variant | One-line | Best for |
|---|---|---|---|
| A | Standalone rewriting | Rewrite follow-ups to be self-contained | Multi-turn chat |
| B | Multi-query expansion | Generate N variants, retrieve for each | Vague / ambiguous queries |
| C | HyDE | Generate a fake answer, search with it | Question/answer style mismatch |
| D | Decomposition | Split a complex question into sub-questions | Multi-part, comparative, or analytical queries |
| E | Step-back prompting | Generalize the question first, then specialize | Very specific facts that need background |
| F | RAG-Fusion | Multi-query + RRF (the merging of #4 and #B) | When you want max recall without picking variants |
| G | Query routing | Pick *which* index to search | Multi-source systems |

These can be **combined**. Production agents often do: route → decompose → for each sub-query do (standalone rewrite + multi-query + HyDE).

---

## A) Standalone-question rewriting (what we already built)

**Problem.** "What about its limitations?" has no useful signal — embeddings can't read your mind.

**Fix.** Rewrite using chat history into a self-contained question.

```
   prior chat                          new query
       │                                   │
       └──────────► LLM (rewrite) ◄────────┘
                        │
                        ▼
                  standalone query  ──► retrieval
```

**Example.**

```
History:
  user: What does the paper say about transformers?
  bot:  They use self-attention (Vaswani et al. 2017, p. 3).

User says: "What about its limitations?"

LLM rewrite: "What are the limitations of transformers?"
```

**Code sketch.** See `query_rewriter.py`. Two tiny tricks:
1. Tell the model to return the question UNCHANGED if it's already standalone — prevents "improving" already-fine queries.
2. If history is empty (first message), skip the LLM call entirely — saves a round trip.

**Caveat.** Use temperature = 0 for the rewrite. We want it faithful to the original intent, not creative.

---

## B) Multi-query expansion

**Problem.** A single query can be phrased many ways. Embedding only one phrasing means missing chunks that happen to match a different phrasing.

**Fix.** Ask the LLM for N rephrasings, retrieve for each, dedupe (or fuse).

```
                 "AI safety"
                      │
                      ▼
                ┌──────────┐
                │   LLM    │ "Give me 3 alternative phrasings"
                └─────┬────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   "AI alignment"  "ML risks"  "responsible AI"
        │             │             │
        ▼             ▼             ▼
   retrieval     retrieval     retrieval
        │             │             │
        └─────────────┼─────────────┘
                      ▼
            merged candidate pool
```

**Example.**

```
Original: "AI safety"

LLM generates:
  1. "AI alignment and value learning"
  2. "Machine learning model robustness and reliability"
  3. "Responsible AI deployment and governance"
```

Each one is a separate retrieval. Their unioned results are way broader than the original.

**Code sketch.**

```python
def multi_query(client, model, query, n=3):
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content":
                f"Generate {n} alternative phrasings of the user's question "
                f"that capture different angles/keywords. One per line."},
            {"role": "user", "content": query},
        ],
        temperature=0.7,   # ← higher temperature for VARIETY this time
    )
    return [q.strip("-* 0123456789.") for q in resp.choices[0].message.content.splitlines() if q.strip()]
```

Note the temperature flip: standalone rewriting wants T=0 (faithful), multi-query wants T≈0.7 (diverse).

**When it shines.** Vague queries, single-word queries, exploratory questions. Less useful when the user is already specific.

**Cost.** N× retrieval cost. Often paired with re-ranking or RRF so the LLM only ever sees the best dedup'd chunks.

---

## C) HyDE — Hypothetical Document Embeddings

This is the slickest trick in the toolkit. **Read this section even if you skip the others.**

### The asymmetry that HyDE exploits

Questions look like questions. Answers look like answers. Their embeddings often live in different regions of vector space:

```
   region: questions       region: answers (the actual chunks)
       ●  "what are X?"         ●●●●●  long explanatory text
       ●  "explain Y"           ●●●●●  full paragraphs
       ●  "tell me about Z"     ●●●●●  with examples and figures
                                         ▲
                                         │
                          this is what we want to match
```

A query embedding gets dragged toward "question text" rather than "answer text". The match works, but it's noisy.

### The fix

Have the LLM **write a fake answer** to the question. Embed that fake answer. Search with it.

```
   query   ──►  LLM hallucinates an answer  ──►  embed THAT  ──►  search
```

The fake answer doesn't have to be correct! It just has to **look like the kind of paragraph that would contain the real answer**. Because it does, its vector lands in the answer region of space, right where the real answer chunks live.

```
                   ●  query
                                  ●  fake answer
                                              ●●●●  real answer chunks
                                          ▲
                                          │ search from HERE
                                          │ matches real answers easily
```

### Example

```
Query: "What is the moon made of?"

LLM hallucinates an "answer":
  "The Moon is primarily composed of silicate rocks, with a crust rich in
   anorthosite, mantle material similar to Earth's, and a small iron core.
   Surface samples returned by the Apollo missions revealed..."

We DON'T return this to the user.
We embed it and search FAISS with that embedding.
The real chunk in the textbook lights up because its text *looks like* this.
```

### Code sketch

```python
def hyde(client, model, embedder, index, chunks, query, k=10):
    # 1. Generate a plausible (possibly hallucinated) answer
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content":
                "Write a paragraph that answers the user's question. "
                "Make it look like an excerpt from a textbook or article. "
                "Length: 3-5 sentences."},
            {"role": "user", "content": query},
        ],
        temperature=0.5,
    )
    fake_answer = resp.choices[0].message.content

    # 2. Embed the FAKE ANSWER (not the question)
    vec = embedder.encode([fake_answer], normalize_embeddings=True).astype("float32")

    # 3. Search FAISS with it
    _, idx = index.search(vec, k)
    return [chunks[i] for i in idx[0]]
```

### Why this works even when the LLM is wrong

The model doesn't have to know the answer. It just needs to know **how the answer would be written**. Even if the fake answer is factually wrong, its *style* matches real answer chunks. Retrieval is style-sensitive at the embedding level, not fact-sensitive.

### When it shines

- Q/A over technical or specialized corpora where queries are short and chunks are long.
- Domains the embedding model wasn't well-trained on — HyDE's fake answer pulls the search vector into the domain's vocabulary.
- Cold-start RAG where you haven't tuned embeddings yet.

### Caveats

- Adds an LLM call per query.
- If the model hallucinates very *wrongly* (mentions the wrong subject), the fake answer drags retrieval off-topic. Mitigation: low temperature and require it stick close to the query's topic.
- Doesn't help if the corpus itself uses question-like phrasing (FAQs).

### When HyDE actively makes retrieval WORSE

The whole pitch rests on one assumption: *the fake answer's vocabulary and structure match the real chunk's*. When that assumption breaks, HyDE doesn't just fail to help — it **destroys signal** the user typed and makes search worse than the plain query baseline.

**Four failure modes:**

1. **Wrong subject.** Query "tell me about Mercury" → LLM writes about the planet, real chunk is about the Roman god. The plain query would have landed *between* the two regions of vector space; HyDE confidently lands in the wrong one.
2. **Vocabulary divergence.** Query "treatment for MI" → LLM uses "heart attack", real chunks use "myocardial infarction". A general-purpose embedder sees those as only kind-of-similar; HyDE's vector misses the clinical region.
3. **Custom/internal jargon.** Query "how does PROJECT-NEPTUNE refresh tokens?" → LLM has never heard of `PROJECT-NEPTUNE` and writes generic OAuth text. The single most informative token in the query is **deleted from the search vector**.
4. **Wrong style/register.** Query "close a stream in Java?" → LLM writes prose, real chunk is a code block. Wrong region of embedding space.

**Why this matters:** in failure modes (1), (3), and (4), searching with the plain query would have done better. HyDE doesn't fail gracefully — it fails *confidently*.

### How production systems make HyDE safe

HyDE almost never runs alone. Layered defenses:

1. **Q + H, not H alone.** The original paper's actual recipe: search with both the query AND the hypothetical answer (average the embeddings, or do two searches and merge). Guarantees the user's literal tokens — including jargon — stay in the search vector. **You can't do worse than baseline.**
2. **Multiple hypotheses.** Generate N=8 fake answers at T=0.7, average their embeddings. One bad sample gets diluted by seven decent ones.
3. **Hybrid search saves you.** BM25 on the *original* query catches exact tokens (`PROJECT-NEPTUNE`, `MI`, code symbols) that HyDE deleted. RRF fuses BM25 + HyDE-dense. This is why hybrid search is built BEFORE HyDE in this repo.
4. **Cross-encoder rerank uses the ORIGINAL query.** Even if HyDE retrieved 20 wrong candidates, if the right chunk made it into the pool via BM25 or query-only dense, the reranker (scoring `(original_query, chunk)`) promotes it. The reranker is the safety net.
5. **Score-based fallback.** If the HyDE retrieval's max similarity is suspiciously low, skip it for this question and use query-only.

### Rule of thumb

| Corpus | HyDE alone? | HyDE with safety net? |
|---|---|---|
| Well-trodden general knowledge (Wikipedia, news, popular science) | Usually fine | Always fine |
| Specialized but well-known domain (medicine, law) | Risky — vocab gaps | Fine with Q+H + hybrid |
| Internal docs / custom jargon / niche product | **Avoid** | Marginal, needs heavy reranking |
| Code, tables, structured data | **Avoid** | Use sparse-dominant retrieval instead |

The mental model: HyDE is a **bet** that the LLM knows the domain well enough to write stylistically-close answers. Win or lose depends on the corpus, not the technique.

---

## D) Query decomposition (sub-questions)

**Problem.** A single complex query has multiple intents. Retrieval can only find chunks that match *all* of them at once — usually nothing.

**Fix.** Have the LLM split the query into smaller sub-questions, retrieve for each, then either merge results or have the LLM compose a final answer from the per-question contexts.

```
   "Compare transformers and RNNs for long sequences"
                       │
                       ▼
                 ┌──────────┐
                 │   LLM    │  decompose
                 └─────┬────┘
                       │
       ┌───────────────┼────────────────┐
       ▼               ▼                ▼
"What are        "What are RNNs?"  "What are the long-seq
 transformers?"                     limits of each?"
       │               │                │
       ▼               ▼                ▼
   retrieve        retrieve         retrieve
       │               │                │
       └───────────────┼────────────────┘
                       ▼
            LLM composes final answer
```

**When it shines.**
- Comparative questions ("compare X and Y").
- Multi-hop questions ("what year did the founder of X's previous company graduate?").
- Analytical questions where the answer requires synthesizing multiple facts.

**Caveats.**
- N× retrieval cost.
- Errors compound: if any sub-question retrieves poorly, the final synthesis suffers.
- Decomposition is itself a skill — small/cheap LLMs can split badly.

---

## E) Step-back prompting

**The idea.** For very specific factual questions, a step-back to a more *general* question retrieves background that helps answer the specific one.

This came from a 2024 Google DeepMind paper ("Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models"). Surprisingly effective.

**Example.**

```
Specific: "What was Estella Leopold's school in Aug-Nov 1954?"

Step back: "What is Estella Leopold's education history?"

Now retrieve for BOTH and feed both contexts to the LLM.
The specific question retrieves the exact fact; the broad question
retrieves the surrounding biographical paragraph that often contains
it AND its relevant context.
```

**Pattern.**

```
   specific query                 step-back query
        │                              │
        ▼                              ▼
    retrieve A                     retrieve B
        │                              │
        └──────────────┬───────────────┘
                       ▼
                  merge contexts
                       │
                       ▼
                      LLM
```

**When it shines.** Fact-finding in encyclopedic or biographical content. Domains where context matters for interpreting a fact.

**Caveat.** Two retrievals per question. Sometimes the step-back is too broad and adds noise.

---

## F) RAG-Fusion

**One sentence.** Multi-query expansion (B) + Reciprocal Rank Fusion (the same RRF from hybrid search, applied to the rankings of the rephrasings).

```
   query
     │
     ▼
   LLM ──► [variant 1, variant 2, variant 3]
                │           │           │
                ▼           ▼           ▼
            retrieve    retrieve    retrieve
                │           │           │
                └─────┬─────┴───────────┘
                      ▼
                RRF fusion
                      │
                      ▼
              merged ranking
```

**Why it's nice.** You get the recall benefit of multi-query without picking arbitrarily between variants. Chunks that consistently rank well across MULTIPLE phrasings of the question float to the top — robust signal of relevance.

**Caveat.** Same N× cost as multi-query. Pair with a re-ranker since RRF can over-rank common chunks.

---

## G) Query routing

**The idea.** In a multi-source system, not every query should hit every source. Use an LLM (or a small classifier) to route the query to the right index.

**Example.**

```
   user query
       │
       ▼
   router (LLM with a short prompt)
       │
       │  decision tree:
       │  - "Is this a code question?"          → search the code index
       │  - "Is this a billing question?"       → search the policy index
       │  - "Is this a how-to question?"        → search the tutorial index
       │  - "Otherwise"                         → search the general index
       │
       ▼
   index_X
       │
       ▼
   retrieval → rerank → LLM
```

**When it shines.** Multi-corpus systems (code + docs + tickets + email). Cheaper than searching every source.

**Caveats.** Routing errors are hard to debug — if the router picks the wrong index, retrieval can't recover.

---

## Which technique when?

| Symptom you're seeing | Try... |
|---|---|
| Follow-up questions retrieve garbage | **A. Standalone rewriting** |
| Single-word or vague queries retrieve unrelated chunks | **B. Multi-query** or **F. RAG-Fusion** |
| Short Q vs long expository chunks; semantic mismatch | **C. HyDE** |
| "Compare X and Y" / multi-hop questions | **D. Decomposition** |
| Very specific factual questions over biographical/encyclopedic data | **E. Step-back** |
| Multi-corpus / multi-tool system | **G. Routing** |

In a serious production stack, you'd combine: **routing → decomposition → standalone rewrite per sub-question → optional HyDE per sub-query → retrieve → fuse → rerank**. Heavy, but maximally robust.

---

## Implementation pattern (generic interface)

All seven variants can be expressed as `query → list[query]`:

```python
class QueryRewriter:
    def rewrite(self, query: str, history: list[dict] | None = None) -> list[str]:
        """Return one or more queries to actually search with."""
        ...
```

- Standalone rewriting → returns `[standalone_query]`
- Multi-query / decomposition / step-back / RAG-Fusion → returns `[q1, q2, q3, ...]`
- HyDE → returns `[hypothetical_answer_text]`
- Routing → orthogonal: it picks an *index*, not a query

This makes them composable. A pipeline might call `decomposer.rewrite()`, then for each sub-query call `hyde.rewrite()`, then run all the resulting queries against the index, then fuse with RRF.

---

## Common pitfalls

| Pitfall | Notes |
|---|---|
| Rewrite hallucinations | The LLM "helpfully" changes the meaning. Mitigation: T=0 for faithful rewrites, T>0 only for variety-seeking ones. Always tell the model: "if the question is already standalone, output it unchanged." |
| Latency stacking | Each rewrite is an LLM call. Multi-query + HyDE could mean 4 LLM calls before a single retrieval. Use small/fast models (e.g. Llama 3.1 8B on Groq) for rewrite steps. |
| Cost stacking | Same problem with API spend. Cache rewrites for repeat queries. |
| Cascading errors | A bad decomposition or a bad route can poison everything downstream. Add fallbacks: "if route is uncertain, search ALL indexes." |
| Forgetting the original | Use the rewrite for **retrieval** but the original (plus history) for **generation**. The user's actual phrasing matters for tone and intent. |
| "Improving" already-good queries | If the user typed a precise query, don't let the rewriter rephrase it into something worse. Build in a "no-op" path. |

---

## Key concepts this teaches

- **LLMs as text transformers**, not just generators. Half of modern RAG is "use an LLM to massage text before searching."
- **The question/answer style asymmetry** — once you see HyDE work, you understand why short-query vs long-chunk is a real problem you should design around.
- **Rewriting as a fan-out**, retrieval as a merge — the same fan-out/merge pattern shows up in agentic RAG, tool use, multi-step reasoning, and tree-search agents.
- **Composability** — every variant is `query → list[query]`. That uniform interface is the same one used internally by LangChain's query transformers, LlamaIndex's `QueryTransform`, and most agent frameworks.
- **The temperature dial** — faithful rewrites use T=0; diverse expansions use T≈0.7. Same model, opposite goals.

→ Back to the [index](README.md).
