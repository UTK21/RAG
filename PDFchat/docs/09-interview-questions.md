# 9. RAG Interview Questions — practice set with model answers

> A self-test bank of 23 RAG interview-style questions across foundations,
> design, debugging, system design, evaluation, and behavioral storytelling.
> Each question is followed by a model "strong candidate" answer — what an
> interviewer would consider hire-signal.
>
> **How to use this doc:**
> - First time through: try to answer each question yourself BEFORE peeking
>   at the answer below it. Cover the screen or scroll one question at a time.
> - Second time: re-read the model answers as reference material.
> - For maximum value: rehearse the behavioral stories (Q21–Q23) out loud
>   on a timer until they land in ~80 seconds each.

---

# 📚 Category 1 — Foundations

### Q1. What is the difference between a bi-encoder and a cross-encoder, and where does each fit in a RAG pipeline?

**Strong answer.**
A **bi-encoder** encodes the query and chunk **separately** into vectors, then computes their similarity (dot product or cosine). Because chunks can be encoded once and reused, it scales to millions of chunks but its scoring is rough — query and chunk never "see" each other inside the model. Used in stage 1 (recall) for FAISS dense retrieval.

A **cross-encoder** concatenates query and chunk into ONE input and runs them through the model together. Full attention between them, so scoring is much more accurate — but you can't precompute, you have to re-run for every (query, chunk) pair. Used in stage 2 (precision) to rerank a small set of candidates.

The standard cascade: bi-encoder narrows millions → 20, cross-encoder narrows 20 → 4. Speed and accuracy without compromising either.

---

### Q2. In one sentence: why do we normalize vectors before inserting into FAISS with `IndexFlatIP`?

**Strong answer.**
Cosine similarity equals dot product **only when both vectors are unit-length**. FAISS's `IndexFlatIP` computes inner products. If we normalize vectors at index and query time, the inner product *becomes* exact cosine similarity — no extra math, no performance cost.

---

### Q3. You're chunking a 100-page PDF. Your colleague says "let's use 50-word chunks for precise retrieval." What's your response?

**Strong answer.**
They're half right. Smaller chunks do produce more focused vectors. But a 50-word chunk loses context — you'd retrieve "and therefore the rate increased to 4.5%" without knowing what rate or when. You also balloon the index 4–5× and add retrieval noise.

The production answer is **parent-child**: embed small chunks (~200 words) for precise *matching*, but send their larger parent (~1000 words) to the LLM for rich *context*. Small for retrieval, big for generation. Best of both, achieved with a single `parent_idx` metadata pointer.

---

# 🔍 Category 2 — Retrieval design

### Q4. Explain Reciprocal Rank Fusion in your own words. Why can't we just normalize and add scores?

**Strong answer.**
We have multiple ranked lists from different retrievers (dense, sparse, HyDE, etc.). Their **raw scores live on different scales** — cosine is bounded `[-1, 1]`, BM25 is unbounded. Normalizing per-corpus, per-query is endless tuning.

RRF throws away the scores entirely and uses only the **rank position**:
```
fused_score(doc) = Σ over each list of  1 / (60 + rank_in_that_list)
```

Three properties make it work in practice: it's **scale-free** (only ranks matter), it **rewards agreement** (a doc in multiple lists stacks score), and the `k=60` constant **dampens** so rank 1 doesn't dominate everything. Used by Elasticsearch, Vespa, Weaviate.

---

### Q5. What's the difference between dense and sparse retrieval? Give one concrete example of a query that each handles better than the other.

**Strong answer.**

| Retriever | Strength | Failure mode |
|---|---|---|
| Dense | Captures **meaning** — "vehicle" matches "car" | Rare exact tokens get smoothed away ("Vaswani" → "academic word") |
| Sparse (BM25) | Exact tokens — "Vaswani", `AX-9281`, product codes light up instantly | Synonyms — "car" ≠ "automobile" to BM25 |

Two concrete examples:
- Query `"treatment for MI"` — sparse retrieval nails `"MI"` if the corpus literally uses that token. Dense, trained on general English, doesn't strongly associate "MI" with "myocardial infarction".
- Query `"treatment for heart attack"` over a corpus that uses `"myocardial infarction"` — dense bridges, sparse fails.

That's why hybrid (both, then RRF-fused) wins consistently — each retriever's failure mode is the other's strength.

---

### Q6. Walk me through what happens, step by step, when a user asks a follow-up question like "what about its limitations?" in your RAG system.

**Strong answer.**
1. User types `"what about its limitations?"`
2. We take `history[-N:]` from the chat memory.
3. `query_rewrite.standalone(history, query)` calls a small LLM (e.g. llama-3.1-8b-instant) to produce a self-contained version: `"What are the limitations of transformers?"`
4. Dense retrieval (FAISS) on standalone query → top 20 children.
5. Sparse retrieval (BM25) on standalone query → top 20 children.
6. RRF fuses both ranked lists → ~30 unique candidates.
7. Cross-encoder reranks all candidates against the **standalone query** → top 4 children.
8. `children_to_parents()` dedupes — multiple children of the same parent become one parent in best-first order.
9. LLM generates the answer using the **original query** + recent history + parent chunks. (Crucially, we use the original for generation, not the rewrite — the rewrite was only for retrieval.)
10. Stream tokens to user; append `(user, assistant)` to history.

The split — rewrite for retrieval, original for generation — is the defensive pattern that saved my eval when the rewriter misbehaved.

---

### Q7. Your friend says "I'm going to add HyDE to my RAG to improve retrieval." What three questions would you ask before agreeing it's a good idea?

**Strong answer.**
1. **Does the LLM know my domain?** If the corpus uses custom jargon (internal project names, niche acronyms), HyDE will generate a fake answer in *generic* vocabulary and miss the actual chunks. The rare-token signal in the query gets deleted from the search vector.
2. **What's my baseline retrieval recall?** If it's already 0.95+, HyDE has nowhere to improve. Adding an extra LLM call for no gain is pure tax.
3. **Do I have hybrid + rerank already?** They're cheap safety nets that catch HyDE's failure modes. With both in place, HyDE can only add signal (Q+H pattern). Without them, HyDE alone is risky.

Bonus: "How am I going to measure whether it helped?" If you can't A/B it, don't add it.

---

# ⚖️ Category 3 — Trade-offs

### Q8. You're given a 50-page company policy document and a chatbot needs to answer questions about it. Build a decision tree: when would you choose RAG vs CAG vs fine-tuning? Defend each branch.

**Strong answer.**
50 pages ≈ 30K tokens, comfortably inside modern context windows. So:

```
   Is the doc static and small enough to fit in context?
       │
       └── YES → CAG (Cache-Augmented Generation)
                  Preload the whole doc into a cached system prompt.
                  Pros: zero retrieval miss, simple, low cost per query
                  with prompt caching.
                  Use this unless one of the below applies.
       │
       └── NO, it changes weekly → RAG
                  Cache invalidation makes CAG painful. Hybrid + rerank
                  pipeline gives flexibility for frequent updates.
       │
       └── Do answers need a specific TONE / FORMAT?
              YES → Fine-tune (on top of RAG/CAG, not instead of)
              NO  → don't bother

   ❌ Fine-tuning ALONE is wrong here.
      You'd bake facts into model weights — brittle when the policy
      updates and you'd have to retrain just to change one number.
```

For most static 50-page policy docs in 2026, **CAG with prompt caching is the right answer.** RAG is over-engineering when the corpus fits.

---

### Q9. A junior engineer on your team wants to use `child_size=50` because "smaller chunks = more precise retrieval." Where is their thinking right? Where is it wrong?

**Strong answer.**

**Right:** smaller chunks produce more focused vectors. A 50-word chunk's vector represents one specific idea more cleanly than a paragraph's.

**Wrong (three angles):**
1. **Context loss at the chunk level.** A 50-word chunk often can't stand alone — you'd retrieve "and therefore the rate increased to 4.5%" with no anchor.
2. **Index bloat.** 4–5× more vectors per document → bigger index, slower searches, more retrieval noise.
3. **Generation starvation.** Even if retrieval finds the right chunk, the LLM gets too little surrounding info to write a good answer.

**The synthesis:** they want precision-of-matching. The way to get it without sacrificing generation context is **parent-child chunking** — embed small for retrieval, send large for generation. ~200-word children inside ~1000-word parents, linked by `parent_idx`.

---

### Q10. Your RAG bot is slow — average response time is 8 seconds. You have three knobs: (a) reduce `retrieve_k`, (b) drop the cross-encoder, (c) switch to a smaller LLM. In what order would you investigate, and why?

**Strong answer.**
First: **find where the time is going.** Add timing logs per stage. Don't tune blind.

Most likely culprit is the cross-encoder rerank — it runs N forward passes sequentially on CPU. With `retrieve_k=20`, that's often 60–70% of latency.

Investigation order:
1. **Profile** (free, 5 min). Confirm where the time actually goes.
2. **Reduce `retrieve_k` 20 → 10.** Halves reranker load. Run eval to confirm quality didn't drop. Cheapest win.
3. **Smaller cross-encoder** (`ms-marco-MiniLM-L-6-v2`, ~90MB vs BGE's 1.1GB). 5× speedup, modest quality dip.
4. **Smaller LLM only after measuring.** A 70B → 8B switch usually tanks the answer model's instruction-following. Verify with the eval.
5. **Drop the cross-encoder entirely** is the nuclear option. Big quality hit. Only if the cost/latency budget demands it.

Critical principle: never change the model until you've tried the cheap knobs and **measured** the trade-off. Most "RAG is slow" complaints get fixed at `retrieve_k`.

---

# 🐛 Category 4 — Debugging scenarios

### Q11. Your bot is hallucinating — confidently giving wrong answers that aren't in the source PDFs. List the diagnostic steps in order, from cheapest to most expensive.

**Strong answer.**
1. **Read the system prompt.** Does it have the grounding rule? "Use ONLY the provided context. Say I don't know if it's not there." Often the prompt was edited and the rule got dropped.
2. **Inspect retrieved chunks** for one hallucinated answer (use the `RetrievalTrace`). Did the right chunks make it in?
3. **If chunks are wrong** → retrieval problem. Look at chunking, embedding model, hybrid weights.
4. **If chunks are right, answer is still wrong** → generation problem. Lower temperature to 0.0–0.1; check that the prompt format clearly delimits context vs question.
5. **Check for prompt injection** in retrieved content — does a PDF contain `"ignore previous instructions"`? Indirect injection is real.
6. **Try a stronger answer model.** If hallucination only happens on the 8B and not the 70B, the prompt is too weak.
7. **Add a Self-RAG critique step.** Expensive (extra LLM call per query). Worth it for high-stakes domains.

Going model-first is the most common mistake. Almost always cheaper to fix the prompt or chunks.

---

### Q12. The eval harness shows `retrieval_recall = 0.95` (great) but `citation_match = 0.20` (terrible). What's likely broken? How would you verify?

**Strong answer.**
Diagnosis: **retrieval is fine, citation is broken.** Decoupling these two metrics is exactly why per-stage eval matters.

Likely causes, in order of probability:
1. **Citation regex too strict.** This was issue 02 in my project — I expected `(doc.pdf p. N)` but the model emitted `doc.pdf (p. N)`. False negatives at the metric level look identical to real failures.
2. **Model isn't citing at all.** The system prompt's citation rule is too soft, or the model is dropping citations under temperature.
3. **Model is citing in a format the regex doesn't recognize** — bullet lists, footnotes, etc.

How to verify (5 minutes):
- Print 2-3 actual bot answers, look at them.
- Manually run the regex against a known good answer to see if it matches.
- If absent → strengthen prompt ("EVERY claim must include `(doc.pdf p. N)`").
- If present but different format → loosen the regex.

---

### Q13. You added HyDE and the aggregate metrics didn't move. Three possible explanations and how you'd distinguish between them.

**Strong answer.**
1. **Retrieval recall was already at ceiling.** If `retrieval_recall` was 1.00 baseline, HyDE has nothing to improve.
2. **Wrong bottleneck.** The failing metrics might be generation-side (citation, keyword). HyDE only touches retrieval — no amount of better retrieval helps a citation regex bug.
3. **Corpus too small.** If your index has so few chunks that retrieval returns the same set regardless of query phrasing, HyDE can't change the candidate pool.

How to distinguish:
- Check baseline `retrieval_recall`. At ceiling? → cause #1.
- Check which metrics are failing. Generation-side? → cause #2.
- Count chunks per query. Always the same set? → cause #3.

This is exactly what I observed on my own toy corpus — 2 chunks total, retrieval always returned both, HyDE was decorative.

---

### Q14. Your bot worked fine yesterday. Today it's giving stale/wrong answers. The code hasn't changed. What's your first hypothesis, and how do you check?

**Strong answer.**
First hypothesis: **stale index.** Someone updated a PDF in `data/` but didn't re-run `ingest.py`. The bot is searching old embeddings against new content. Confidently wrong because retrieval is "working" against the wrong data.

Verification (5 seconds): compare `indexes/manifest.json::doc_hashes` against current hashes in `data/`. My system actually enforces this — `chat.py` refuses to start with a stale index, exactly to prevent this footgun.

Second hypothesis: **upstream change**. Groq deprecated the model, rate-limited the key, changed an endpoint. Check logs for API errors.

Third hypothesis: **env mutation**. Someone edited `.env` or upgraded a dep that changed embedding behavior.

Order to check:
1. Manifest freshness (instant)
2. API error logs (instant)
3. Env / dep diff (`git diff` on `requirements.txt` and `.env`)

Code is the *last* place I'd look when "no code changed" — by definition, that's not where it broke.

---

# 🏗️ Category 5 — System design

### Q15. Design a RAG system for 10 million documents. What changes vs your current design? Walk through retrieval, storage, eval, and cost.

**Strong answer.**
Multiple things change at once. Walking through layer by layer:

**Storage / index.** FAISS in-memory is out — 10M × 384-dim floats is ~15GB just for vectors. Options:
- **HNSW with PQ (product quantization)** — fits in memory, fast lookup, small quality loss.
- **DiskANN** — disk-backed, scales further, slightly higher latency.
- **Managed vector DBs** — Pinecone (serverless), Qdrant, Weaviate, Milvus. Most teams pick this in 2026.

**Indexing pipeline.** Can't re-embed 10M docs on every change. Need an incremental ingest stream — delta detection, batched embedding, atomic index updates. Schedule full rebuilds quarterly for cleanup.

**Retrieval.**
- Must use ANN (approximate). `IndexFlatIP` is exact and O(N) — death at 10M.
- **Metadata pre-filtering is critical.** Filter by `tenant_id`, `doc_type`, `date_range` BEFORE the similarity search, not after. Vector DBs all support this.
- Bi-encoder pulls 200 candidates → cross-encoder narrows to 10.
- Hybrid is harder at scale: either use a system that supports it natively (Elasticsearch + dense, OpenSearch, Weaviate), or run two separate retrievals + RRF.

**Eval.** Move beyond a hand-curated golden set:
- **Golden set** (50–200 hand-curated, regression gate).
- **Distributional eval** sampled from real production traffic.
- Drift monitoring — daily run, dashboard, page on anomalies.

**Cost / operational.**
- Shard the index by domain or tenant to reduce per-query embed cost.
- Semantic cache for repeat queries (huge cost saver in production).
- Smaller embedder if quality permits; quantize to int8.
- Index versioning, blue/green deploys, rollback plan.

---

### Q16. Design a multi-tenant RAG SaaS. Customer A and Customer B both upload PDFs. Neither should ever see the other's content. How do you guarantee isolation at the retrieval layer? What goes wrong if you get it wrong?

**Strong answer.**
**The non-negotiable insight:** tenant isolation lives at the **retrieval layer**, not at the application layer. Filtering responses after retrieval is too late — the chunks have already crossed the boundary.

Implementation:
- Every chunk has `tenant_id` in its metadata.
- Every retrieval is **pre-filtered** by `tenant_id` BEFORE the similarity search.
- Production vector DBs support this natively — Pinecone *namespaces*, Qdrant *collections*, Weaviate *filters*.

**Defense in depth:**
1. `tenant_id` on every chunk.
2. `tenant_id` baked into the auth token, validated at every API call.
3. Separate namespaces/collections per tenant.
4. Conversation history kept tenant-scoped; never cross-pollinate.
5. Adversarial test cases: deliberately try queries from tenant A that mention tenant B's name. Should retrieve nothing from B.
6. Audit log: every retrieval logged with `(user, tenant_id, retrieved_chunk_ids)` for forensics.

**What goes wrong if you get it wrong:** customer A's question pulls customer B's confidential chunks → catastrophic data breach. Single largest risk in multi-tenant LLM systems. Companies have died from this.

---

### Q17. Your company wants to "add RAG to our app" — that's the entire brief. What clarifying questions do you ask before writing any code? Aim for at least 6.

**Strong answer.**
Whatever you don't ask now will bite later. Here are mine, ranked by typical impact:

1. **What problem are we solving?** Search, FAQ, synthesis, analysis — different designs.
2. **What's the data?** Format, size, sensitivity (PII?), change frequency, source of truth.
3. **Who uses it?** Internal? External? Anonymous? Multi-tenant?
4. **What's the quality bar?** "Some hallucinations are OK" vs "must cite sources" → totally different stack.
5. **Latency budget?** Interactive (<2s) vs batch (minutes OK) → wildly different architectures.
6. **Cost budget?** Per-query, monthly, infra.
7. **How do we measure success?** From day one we need an answer; otherwise we're vibing.
8. **Deployment target?** Cloud (which?), on-prem, regulatory restrictions?
9. **Maintenance owner?** Who updates the corpus, who responds to bad answers?
10. **Timeline?** Two weeks vs six months → totally different scope.

Bonus: "Why do you think RAG is the right answer here?" Sometimes the real answer is fine-tuning, sometimes CAG, sometimes a simple FAQ search would do.

---

# 📊 Category 6 — Evaluation discipline

### Q18. You change one thing — bigger chunks. The aggregate eval score *drops* by 3%. Should you revert? Defend your answer.

**Strong answer.**
Maybe. Aggregates hide direction.

First: look at the **per-metric breakdown.** Did `retrieval_recall` drop (more context = noisier retrieval) but `keyword_coverage` rise (more context = richer answer)? That's a real trade-off, not a regression.

Second: **what's the noise floor?** With 8 questions, 3% is ~0.24 questions — within noise. With 100 questions, 3% is meaningful.

Third: **which metric matters most for the use case?** If you're a high-citation product, prioritize `citation_match`. If completeness matters, lean on `keyword_coverage`.

Decision: revert if the failing metric matters most and the gain doesn't compensate. Keep if the swap aligns with use-case priorities and the loss is within noise.

The right answer is almost never "revert because aggregate dropped." It's "tell me what swapped, and decide whether the swap was a net win for *us*."

---

### Q19. Your eval suite has 8 questions and is passing 100%. Your manager says "great, ship it." What's the honest response?

**Strong answer.**
Honest response: "100% on 8 questions is **suspicious**, not validating."

Concerns:
- Statistically insignificant. 8 questions, 100% pass — tells us almost nothing about deployment behavior.
- Likely cherry-picked. If we wrote them to be passable, of course they pass.
- We have no signal on the questions we *didn't* write.

Push for before shipping:
- Expand dataset to **30–50 questions** with diverse failure modes:
  - Out-of-corpus (bot must say "I don't know")
  - Multi-document conflicts (bot must surface both)
  - Negation (does X include Y? answer no)
  - Adversarial paraphrasing
  - Long-tail rare-vocabulary queries
- Pin **distributional eval** once we have user queries — sample real traffic, check patterns hold.
- Set a **regression gate**: any metric drop > 5% blocks merge.

Ship a beta to a small set of internal users for shadow eval — actual diversity beats hand-curated diversity. But "100% on our hand-picked 8" is not a release criterion.

---

### Q20. What's the difference between LLM-as-judge and keyword coverage as metrics? Name one failure mode of each.

**Strong answer.**

| Metric | Strengths | Failure modes |
|---|---|---|
| **Keyword coverage** | Deterministic, cheap, fast, transparent (you can debug a fail in 10 seconds) | Brittle. `"no cream"` vs `"does not use cream"` — semantically identical, mechanically different. Synonyms break it. |
| **LLM-judge** | Holistic, captures semantic correctness, catches subtle wrongness | Biased: **position bias** (rates 2nd answer higher), **length bias** (longer = better), **self-preference** (judge from same family rates higher), **sycophancy** (defaults to nice scores) |

Use them together. Keyword for objective facts, LLM-judge for nuanced quality. Periodically pin human labels on judged samples to calibrate the judge. Never use LLM-judge as your only signal — it's noisy enough to make wrong shipping decisions.

---

# 🎬 Category 7 — Behavioral / story

### Q21. Tell me about a bug you debugged in a production-style RAG system.

**Strong answer (rehearse this until it lands in ~80 seconds).**

> "While building a multi-PDF RAG bot, I noticed in my debug log that the query rewriter — a small/fast LLM that was supposed to convert follow-up questions into standalone form — was generating *answers* instead of *questions*. The user asked 'what about for a beginner?' and the rewriter said 'For a beginner, I'd recommend the traditional version' — that's not a question, that's an opinion.
>
> Here's what made it interesting: the user-visible final answer was still correct. My pipeline used the original query for generation, not the rewrite. The damage stayed contained downstream.
>
> Two fixes: few-shot examples in the rewriter prompt (small models obey examples, not abstract rules), and a 5-line validator that falls back to the original query if the rewrite looks like an answer. Also added a per-stage metric — `rewrite_quality` — that catches this specific failure. It went from 0.67 to 1.00 after the fix.
>
> Two lessons stuck with me. First: small LLMs need examples, not rules. Second: end-to-end metrics miss bugs that defensive pipeline design absorbs. You need per-stage metrics or these regressions go silent."

Roughly 80 seconds spoken. Hits: symptom → diagnosis → defensive-design surprise → fix → measurement → general lesson. Strong answer.

---

### Q22. Describe a time you chose NOT to add a fancy technique to your system. Why, and how did you justify the decision?

**Strong answer.**

> "I A/B'd HyDE on my corpus using my eval harness. HyDE adds an LLM call per query — meaningful latency and cost. The result: zero metric change across all four metrics.
>
> Why? My corpus was small enough that retrieval was already at ceiling — `retrieval_recall` was 1.00 baseline. HyDE had nothing to improve.
>
> The lesson: techniques don't transfer automatically. A paper showing HyDE helps on Wikipedia doesn't tell you it'll help on YOUR corpus. The eval is what tells you that.
>
> I removed it from the default config and wrote up the negative result. A negative result is just as valuable as a positive one — it saves us cost and complexity forever. Default-on by hype is one of the most expensive mistakes I see in production RAG."

---

### Q23. Your team is debating between LangChain, LlamaIndex, and "just write it ourselves." You've done the third. Make a one-paragraph case for each option as if you were arguing for it. Then say which you'd actually pick and why.

**Strong answer.**

> **For LangChain:** mature ecosystem, especially **LangGraph** for agentic flows and **LangSmith** for observability and tracing. Strong community, lots of plug-ins. Good for "we need to ship something agentic with monitoring tomorrow."
>
> **For LlamaIndex:** cleaner abstractions, often better for pure RAG, less kitchen-sink. Good for "we want idiomatic RAG without over-engineering."
>
> **For DIY:** total control, no surprises, perfect understanding of every component. Good for "we have specific constraints no framework covers, and we want to actually know how this works."
>
> **My pick depends on context.** For learning, DIY (which is what I did — built mine without LangChain to understand the pieces). For a small team shipping production with limited time, LangGraph + LangSmith — the agentic capabilities and observability win. For a company with bespoke retrieval needs (custom embedders, domain-specific rerankers), start DIY and port to a framework only when it adds clear value.
>
> Honest meta-point: framework choice matters less than people think. Understanding the underlying patterns matters more. I'd never hire someone based on which framework they know.

---

# 🎯 Self-assessment cheat sheet

After comparing your answers to the model answers, rough level rubric:

| Pattern in your answers | Probable level |
|---|---|
| Got the foundations (Q1–Q6) without hesitation, struggled on system design (Q15–Q17) | **Junior-to-mid**, very normal |
| Got foundations + trade-offs + debugging, system design felt rough but you asked the right clarifying questions | **Solid mid**, ready for offers |
| Above + clean system design with multi-tenant security awareness | **Senior-track** |
| Above + cited specific production frameworks / experiences (Pinecone, Langfuse, named cost numbers) | **Staff-track** — and you need actual production scars at this level |

## What to practice if you want to level up further

| Gap | Drill |
|---|---|
| Stumbled on Q15 (10M scale) | Read 1–2 Pinecone / Qdrant "production case study" blogs. Get the vocabulary. |
| Stumbled on Q16 (multi-tenant) | Build a tiny demo with two `tenant_id`s and confirm isolation. 2 hours. |
| Stumbled on Q21 (bug story) | Rehearse the rewriter story out loud, on a timer. 5 reps. Should land in ~80 sec. |
| Stumbled on Q23 (framework comparison) | 30 min reading each framework's quick-start. Just enough to say one sentence per tool. |

---

→ Back to the [index](README.md).
