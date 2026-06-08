# 3. Re-ranking — bi-encoder for recall, cross-encoder for precision

> **TL;DR:** Retrieve a wide pool of candidates with the fast bi-encoder (say 20), then re-score them with a slow but accurate cross-encoder and keep the best 4.

## The problem

A bi-encoder is fast because it encodes the query and chunks **separately**.
That speed has a cost: the model never gets to "compare" them directly.
So its top 4 has noise — the *right* chunk might be at rank 7 instead of rank 1.

Simple fix: retrieve more (say 20), then have a smarter model pick the winners.

## The big idea: bi-encoder vs cross-encoder

```
BI-ENCODER  (fast, rough)

     query  ──► encoder ──► vec_q ─┐
                                   ├──► cosine  ──► score
     chunk  ──► encoder ──► vec_c ─┘

     Pro: encode every chunk ONCE, then querying is O(N) dot products. Scales to millions.
     Con: query and chunk never see each other inside the model. Rough ranking.


CROSS-ENCODER  (slow, accurate)

     [CLS] query [SEP] chunk [SEP] ──► encoder ──► score

     Pro: full attention between query tokens and chunk tokens. Very accurate.
     Con: must re-run for every (query, chunk) pair. Can't precompute. Slow.
```

So we use **both**, each at the stage they're good for:

```
   ┌──────────────────────────────────────────────────────────┐
   │                  TWO-STAGE RETRIEVAL                     │
   │                                                          │
   │   Stage 1: bi-encoder + FAISS    ──►  top 20 candidates  │
   │           (RECALL: wide net, fast)                       │
   │                                                          │
   │   Stage 2: cross-encoder rerank  ──►  top 4 chunks       │
   │           (PRECISION: best of the candidates)            │
   │                                                          │
   │   Stage 3: LLM answers using the top 4                   │
   └──────────────────────────────────────────────────────────┘
```

## The flow

```
         question
            │
            ▼
   ┌──────────────────────┐
   │  bi-encoder + FAISS  │  k=20 candidates
   └──────────┬───────────┘
              │
              ▼  20 candidates
   ┌──────────────────────┐
   │   cross-encoder      │  scores each (query, candidate) pair
   │                      │  sorts by score
   └──────────┬───────────┘
              │
              ▼  top 4
            LLM
```

## Code

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-base")  # ~280MB, free, runs on CPU

def rerank(reranker, query, candidates, top_k):
    if not candidates:
        return []
    pairs = [[query, c.text] for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_k]]
```

In the chat loop:

```python
candidates = retrieve(query, embedder, index, chunks, k=20)   # wide
top_chunks = rerank(reranker, query, candidates, top_k=4)     # narrow
```

**In this repo:** `reranker.py`. Used by `main.py`.

## Free re-ranker models

| Model | Size | Notes |
|---|---|---|
| `BAAI/bge-reranker-base` | ~280 MB | Solid default. English. |
| `BAAI/bge-reranker-large` | ~1.1 GB | More accurate, slower. |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~90 MB | Tiny, fast. Older but still usable. |
| `mixedbread-ai/mxbai-rerank-base-v1` | ~280 MB | Strong open competitor to BGE. |

Paid (better, but cost money): Cohere Rerank 3, Voyage Rerank.

## When to use it

- **Almost always.** Re-ranking is the single biggest quality bump in modern RAG.
- Customer support / KB search where precision matters more than recall.
- Any RAG where the LLM context is small and you want the best 3–5 chunks, not 20.

## When NOT to use it

- Tight latency budgets and very large candidate pools.
  - Cross-encoders are ~100–1000× slower than bi-encoders per pair.
  - Reranking 20 candidates is fine on CPU. Reranking 1000 is not.
- Pure prototypes where every ms of startup counts.

## Caveats

| Caveat | Notes |
|---|---|
| Latency | Each (query, chunk) pair is a forward pass. Keep candidate count ≤ a few dozen on CPU. |
| Model download | ~280 MB on first run. Cached in `~/.cache/huggingface` after that. |
| Language matters | `bge-reranker-base` is English. Use a multilingual cross-encoder for non-English content. |
| Doesn't fix bad recall | If the right chunk isn't in the candidate pool at all, reranking can't save you. **Fix recall first** (better chunking, hybrid search). |

## Key concepts this teaches

- **Recall vs precision** as two separate retrieval problems.
- **Bi-encoder vs cross-encoder** — the most important architectural distinction in modern retrieval. Once you grok it, half the RAG papers become readable.
- The **two-stage retrieval cascade**: a pattern that appears everywhere (search engines, recommender systems, ad ranking).
- Why you should over-retrieve and then filter, not just trust top-k.

→ Next: [Hybrid Search](04-hybrid-search.md)
