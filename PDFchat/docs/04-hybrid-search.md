# 4. Hybrid Search — combine meaning (dense) and exact words (sparse)

> **TL;DR:** Run both a dense retriever (embeddings) and a sparse retriever (BM25), then merge their rankings using Reciprocal Rank Fusion. Covers each method's blind spots.

## The problem

| Retriever alone | Fails on |
|---|---|
| Dense (embeddings) | Exact tokens — names, IDs, acronyms, product codes |
| Sparse (BM25) | Synonyms, paraphrasing, meaning |

You can pick the wrong one for any single question. Run both — neither has all the answers.

## The fix in one picture

```
                question
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
  ┌──────────┐         ┌──────────┐
  │  DENSE   │         │  SPARSE  │
  │  (FAISS) │         │  (BM25)  │
  │          │         │          │
  │ meaning  │         │ exact    │
  │ search   │         │ tokens   │
  └────┬─────┘         └────┬─────┘
       │ ranked list        │ ranked list
       │                    │
       └──────────┬─────────┘
                  ▼
          ┌──────────────┐
          │  RRF fusion  │   ignore scores, use only ranks
          │  (hybrid.py) │
          └──────┬───────┘
                 │
                 ▼
            merged ranking ─► rerank ─► LLM
```

## BM25 in 60 seconds

BM25 = "TF-IDF, tuned for retrieval". For each query word, score a chunk on:

```
   TF (term frequency)       — how often the word appears (saturates → spam-proof)
   ×
   IDF (inverse doc freq.)   — rare words count more than "the"/"and"
   ×
   length normalization      — penalize long chunks that match by accident
```

Sum these for every query word. No neural net, no model download, just word counts.

This is why "Vaswani", "AX-9281", "GPT-3.5" — rare exact tokens — light up BM25 instantly.

## The trap when combining: don't add scores

```
  Dense score:    -1 ──────────────────────── 1
  BM25 score:     0  ───────────────────► ∞ (unbounded)

  dense=0.81  +  bm25=14.2  =  15.01    ← BM25 always wins. Broken.
```

The scores are on different scales. You'd need to normalize per-corpus, per-query — endless tuning.

## The fix: Reciprocal Rank Fusion (RRF)

Throw away the scores. Use only the **ranks**.

```
                          1
   RRF_score(doc)  =   Σ  ──────────────
                      each   k + rank
                      list                ( k = 60, classic constant )
```

### Worked example

Suppose `chunk_7` is **rank 1** in dense and **rank 3** in sparse:

```
   from dense:    1 / (60 + 1)  =  0.01639
   from sparse:   1 / (60 + 3)  =  0.01587
                              ───────────
   total RRF:                      0.03226
```

A chunk only one retriever liked tops out at ~0.01639. A chunk both liked starts adding. Agreement wins automatically.

### Why this works so well

| Property | Why it matters |
|---|---|
| Scale-free | Doesn't matter that one score is cosine and the other is BM25. Ranks only. |
| Agreement is rewarded | Both retrievers like a chunk → its score stacks. |
| Damped | `k=60` keeps rank 1 from dominating. Lower-ranked items still contribute. |
| No tuning | The constant `k=60` is from the original 2009 paper. Almost never needs changing. |

## Code

**BM25 index (`bm25_index.py`):**

```python
import re, numpy as np
from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)

def tokenize(text):
    return _TOKEN_RE.findall(text.lower())

def build_bm25(chunks):
    return BM25Okapi([tokenize(c.text) for c in chunks])

def bm25_search(bm25, chunks, query, k):
    tokens = tokenize(query)
    if not tokens:
        return []
    scores = bm25.get_scores(tokens)
    top = np.argsort(scores)[::-1][:k]
    return [chunks[int(i)] for i in top]
```

**RRF fusion (`hybrid.py`):**

```python
def reciprocal_rank_fusion(rank_lists, k_rrf=60):
    scores, objs = {}, {}
    for ranked in rank_lists:
        for rank, chunk in enumerate(ranked, start=1):
            key = id(chunk)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k_rrf + rank)
            objs[key] = chunk
    return sorted(objs.values(), key=lambda c: scores[id(c)], reverse=True)
```

**In the chat loop (`main.py`):**

```python
dense_hits  = retrieve(query, embedder, dense_index, chunks, k=20)
sparse_hits = bm25_search(bm25, chunks, query, k=20)
fused       = reciprocal_rank_fusion([dense_hits, sparse_hits], k_rrf=60)
top_chunks  = rerank(reranker, query, fused, top_k=4)
```

## When to use it

- **Default it on** for any serious RAG. The cost is tiny (BM25 is cheap), the recall gain is huge.
- Legal, medical, finance — anywhere documents are full of proper nouns, codes, identifiers.
- Code search and API docs — exact symbol names matter.
- Multi-tenant search where you can't predict how users will phrase queries.

## When you can skip it

- Single-doc demos with short, semantic questions where dense alone works fine.
- Very specialized embedders fine-tuned on your domain may already cover exact-token recall (rare in practice).

## Caveats

| Caveat | Notes |
|---|---|
| Tokenization is crude | We split on `\w+` and lowercase. Production: stemming (`runs` → `run`), real tokenizer, or stopword removal. |
| Memory | BM25Okapi keeps a tokenized copy of every chunk. Fine for thousands; for millions, use Tantivy / Elasticsearch / Vespa. |
| BM25 ≠ semantic | BM25 alone fails on synonyms. Always pair it with dense in hybrid mode. |
| RRF constant | `k=60` is the standard. Tuning it rarely moves the needle. |

## Aliases & relatives

| Name | Notes |
|---|---|
| Hybrid search / Hybrid retrieval | Common name. |
| BM25 + dense ensemble | Same idea. |
| Linear fusion / Weighted fusion | Alternative to RRF — combine normalized scores with weights. More tuning required. Often used in vendor systems. |
| Cross-encoder fusion | Instead of RRF, throw both lists to a cross-encoder and let it re-score. More accurate, slower. |

## Key concepts this teaches

- **Dense vs sparse retrieval** — the second-most-important distinction in modern search (after bi/cross encoder).
- **BM25 fundamentals** — TF saturation, IDF, length normalization. These show up in *every* search engine.
- **Why scores from different retrievers aren't comparable.**
- **Rank-based fusion (RRF)** — a universal trick that works for ANY collection of ranked lists, not just dense + sparse. You can RRF dense + sparse + LLM-rewritten queries + metadata filters all together.
- The general **wide-recall → fuse → precision-rerank** cascade used by Elasticsearch, Vespa, Weaviate, OpenSearch.

→ Back to the [index](README.md).
