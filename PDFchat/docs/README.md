# RAG Study Notes

Personal reference for the RAG techniques implemented in this repo.
Each note is self-contained: problem → diagram → code → use cases → caveats.

## Reading order

The notes build on each other. The same chatbot upgrades through each stage:

| # | Note | What it adds | Status |
|---|------|--------------|--------|
| 1 | [Naive RAG](01-naive-rag.md) | The baseline pipeline. PDF → chunks → embeddings → FAISS → LLM. | ✅ Built |
| 2 | [Conversational Memory](02-conversational-memory.md) | Standalone-question rewriting so follow-ups work. | ✅ Built |
| 3 | [Re-ranking](03-reranking.md) | Cross-encoder second stage for precision. | ✅ Built |
| 4 | [Hybrid Search](04-hybrid-search.md) | BM25 + dense + Reciprocal Rank Fusion. | ✅ Built |

## Current pipeline (everything turned on)

```
                          user question
                                │
                                ▼
                       query_rewriter ─── note #2
                                │
                                ▼
                ┌─── dense retrieval (FAISS)
                │                              ─── note #4
                └─── sparse retrieval (BM25)
                                │
                                ▼
                       RRF fusion ─── note #4
                                │
                                ▼
                       cross-encoder re-rank ─── note #3
                                │
                                ▼
                              LLM
                                │
                                ▼
                       grounded answer
```

## File ↔ note map

| Code file | Implements |
|-----------|------------|
| `pdf_loader.py` | Naive RAG |
| `embeddings.py` | Naive RAG |
| `retriever.py` | Naive RAG (dense half) |
| `llm.py` | Naive RAG + chat history (note #2) |
| `query_rewriter.py` | Conversational Memory |
| `reranker.py` | Re-ranking |
| `bm25_index.py` | Hybrid Search (sparse half) |
| `hybrid.py` | Hybrid Search (RRF fusion) |
| `config.py` | Settings for all of the above |
| `main.py` | Wires every stage together |

## Quick-start (run it locally)

```bash
cd PDFchat
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# paste your free Groq API key from https://console.groq.com/keys

python main.py path/to/your.pdf
```

## What I (still) want to learn next

Roughly in order of "next things I should build":

- Persistent indexes (save/load FAISS + BM25 so I don't re-embed every run)
- Parent-child chunking (embed small, send large)
- Multi-PDF ingestion (one bot, many sources, cite by doc name)
- Streaming responses (word-by-word output)
- Evaluation harness (measure actual quality, not vibes)
- Agentic RAG (multi-step search loops)
- GraphRAG (relationships, not just chunks)
