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
| 5 | [Parent-Child Chunking](05-parent-child-chunking.md) | Embed small children for matching, send big parents to the LLM. | ✅ Built |
| 6 | [Query Rewriting](06-query-rewriting.md) | Deep dive: standalone, multi-query, HyDE, decomposition, step-back, RAG-Fusion, routing. | 📖 Concept reference |
| 7 | [Evaluation Harness](07-evaluation-harness.md) | Datasets, per-stage metrics, LLM-as-judge, production patterns. Built in `PDFchat-app/eval/`. | 📖 Concept + product reference |
| 8 | [Tier 2 Roadmap](08-tier-2-roadmap.md) | Contextual Retrieval, Self-RAG, Agentic, Multimodal, GraphRAG, fine-tuning, CAG — topic by topic. | 🗺️ Forward-looking |
| 9 | [Interview Questions](09-interview-questions.md) | 23 questions with model answers across foundations, design, debugging, system design, eval, and behavioral. | 🎯 Self-test |
| 10 | [Agentic RAG — Manual Loop](10-agentic-rag-manual-loop.md) | Hand-rolled ReAct loop (no framework). Built in `PDFchat-app/pdfchat/agent.py`. Prerequisite for LangGraph + MCP + multi-agent. | ✅ Built |
| ★ | [Cheatsheet](cheatsheet.md) | Worked example + annotated pipeline + mental models table. | 📖 Quick reference |

## Current pipeline (everything turned on)

```
   PDF ──► parents (big)  +  children (small, point to parent) ─── note #5
                                │
                       ┌────────┴────────┐
                       ▼                 ▼
                    FAISS              BM25       (built on CHILDREN)
                                │
        ────────────────────────────────────────  (per question)
                          user question
                                │
                                ▼
                       query_rewriter ─── note #2
                                │
                                ▼
                ┌─── dense retrieval (children)
                │                                ─── note #4
                └─── sparse retrieval (children)
                                │
                                ▼
                       RRF fusion ─── note #4
                                │
                                ▼
                cross-encoder re-rank (children) ─── note #3
                                │
                                ▼
              top-k children  ─►  PARENTS (dedupe) ─── note #5
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
| `pdf_loader.py` | Naive RAG + Parent-Child (note #5) |
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

- Multi-PDF ingestion (one bot, many sources, cite by doc name)
- Persistent indexes (save/load FAISS + BM25 so I don't re-embed every run)
- Streaming responses (word-by-word output)
- Evaluation harness (measure actual quality, not vibes)
- Agentic RAG (multi-step search loops)
- GraphRAG (relationships, not just chunks)
