# PDFchat-app

Product-shaped RAG chatbot over a folder of PDFs. Persistent indexes,
streaming output, hybrid search + rerank + parent-child + (optional) HyDE,
and an eval harness so you can measure changes instead of vibing.

Companion to `../PDFchat/` (the study version with heavy comments + docs).

## Setup

```bash
cd PDFchat-app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# paste your free Groq key from https://console.groq.com/keys
```

Drop one or more PDFs into `data/`.

## Three CLIs

### 1. `ingest.py` — build the indexes

```bash
python ingest.py
```

Reads every `.pdf` in `data/`, builds FAISS + BM25 over parent/child chunks,
writes everything to `indexes/` along with a manifest of file hashes and
chunk settings.

Run this **once** after adding or changing PDFs. `chat.py` and `eval.py`
read from the saved indexes — they don't re-embed.

### 2. `chat.py` — interactive REPL

```bash
python chat.py
```

Loads the saved indexes, refuses to start if the PDFs have changed since
the last ingest (run `ingest.py` again in that case), then drops into a
chat loop with streaming output and per-question source citations.

### 3. `eval.py` — measure quality

```bash
python eval/seed.py            # auto-bootstrap a dataset stub
# now hand-edit eval/dataset.yaml — delete bad questions, fix wrong pages

python eval.py                 # run the suite
python eval.py --no-judge      # skip the LLM-judge metric (cheaper)

# A/B without editing .env:
python eval.py --config USE_HYDE=true
python eval.py --config USE_HYDE=true --config TOP_K=6
```

The harness reports four metrics per question:

| Metric | What it checks |
|---|---|
| `retrieval_recall` | Did the expected (doc, page) end up in the top-k chunks? |
| `citation_match` | Did the answer cite `(doc.pdf p. N)` correctly? |
| `keyword_coverage` | What fraction of expected keywords are in the answer? |
| `llm_judge` | Holistic 1–5 score from a separate LLM call. |

## Project layout

```
PDFchat-app/
├── ingest.py / chat.py / eval.py     # CLI entry points
├── pdfchat/                          # importable package
│   ├── config.py
│   ├── loader.py                     # multi-PDF + parent-child + doc_name
│   ├── embeddings.py / bm25.py       # build indexes
│   ├── retrieval.py / hybrid.py / rerank.py
│   ├── query_rewrite.py              # standalone + HyDE
│   ├── llm.py                        # streaming + grounded prompt
│   ├── storage.py                    # save/load + manifest + freshness
│   └── pipeline.py                   # orchestrator (used by chat & eval)
├── eval/
│   ├── dataset.yaml                  # hand-curated ground truth
│   ├── seed.py                       # LLM-assisted dataset bootstrapper
│   └── metrics.py
├── data/                             # gitignored — drop your PDFs here
└── indexes/                          # gitignored — built by ingest.py
```

## Flags worth knowing

All set via `.env` or `--config KEY=VALUE` to `eval.py`:

| Flag | Default | What it does |
|---|---|---|
| `USE_HYDE` | `false` | Adds HyDE retrieval alongside dense + sparse (Q+H safety pattern; never replaces query-based search). |
| `RETRIEVE_K` | `20` | How many candidates each retriever returns before fusion. |
| `TOP_K` | `4` | How many parents the LLM finally sees. |
| `PARENT_SIZE` / `CHILD_SIZE` | `1200` / `240` | Chunk sizes (words) for parent-child retrieval. |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Main answer model. |
| `GROQ_REWRITE_MODEL` | `llama-3.1-8b-instant` | Used for standalone rewrites, HyDE, LLM-judge — small/fast to keep cost down. |
