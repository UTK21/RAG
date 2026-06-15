# PDFchat-app — Tier 1 Architecture

Reference walkthrough of the product version: what each part does, how the
pieces fit together, and which new ideas Tier 1 introduces on top of the
five techniques from the study version.

> If you want the *retrieval theory* — bi-encoder vs cross-encoder, RRF,
> parent-child, etc. — see `../PDFchat/docs/`. This doc covers what's new
> in the **product layer** built around that theory.

---

## 1. What Tier 1 is for

The study version (`PDFchat/`) is a single-PDF teaching artifact: heavily
commented, indexes rebuilt on every run, output prints in one blob,
"is this better?" answered by gut feeling.

Tier 1 turns that into something you could actually deploy:

| Concern | Study version | Tier 1 |
|---|---|---|
| Source data | one PDF as a CLI arg | **a folder of PDFs**, doc tracked per chunk |
| Startup time | re-embeds every run | **persisted indexes** loaded in seconds |
| Output UX | answer appears all at once | **streaming** token-by-token |
| Query enhancement | standalone rewrites only | + **HyDE** as a togglable extra path |
| Quality assurance | "feels right" | **eval harness** with four quantitative metrics |
| Code shape | one folder, educational comments | importable package + three CLIs |

These five upgrades are the deliverables. The rest of this doc explains
how they fit together.

---

## 2. The architectural shift: index time vs query time

In the study version, building the index and answering questions happened
in the same script. Tier 1 splits them — this is the single biggest
structural change.

```
   ─── INDEX TIME (run once when PDFs change) ───
                        │
                  data/*.pdf
                        │
                        ▼
                   ingest.py
                        │
                        ▼
              indexes/  (FAISS + BM25 + chunks + manifest)


   ─── QUERY TIME (run any number of times) ───
                        │
                   chat.py / eval.py
                        │
                        ▼
                  loads indexes/
                        │
                        ▼
                Pipeline.answer(...)
                        │
                        ▼
                  grounded reply
```

Why split them? Three reasons:

1. **Cost.** Embedding hundreds of chunks takes seconds-to-minutes. Doing
   that on every chat startup is wasteful.
2. **Determinism.** The same query against the same data should retrieve
   the same chunks. Persisting the index removes "did the embedder
   initialize differently this time?" as a variable.
3. **Operational sanity.** Treating ingest as a discrete step makes it a
   thing you can monitor, log, schedule, redo.

The price is a new problem: indexes can go **stale** when PDFs change.
We solve that with a manifest (next section).

---

## 3. The two lifecycles, in detail

### 3.1 Index time (`ingest.py`)

```
   for each *.pdf in data/:
        │
        ▼
   load_pdf()                                          loader.py
        │
        ▼
   per page:
       split into PARENTS (~1200 words)
       each parent → split into CHILDREN (~240 words)
       every chunk tagged with doc_name + page
        │
        ▼
   build_index(children, embedder)  ─►  FAISS dense    embeddings.py
   build_bm25(children)             ─►  BM25 sparse    bm25.py
        │
        ▼
   manifest = {
       doc_hashes:           {pdf -> sha256},
       settings_fingerprint: {chunk sizes, embed_model},
       doc_names, n_parents, n_children,
   }
        │
        ▼
   storage.save() writes 4 files:                      storage.py
       indexes/dense.faiss         FAISS binary
       indexes/sparse.bm25.pkl     pickled BM25Okapi
       indexes/chunks.pkl          pickled {parents, children}
       indexes/manifest.json       fingerprint
```

### 3.2 Query time (`chat.py` / `eval.py`)

```
   storage.load(index_dir)         ─►  LoadedIndex     storage.py
        │
        ▼
   check_fresh(loaded, data/, current_settings_fingerprint)
        │
        ├─► if PDFs changed since ingest    → refuse to start
        └─► if chunk settings changed       → refuse to start
        │
        ▼
   load embedder + reranker (one-time)
        │
        ▼
   Pipeline(settings, loaded, embedder, reranker, client)
        │
        ▼
   ─── per question loop ───
       │
       │  user types a question
       ▼
   pipe.answer_stream(query, history)
       │
       ├── 1. standalone-rewrite the query  (query_rewrite.standalone)
       │
       ├── 2. dense_hits   = retrieve(standalone_query)
       │    sparse_hits  = bm25_search(standalone_query)
       │    rank_lists   = [dense_hits, sparse_hits]
       │
       ├── 2a. (if USE_HYDE)
       │      hyde_text  = hyde(standalone_query)
       │      hyde_hits  = retrieve_with_vector(embed(hyde_text))
       │      rank_lists.append(hyde_hits)
       │
       ├── 3. fused      = reciprocal_rank_fusion(rank_lists)
       │
       ├── 4. top_children = rerank(standalone_query, fused, top_k)
       │      (rerank scores against the ORIGINAL standalone query,
       │       NEVER the HyDE text — this is HyDE's safety net)
       │
       ├── 5. parents = children_to_parents(top_children, parents)
       │      (dedupe — same parent referenced by multiple children = once)
       │
       ├── 6. tokens, trace = llm.answer_stream(parents + history + original)
       │
       ├── 7. chat.py prints trace.standalone_query + sources first
       │
       └── 8. chat.py streams tokens, accumulates into reply
       │
       ▼
   history.append(user + assistant)
```

The whole flow lives in `pdfchat/pipeline.py`. Chat and eval call it
differently:
- `chat.py` calls `answer_stream` → prints tokens as they arrive.
- `eval.py` calls `answer` → gets full reply + the same `RetrievalTrace`,
  uses both for scoring.

---

## 4. Module dependency graph

Who depends on whom inside the package:

```
                        ┌───────────────┐
                        │ pipeline.py   │  ◄── chat.py, eval.py
                        └──────┬────────┘
              ┌────────────────┼─────────────────────┐
              │                │                     │
              ▼                ▼                     ▼
         retrieval.py    query_rewrite.py        llm.py
              │                │                     │
              ▼                ▼                     │
         embeddings.py        groq                   │
              │                                      │
              │      ┌─────────────────┐             │
              └─────►│   loader.py     │◄────────────┘
                     │  (data model)   │
                     └─────────────────┘
                              ▲
                              │
                     ┌────────┴────────┐
                     │                 │
                hybrid.py         bm25.py
                                       │
                                       ▼
                                rank_bm25
```

`loader.py` is the data-model hub — `Chunk`, `ParentChunk`,
`children_to_parents()`. Every retrieval module imports from it.

`storage.py` is intentionally absent from this diagram — it's only used
by `ingest.py` (writer) and `chat.py`/`eval.py` (reader at startup). It
never participates in the per-question hot path.

---

## 5. The five deliverables, in depth

### 5.1 Multi-PDF ingestion

**Problem.** Real knowledge bases are multiple documents. The bot needs
to know *which* doc each fact came from so it can cite, and it needs to
behave sensibly when two docs disagree.

**Solution.** Two small additions:

1. **A new metadata field.** `doc_name` now lives on every chunk:
   ```python
   @dataclass
   class Chunk:
       text: str
       page: int
       parent_idx: int
       doc_name: str    # ◄ new
   ```
2. **Conflict-aware prompt.** The system prompt in `llm.py` now includes:
   > "If sources DISAGREE, name each source and what it says — do not
   > silently pick one."

**Flow.**

```
   data/
      ├── recipes_italian.pdf  ────► chunks tagged doc_name=recipes_italian.pdf
      ├── recipes_french.pdf   ────► chunks tagged doc_name=recipes_french.pdf
      └── kitchen_techniques.pdf ─► chunks tagged doc_name=kitchen_techniques.pdf
                                          │
                                          ▼
                            single FAISS + BM25 over ALL chunks
                                          │
                                          ▼
                        retrieval returns chunks from any/all docs
                                          │
                                          ▼
                      LLM cites e.g. (recipes_italian.pdf p.4)
                      and names disagreements when sources conflict
```

**Files involved:** `pdfchat/loader.py` (added `doc_name`, added
`load_directory()`), `pdfchat/llm.py` (citation prompt).

**Important concept — metadata pointers, again.**
This is the second time we've leaned on the "tiny field with huge
leverage" pattern. `parent_idx` unlocked parent-child retrieval; now
`doc_name` unlocks source attribution. In Tier 2 you'll see the same
pattern again with `section_id`, `timestamp`, `chunk_summary`, etc.

---

### 5.2 Persistent indexes

**Problem.** Re-embedding every chunk on every chat startup is slow and
wasteful. Plus, if the embedding result varied at all (e.g. due to model
non-determinism), you'd get inconsistent retrieval.

**Solution.** Save everything to disk; reload from disk; refuse to use
stale data.

```
   indexes/
   ├── dense.faiss          ◄── faiss.write_index / faiss.read_index
   ├── sparse.bm25.pkl      ◄── pickle.dump / load (BM25Okapi)
   ├── chunks.pkl           ◄── pickle.dump / load ({parents, children})
   └── manifest.json        ◄── what these were built from
```

**The manifest** is the load-bearing piece:

```json
{
  "doc_names": ["recipes_italian.pdf", "recipes_french.pdf"],
  "n_parents": 47,
  "n_children": 218,
  "doc_hashes": {
    "recipes_italian.pdf": "8a1f0e...",
    "recipes_french.pdf":  "3c52a9..."
  },
  "settings_fingerprint": {
    "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
    "parent_size": 1200,
    "parent_overlap": 200,
    "child_size": 240,
    "child_overlap": 40
  }
}
```

**The freshness check** runs on every `chat.py` / `eval.py` startup:

```
   Current data/ hashes != manifest.doc_hashes
       → "PDFs changed since last ingest" → refuse to start
   Current chunk settings != manifest.settings_fingerprint
       → "Chunk settings changed" → refuse to start
```

This stops a real footgun: editing a PDF and forgetting to re-ingest,
then getting confidently wrong answers from a stale index. The bot will
just refuse to run until you `python ingest.py` again.

**Files involved:** `pdfchat/storage.py` (all of it), `ingest.py`
(writer), `chat.py` + `eval.py` (readers + freshness gate).

**Important concept — index lifecycle.**
Production search systems treat indexes as artifacts with their own
build/version/deploy story. The manifest is the most minimal version of
that: a JSON file that says "this index was built from exactly these
inputs with exactly these settings." That alone catches 90% of
"why is the bot wrong?" incidents.

---

### 5.3 Streaming responses

**Problem.** Non-streaming LLM calls block until the entire reply is
generated. A long answer takes 5–15 seconds of silence then arrives all
at once. Bad UX.

**Solution.** Use Groq's streaming API; print each token delta as it
arrives.

```
   client.chat.completions.create(..., stream=True)
        │
        ▼
   for chunk in stream:
       delta = chunk.choices[0].delta.content
       if delta:
           yield delta              ◄ pdfchat/llm.py::answer_stream
                ▼
       chat.py: print(delta, end="", flush=True)
                accumulate into reply_parts
                ▼
       (after loop) full_reply = "".join(reply_parts)
                    history.append({"role": "assistant", "content": full_reply})
```

**Two design notes:**

1. **Both functions, not just one.** `llm.py` exposes both `answer()`
   (returns full string) and `answer_stream()` (yields deltas).
   Streaming is great for chat but awkward for eval (you'd assemble the
   string anyway). Eval calls `answer()`; chat calls `answer_stream()`.

2. **Citations print BEFORE the first token.** In `chat.py`:
   ```
   you> ...
       (rewrote → ...)               ◄ instant, comes from trace
       (sources: recipe.pdf p.4, ...) ◄ instant, comes from trace
   bot> answer streams in here word by word...
   ```
   `pipeline.answer_stream` returns `(token_iter, trace)` together so
   the trace is fully populated before the first token. Sources show up
   immediately even though the text is still arriving.

**Files involved:** `pdfchat/llm.py::answer_stream`, `chat.py`,
`pdfchat/pipeline.py::answer_stream`.

---

### 5.4 HyDE (with safety pattern)

**Problem.** Sometimes the query and the answer are written in such
different styles that dense retrieval struggles. (See
`../PDFchat/docs/06-query-rewriting.md` for the full theory.)

**Solution.** Generate a hypothetical answer; search with its embedding
*in addition to* normal retrieval; let the cross-encoder filter.

**The Q+H safety pattern — why HyDE can't make things worse here.**

```
                       standalone_query
                              │
                  ┌───────────┼───────────┐
                  │           │           │
                  ▼           ▼           ▼
              dense       BM25       (if USE_HYDE)
              search      search        hyde_text = LLM.generate()
                  │           │           hyde_vec  = embed(hyde_text)
                  │           │           dense_search(hyde_vec)
                  │           │                   │
                  └───────────┼───────────────────┘
                              ▼
                         RRF fusion
                              │
                              ▼
                   rerank(STANDALONE_QUERY, fused)  ◄ HyDE's safety net
                              │                       reranker scores
                              ▼                       against ORIGINAL
                          top children                query, never HyDE
```

Two layers of safety:

1. **HyDE results SUPPLEMENT, never REPLACE.** Dense and BM25 searches
   on the original query always run. Worst case HyDE adds noise; the
   query's own results are still in the pool.
2. **Reranker uses the original query.** Even if HyDE retrieved 20 wrong
   candidates, the cross-encoder filters them by scoring `(original
   query, chunk)`. Off-topic HyDE candidates get demoted and dropped.

So HyDE is a strictly additive signal. The only "cost" is one extra LLM
call per question.

**Files involved:** `pdfchat/query_rewrite.py::hyde`,
`pdfchat/retrieval.py::retrieve_with_vector`,
`pdfchat/pipeline.py::_retrieve` (the orchestration).

**Important concept — additive vs replacing rewrites.**
A naive HyDE implementation searches only with the fake answer (replacing
the query). Q+H searches with both. The general lesson: when you add a
query transformation, **prefer designs where the transformation augments
rather than replaces the original**. That keeps the system robust to bad
transformations.

---

### 5.5 Evaluation harness

**Problem.** Without measurement, "did this change help?" is a vibe.
Every retrieval tweak, prompt edit, or new technique might be a step
forward or backward — and you'd never know.

**Solution.** A small but well-designed harness:

```
   ┌─────────────────────┐
   │  eval/dataset.yaml  │  hand-curated (question, doc, page, keywords)
   └──────────┬──────────┘
              │
              ▼
   ┌─────────────────────────────────────┐
   │           eval.py                   │
   │                                     │
   │  for each case:                     │
   │     reply, trace = pipe.answer(q)   │
   │     measure:                        │
   │       retrieval_recall (trace)      │
   │       citation_match   (reply)      │
   │       keyword_coverage (reply)      │
   │       llm_judge        (reply)      │
   │                                     │
   │  print per-case + aggregate         │
   └─────────────────────────────────────┘
```

**The four metrics, by what stage they catch problems at:**

| Metric | Stage it catches | How |
|---|---|---|
| `retrieval_recall` | retrieval | Did the right (doc, page) end up in `trace.top_children`? Isolates retrieval failures from generation failures. |
| `citation_match` | generation (faithfulness) | Did the answer cite the right `(doc.pdf p. N)`? Regex over the answer. |
| `keyword_coverage` | generation (completeness) | What fraction of expected key terms appear in the answer? Lowercase substring check. |
| `llm_judge` | generation (holistic) | 1–5 score from a separate LLM call. Noisier, costs tokens. Useful sanity check. |

The first three are deterministic and cheap. `llm_judge` is the only one
that costs API tokens — `--no-judge` skips it.

**Important concept — separating retrieval from generation.**
If the bot gives a wrong answer, was it because (a) the wrong chunk was
retrieved or (b) the right chunk was retrieved but the LLM messed up?
The harness separates these:
- Low `retrieval_recall` → fix retrieval (chunking, hybrid mix, HyDE).
- High `retrieval_recall` but low `citation_match`/`keyword_coverage` →
  fix the prompt or pick a stronger model.

You almost never improve a RAG system without that diagnosis.

**The dataset is what matters.**
The harness is plumbing; the dataset is the truth. `eval/seed.py`
LLM-bootstraps a stub, but the seeded questions are mediocre at best.
**You have to hand-edit.** Delete uninteresting questions. Fix wrong
pages. Add edge cases (negations, multi-doc, conflicting info, "answer
is not in the docs"). 10 great questions beat 100 mediocre ones.

**The A/B pattern.**
`eval.py --config KEY=VALUE` overrides settings for one run. So:

```bash
python eval.py                                 # baseline
python eval.py --config USE_HYDE=true          # HyDE on
python eval.py --config TOP_K=6                # more context
```

You compare aggregate metrics across runs. That's how you'll learn
whether HyDE helps *your* corpus, whether the default chunk size is right,
etc. — without arguing from intuition.

**Files involved:** `eval/dataset.yaml`, `eval/seed.py`,
`eval/metrics.py`, `eval.py`, `pdfchat/config.py::load_settings`
(supports `overrides=`).

---

## 6. File-by-file reference

| File | Role |
|---|---|
| `requirements.txt` | All deps. `pyyaml` is the only new one over the study version. |
| `.env.example` | Tunable knobs. Copy to `.env` and edit. |
| **CLIs** | |
| `ingest.py` | Build & persist indexes. Run when PDFs change. |
| `chat.py` | Interactive REPL with streaming + citations. Refuses stale indexes. |
| `eval.py` | Runs the eval suite. Supports `--config` overrides. |
| **Package: `pdfchat/`** | |
| `config.py` | `Settings` dataclass + `load_settings(overrides)`. Centralized knobs. |
| `loader.py` | `Chunk`/`ParentChunk` dataclasses, `load_pdf`, `load_directory`, `children_to_parents`. Every chunk has `doc_name`. |
| `embeddings.py` | Loads sentence-transformer; builds the FAISS dense index. |
| `bm25.py` | Tokenizer + BM25Okapi sparse index. |
| `retrieval.py` | Dense top-k search. Two variants: from text query, from precomputed vector (HyDE). |
| `hybrid.py` | RRF over N ranked lists. |
| `rerank.py` | Cross-encoder rerank. |
| `query_rewrite.py` | `standalone()` (chat history → standalone question) + `hyde()` (hypothetical answer). |
| `llm.py` | Grounded system prompt, `answer()`, `answer_stream()`. |
| `storage.py` | Save/load + manifest + `check_fresh` + `settings_fingerprint`. The lifecycle module. |
| `pipeline.py` | The orchestrator. `Pipeline.answer` and `Pipeline.answer_stream`. Both return a `RetrievalTrace` for transparency. |
| **`eval/`** | |
| `dataset.yaml` | Hand-curated ground truth. |
| `seed.py` | LLM-bootstraps a stub dataset from your PDFs. |
| `metrics.py` | Four metric functions. |
| **Generated** | |
| `data/` | You drop PDFs here (gitignored). |
| `indexes/` | Built artifacts (gitignored). |

---

## 7. Configuration knobs

All read from environment / `.env`, overridable at eval time with
`--config KEY=VALUE`:

| Var | Default | Effect |
|---|---|---|
| `GROQ_API_KEY` | — | Required. |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Main answer model. |
| `GROQ_REWRITE_MODEL` | `llama-3.1-8b-instant` | Used for standalone rewrite, HyDE, LLM-judge. Small/fast on purpose. |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Bi-encoder. Changing this invalidates the index. |
| `RERANK_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder. Not in the manifest because it doesn't affect the index. |
| `DATA_DIR` / `INDEX_DIR` | `data` / `indexes` | Where PDFs / persisted artifacts live. |
| `PARENT_SIZE` / `PARENT_OVERLAP` | `1200` / `200` | Parent chunking. Invalidates index. |
| `CHILD_SIZE` / `CHILD_OVERLAP` | `240` / `40` | Child chunking. Invalidates index. |
| `RETRIEVE_K` | `20` | Per-retriever candidate count. |
| `RRF_K` | `60` | Reciprocal Rank Fusion constant. Almost never change. |
| `TOP_K` | `4` | Final number of parents the LLM sees. |
| `TEMPERATURE` | `0.2` | Answer temperature. |
| `HISTORY_TURNS` | `5` | How many past messages we keep. |
| `USE_HYDE` | `false` | Add HyDE retrieval path. |
| `HYDE_TEMPERATURE` | `0.5` | Diversity of HyDE's fake answer. |

**The four chunking/embedding settings are part of the manifest** — change
any of them and `chat.py` will refuse to start until you re-ingest. This
is intentional. A FAISS index built with `child_size=240` is incoherent
with chunks split at `child_size=400`.

---

## 8. End-to-end verification

```bash
cd PDFchat-app
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # paste your Groq key from console.groq.com

# 1. Drop 2-3 PDFs into data/  (use ones with overlapping content)

# 2. Build indexes
python ingest.py
#   expect: "N docs, M parents, K children" then "Done in Xs"
#   confirm: indexes/ now has 4 files including manifest.json

# 3. Chat — streaming + citations
python chat.py
#   try a follow-up: see "(rewrote → ...)" print
#   try a multi-doc question — confirm citations include doc names
#   try a question whose answer is in two docs differently —
#     bot should name both sources

# 4. Edit one of the PDFs, then run chat.py again
#   expect: "Stale index: PDFs in data/ have changed..." and exit
#   re-run ingest.py to fix

# 5. Eval
python eval/seed.py --per-doc 3            # auto-bootstrap dataset stub
#   hand-edit eval/dataset.yaml: keep 5-10 good questions
python eval.py                             # baseline
python eval.py --config USE_HYDE=true      # HyDE on
#   compare the aggregate scores between runs — this is your first
#   real measurement of "did changing X actually help?"
```

If every step works, Tier 1 is delivered.

---

## 9. New concepts Tier 1 introduces

Independent of any specific implementation file, these are the
**generalizable ideas** Tier 1 teaches that go beyond the retrieval
theory of the study version:

| Concept | Generalization |
|---|---|
| **Splitting index time from query time** | Same pattern in: search engines (indexer vs. searcher), ML systems (training vs. inference), build systems (compile vs. run). |
| **Manifest-based freshness** | Same pattern in: Docker image digests, Bazel build hashes, npm lockfiles. The manifest is a *content-addressed fingerprint* of "what produced this artifact." |
| **Metadata pointers as small architectural levers** | We've now seen `parent_idx` (small→big) and `doc_name` (attribution). Future ones: `section_id`, `timestamp`, `chunk_summary`. Each unlocks features wildly out of proportion to its size. |
| **Additive vs. replacing transformations** | The Q+H safety pattern. Generalize: when adding query rewriting, query expansion, candidate generation, prefer designs where new signals supplement rather than overwrite existing ones. |
| **Separating retrieval failures from generation failures** | The four-metric design. Generalize: in any pipeline, build metrics for each *stage*, not just the end-to-end result. End-to-end metrics tell you something broke; per-stage metrics tell you where. |
| **A/B via runtime overrides** | `--config KEY=VALUE`. Generalize: any system with knobs deserves runtime override paths so you can compare configurations without editing files. |
| **Streaming as decoupled iteration** | `(token_iter, trace)` returned together — trace is final, tokens are still arriving. Generalize: streaming systems often have *out-of-band* metadata available before the data. Surface it.

---

## 10. What this builds toward

Tier 1 is the foundation Tier 2 needs.
- Multi-PDF + `doc_name` makes Contextual Retrieval feasible (you need
  per-document context for chunk-prefixing summaries).
- Persistent indexes are required by any technique that needs to
  precompute extra metadata (summaries, entity extractions, graph edges).
- The eval harness is what makes Self-RAG, agentic RAG, or any new
  technique worth comparing against.

Tier 2 plugs into the spots Tier 1 designed in: a new retriever in
`pipeline.py::_retrieve`, a new metric in `eval/metrics.py`, a new
metadata field on `Chunk`. The shape is set.
