# 7. Evaluation Harness — measuring RAG quality instead of vibing

> **TL;DR:** An eval harness is a tiny dataset of *what the bot should answer* plus code that compares the bot's actual answers against it. Without one, every change you make to a RAG system is a guess. With one, you can say "switching HyDE on raised retrieval recall from 62% to 78% on our test set."

---

## The problem (and why it's bigger than it sounds)

You change something — different chunk size, different reranker, new prompt — and ask: "did that help?"

There are three ways to answer that question:

```
   Method                    Honest assessment
   ───────────────────────   ────────────────────────────────────────────────
   1. "It feels better"  ──► useless. You have a strong narrative bias
                             toward the change you just made working.
   2. "Try a few queries" ─► better, but you'll cherry-pick. You'll also
                             miss regressions on queries you didn't try.
   3. Run an eval suite  ──► boring, slower, undeniable.
```

Without method 3, you keep making changes that *might* be improvements,
shipping them, finding out two weeks later something's broken, rolling
back. You move sideways forever.

An evaluation harness is method 3. It's the *single* skill that
separates someone who fiddles with RAG from someone who *develops* RAG.

---

## What an eval harness actually is

Three pieces. That's it.

```
   ┌───────────────────────┐
   │     1. DATASET        │  hand-curated (question, expected outcome)
   │       (the gold)      │
   └──────────┬────────────┘
              │
              ▼
   ┌───────────────────────┐
   │     2. METRICS        │  functions that score actual vs expected
   │  (how we measure)     │
   └──────────┬────────────┘
              │
              ▼
   ┌───────────────────────┐
   │     3. RUNNER         │  loop over dataset, call the bot, score,
   │  (the orchestrator)   │  print aggregates
   └───────────────────────┘
```

In our Tier 1 implementation:

| Piece | File | Role |
|---|---|---|
| Dataset | `PDFchat-app/eval/dataset.yaml` | Hand-curated list of (question, expected_doc, expected_page, expected_keywords) tuples. |
| Metrics | `PDFchat-app/eval/metrics.py` | Four functions: `retrieval_recall`, `citation_match`, `keyword_coverage`, `llm_judge`. |
| Runner | `PDFchat-app/eval.py` | Loads dataset, runs each question through `Pipeline.answer`, scores, prints a per-question table + aggregates. |
| Seed helper | `PDFchat-app/eval/seed.py` | LLM-bootstraps a dataset stub from your PDFs. You then prune and fix by hand. |

---

## Eval vs. tests — they are NOT the same thing

A common beginner confusion:

```
   UNIT TESTS                              EVALUATION
   ──────────                              ──────────
   Check that CODE behaves correctly       Check that BEHAVIOR is correct
   Deterministic — same input, same out    Probabilistic — LLMs vary
   Pass/fail boolean                       Scores on a spectrum
   Fast (milliseconds)                     Slow (seconds per question)
   Run on every commit                     Run before merges / nightly
   Written when feature is built           Curated over the lifetime of the system

   Example: assert add(2, 3) == 5          Example: when asked about cream
                                                    in carbonara, the bot
                                                    must cite recipes.pdf p.4
                                                    AND mention "not authentic"
```

You need both. Unit tests prove the chunker doesn't crash on empty
pages. Eval proves the bot still answers carbonara correctly after you
changed the chunker.

---

## The dataset is the asset

This is the most counter-intuitive thing about evals:

> The eval harness *code* is plumbing.
> The eval *dataset* is the real product.

You can rewrite the harness in an afternoon. Building a dataset of 50
genuinely useful (question, expected answer) tuples takes weeks of
careful work — and once you have it, it's *worth more than your code*.

### What makes a good eval question

```
   GOOD                                  BAD
   ────                                  ────
   "Does carbonara use cream?"           "Tell me about the recipes"
       ► tests negation handling             ► too vague, no clear pass/fail
       ► single doc, single page

   "Which doc says spicy and which       "What is in carbonara?"
    says mild for the same dish?"            ► doesn't test anything specific
       ► tests multi-doc conflict             ► answer is hard to score
       ► forces the bot to name sources

   "What's the address of HQ?"           "Who is the founder?"
       ► answer is "not in the docs"          ► may or may not be in the docs;
       ► tests refusal behavior                  flaky test
```

Good questions usually fall into one of these categories:

| Category | What it tests |
|---|---|
| **Single-fact lookups** | Basic retrieval + correct citation |
| **Negation handling** | "Does X include Y?" where the answer is no |
| **Multi-doc conflicts** | When two docs disagree, does the bot surface both? |
| **Out-of-corpus** | Questions whose answer is genuinely not in the PDFs — the bot should say "I don't know", not invent |
| **Follow-ups** | Test conversational memory: "what about its price?" after a question about a product |
| **Synonyms / paraphrasing** | The doc says "vehicle"; you ask about "car" — tests dense retrieval |
| **Exact identifiers** | "What does section 4.2.1 say?" — tests sparse retrieval |
| **Multi-part questions** | "Compare X and Y" — tests synthesis across chunks |

**Rule of thumb:** 10 carefully chosen questions beat 100 mediocre ones.
A small dataset that hits every category above is more diagnostic than
500 lookup-style questions.

### Bootstrapping the dataset (the hybrid approach)

Starting from zero is paralyzing. Our `seed.py` solves the cold start:

```
   ┌─────────────────────────┐
   │  for each PDF in data/  │
   └────────────┬────────────┘
                │
                ▼
   ┌─────────────────────────────┐
   │  LLM proposes Q/A pairs     │  ◄── strict JSON output
   │  with page + keywords       │
   └────────────┬────────────────┘
                │
                ▼
   ┌─────────────────────────────┐
   │  writes eval/dataset.yaml   │  ◄── auto-seeded stub
   │  (NOT yet trustworthy)      │
   └────────────┬────────────────┘
                │
                ▼
   ┌─────────────────────────────┐
   │  YOU hand-edit:             │  ◄── this is the real work
   │  - delete bad questions     │
   │  - fix wrong answers/pages  │
   │  - add edge cases LLM       │
   │    wouldn't think of        │
   └─────────────────────────────┘
```

Without the human pass, your "eval" measures how well the bot
reproduces *the LLM's own guesses* — circular and worthless.

---

## The four metrics, and which problem each catches

Different metrics catch failures at different stages of the pipeline.
This is the single most important design insight in evaluation.

```
                   user question
                          │
                          ▼
                  RETRIEVAL pipeline    ◄── if this stage fails, the right
                  (dense + sparse +         chunks never reach the LLM
                   RRF + rerank)
                          │
                          ▼
                  retrieved chunks      ◄── retrieval_recall measures HERE
                          │
                          ▼
                  GENERATION (LLM)      ◄── if this stage fails, the LLM
                  using the chunks          had the right info but bungled
                          │                 the response
                          ▼
                  answer + citations    ◄── citation_match, keyword_coverage,
                                            llm_judge measure HERE
```

So when a question fails, the metric pattern tells you *where to look*:

| `retrieval_recall` | `citation_match` | `keyword_coverage` | What's broken |
|---|---|---|---|
| ✗ low | ✗ low | ✗ low | **Retrieval is broken** — fix chunking, hybrid mix, embeddings. The LLM can't cite what it didn't see. |
| ✓ high | ✗ low | ✓ high | **LLM ignored the citation instruction** — fix the system prompt. Right info, sloppy attribution. |
| ✓ high | ✓ high | ✗ low | **LLM is paraphrasing too loosely** — turn temperature down, tighten the prompt. Cited correctly but missed expected facts. |
| ✓ high | ✓ high | ✓ high | **Likely a true pass** — sanity-check with `llm_judge`. |

Without per-stage metrics you'd just see "this question failed" and
have no idea what to fix. **The four-metric design is doing diagnosis,
not just scoring.**

### A closer look at each

#### 1. `retrieval_recall` (deterministic, free)

Did the expected `(doc, page)` end up in the chunks the LLM saw?

```python
hits = [c for c in top_children
        if c.doc_name == expected_doc
        and (expected_page is None or c.page == expected_page)]
passed = len(hits) > 0
```

The "before generation" metric. If this fails, nothing the LLM does
can save you — you fed it the wrong information.

#### 2. `citation_match` (deterministic, free)

Did the bot's answer actually cite the right `(doc.pdf p. N)`?

```python
cites = re.findall(r"\(\s*([^()\s]+\.pdf)\s+p\.?\s*(\d+)\s*\)", answer)
```

Tests *faithful attribution*. The bot might generate a perfect answer
but cite the wrong page — that's a faithfulness bug that scares users.

#### 3. `keyword_coverage` (deterministic, cheap)

Of the expected key terms, how many appear in the answer?

```python
matched = [kw for kw in expected_keywords if kw.lower() in answer.lower()]
score = len(matched) / len(expected_keywords)
```

A blunt instrument but surprisingly effective. If you say the answer
*must* mention "not authentic" and "cream", and it doesn't, the bot is
giving a shallow or evasive answer.

**Trap:** keyword matching is brittle. If the dataset says `"PCI"` and
the answer says `"percutaneous coronary intervention"`, you score 0.
Mitigation: list both in `expected_keywords` and tune your threshold.

#### 4. `llm_judge` (probabilistic, costs API tokens)

Ask a separate LLM: "Score this answer 1–5 against the expected
behavior." Catches subtle wrongness that keyword checks miss.

**The biases you should know about:**

| Bias | What it looks like |
|---|---|
| Position bias | If you show "candidate A" then "candidate B", LLMs tend to rate B higher. Randomize order in pairwise comparisons. |
| Length bias | Longer answers are perceived as more thorough, even when wrong. |
| Self-preference | An LLM judging answers from the same model family rates them higher. Use a *different* model for judging when possible. |
| Sycophancy | LLMs are trained to be agreeable — they may rate answers higher than warranted. |

Treat `llm_judge` as a **sanity check, not a primary metric**. Pin a
human label on a sample of judged answers periodically to calibrate.

---

## The development feedback loop

Once you have a working eval, your daily flow changes:

```
   BEFORE EVAL                             AFTER EVAL
   ───────────                             ──────────

   1. Have an idea                         1. Have an idea
   2. Code it                              2. Code it
   3. Try 2 questions in chat              3. Run eval.py (baseline → new)
   4. "Feels good, ship it"                4. Look at per-metric diff:
   5. ... discover regression 2 weeks         "recall +12%, keyword -3%"
      later from a user report             5. Decide: keep, revert, or tune
                                           6. Add the regression question to
                                              the dataset so it never breaks
                                              silently again
```

### A/B comparisons

This is what the `--config KEY=VALUE` flag on `eval.py` is for:

```bash
python eval.py                                 # baseline
python eval.py --config USE_HYDE=true          # HyDE on
python eval.py --config TOP_K=8                # more context
python eval.py --config CHILD_SIZE=400         # bigger chunks
```

Aggregate scores from each run are directly comparable. You can keep
a running notebook:

```
   Baseline         recall=0.62  cite=0.71  kw=0.55  judge=3.8
   + HyDE           recall=0.78  cite=0.74  kw=0.61  judge=3.9   ← wins
   + TOP_K=8        recall=0.79  cite=0.65  kw=0.60  judge=3.7   ← cite dropped
   + CHILD_SIZE=400 recall=0.59  cite=0.69  kw=0.52  judge=3.5   ← regression
```

No vibes. No narrative bias. Each row is a measurable claim.

---

## Why this matters at production scale

A single-PDF demo can survive without evals. A real product cannot.
Here's where evals earn their keep in production:

### 1. Pre-merge regression gates

```
   developer pushes a PR
            │
            ▼
   CI runs eval.py against the golden dataset
            │
   ┌────────┴────────┐
   ▼                 ▼
   metrics improved  metrics dropped >5%
   ► merge OK        ► merge BLOCKED, needs human review
```

Without this, a sloppy prompt edit can quietly tank quality. Users
notice within a day. With this, the regression is caught in 90 seconds.

### 2. Triage and root-cause for production failures

```
   user reports: "the bot keeps citing wrong pages on legal docs"
                                 │
                                 ▼
                Can you reproduce in the eval suite?
                                 │
                  ┌──────────────┴──────────────┐
                  ▼                             ▼
              YES                             NO
   ► the metric pattern already       ► add this question to the dataset
     told you which stage broke         ► reproduce, fix, re-run eval
   ► fix, re-run eval, ship             ► permanent regression test
```

The dataset *grows* in response to every customer issue. Over a year,
a production RAG system's dataset becomes a precise map of the failure
modes that actually matter to users.

### 3. Model and provider migration

When a new model drops (Llama 4, Sonnet 4.7, GPT-5), the question
"should we switch?" used to be guesswork. With an eval:

```
   eval --config GROQ_MODEL=llama-3.3-70b-versatile   ◄ baseline
   eval --config GROQ_MODEL=llama-4-70b               ◄ candidate
   eval --config GROQ_MODEL=mixtral-8x22b             ◄ alternative
```

Now you have an objective basis for the switch — and a record of *why*
you made it that survives staff turnover.

### 4. Hyperparameter tuning

Production RAG has dozens of knobs: chunk sizes, k values, rerank depth,
temperature, embedding model, reranker model. Tuning by intuition is a
losing battle.

```
   for child_size in [120, 240, 360, 480]:
       for top_k in [3, 4, 6]:
           run eval, record metrics
   ► pick the (child_size, top_k) pair with the best aggregate score
```

This is just grid search, but it's only possible *because* you have
numbers to compare.

### 5. Long-running quality dashboards

In serious deployments, the eval suite runs nightly and posts to a
dashboard:

```
   recall@4        ─►  78% (last week 76%) ▲
   citation match  ─►  91% (last week 92%) ▼ slight drift
   judge avg       ─►  4.1 (last week 4.0) ▲
   p95 latency     ─►  1.4s (last week 1.3s) ▼
```

Drift detection. If quality slowly degrades over weeks (e.g. because
PDFs were added that the chunker handles poorly), you see it before
users do.

### 6. Distributional vs. golden eval

There are two complementary datasets in mature production setups:

| Type | What it is | Used for |
|---|---|---|
| **Golden** | Small (50–200), hand-curated, covers edge cases and known regressions | Merge gates, regression detection. High signal per question. |
| **Distributional** | Large (thousands), sampled from real user queries (with PII scrubbed) | Tracks actual deployment quality. Catches issues the golden set misses because the team didn't anticipate them. |

You start with golden (which is what we built). You graduate to
distributional once you have real user traffic.

### 7. Online evals — the next horizon

Everything so far is *offline eval* (run on a fixed dataset). Some
high-stakes systems also run *online eval*: every Nth production
response gets shadow-evaluated in real time. Drifts trigger pages.
This is heavy infrastructure but it's how the most reliable
LLM-powered systems operate.

---

## Common beginner traps

| Trap | What goes wrong | The fix |
|---|---|---|
| **Treating LLM-judge as ground truth** | The judge has its own biases — circular evaluation. | Pin human labels periodically. Combine judge with deterministic metrics. |
| **Only end-to-end metrics** | "The bot failed" — but at which stage? You can't fix what you can't isolate. | Always include per-stage metrics (`retrieval_recall`). |
| **Letting the dataset stagnate** | Bot gets really good at the eval questions, terrible at real ones. | Add every production failure to the dataset. Refresh quarterly. |
| **Optimizing on the wrong thing** | You drive `keyword_coverage` up by stuffing keywords into the prompt — recall and citation tank. | Watch *all* metrics in A/B comparisons. A 10% gain in one with a 15% loss in another is a regression. |
| **No regression baseline** | You change ten things at once. Eval drops 5%. Which change caused it? | A/B one variable at a time when possible. Keep a results notebook. |
| **Tiny dataset, big claims** | "Recall went from 65% to 75%" on 8 questions. That's noise. | Confidence intervals matter. Aim for 50+ questions. Bootstrap-resample to get error bars. |
| **Eval set leaks into training** | If you ever fine-tune an embedder or model, NEVER include eval questions in the training set. The model will memorize them. | Keep eval data strictly separated. Mark files clearly. |
| **Skipping the human review step on seeded data** | LLM-generated stubs answer their own questions — circular and worthless. | Hand-edit. Always. |

---

## A complete mental model

```
   ┌───────────────────────────────────────────────────────────────┐
   │   Curated dataset (the gold) + per-stage metrics             │
   │   = the only honest way to answer "did this change help?"    │
   └───────────────────────────────────────────────────────────────┘
                                │
       ┌────────────────────────┼────────────────────────┐
       ▼                        ▼                        ▼
   pre-merge gates       triage production         A/B everything
   stop regressions      failures into             — HyDE, models,
                         the dataset               chunk sizes, prompts


   The dataset grows over time. Old failures never come back silently.
   The harness becomes the institutional memory of "what we know about
   how this bot should behave."
```

This is what separates a hobby RAG from a production RAG. Not better
chunking. Not fancier reranking. Not GraphRAG. **The discipline of
measuring before you ship.**

---

## Aliases & related concepts

| Name | Notes |
|---|---|
| **Eval / Evals** | Generic term, especially in LLM-land. |
| **Golden dataset / Test set** | The curated ground truth. "Golden" emphasizes carefully chosen, "test set" emphasizes never-trained-on. |
| **Regression suite** | The eval set, viewed as a tool for catching backsliding. |
| **LLM-as-judge** | The 4th metric pattern. Whole sub-field with its own quirks (DeepEval, RAGAS, Promptfoo). |
| **RAGAS** | A popular open-source library of RAG-specific metrics: faithfulness, answer relevancy, context precision/recall. Worth reading their definitions even if you don't use the library. |
| **DeepEval** | Another eval framework; pytest-style integration. |
| **Promptfoo** | A YAML-driven eval runner, popular for prompt iteration. |
| **MMLU / TruthfulQA / etc.** | Academic eval datasets — useful for *model selection*, not for evaluating *your specific RAG*. |
| **Online eval / shadow eval** | Running evals in real time against production traffic. |
| **A/B testing in production** | The grown-up version of `--config` flags. Route X% of traffic to the new config, compare. |

---

## Key concepts this teaches

- **Eval ≠ tests.** Tests check code; eval checks behavior on real-world inputs against curated expectations.
- **The dataset is the asset.** Code is replaceable in an afternoon. A well-curated 50-question dataset is worth weeks.
- **Per-stage metrics enable diagnosis.** End-to-end metrics tell you something broke; per-stage tells you *where*.
- **LLM-as-judge is useful but biased.** Pair it with deterministic metrics. Never trust it alone.
- **A/B everything you change.** Once you have an eval, every claim ("this is better") becomes falsifiable.
- **The dataset grows.** Every production failure becomes a permanent regression test.
- **Eval discipline is the dividing line.** It's what turns a working demo into a system you can operate.

→ Back to the [index](README.md).
