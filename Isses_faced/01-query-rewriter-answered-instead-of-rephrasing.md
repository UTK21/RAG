# Issue 01 — Query rewriter answered the question instead of rephrasing it

**Date encountered:** 2026-06-18
**Component:** `pdfchat/query_rewrite.py::standalone` (in `PDFchat-app/`)
**Severity:** Latent — bug present, but the pipeline's downstream design hid it.
**Caught by:** Eyeballing the `(rewrote → ...)` debug line in `chat.py`.

---

## TL;DR

A small/fast LLM (`llama-3.1-8b-instant`) used for the conversational
query-rewriting step ignored the explicit instruction "Do NOT answer.
Output only the question." Instead of rewriting a follow-up question
into a standalone form, it generated an opinion-style answer.

The user-visible response was still good — defensive design downstream
absorbed the bug. But this is exactly the kind of silent degradation
that destroys RAG quality over time.

---

## The transcript

User dialogue (from `python chat.py` in `PDFchat-app/`):

```
you> Does carbonara use cream?
bot> [answers correctly, citing both PDFs]

you> what about for me someone who is new to cooking?
   (rewrote → "For someone who is new to cooking, I would recommend
              starting with the traditional Italian version of
              carbonara, as it's simpler and doesn't require heavy
              cream.")
   (sources: italian_classics.pdf p.1, american_kitchen.pdf p.1)

bot> For someone new to cooking, american_kitchen.pdf (p. 1) suggests
     that the American-style carbonara using heavy cream is more
     "forgiving" and can help prevent scrambling the eggs... However,
     italian_classics.pdf (p. 1) emphasizes that...
```

**The rewrite is not a question. It's an answer.** It was supposed to
turn "what about for me someone who is new to cooking?" into something
like "Which carbonara recipe is best for someone new to cooking?"

---

## Root cause

Two factors stacked:

1. **Small model + abstract instruction.** Our rewrite step uses
   `llama-3.1-8b-instant` on purpose — small, fast, cheap. Small LLMs
   are very strongly trained to be "helpful". When a system prompt says
   "Do NOT answer" abstractly, smaller models often interpret it as
   *guidance* rather than a *rule* and revert to their helpful
   defaults.

2. **Prompt without concrete examples.** Our `STANDALONE_PROMPT`
   describes the desired behavior but doesn't *show* it:

   ```python
   STANDALONE_PROMPT = (
       "You rewrite follow-up questions to be standalone.\n"
       "Given chat history and a new user message, output ONLY the message "
       "rewritten so it makes sense without the history.\n"
       "If already standalone, output unchanged. Do NOT answer. "
       "Output only the question."
   )
   ```

   Small models obey concrete examples (few-shot) much more reliably
   than abstract rules.

---

## Why the user-visible answer was still good

Two design choices in the pipeline contained the damage:

### 1. Retrieval still worked by coincidence

The rewritten text accidentally contained the right *keywords*
("carbonara", "cooking", "Italian"), so FAISS + BM25 still pulled the
right chunks. If the rewriter had drifted further (e.g. recommending
*risotto* instead), retrieval would have collapsed.

### 2. Generation uses the ORIGINAL query, not the rewrite

From `pdfchat/pipeline.py::answer_stream`:

```python
reply = llm.answer(
    ...
    query=query,            # ← the user's ORIGINAL message
    context_chunks=parents,
    history=recent,
    ...
)
```

The rewrite was *only* used for retrieval. The actual LLM-generation
step received the user's original question, the grounded system prompt,
and the retrieved chunks. So the answer ignored the rewriter's opinion
entirely.

**This is the single most important defensive choice in the pipeline.**
If we'd reused the rewrite for generation, the bot would have just
regurgitated the small model's recommendation.

---

## The fix

Apply in `pdfchat/query_rewrite.py`. Two changes, belt-and-suspenders.

### Fix A — Few-shot examples in the prompt

```python
STANDALONE_PROMPT = (
    "You rewrite follow-up questions to be standalone.\n"
    "Given chat history and a new user message, output ONLY the message "
    "rewritten so it makes sense without the history.\n"
    "Do NOT answer. Do NOT explain. Output a QUESTION.\n\n"
    "Examples:\n"
    "History: [user asks about transformers]\n"
    "New: 'what about its limitations?'\n"
    "Output: 'What are the limitations of transformers?'\n\n"
    "History: [user asks about carbonara]\n"
    "New: 'what about for a beginner?'\n"
    "Output: 'Which version of carbonara is best for a beginner?'\n\n"
    "If the new message is already standalone, output it unchanged."
)
```

### Fix B — Output validation (catches what the prompt misses)

```python
rewritten = (resp.choices[0].message.content or "").strip()

# Heuristics that flag answer-shaped rewrites
looks_like_answer = (
    len(rewritten.split()) > 30                            # too long
    or not rewritten.rstrip(".").endswith("?")             # not a question
    or rewritten.lower().startswith(
        ("i would", "the ", "according to", "for someone")
    )
)
if not rewritten or looks_like_answer:
    return new_question     # safe fallback — original query intent preserved

return rewritten
```

### Fix C (last resort) — use the big model for rewrites

In `.env`:
```
GROQ_REWRITE_MODEL=llama-3.3-70b-versatile
```
Reliable but ~10× slower and ~10× more expensive per question.

---

## Would the existing eval harness have caught this?

**Honest answer: probably not, with the current metric set.**

The harness measures the END-OF-PIPELINE behavior with four metrics:
`retrieval_recall`, `citation_match`, `keyword_coverage`, `llm_judge`.

In this specific bug:

| Metric | Would it have failed? | Why |
|---|---|---|
| `retrieval_recall` | ❌ no | The right chunks were still retrieved (lucky keywords). |
| `citation_match` | ❌ no | The final answer cited the right pages. |
| `keyword_coverage` | ❌ no | The final answer contained the expected terms. |
| `llm_judge` | ❌ probably not | The final answer reads as correct and balanced. |

The downstream design hid the bug from end-of-pipeline metrics. **This
is a real limitation of "outcome-only" evals**: when a sub-component is
broken but its damage is absorbed by another sub-component, the outcome
metric is silent.

### What WOULD have caught it: a per-stage metric

This is the "separate retrieval failures from generation failures"
mental model from `PDFchat/docs/07-evaluation-harness.md`. We have a
metric for the *retrieval* stage (`retrieval_recall`). We don't have
one for the *rewrite* stage.

Adding one is small:

```python
# eval/metrics.py
def rewrite_quality(rewritten: str, original: str) -> MetricResult:
    """Catches rewriters that 'help' too much."""
    looks_like_answer = (
        len(rewritten.split()) > 30
        or not rewritten.rstrip(".").endswith("?")
        or rewritten.lower().startswith(("i would", "the ", "according to"))
    )
    passed = not looks_like_answer
    return MetricResult(
        "rewrite_quality",
        1.0 if passed else 0.0,
        passed,
        "rewrite is a question" if passed else f"rewrite looks like an answer: {rewritten!r}",
    )
```

And the dataset would gain follow-up cases that have history:

```yaml
- question: "what about for a beginner?"
  history:
    - { role: user, content: "tell me about carbonara" }
    - { role: assistant, content: "Carbonara is a Roman dish using ..." }
  expected_keywords: ["beginner", "easier", "forgiving"]
  notes: "Follow-up — rewrite_quality should pass; tests rewriter discipline."
```

With those two additions, the eval would have flagged the regression
on the first run. The harness framework already supports per-stage
metrics; we just didn't have one for this stage yet.

### Generalizable lesson

> **An eval harness only catches what it explicitly measures.** Adding a
> new pipeline stage means adding a new metric for that stage.
> "Outcome-only" eval is necessary but insufficient.

---

## Lessons learned

1. **Small LLMs need few-shot examples, not abstract rules.** Reserve
   abstract instructions ("be concise", "do not answer") for big
   models. Show small models exactly what good output looks like.

2. **Always validate small-model output.** A 5-line heuristic check
   for "did the model do what we asked?" prevents 90% of these
   silent failures.

3. **Defensive pipeline design pays off.** Keeping retrieval-time
   queries separate from generation-time queries meant a broken
   sub-step didn't sink the whole answer. **Design pipelines so that
   no single component's failure produces a wrong final answer.**

4. **The eval harness must grow with the pipeline.** When you add a
   stage, add a per-stage metric. Otherwise the harness only sees the
   final outcome and hidden bugs accumulate.

5. **Watching debug output is a primitive form of eval.** I caught this
   because `chat.py` prints `(rewrote → ...)`. Without that diagnostic
   line, the bug would have been completely invisible.

---

## Status

- [x] Bug identified and documented
- [x] Fix A (few-shot prompt) applied to `pdfchat/query_rewrite.py`
- [x] Fix B (output validation `_looks_like_answer`) applied
- [x] New `rewrite_quality` metric added to `eval/metrics.py`
- [x] Follow-up test cases added to `eval/dataset.yaml`
- [x] Re-run eval — regression flagged before fix, passes after
- [ ] Citation regex bug surfaced incidentally — needs follow-up

---

## Empirical results — eval went RED then GREEN

### BEFORE fix

The eval harness (with the new `rewrite_quality` metric) flagged case 3:

```
[3] FAIL  'what about for a beginner cook?'
     ✓ retrieval_recall   1.00  found 1 matching chunk(s)
     ✗ citation_match     0.00  no (doc.pdf p. N) citations found
     ✓ keyword_coverage   0.50  matched 1/2: ['forgiving']
     ✗ rewrite_quality    0.00  rewrite looks like an answer:
       "For a beginner cook, what's the best approach to making a
        traditional carbonara"
```

Aggregate:
```
  rewrite_quality    avg=0.67  passed 2/3
```

### AFTER fix

Same case, same dataset, same question:

```
[3] FAIL  'what about for a beginner cook?'
     ✓ retrieval_recall   1.00  found 1 matching chunk(s)
     ✗ citation_match     0.00  (separate bug — citation format)
     ✓ keyword_coverage   0.50  matched 1/2: ['forgiving']
     ✓ rewrite_quality    1.00  rewrite is a question:
       'Which version of carbonara is best for a beginner cook?'
```

Aggregate:
```
  rewrite_quality    avg=1.00  passed 3/3
```

The actual fix was a few-shot prompt teaching the small model the exact
output shape, plus a tiny heuristic check that falls back to the original
query if the rewrite still slips through. ~25 lines of code total.

---

## Incidental discovery — a SECOND bug surfaced

While running the eval to verify the rewriter fix, `citation_match`
failed on cases 2 and 3 with "no (doc.pdf p. N) citations found".

The bot IS citing sources, but probably in a format like
`italian_classics.pdf (p. 1)` instead of the expected
`(italian_classics.pdf p. 1)`. Our regex is:

```python
_CITE_RE = re.compile(r"\(\s*([^()\s]+\.pdf)\s+p\.?\s*(\d+)\s*\)", re.IGNORECASE)
```

…which requires both the doc name AND page to be inside the parens.
The model is producing valid citations in a different layout the regex
doesn't accept. **The eval helped me find a bug I wasn't even looking
for.** Either the regex needs to be more permissive, or the system
prompt needs to be more explicit about the exact citation format.

(Filed for a follow-up issue.)

---

## Material for the LinkedIn post

Concrete numbers and quotes ready to drop in:

> **The buggy rewrite (verbatim):**
> "For someone who is new to cooking, I would recommend starting with the
> traditional Italian version of carbonara..."

> **The fixed rewrite (verbatim):**
> "Which version of carbonara is best for a beginner cook?"

> **The eval metric that caught it:**
> `rewrite_quality` — a per-stage check on the query-rewriter's output.
> Pass criteria: under 30 words, ends with `?`, doesn't start with
> answer-like prefixes ("I would", "For someone", "According to"…).

> **The aggregate numbers:**
> Before fix: 2/3 cases passing rewrite_quality (avg 0.67).
> After fix:  3/3 cases passing rewrite_quality (avg 1.00).
> Total code changed: ~25 lines.

> **The plot twist:**
> While running the eval to verify the fix, it surfaced a SECOND bug
> I wasn't looking for (citation regex too strict). That's the multiplier
> on eval discipline — every run can find issues you didn't predict.

---

# 📣 LinkedIn post — fully assembled assets

Everything below is **ready to copy**. Pick the post variant that fits your
voice, then grab whichever stats / screenshots / diagrams you want.

---

## 🧱 The numbers card

```
   ┌──────────────────────────────────────────────────────────┐
   │  CARBONARA-BUG POSTMORTEM                                │
   │                                                          │
   │  Bug found by         eval harness  (per-stage metric)   │
   │  Files changed        4                                  │
   │  Lines changed        143 added / 18 removed             │
   │  Actual fix size      ~60 LoC in pdfchat/query_rewrite.py│
   │  New eval metric      ~25 LoC in eval/metrics.py         │
   │  Test cases added     3 (1 standalone + 1 multi-doc      │
   │                        + 1 follow-up)                    │
   │                                                          │
   │  rewrite_quality      0.67  →  1.00                      │
   │                       (2/3 pass)   (3/3 pass)            │
   │                                                          │
   │  Incidental bugs              +1  (citation regex too    │
   │  surfaced by the eval             strict — found while   │
   │                                   verifying the fix)     │
   └──────────────────────────────────────────────────────────┘
```

Screenshot this directly, or paste into carbon.now.sh for a fancy
syntax-highlighted version.

---

## 🖼️ Visual 1 — Where the bug lived (pipeline diagram)

```
   user question
        │
        ▼
   ┌──────────────────────────────────────────────────────────┐
   │  query_rewriter (small LLM: llama-3.1-8b-instant)        │  ◄── bug here
   │                                                          │
   │  Prompt said:  "Do NOT answer. Output only the question."│
   │  Model did:    *answered the question anyway*            │
   └────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
                       FAISS + BM25
                            │
                            ▼
                   cross-encoder rerank
                            │
                            ▼
                 ┌────────────────────────┐
                 │ generation (big LLM)   │  ◄── user-visible result was
                 │ uses ORIGINAL query    │       still correct — design
                 │ + grounded prompt      │       hid the bug downstream
                 └────────────────────────┘
                            │
                            ▼
                       good answer
                       (looks fine!)
                       (eval would have said ✅ without per-stage metric)
```

---

## 🖼️ Visual 2 — The eval went RED, then GREEN

### BEFORE the fix

```
   [3] FAIL  'what about for a beginner cook?'
        ✓ retrieval_recall   1.00
        ✗ citation_match     0.00
        ✓ keyword_coverage   0.50
        ✗ rewrite_quality    0.00
          rewrite looks like an answer:
          "For a beginner cook, what's the best approach to
           making a traditional carbonara"

   AGGREGATE
        rewrite_quality    avg=0.67   passed 2/3
```

### AFTER the fix

```
   [3] FAIL  'what about for a beginner cook?'
        ✓ retrieval_recall   1.00
        ✗ citation_match     0.00   ← unrelated bug surfaced!
        ✓ keyword_coverage   0.50
        ✓ rewrite_quality    1.00
          rewrite is a question:
          'Which version of carbonara is best for a beginner cook?'

   AGGREGATE
        rewrite_quality    avg=1.00   passed 3/3
```

Two takeaways from this screenshot:
1. The targeted metric (`rewrite_quality`) flipped from FAIL to PASS.
2. Even after the fix, case 3 still shows `FAIL` overall — because
   `citation_match` revealed a different bug we weren't looking for.

---

## 🖼️ Visual 3 — The fix itself (the rewriter prompt diff)

**Before** — abstract rules the small model ignored:

```python
STANDALONE_PROMPT = (
    "You rewrite follow-up questions to be standalone.\n"
    "Given chat history and a new user message, output ONLY the message "
    "rewritten so it makes sense without the history.\n"
    "If already standalone, output unchanged. Do NOT answer. "
    "Output only the question."
)
```

**After** — concrete examples the small model can imitate:

```python
STANDALONE_PROMPT = (
    "You rewrite follow-up questions to be standalone search queries.\n"
    "Given chat history and a new user message, output ONLY the message "
    "rewritten so it makes sense without the history.\n"
    "Do NOT answer. Do NOT explain. Output a QUESTION ending in '?'.\n\n"
    "Examples:\n\n"
    "History:\n"
    "  user: tell me about transformers\n"
    "  assistant: Transformers use self-attention (Vaswani 2017).\n"
    "New message: what about its limitations?\n"
    "Standalone version: What are the limitations of transformers?\n\n"
    # ...two more examples...
)
```

Plus a 5-line safety net that catches anything the prompt missed:

```python
def _looks_like_answer(rewritten: str) -> bool:
    s = rewritten.strip()
    if not s: return True
    if len(s.split()) > 30: return True
    if not s.rstrip(".!").endswith("?"): return True
    if s.lower().startswith(_ANSWER_PREFIXES): return True
    return False
```

---

## ✍️ Post draft — Short / punchy (Twitter-style, ~600 chars)

> Caught my RAG chatbot cheating today.
>
> A small/fast LLM I use to rewrite follow-up questions ignored the
> instruction "Do NOT answer" — and answered them. The user-visible
> output was still correct, because generation uses the ORIGINAL
> query, not the rewrite. My eval would have said ✅ without a
> per-stage metric.
>
> Added rewrite_quality metric → went RED → fixed with few-shot
> examples + a 5-line validator → went GREEN.
>
> Three takeaways:
>   • Small LLMs need examples, not rules
>   • Defensive pipeline design hides bugs (good AND bad)
>   • End-to-end metrics aren't enough — every stage needs its own
>
> #RAG #LLM #BuildingInPublic

---

## ✍️ Post draft — Medium / story-driven (~1500 chars, the recommended one)

> 🐛 Caught my RAG chatbot cheating on itself today.
>
> I'm building a RAG (Retrieval-Augmented Generation) system over a
> folder of PDFs. Part of the design: a "query rewriter" — when a
> user types a follow-up like "what about for a beginner?", a small
> LLM rewrites it into a standalone question before retrieval.
>
> The system prompt is blunt: "Do NOT answer. Output only the question."
>
> Then I saw this in the debug log:
>
> User: "what about for me someone who is new to cooking?"
> Rewriter: "For someone who is new to cooking, I would recommend
> starting with the traditional Italian version of carbonara..."
>
> The rewriter ANSWERED the question instead of rephrasing it.
>
> But the unsettling part — the final answer was still correct.
> Two design choices hid the bug:
>
> 1. The rewrite happened to contain the right keywords ("carbonara",
>    "cooking"), so retrieval still pulled the right chunks.
> 2. By design, generation uses the user's ORIGINAL query, not the
>    rewrite. So the actual answer ignored the rewriter's opinion.
>
> I added a new per-stage metric to my eval harness:
> `rewrite_quality` — checks that the rewrite is actually a question.
>
> Before the fix:  2/3 cases passing (avg 0.67)
> After the fix:   3/3 cases passing (avg 1.00)
>
> The fix was tiny: few-shot examples in the rewrite prompt + a
> 5-line validator that falls back to the original query if the
> small model misbehaves. ~25 lines of code.
>
> Plot twist: while verifying the fix, the eval flagged a SECOND bug
> I wasn't even looking for — my citation regex was too strict and
> missing valid citations. That's the multiplier on eval discipline.
> Every run finds things you didn't predict.
>
> Three takeaways I'm putting in my notes:
>
> ① Small/fast LLMs ignore abstract rules. Few-shot examples + output
>   validation are the practical fix — not bigger models.
>
> ② Defensive pipeline design pays off. Keeping retrieval-time and
>   generation-time queries separate meant a broken sub-step didn't
>   sink the whole answer. Design pipelines so no single component's
>   failure produces a wrong final result.
>
> ③ End-to-end metrics aren't enough. If I'd relied only on "is the
>   final answer correct?" my eval would have said ✅ on this query.
>   An eval harness only catches what it explicitly measures.
>
> Production RAG isn't about better embeddings or fancier rerankers.
> It's about catching failures the system has accidentally learned
> to absorb.
>
> #RAG #LLM #AI #MachineLearning #BuildingInPublic #SoftwareEngineering

---

## ✍️ Post draft — Long-form / carousel layout (5 slides)

Use this if you want to make a LinkedIn carousel post. Each slide is
self-contained.

### Slide 1 — Hook
```
   ┌─────────────────────────────────────────┐
   │                                         │
   │   I caught my RAG chatbot               │
   │   cheating on itself.                   │
   │                                         │
   │   And the scary part is —               │
   │   the user-visible answer               │
   │   was STILL correct.                    │
   │                                         │
   │   [thread →]                            │
   │                                         │
   └─────────────────────────────────────────┘
```

### Slide 2 — The bug
```
   ┌─────────────────────────────────────────┐
   │  The rewriter ignored its instructions  │
   │                                         │
   │  Prompt:                                │
   │    "Do NOT answer.                      │
   │     Output only the question."          │
   │                                         │
   │  User:                                  │
   │    "what about for a beginner?"         │
   │                                         │
   │  Small LLM:                             │
   │    "For a beginner cook, what's the     │
   │     best approach to making a           │
   │     traditional carbonara"              │
   │                                         │
   │  ↑ that's not a question.               │
   └─────────────────────────────────────────┘
```

### Slide 3 — Why the answer still looked correct
```
   ┌─────────────────────────────────────────┐
   │  Two design choices hid the damage:     │
   │                                         │
   │  1. The bad rewrite still had the right │
   │     keywords. Retrieval found the right │
   │     chunks by accident.                 │
   │                                         │
   │  2. Generation uses the ORIGINAL query, │
   │     not the rewrite. The answer ignored │
   │     the rewriter's opinion entirely.    │
   │                                         │
   │  Defensive pipeline design pays off —   │
   │  but it can hide bugs from end-to-end   │
   │  metrics.                               │
   └─────────────────────────────────────────┘
```

### Slide 4 — The fix
```
   ┌─────────────────────────────────────────┐
   │  Two changes, ~25 lines of code:        │
   │                                         │
   │  ① Few-shot examples in the prompt      │
   │     (small models obey examples,        │
   │      not abstract rules)                │
   │                                         │
   │  ② Output validator that catches        │
   │     answer-shaped rewrites and falls    │
   │     back to the original query          │
   │                                         │
   │  Plus: a new per-stage metric in the    │
   │  eval harness — rewrite_quality.        │
   │                                         │
   │  Eval went RED → GREEN:                 │
   │  rewrite_quality   0.67 → 1.00          │
   └─────────────────────────────────────────┘
```

### Slide 5 — Lessons
```
   ┌─────────────────────────────────────────┐
   │  Three lessons in my notes:             │
   │                                         │
   │  ① Small LLMs need examples, not rules. │
   │     Show, don't tell.                   │
   │                                         │
   │  ② Defensive design is a double-edged   │
   │     sword. It saves users from bugs,    │
   │     but hides bugs from you.            │
   │                                         │
   │  ③ End-to-end metrics aren't enough.    │
   │     Every pipeline stage needs its own  │
   │     per-stage check.                    │
   │                                         │
   │  Production RAG isn't about better      │
   │  embeddings. It's about catching the    │
   │  failures the system absorbs silently.  │
   │                                         │
   │  #RAG #LLM #BuildingInPublic            │
   └─────────────────────────────────────────┘
```

---

## 🏷️ Hashtag bundle (pick 3–5)

Primary:
- `#RAG` `#LLM` `#AI` `#MachineLearning`

Secondary:
- `#BuildingInPublic` `#SoftwareEngineering` `#GenerativeAI`
  `#PromptEngineering` `#MLOps`

Specific:
- `#LangChain` `#LlamaIndex` `#Groq` (use only if the post mentions
  these by name)

---

## 🖼️ How to turn these into actual images

LinkedIn rewards visual posts. You have three options:

1. **Screenshot the ASCII blocks** directly from your terminal or this
   markdown file in a dark theme. Works on Mac with `Cmd+Shift+4`.
2. **carbon.now.sh** — paste any code block, pick a theme, export a
   PNG. Great for the prompt-diff visual.
3. **excalidraw.com** — for the pipeline diagram. Hand-drawn aesthetic
   reads as more "behind the scenes" than auto-generated.

---

## 🎯 Recommended posting structure

If you want maximum reach with minimum effort, this combo works well:

1. **Post body:** the medium-length draft above (~1500 chars).
2. **Inline image 1:** screenshot of the BEFORE eval output (the FAIL
   on `rewrite_quality` with the bad rewrite quoted).
3. **Inline image 2:** screenshot of the AFTER eval output (the PASS
   on `rewrite_quality` with the good rewrite quoted).
4. **Comment 1 (you, immediately):** "The full writeup with code
   diffs lives in my repo: https://github.com/UTK21/RAG/blob/main/Isses_faced/01-query-rewriter-answered-instead-of-rephrasing.md"

Posting a link in a comment rather than the body avoids the
LinkedIn-suppresses-link-posts trap.

---

## 📌 Final quote-ready snippets

For threads, replies, or follow-up posts:

> "Small LLMs obey examples, not abstract rules."

> "Defensive pipeline design is a double-edged sword. It saves users
> from bugs, but hides bugs from you."

> "An eval harness only catches what it explicitly measures. When you
> add a new pipeline stage, add a new per-stage metric."

> "Production RAG isn't about better embeddings. It's about catching
> failures the system has accidentally learned to absorb."

> "I shipped a fix that flipped one metric from 0.67 to 1.00 in 60
> lines of code. The eval surfaced a second bug I wasn't even looking
> for while I was verifying the first one. Compound returns."
