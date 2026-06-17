# Issue 02 — Citation regex too strict; missed valid citations

**Date encountered:** 2026-06-18
**Component:** `PDFchat-app/eval/metrics.py::_CITE_RE`
**Severity:** Eval blindspot — eval falsely reported `citation_match` failures even when the bot was citing correctly.
**Caught by:** While verifying the fix for Issue 01 (rewriter bug). Classic "fix one thing, expose another."

---

## TL;DR

The citation-match metric used a regex that required both the document
name AND page number to be inside the same parentheses:
`(doc.pdf p. 1)`.

In reality, the answer model (`llama-3.3-70b-versatile`) consistently
emits citations with the document name *outside* the parens:
`doc.pdf (p. 1)`.

Result: the metric scored 0/3 on perfectly valid citations. We were
penalizing the bot for the eval harness's pattern-matching error.

---

## The mismatch

```
   We told the model:   "Cite using (doc.pdf p. N)"
                                  ↑   ↑
                       both pieces inside the parens

   The model actually emits:   doc.pdf (p. N)
                                ↑       ↑
                            doc OUTSIDE, page INSIDE

   Our regex required:   r"\(\s*([^()\s]+\.pdf)\s+p\.?\s*(\d+)\s*\)"
                          ↑                                       ↑
                          MUST start with (                       MUST end with )
                          MUST contain doc + page between them
```

Verbatim bot output captured during diagnosis:

```
It depends on the source. According to american_kitchen.pdf (p. 1), yes,
the American-style carbonara uses heavy cream... However, italian_classics.pdf
(p. 1) states that authentic Italian carbonara does not use cream...
```

Three perfectly valid citations in one answer; the regex found zero.

---

## Root cause

System prompts are *suggestions* to LLMs, not contracts. The system
prompt asked for a specific format but the model has its own training-data
priors about what looks "natural" — and `doc.pdf (p. 1)` is the more
common pattern in real-world academic writing. The model deferred to
its own style instead of obeying the prompt literally.

This is a general failure mode: **never trust the model to produce a
single rigid output format unless you constrain it structurally** (JSON
mode, tool calls, grammars). For looser formats like prose with
citations, the eval has to be permissive.

---

## The fix

Loosen the regex to match the common variants the model is likely to
produce, while still rejecting unrelated `pdf` mentions and stray page
numbers far apart in the text.

**Before:**
```python
_CITE_RE = re.compile(
    r"\(\s*([^()\s]+\.pdf)\s+p\.?\s*(\d+)\s*\)",
    re.IGNORECASE,
)
```

**After:**
```python
_CITE_RE = re.compile(
    r"([\w\-]+\.pdf)"     # doc name
    r"[\s,()]{0,4}"       # up to 4 chars of separator (space, comma, parens)
    r"p\.?\s*"            # 'p' or 'p.'
    r"(\d+)",             # page number
    re.IGNORECASE,
)
```

Accepted formats now:

| Format | Matches? |
|---|---|
| `(doc.pdf p. 1)` (original spec) | ✓ |
| `doc.pdf (p. 1)` (what the model actually emits) | ✓ |
| `doc.pdf, p. 1` | ✓ |
| `doc.pdf p. 1` (bare) | ✓ |
| `doc.pdf p.1` (no space after `p.`) | ✓ |
| `doc.pdf is great because... p. 1 of the manual` (false positive guard) | ✗ — separator > 4 chars |

---

## Empirical result — eval went RED then GREEN

### BEFORE the regex fix

```
[1] PASS
     ✓ citation_match   1.00  cited italian_classics.pdf p.1
[2] FAIL
     ✗ citation_match   0.00  no (doc.pdf p. N) citations found
[3] FAIL
     ✗ citation_match   0.00  no (doc.pdf p. N) citations found

AGGREGATE
  citation_match     avg=0.33  passed 1/3
```

Case 1 happened to pass because the bot used the strict format that
time. Cases 2 and 3 used the loose format — perfectly valid — and the
regex rejected them. Pure metric error.

### AFTER the regex fix

```
[1] PASS
     ✓ citation_match   1.00  cited italian_classics.pdf p.1
[2] FAIL
     ✓ citation_match   1.00  cited italian_classics.pdf p.1
[3] PASS
     ✓ citation_match   1.00  cited american_kitchen.pdf p.1

AGGREGATE
  citation_match     avg=1.00  passed 3/3
```

Case 3 went from overall FAIL to overall PASS — its only blocker
*was* the bogus citation_match score.

Case 2 still fails *overall*, but now for a real reason: `keyword_coverage`
is 0.00 because my dataset asked for the exact strings `"no cream"` and
`"authentic"`, and the bot used synonyms (`"does not use cream"`, `"traditional"`).
That's a dataset hygiene issue, not a code bug — different fix.

---

## Lessons learned

1. **Brittle regex is a leaky eval.** Strict patterns produce false
   negatives that look like real failures. Always inspect what the
   actual output looks like before writing the matcher.

2. **System prompts cannot enforce output format on their own.** If you
   need a specific structure, use structured outputs (JSON mode, tool
   calls, grammars) or be permissive in your matcher.

3. **Fixing one bug routinely surfaces another.** Issue 01 used the
   eval to verify a fix. The eval was wrong, but in a way that was only
   visible *after* Issue 01 was fixed (because the rewriter bug was
   dominating the failures). Compound debugging.

4. **Eval bugs and code bugs feel the same on the dashboard.** Both
   show up as low scores. Diagnose every red metric, every time. Some
   are real, some are infrastructure.

5. **`keyword_coverage` is itself brittle.** Case 2 still fails because
   `"no cream"` and `"does not use cream"` are scored as different —
   meaningfully the same, mechanically different. Future fix: either
   expand the keyword lists in the dataset to include synonyms, or use
   the `llm_judge` metric as the primary signal for semantic-match
   cases.

---

## Status

- [x] Bug identified
- [x] Regex loosened to accept all common citation formats
- [x] Re-ran eval — `citation_match` went 0.33 → 1.00
- [x] Documented (this file)
- [ ] Follow-up: `keyword_coverage` dataset hygiene — expand keyword
      lists to include synonyms (filed for future)
