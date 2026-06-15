"""Four simple metrics for grading a RAG answer.

Each takes the expected example + the bot's actual response (text + retrieval
trace) and returns either a bool or a 0..1 score plus a short reason string.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from groq import Groq

from pdfchat.loader import Chunk, ParentChunk


@dataclass
class MetricResult:
    name: str
    score: float            # 0.0 to 1.0
    passed: bool            # threshold-based shortcut
    detail: str             # human-readable explanation


# ---------------------------------------------------------------------------
# 1. Retrieval recall@k
#    Did the expected (doc, page) end up in the candidates the LLM saw?
# ---------------------------------------------------------------------------

def retrieval_recall(
    expected_doc: str,
    expected_page: int | None,
    top_children: list[Chunk],
) -> MetricResult:
    hits = [
        c for c in top_children
        if c.doc_name == expected_doc
        and (expected_page is None or c.page == expected_page)
    ]
    passed = len(hits) > 0
    detail = (
        f"found {len(hits)} matching chunk(s) out of {len(top_children)} retrieved"
        if passed
        else f"expected {expected_doc} p.{expected_page} NOT in top {len(top_children)} retrieved"
    )
    return MetricResult("retrieval_recall", 1.0 if passed else 0.0, passed, detail)


# ---------------------------------------------------------------------------
# 2. Citation match
#    Does the bot's answer cite (doc.pdf p. N) for the right doc + page?
# ---------------------------------------------------------------------------

_CITE_RE = re.compile(r"\(\s*([^()\s]+\.pdf)\s+p\.?\s*(\d+)\s*\)", re.IGNORECASE)


def citation_match(
    expected_doc: str,
    expected_page: int | None,
    answer_text: str,
) -> MetricResult:
    cites = _CITE_RE.findall(answer_text)
    if not cites:
        return MetricResult("citation_match", 0.0, False, "no (doc.pdf p. N) citations found")

    for doc, page in cites:
        if doc.lower() == expected_doc.lower() and (
            expected_page is None or int(page) == expected_page
        ):
            return MetricResult(
                "citation_match", 1.0, True,
                f"cited {doc} p.{page}",
            )
    return MetricResult(
        "citation_match", 0.0, False,
        f"cited {cites!r} but expected {expected_doc} p.{expected_page}",
    )


# ---------------------------------------------------------------------------
# 3. Keyword coverage
#    What fraction of the expected key terms appear in the answer?
# ---------------------------------------------------------------------------

def keyword_coverage(
    expected_keywords: list[str],
    answer_text: str,
) -> MetricResult:
    if not expected_keywords:
        return MetricResult("keyword_coverage", 1.0, True, "no keywords specified")
    a = answer_text.lower()
    matched = [kw for kw in expected_keywords if kw.lower() in a]
    score = len(matched) / len(expected_keywords)
    passed = score >= 0.5
    return MetricResult(
        "keyword_coverage", score, passed,
        f"matched {len(matched)}/{len(expected_keywords)}: {matched}",
    )


# ---------------------------------------------------------------------------
# 4. LLM-judge score
#    Holistic 1..5 score from a separate LLM call. Noisier, costs API tokens.
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = (
    "You are evaluating a question-answering bot. Score the actual answer "
    "against the expected behavior on a 1-5 scale:\n"
    "  5 = correct, complete, faithful to the source\n"
    "  4 = mostly correct, minor omission\n"
    "  3 = partially correct\n"
    "  2 = misses the point or makes claims not in expected behavior\n"
    "  1 = wrong or hallucinated\n"
    "Output ONLY a single digit 1-5."
)


def llm_judge(
    client: Groq,
    model: str,
    question: str,
    expected_keywords: list[str],
    actual_answer: str,
) -> MetricResult:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _JUDGE_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n"
                    f"Expected to mention: {expected_keywords}\n"
                    f"Actual answer:\n{actual_answer}\n\n"
                    f"Score (1-5):"
                ),
            },
        ],
        temperature=0.0,
    )
    raw = (resp.choices[0].message.content or "").strip()
    digits = re.findall(r"[1-5]", raw)
    if not digits:
        return MetricResult("llm_judge", 0.0, False, f"judge returned non-digit: {raw!r}")
    n = int(digits[0])
    return MetricResult(
        "llm_judge", n / 5.0, n >= 4,
        f"score {n}/5",
    )
