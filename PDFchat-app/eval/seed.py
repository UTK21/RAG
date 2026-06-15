"""Bootstrap eval/dataset.yaml from your PDFs by asking the LLM to propose
Q&A pairs. Designed to give you a starting point — you MUST hand-edit the
output before trusting it.

Usage:
    python eval/seed.py             # writes eval/dataset.yaml (overwrites)
    python eval/seed.py --per-doc 3 # how many pairs per PDF
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from pdfchat.config import settings
from pdfchat.llm import make_client
from pdfchat.loader import load_pdf

SEED_PROMPT = (
    "You generate evaluation questions for a RAG (Retrieval-Augmented Generation) bot.\n"
    "Given a PDF excerpt, produce {n} diverse factual questions that someone might "
    "realistically ask. For EACH question, also provide:\n"
    "  - the page the answer comes from\n"
    "  - 2-4 keywords that the correct answer SHOULD contain\n"
    "  - a one-line note describing what's being tested\n\n"
    "Return STRICT JSON (no markdown, no commentary) as a list of objects with "
    "keys: question, expected_page, expected_keywords, notes."
)


def _excerpt(parents, max_chars: int = 6000) -> str:
    """Build a representative excerpt from the parents (truncated)."""
    parts: list[str] = []
    total = 0
    for p in parents:
        block = f"[p. {p.page}]\n{p.text}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts)


def seed_pdf(client, model: str, doc_name: str, parents, per_doc: int) -> list[dict]:
    excerpt = _excerpt(parents)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SEED_PROMPT.format(n=per_doc)},
            {"role": "user", "content": f"Document: {doc_name}\n\n{excerpt}"},
        ],
        temperature=0.4,
    )
    raw = (resp.choices[0].message.content or "").strip()
    # Be forgiving — strip code fences if present.
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  warning: could not parse JSON for {doc_name}; skipping", file=sys.stderr)
        return []

    out: list[dict] = []
    for it in items:
        out.append({
            "question": it.get("question", "").strip(),
            "expected_doc": doc_name,
            "expected_page": it.get("expected_page"),
            "expected_keywords": it.get("expected_keywords", []),
            "notes": it.get("notes", ""),
        })
    return out


def _yaml_dump(items: list[dict]) -> str:
    """Minimal hand-rolled YAML writer — keeps the file readable without
    depending on PyYAML's quoting quirks."""
    out: list[str] = []
    for it in items:
        out.append(f"- question: {json.dumps(it['question'])}")
        out.append(f"  expected_doc: {json.dumps(it['expected_doc'])}")
        page = it.get("expected_page")
        out.append(f"  expected_page: {page if page is not None else 'null'}")
        kws = it.get("expected_keywords", [])
        out.append("  expected_keywords:")
        for kw in kws:
            out.append(f"    - {json.dumps(kw)}")
        out.append(f"  notes: {json.dumps(it.get('notes', ''))}")
        out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-doc", type=int, default=3)
    parser.add_argument(
        "--out", default=os.path.join("eval", "dataset.yaml"),
        help="output YAML path",
    )
    args = parser.parse_args()

    if not settings.groq_api_key:
        print("GROQ_API_KEY missing.", file=sys.stderr)
        return 1
    if not os.path.isdir(settings.data_dir):
        print(f"No data dir: {settings.data_dir}", file=sys.stderr)
        return 1

    client = make_client(settings.groq_api_key)
    all_items: list[dict] = []
    for f in sorted(os.listdir(settings.data_dir)):
        if not f.lower().endswith(".pdf"):
            continue
        print(f"Seeding from: {f}")
        parents, _ = load_pdf(
            os.path.join(settings.data_dir, f),
            parent_size=settings.parent_size,
            parent_overlap=settings.parent_overlap,
            child_size=settings.child_size,
            child_overlap=settings.child_overlap,
        )
        all_items.extend(seed_pdf(client, settings.groq_rewrite_model, f, parents, args.per_doc))

    header = (
        "# Auto-seeded eval dataset. REVIEW AND EDIT before trusting.\n"
        "# Wrong/uninteresting questions = wrong/uninteresting scores.\n\n"
    )
    with open(args.out, "w") as f:
        f.write(header + _yaml_dump(all_items))

    print(f"Wrote {len(all_items)} stub questions to {args.out}")
    print("NEXT: open it, delete bad ones, fix wrong answers/pages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
