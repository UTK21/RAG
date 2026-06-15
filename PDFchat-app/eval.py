"""Run the eval suite. Loads dataset.yaml, runs each question through the
pipeline, scores it, prints a table + aggregates.

Usage:
    python eval.py
    python eval.py --config USE_HYDE=true
    python eval.py --config USE_HYDE=true --config TOP_K=6
    python eval.py --no-judge        # skip the LLM-judge metric

The --config flag lets you A/B without editing .env: every value you pass
overrides the corresponding env var for THIS run only.
"""
from __future__ import annotations

import argparse
import os
import sys

import yaml

from eval.metrics import (
    MetricResult,
    citation_match,
    keyword_coverage,
    llm_judge,
    retrieval_recall,
)
from pdfchat import storage
from pdfchat.config import load_settings
from pdfchat.embeddings import load_embedder
from pdfchat.llm import make_client
from pdfchat.pipeline import Pipeline
from pdfchat.rerank import load_reranker


def _parse_overrides(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"Bad --config: {it!r}. Expected KEY=VALUE.")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _print_row(idx: int, question: str, results: list[MetricResult]) -> None:
    status = "PASS" if all(r.passed for r in results) else "FAIL"
    print(f"\n[{idx}] {status}  {question!r}")
    for r in results:
        marker = "✓" if r.passed else "✗"
        print(f"     {marker} {r.name:18s} {r.score:.2f}  {r.detail}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=os.path.join("eval", "dataset.yaml"))
    parser.add_argument(
        "--config", action="append", default=[],
        help="KEY=VALUE override (repeatable)",
    )
    parser.add_argument("--no-judge", action="store_true",
                        help="skip the LLM-judge metric (cheaper)")
    args = parser.parse_args()

    overrides = _parse_overrides(args.config)
    settings = load_settings(overrides=overrides)

    if not settings.groq_api_key:
        print("GROQ_API_KEY missing.", file=sys.stderr)
        return 1

    with open(args.dataset) as f:
        cases = [c for c in (yaml.safe_load(f) or []) if c]
    if not cases:
        print(f"No cases in {args.dataset}. Seed it with: python eval/seed.py",
              file=sys.stderr)
        return 1

    print(f"Loading indexes from: {settings.index_dir}")
    loaded = storage.load(settings.index_dir)
    print(f"Loading embedder + reranker...")
    embedder = load_embedder(settings.embed_model)
    reranker = load_reranker(settings.rerank_model)
    client = make_client(settings.groq_api_key)
    pipe = Pipeline(
        settings=settings, loaded_index=loaded,
        embedder=embedder, reranker=reranker, client=client,
    )

    print(f"\nRunning {len(cases)} cases  (HyDE={'on' if settings.use_hyde else 'off'})")

    per_metric: dict[str, list[float]] = {}

    for i, case in enumerate(cases, start=1):
        question = case["question"]
        expected_doc = case["expected_doc"]
        expected_page = case.get("expected_page")
        expected_keywords = case.get("expected_keywords") or []

        reply, trace = pipe.answer(question)

        results: list[MetricResult] = [
            retrieval_recall(expected_doc, expected_page, trace.top_children),
            citation_match(expected_doc, expected_page, reply),
            keyword_coverage(expected_keywords, reply),
        ]
        if not args.no_judge:
            results.append(
                llm_judge(client, settings.groq_rewrite_model,
                          question, expected_keywords, reply)
            )

        _print_row(i, question, results)
        for r in results:
            per_metric.setdefault(r.name, []).append(r.score)

    # ---- aggregates -------------------------------------------------------
    print("\n" + "=" * 60)
    print("AGGREGATE")
    print("=" * 60)
    for name, scores in per_metric.items():
        avg = sum(scores) / len(scores)
        n_pass = sum(1 for s in scores if s >= 0.5)
        print(f"  {name:18s} avg={avg:.2f}  passed {n_pass}/{len(scores)}")
    print()
    print(f"Config snapshot: HyDE={settings.use_hyde}  TOP_K={settings.top_k}  "
          f"RETRIEVE_K={settings.retrieve_k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
