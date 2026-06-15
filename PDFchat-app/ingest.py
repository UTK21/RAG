"""Build FAISS + BM25 indexes over every PDF in data/ and persist them.

Run once per change to your PDF folder. chat.py and eval.py read from the
saved indexes — no rebuild on every startup.

Usage:
    python ingest.py
"""
from __future__ import annotations

import sys
import time

from pdfchat import bm25 as bm25_mod
from pdfchat import storage
from pdfchat.config import settings
from pdfchat.embeddings import build_index, load_embedder
from pdfchat.loader import load_directory


def main() -> int:
    t0 = time.time()

    print(f"Reading PDFs from: {settings.data_dir}")
    try:
        parents, children = load_directory(
            settings.data_dir,
            parent_size=settings.parent_size,
            parent_overlap=settings.parent_overlap,
            child_size=settings.child_size,
            child_overlap=settings.child_overlap,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    doc_names = sorted({p.doc_name for p in parents})
    print(f"  {len(doc_names)} docs, {len(parents)} parents, {len(children)} children")

    print(f"Loading embedder: {settings.embed_model}")
    embedder = load_embedder(settings.embed_model)

    print("Building FAISS index (dense, over children)...")
    dense_index = build_index(children, embedder)

    print("Building BM25 index (sparse, over children)...")
    bm25 = bm25_mod.build_bm25(children)

    manifest = {
        "doc_names": doc_names,
        "n_parents": len(parents),
        "n_children": len(children),
        "doc_hashes": storage.doc_hashes(settings.data_dir),
        "settings_fingerprint": storage.settings_fingerprint(settings),
    }

    print(f"Saving to: {settings.index_dir}")
    storage.save(
        settings.index_dir,
        dense_index=dense_index,
        bm25=bm25,
        parents=parents,
        children=children,
        manifest=manifest,
    )

    print(f"Done in {time.time() - t0:.1f}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
