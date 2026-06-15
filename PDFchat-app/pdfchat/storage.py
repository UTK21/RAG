"""Persist + reload FAISS index, BM25, parents, children, plus a manifest
describing what they were built from (so we can detect staleness)."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from typing import Any

import faiss

from pdfchat.loader import Chunk, ParentChunk

# File names inside the index dir.
_FAISS_FILE = "dense.faiss"
_BM25_FILE = "sparse.bm25.pkl"
_CHUNKS_FILE = "chunks.pkl"
_MANIFEST_FILE = "manifest.json"


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def doc_hashes(data_dir: str) -> dict[str, str]:
    """Hash every PDF in data_dir. The fingerprint that drives freshness checks."""
    out: dict[str, str] = {}
    for f in sorted(os.listdir(data_dir)):
        if f.lower().endswith(".pdf"):
            out[f] = _sha256_file(os.path.join(data_dir, f))
    return out


def save(
    index_dir: str,
    dense_index: faiss.IndexFlatIP,
    bm25: Any,
    parents: list[ParentChunk],
    children: list[Chunk],
    manifest: dict[str, Any],
) -> None:
    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(dense_index, os.path.join(index_dir, _FAISS_FILE))
    with open(os.path.join(index_dir, _BM25_FILE), "wb") as f:
        pickle.dump(bm25, f)
    with open(os.path.join(index_dir, _CHUNKS_FILE), "wb") as f:
        pickle.dump({"parents": parents, "children": children}, f)
    with open(os.path.join(index_dir, _MANIFEST_FILE), "w") as f:
        json.dump(manifest, f, indent=2)


@dataclass
class LoadedIndex:
    dense_index: faiss.IndexFlatIP
    bm25: Any
    parents: list[ParentChunk]
    children: list[Chunk]
    manifest: dict[str, Any]


def load(index_dir: str) -> LoadedIndex:
    if not os.path.isdir(index_dir):
        raise FileNotFoundError(f"Index dir not found: {index_dir}")
    for f in (_FAISS_FILE, _BM25_FILE, _CHUNKS_FILE, _MANIFEST_FILE):
        if not os.path.isfile(os.path.join(index_dir, f)):
            raise FileNotFoundError(
                f"Index incomplete: missing {f}. Run ingest.py first."
            )

    dense_index = faiss.read_index(os.path.join(index_dir, _FAISS_FILE))
    with open(os.path.join(index_dir, _BM25_FILE), "rb") as f:
        bm25 = pickle.load(f)
    with open(os.path.join(index_dir, _CHUNKS_FILE), "rb") as f:
        store = pickle.load(f)
    with open(os.path.join(index_dir, _MANIFEST_FILE), "r") as f:
        manifest = json.load(f)

    return LoadedIndex(
        dense_index=dense_index,
        bm25=bm25,
        parents=store["parents"],
        children=store["children"],
        manifest=manifest,
    )


def check_fresh(loaded: LoadedIndex, data_dir: str, settings_fingerprint: dict[str, Any]) -> tuple[bool, str]:
    """Compare loaded manifest against current data dir + settings.
    Returns (is_fresh, human_message)."""
    current_hashes = doc_hashes(data_dir)
    if current_hashes != loaded.manifest.get("doc_hashes"):
        return False, "PDFs in data/ have changed since last ingest."
    if settings_fingerprint != loaded.manifest.get("settings_fingerprint"):
        return False, "Chunking/embedding settings have changed since last ingest."
    return True, "fresh"


def settings_fingerprint(settings) -> dict[str, Any]:
    """Subset of settings that, if changed, invalidates the index."""
    return {
        "embed_model": settings.embed_model,
        "parent_size": settings.parent_size,
        "parent_overlap": settings.parent_overlap,
        "child_size": settings.child_size,
        "child_overlap": settings.child_overlap,
    }
