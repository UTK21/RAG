"""Multi-PDF loader with parent-child chunking. Every chunk carries doc_name."""
from __future__ import annotations

import os
from dataclasses import dataclass

from pypdf import PdfReader


@dataclass
class ParentChunk:
    text: str
    page: int
    doc_name: str


@dataclass
class Chunk:
    text: str
    page: int
    parent_idx: int
    doc_name: str


def _window(words: list[str], size: int, overlap: int) -> list[str]:
    if not words:
        return []
    out: list[str] = []
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        piece = " ".join(words[start : start + size])
        if piece:
            out.append(piece)
        if start + size >= len(words):
            break
    return out


def load_pdf(
    path: str,
    parent_size: int,
    parent_overlap: int,
    child_size: int,
    child_overlap: int,
) -> tuple[list[ParentChunk], list[Chunk]]:
    """Read one PDF; return its (parents, children). doc_name = basename."""
    doc_name = os.path.basename(path)
    reader = PdfReader(path)

    parents: list[ParentChunk] = []
    children: list[Chunk] = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        for parent_text in _window(text.split(), parent_size, parent_overlap):
            parent_idx = len(parents)
            parents.append(ParentChunk(text=parent_text, page=page_num, doc_name=doc_name))
            for child_text in _window(parent_text.split(), child_size, child_overlap):
                children.append(
                    Chunk(
                        text=child_text,
                        page=page_num,
                        parent_idx=parent_idx,
                        doc_name=doc_name,
                    )
                )

    return parents, children


def load_directory(
    data_dir: str,
    parent_size: int,
    parent_overlap: int,
    child_size: int,
    child_overlap: int,
) -> tuple[list[ParentChunk], list[Chunk]]:
    """Walk `data_dir`, parse every .pdf. parent_idx is global across all docs."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data dir not found: {data_dir}")

    pdf_paths = sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(".pdf")
    )
    if not pdf_paths:
        raise FileNotFoundError(f"No .pdf files found in {data_dir}")

    all_parents: list[ParentChunk] = []
    all_children: list[Chunk] = []

    for path in pdf_paths:
        parents, children = load_pdf(
            path, parent_size, parent_overlap, child_size, child_overlap
        )
        # parent_idx is local to load_pdf(); rebase onto the global parents list.
        offset = len(all_parents)
        for c in children:
            c.parent_idx += offset
        all_parents.extend(parents)
        all_children.extend(children)

    return all_parents, all_children


def children_to_parents(
    matched_children: list[Chunk], parents: list[ParentChunk]
) -> list[ParentChunk]:
    """Map ranked children to deduped parents, preserving order."""
    seen: set[int] = set()
    out: list[ParentChunk] = []
    for child in matched_children:
        if child.parent_idx in seen:
            continue
        seen.add(child.parent_idx)
        out.append(parents[child.parent_idx])
    return out
