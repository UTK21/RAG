"""
pdf_loader.py
=============
Pipeline stage 1: turn a PDF into TWO levels of chunks.

In naive RAG we had one chunk size doing two jobs at once:
  * Be SMALL enough to embed precisely.
  * Be LARGE enough to give the LLM enough surrounding context.

Those two goals fight each other. Parent-child chunking decouples them:

    ┌────────────────────────────────────────────┐
    │  PARENT  (~1200 words, paragraph-sized)    │
    │                                            │
    │   ┌─────────┐ ┌─────────┐ ┌─────────┐      │
    │   │ child 1 │ │ child 2 │ │ child 3 │ ...  │  ◄── ~240-word
    │   └─────────┘ └─────────┘ └─────────┘      │     children
    └────────────────────────────────────────────┘

We EMBED the children (precise matching) but SEND THE PARENT to the LLM
(rich context). Best of both worlds.

Why per-page parents?
---------------------
Splitting page-by-page first means every parent (and therefore every child)
knows exactly which page it came from — clean page citations.

Caveat: scanned PDFs
--------------------
pypdf only extracts TEXT, not images. Scanned PDFs return empty pages here;
you'd need OCR (e.g. pytesseract) to handle those.
"""

from __future__ import annotations

from dataclasses import dataclass

from pypdf import PdfReader


@dataclass
class Chunk:
    """
    A CHILD chunk — the small, precise unit we actually embed and search.
    `parent_idx` points back into the parents list so we can fetch the
    larger context when a match is found.
    """

    text: str
    page: int
    parent_idx: int  # index into the parents list


@dataclass
class ParentChunk:
    """
    A PARENT chunk — the larger context unit we send to the LLM after a
    child match. Bigger window = more surrounding info for the model.
    """

    text: str
    page: int


def _window(words: list[str], size: int, overlap: int) -> list[str]:
    """
    Generic word-based sliding-window splitter. Reused for BOTH the parent
    pass (large windows) and the child pass (small windows) — only the
    `size` and `overlap` change.

    Overlap exists so an important sentence sitting on a window boundary
    still appears INTACT in at least one chunk. Without overlap the sentence
    would be cut in half across two adjacent chunks and neither half would
    match a query well.
    """
    if not words:
        return []
    out: list[str] = []
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        piece = " ".join(words[start : start + size])
        if piece:
            out.append(piece)
        # Stop once the window has consumed the whole input; otherwise the
        # last partial window gets emitted again as a shorter duplicate.
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
    """
    Read the PDF and produce parents + children.

    For each page:
      1. Split the page into PARENTS (large windows, e.g. ~1200 words).
      2. For each parent, split it further into CHILDREN (small windows,
         e.g. ~240 words). Each child remembers its parent_idx.

    Returns (parents, children) in two parallel lists. Children are what
    we embed and search; parents are what we hand to the LLM.

    Why split children FROM the parent text (not directly from the page)?
    -------------------------------------------------------------------
    So that no child ever crosses a parent boundary. When we later fetch
    "the parent of this child", we get a clean, self-contained block — no
    awkward halves of two different parents glued together.
    """
    reader = PdfReader(path)

    parents: list[ParentChunk] = []
    children: list[Chunk] = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue

        # --- Parent pass --------------------------------------------------
        for parent_text in _window(text.split(), parent_size, parent_overlap):
            parent_idx = len(parents)  # this parent's position in `parents`
            parents.append(ParentChunk(text=parent_text, page=page_num))

            # --- Child pass (within this parent) --------------------------
            for child_text in _window(parent_text.split(), child_size, child_overlap):
                children.append(
                    Chunk(text=child_text, page=page_num, parent_idx=parent_idx)
                )

    return parents, children


def children_to_parents(
    matched_children: list[Chunk],
    parents: list[ParentChunk],
) -> list[ParentChunk]:
    """
    Map a ranked list of matched CHILD chunks to their corresponding
    PARENT chunks, preserving order and deduplicating.

    Why dedupe?
    -----------
    Often multiple children of the SAME parent will all rank high — the
    matching content is concentrated in one part of the doc. Without
    dedupe we'd send the same parent text to the LLM 3 times, burning
    tokens and giving it nothing new.

    Preserving order means the BEST-matching child's parent comes first.
    """
    seen: set[int] = set()
    out: list[ParentChunk] = []
    for child in matched_children:
        if child.parent_idx in seen:
            continue
        seen.add(child.parent_idx)
        out.append(parents[child.parent_idx])
    return out
