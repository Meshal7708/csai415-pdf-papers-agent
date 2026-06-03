"""PDF -> text -> chunks (+overlap) with a page map and full provenance.

The unit of retrieval is a *chunk*. Every chunk carries enough provenance to
render a grounded citation with a page range:

    paper_id, chunk_index, text,
    page_start, page_end,          <- from the page map
    char_start, char_end,          <- offsets into the document's full text
    n_tokens (approx), sha256       <- for dedup / change detection

`parse_pdf` is defensive: pypdf occasionally returns None for a page's text on
malformed PDFs, so we coerce to "" and keep going rather than crashing the run.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from pypdf import PdfReader


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()[:16]


@dataclass
class Page:
    page_number: int          # 1-based, matches what a human sees in a viewer
    text: str
    char_start: int           # offset of this page's text in the document full text
    char_end: int


def parse_pdf(path: str | Path, max_pages: int = 0) -> List[Page]:
    """Return per-page text with char offsets into the concatenated full text.

    `max_pages` (0 = all) caps how many leading pages are parsed.
    """
    reader = PdfReader(str(path))
    src_pages = reader.pages if not max_pages else reader.pages[:max_pages]
    pages: List[Page] = []
    cursor = 0
    for i, pg in enumerate(src_pages):
        try:
            raw = pg.extract_text() or ""
        except Exception:
            raw = ""
        # normalise whitespace; keep paragraph breaks meaningful for chunk edges
        text = re.sub(r"[ \t]+", " ", raw).strip()
        text = re.sub(r"\n{2,}", "\n", text)
        sep = "\n" if i > 0 else ""
        start = cursor + len(sep)
        end = start + len(text)
        pages.append(Page(page_number=i + 1, text=text, char_start=start, char_end=end))
        cursor = end
    return pages


def _page_for_offset(pages: List[Page], offset: int) -> int:
    """Map a character offset in the full text back to its 1-based page number."""
    for p in pages:
        if p.char_start <= offset < p.char_end:
            return p.page_number
    return pages[-1].page_number if pages else 1


@dataclass
class Chunk:
    paper_id: str
    chunk_index: int
    text: str
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    n_tokens: int
    sha256: str

    @property
    def chunk_id(self) -> str:
        return f"{self.paper_id}::{self.chunk_index}"

    def to_doc(self, run_id: str) -> Dict:
        d = {
            "_id": self.chunk_id,
            "paper_id": self.paper_id,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "n_tokens": self.n_tokens,
            "sha256": self.sha256,
            "run_id": run_id,
        }
        return d


def chunk_pages(
    paper_id: str,
    pages: List[Page],
    size: int = 900,
    overlap: int = 150,
    min_chars: int = 120,
) -> List[Chunk]:
    """Sliding-window chunker over the document's full text.

    Windows are cut on the full concatenated text (so a chunk can legitimately
    span a page break — captured as page_start..page_end). We snap the right
    edge to the nearest sentence boundary inside the last 200 chars when possible
    so chunks read cleanly.
    """
    full = "".join(("\n" if p.page_number > 1 else "") + p.text for p in pages)
    chunks: List[Chunk] = []
    n = len(full)
    idx = 0
    start = 0
    step = max(1, size - overlap)
    while start < n:
        end = min(start + size, n)
        # try to end on a sentence boundary for readability
        if end < n:
            window = full[end - 200 : end]
            m = list(re.finditer(r"[.!?]\s", window))
            if m:
                end = (end - 200) + m[-1].end()
        text = full[start:end].strip()
        if len(text) >= min_chars:
            chunks.append(
                Chunk(
                    paper_id=paper_id,
                    chunk_index=idx,
                    text=text,
                    page_start=_page_for_offset(pages, start),
                    page_end=_page_for_offset(pages, max(start, end - 1)),
                    char_start=start,
                    char_end=end,
                    n_tokens=len(text.split()),
                    sha256=_sha(text),
                )
            )
            idx += 1
        start += step
    return chunks


def ingest_pdf(paper_id: str, path: str | Path, size=900, overlap=150, min_chars=120, max_pages=0):
    """Parse one PDF and return (n_pages, list[Chunk]). Empty list if unreadable."""
    pages = parse_pdf(path, max_pages=max_pages)
    if not pages or sum(len(p.text) for p in pages) < min_chars:
        return len(pages), []
    return len(pages), chunk_pages(paper_id, pages, size, overlap, min_chars)
