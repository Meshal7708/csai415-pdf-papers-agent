"""Answer composition with grounded citations and page ranges.

`ExtractiveAnswerer` (default, offline, deterministic) builds the answer only
from retrieved chunk text: it ranks sentences in the top chunks by overlap with
the question, stitches the best ones together, and tags each with a numbered
citation `[n]` that resolves to `title (paper_id), pp.X–Y`. Because every
sentence is lifted from a retrieved chunk, the answer is faithful by construction
— which the evaluation then measures rather than assumes.

`LLMAnswerer` is an optional drop-in: if `OPENAI_API_KEY` is set it asks a chat
model to answer *strictly from the provided context* with the same citation
markers. Same interface, so the executor and evaluation don't change.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List

_STOP = set("the a an of to in for and or on with from by as is are be this that "
            "we our it its using use used can will into over under across".split())


def _boilerplate(s: str) -> bool:
    """Author/affiliation/header lines that pollute extractive answers."""
    if "@" in s or re.search(r"https?://", s):
        return True
    words = s.split()
    if len(words) < 4:
        return True
    caps = sum(1 for w in words if w[:1].isupper())
    return caps / len(words) > 0.6                  # mostly proper nouns -> a name list


def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return [s.strip() for s in parts if len(s.strip()) > 25 and not _boilerplate(s.strip())]


def _content_words(text: str) -> set:
    return {w for w in re.findall(r"[a-zA-Z][a-zA-Z0-9\-]+", text.lower())
            if w not in _STOP and len(w) > 2}


@dataclass
class Answer:
    question: str
    text: str
    citations: List[dict]
    contexts: List[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"question": self.question, "answer": self.text,
                "citations": self.citations, "meta": self.meta}


class ExtractiveAnswerer:
    kind = "extractive"

    def __init__(self, max_sentences: int = 4):
        self.max_sentences = max_sentences

    def answer(self, question: str, cites: List) -> Answer:
        qwords = _content_words(question)
        scored = []
        for rank, c in enumerate(cites):
            for sent in _sentences(c.text):
                overlap = len(qwords & _content_words(sent))
                if overlap == 0:
                    continue
                # prefer sentences with more query overlap, from higher-ranked chunks
                score = overlap + 1.0 / (1 + rank)
                scored.append((score, rank, c, sent))
        scored.sort(key=lambda x: (-x[0], x[1]))

        # Spread coverage across sources: at most 2 sentences per chunk, so the
        # answer cites multiple papers instead of quoting one passage repeatedly.
        picked, per_chunk, pieces, used = [], {}, [], []
        for score, rank, c, sent in scored:
            if len(picked) >= self.max_sentences:
                break
            if sent in [p[3] for p in picked]:
                continue
            if per_chunk.get(c.chunk_id, 0) >= 2:
                continue
            per_chunk[c.chunk_id] = per_chunk.get(c.chunk_id, 0) + 1
            picked.append((score, rank, c, sent))
        picked.sort(key=lambda x: x[1])             # restore reading order by chunk rank

        # assign citation numbers in order of first appearance
        cite_index, citations = {}, []
        for _, _, c, sent in picked:
            if c.chunk_id not in cite_index:
                cite_index[c.chunk_id] = len(citations) + 1
                citations.append({
                    "marker": f"[{len(citations) + 1}]", "paper_id": c.paper_id,
                    "title": c.title, "page_range": c.page_range, "chunk_id": c.chunk_id,
                })
            pieces.append(f"{sent} [{cite_index[c.chunk_id]}]")
            used.append(c.text)

        if not pieces:                              # nothing overlapped -> fall back to top chunk
            if cites:
                c = cites[0]
                citations = [{"marker": "[1]", "paper_id": c.paper_id, "title": c.title,
                              "page_range": c.page_range, "chunk_id": c.chunk_id}]
                snippet = _sentences(c.text)[:2] or [c.text[:240]]
                pieces = [" ".join(snippet) + " [1]"]
                used = [c.text]
            else:
                return Answer(question, "No relevant passages were retrieved.", [], [], {})

        text = " ".join(pieces)
        return Answer(question, text, citations, contexts=used,
                      meta={"answerer": self.kind, "n_sentences": len(pieces)})


class LLMAnswerer:
    kind = "llm"

    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model

    def answer(self, question: str, cites: List) -> Answer:
        ctx = "\n\n".join(f"[{i+1}] {c.title} ({c.paper_id}), {c.page_range}\n{c.text}"
                          for i, c in enumerate(cites))
        msg = [
            {"role": "system", "content": "Answer ONLY from the context. Cite every "
             "claim with its [n] marker. If the context is insufficient, say so."},
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{ctx}"},
        ]
        out = self.client.chat.completions.create(model=self.model, messages=msg,
                                                   temperature=0.0).choices[0].message.content
        citations = [{"marker": f"[{i+1}]", "paper_id": c.paper_id, "title": c.title,
                      "page_range": c.page_range, "chunk_id": c.chunk_id}
                     for i, c in enumerate(cites)]
        return Answer(question, out, citations, contexts=[c.text for c in cites],
                      meta={"answerer": self.kind, "model": self.model})


def get_answerer():
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return LLMAnswerer()
        except Exception:
            pass
    return ExtractiveAnswerer()
