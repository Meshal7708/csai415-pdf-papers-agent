"""Extract real CITES edges from the corpus PDFs.

Scans each paper's full text (all pages, not the 20-page ingest cap) for the
arXiv ids of *other* in-corpus papers. Only genuine references become edges —
nothing is synthesised. The corpus deliberately spans six disjoint topics, so
this graph is sparse by construction; we report the true count and lean on
SIMILAR_TO (semantic) edges for GraphRAG expansion. Writes data/cites.json.
"""
from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT.parent / "D2" / "data" / "pdfs"
OUT = ROOT / "data" / "cites.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

ID = re.compile(r"\b(\d{4}\.\d{4,5})\b")


def main():
    papers = pd.read_csv(ROOT.parent / "D2" / "data" / "papers.csv", dtype={"paper_id": str})
    ids = set(papers["paper_id"])
    pdfs = {os.path.basename(p)[:-4]: p for p in glob.glob(str(PDF_DIR / "*.pdf"))}
    edges = []
    for pid in sorted(ids):
        path = pdfs.get(pid)
        if not path:
            continue
        try:
            rd = PdfReader(path)
            txt = " ".join((pg.extract_text() or "") for pg in rd.pages)
        except Exception:
            continue
        found = ({m for m in ID.findall(txt)} & ids) - {pid}
        for tgt in sorted(found):
            edges.append([pid, tgt])
    OUT.write_text(json.dumps({"n_edges": len(edges), "edges": edges}, indent=2))
    print(f"CITES edges: {len(edges)}  ->  {OUT}")


if __name__ == "__main__":
    main()
