"""Download real full-text PDFs for a curated subset of the D1 arXiv corpus.

D1 fetched 150 papers' metadata (title/abstract/authors). D2 needs the *full
text* with page maps, so here we pull the actual PDFs from arXiv and write
`data/papers.csv` — the brief's required manifest:
    paper_id, title, authors, venue, year, pdf_url, pdf_path, topics

Politeness: arXiv asks for a few seconds between hits; we sleep and retry. Any
paper whose PDF fails to download is simply left out of papers.csv (the pipeline
skips papers with no local PDF), so a flaky download never breaks a run.

Usage:
    python3 scripts/download_pdfs.py --per-topic 10
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
D1_CORPUS = ROOT.parent / "data" / "corpus.parquet"   # reuse D1's pinned metadata
PDF_DIR = ROOT / "data" / "pdfs"
PAPERS_CSV = ROOT / "data" / "papers.csv"
PDF_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "CSAI415-D2/0.1 (academic coursework; contact: team)"}


def download(paper_id: str, dest: Path, retries: int = 3) -> bool:
    if dest.exists() and dest.stat().st_size > 10_000:
        return True
    url = f"https://arxiv.org/pdf/{paper_id}"
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200 and r.content[:4] == b"%PDF":
                dest.write_bytes(r.content)
                return True
        except Exception:
            pass
        time.sleep(3 * (attempt + 1))
    return False


def main(per_topic: int):
    df = pd.read_parquet(D1_CORPUS)
    # curated subset: top `per_topic` per topic (relevance order is preserved)
    subset = df.groupby("topic", group_keys=False).head(per_topic).reset_index(drop=True)
    rows = []
    for i, r in subset.iterrows():
        pid = r["paper_id"]
        dest = PDF_DIR / f"{pid}.pdf"
        ok = download(pid, dest)
        print(f"[{i+1}/{len(subset)}] {pid} ({r['topic']}): {'ok' if ok else 'FAILED'}")
        if ok:
            rows.append({
                "paper_id": pid, "title": r["title"], "authors": r["authors"],
                "venue": r["venue"], "year": int(r["year"]),
                "pdf_url": f"https://arxiv.org/pdf/{pid}", "pdf_path": str(dest),
                "topics": r["topic"],
            })
        time.sleep(2)  # be polite to arXiv
    out = pd.DataFrame(rows)
    out.to_csv(PAPERS_CSV, index=False)
    print(f"\nWrote {len(out)} papers to {PAPERS_CSV}")
    print(out["topics"].value_counts().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-topic", type=int, default=10)
    main(ap.parse_args().per_topic)
