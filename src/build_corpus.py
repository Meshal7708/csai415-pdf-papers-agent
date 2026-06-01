"""created Real-data corpus builder for D1. as you recommeneded prof

Fetches ~150 real papers from the arXiv API across 6 topic clusters.
Output: data/corpus.parquet. Schema unchanged from the previous version so
retriever / gold / automl / online all work without edits:
  paper_id, title, abstract, authors, venue, year, topic, text
"""
from __future__ import annotations
import hashlib
from pathlib import Path

import pandas as pd
import arxiv  # pip install arxiv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

SEED = 42  # kept for run-card continuity

# Each topic -> an arXiv query. Results are sorted by relevance and the
# generated corpus.parquet is committed, so the snapshot is pinned for the team.
TOPIC_QUERIES = {
    "transformers":    'cat:cs.CL AND (abs:"attention" OR abs:"transformer")',
    "rag":             '(abs:"retrieval-augmented generation" OR abs:"dense retrieval" OR abs:"passage retrieval")',
    "online_learning": '(abs:"concept drift" OR abs:"online learning" OR abs:"data stream")',
    "vision":          'cat:cs.CV AND (abs:"image" OR abs:"vision transformer")',
    "rl_agents":       '(abs:"reinforcement learning" AND abs:"agent")',
    "automl":          '(abs:"automl" OR abs:"hyperparameter optimization" OR abs:"neural architecture search")',
}
PER_TOPIC = 25  # 6 * 25 = 150


def fetch_topic(client, topic, query, n, seen):
    """Return up to n unique papers for a topic (skips ids already seen)."""
    search = arxiv.Search(query=query, max_results=n * 3,
                          sort_by=arxiv.SortCriterion.Relevance)
    rows = []
    for r in client.results(search):
        pid = r.get_short_id().split("v")[0]      # e.g. 2305.12345 (strip version)
        if pid in seen:
            continue
        seen.add(pid)
        rows.append({
            "paper_id": pid,
            "title": r.title.strip().replace("\n", " "),
            "abstract": r.summary.strip().replace("\n", " "),
            "authors": ", ".join(a.name for a in r.authors),
            "venue": "arXiv",
            "year": r.published.year,
            "topic": topic,
        })
        if len(rows) >= n:
            break
    return rows


def build_corpus(target_size: int = 150):
    per_topic = target_size // len(TOPIC_QUERIES)
    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=5)
    rows, seen = [], set()
    for topic, query in TOPIC_QUERIES.items():
        got = fetch_topic(client, topic, query, per_topic, seen)
        for row in got:
            row["text"] = f'{row["title"]}. {row["abstract"]}'
        rows.extend(got)
        print(f"{topic}: {len(got)} papers (total {len(rows)})")
    df = pd.DataFrame(rows)
    h = hashlib.sha256("\n".join(df["text"]).encode("utf-8")).hexdigest()[:16]
    df.attrs["corpus_hash"] = h
    return df


if __name__ == "__main__":
    df = build_corpus()
    out = DATA / "corpus.parquet"
    df.to_parquet(out, index=False)
    print(f"Wrote {len(df)} papers to {out}")
    print(df["topic"].value_counts().to_string())
    print(f"corpus_hash={df.attrs['corpus_hash']}")
    (DATA / "corpus_hash.txt").write_text(df.attrs["corpus_hash"])
