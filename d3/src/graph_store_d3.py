"""Enriched knowledge graph for D3 GraphRAG.

Extends the D2 schema (Author/Paper/Topic/Venue) with two paper-paper edges:

* `CITES`       — real references between in-corpus papers (extracted from the
                  PDFs; sparse because the corpus spans six disjoint topics).
* `SIMILAR_TO`  — top-k semantic neighbours by cosine over paper embeddings
                  (mean of a paper's bge chunk vectors). This is the dense,
                  meaningful edge the GraphRAG executor walks to expand recall.

`NetworkxGraphStore` (default, no server) and `Neo4jGraphStore` expose the same
query API. The headline method is `weighted_subgraph(seed_ids)` — the multi-signal
expansion the executor uses — plus semantic neighbours, 2-hop expansion, author
collaboration paths, co-citation, and PageRank authority. Mirrors
`cypher_queries_d3.py`.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


def _authors(s: str) -> List[str]:
    return [a.strip() for a in str(s).split(",") if a.strip()]


def topk_similar(paper_ids: List[str], vectors: np.ndarray, k: int = 8,
                 min_score: float = 0.30) -> Dict[str, List[Tuple[str, float]]]:
    """Cosine k-NN over L2-normalised paper vectors -> SIMILAR_TO adjacency."""
    X = vectors.astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    sims = Xn @ Xn.T
    out: Dict[str, List[Tuple[str, float]]] = {}
    for i, pid in enumerate(paper_ids):
        row = sims[i].copy()
        row[i] = -1.0
        idx = np.argsort(-row)[:k]
        out[pid] = [(paper_ids[j], float(row[j])) for j in idx if row[j] >= min_score]
    return out


class NetworkxGraphStore:
    backend = "networkx"

    def __init__(self):
        import networkx as nx
        self.nx = nx
        self.g = nx.MultiDiGraph()

    # ---------------------------------------------------------------- load ---
    def load(self, papers: List[Dict],
             similar: Optional[Dict[str, List[Tuple[str, float]]]] = None,
             cites: Optional[List[Tuple[str, str]]] = None) -> Dict[str, int]:
        g = self.g
        g.clear()
        for p in papers:
            pid = str(p["paper_id"])
            g.add_node(("Paper", pid), kind="Paper",
                       title=p["title"], year=int(p.get("year", 0)),
                       topic=p.get("topic", ""))
            g.add_node(("Topic", p["topic"]), kind="Topic")
            g.add_edge(("Paper", pid), ("Topic", p["topic"]), rel="ABOUT")
            venue = p.get("venue", "arXiv")
            g.add_node(("Venue", venue), kind="Venue")
            g.add_edge(("Paper", pid), ("Venue", venue), rel="PUBLISHED_IN")
            for a in _authors(p.get("authors", "")):
                g.add_node(("Author", a), kind="Author")
                g.add_edge(("Author", a), ("Paper", pid), rel="WROTE")
        present = {("Paper", str(p["paper_id"])) for p in papers}
        if cites:
            for src, tgt in cites:
                if ("Paper", src) in present and ("Paper", tgt) in present:
                    g.add_edge(("Paper", src), ("Paper", tgt), rel="CITES")
        if similar:
            for src, nbrs in similar.items():
                if ("Paper", src) not in present:
                    continue
                for tgt, score in nbrs:
                    if ("Paper", tgt) in present:
                        g.add_edge(("Paper", src), ("Paper", tgt),
                                   rel="SIMILAR_TO", score=float(score))
        return self.stats()

    # ------------------------------------------------------- helpers ---------
    def _topics_of(self, pid: str) -> set:
        n = ("Paper", pid)
        return {v[1] for _, v, d in self.g.out_edges(n, data=True) if d["rel"] == "ABOUT"}

    def _authors_of(self, pid: str) -> set:
        n = ("Paper", pid)
        return {u[1] for u, _, d in self.g.in_edges(n, data=True) if d["rel"] == "WROTE"}

    def _similar_of(self, pid: str) -> List[Tuple[str, float]]:
        n = ("Paper", pid)
        if n not in self.g:
            return []
        out = []
        for _, v, d in self.g.out_edges(n, data=True):
            if d["rel"] == "SIMILAR_TO":
                out.append((v[1], float(d.get("score", 0.0))))
        return sorted(out, key=lambda x: -x[1])

    # ------------------------------------------- D3 query API ----------------
    def weighted_subgraph(self, seed_ids: List[str], limit: int = 10,
                          w_sim: float = 1.0, w_topic: float = 0.5,
                          w_author: float = 0.5) -> List[Dict]:
        """Multi-signal expansion: rank neighbours of the seed set by a weighted
        blend of semantic similarity + shared topic + shared author."""
        seeds = [str(s) for s in seed_ids]
        seed_set = set(seeds)
        seed_topics = {s: self._topics_of(s) for s in seeds if ("Paper", s) in self.g}
        seed_authors = {s: self._authors_of(s) for s in seeds if ("Paper", s) in self.g}
        scores: Dict[str, float] = defaultdict(float)
        reasons: Dict[str, set] = defaultdict(set)
        for s in seeds:
            if ("Paper", s) not in self.g:
                continue
            # semantic neighbours
            for tgt, sc in self._similar_of(s):
                if tgt not in seed_set:
                    scores[tgt] += w_sim * sc
                    reasons[tgt].add("similar")
        # topic / author overlap against every other paper
        for node, d in self.g.nodes(data=True):
            if d.get("kind") != "Paper":
                continue
            cid = node[1]
            if cid in seed_set:
                continue
            ctopics, cauthors = self._topics_of(cid), self._authors_of(cid)
            for s in seeds:
                if seed_topics.get(s, set()) & ctopics:
                    scores[cid] += w_topic
                    reasons[cid].add("topic")
                if seed_authors.get(s, set()) & cauthors:
                    scores[cid] += w_author
                    reasons[cid].add("author")
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:limit]
        return [{"id": pid, "title": self.g.nodes[("Paper", pid)]["title"],
                 "score": round(sc, 4), "via": sorted(reasons[pid])}
                for pid, sc in ranked]

    def semantic_neighbours(self, paper_id: str, limit: int = 8,
                            min_score: float = 0.30) -> List[Dict]:
        out = [{"id": t, "title": self.g.nodes[("Paper", t)]["title"], "score": round(s, 4)}
               for t, s in self._similar_of(str(paper_id)) if s >= min_score]
        return out[:limit]

    def two_hop_similar(self, paper_id: str, limit: int = 8) -> List[Dict]:
        pid = str(paper_id)
        direct = {t for t, _ in self._similar_of(pid)}
        agg: Dict[str, float] = {}
        for mid, s1 in self._similar_of(pid):
            for tgt, s2 in self._similar_of(mid):
                if tgt == pid or tgt in direct:
                    continue
                agg[tgt] = max(agg.get(tgt, 0.0), s1 * s2)
        ranked = sorted(agg.items(), key=lambda x: -x[1])[:limit]
        return [{"id": t, "title": self.g.nodes[("Paper", t)]["title"],
                 "strength": round(s, 4)} for t, s in ranked]

    def author_collab_path(self, author_a: str, author_b: str) -> Optional[Dict]:
        a, b = ("Author", author_a), ("Author", author_b)
        if a not in self.g or b not in self.g:
            return None
        UG = self.nx.Graph()
        for u, v, d in self.g.edges(data=True):
            if d["rel"] == "WROTE":
                UG.add_edge(u, v)
        try:
            path = self.nx.shortest_path(UG, a, b)
        except (self.nx.NetworkXNoPath, self.nx.NodeNotFound):
            return None
        hops = [n[1] if n[0] == "Author" else self.g.nodes[n]["title"] for n in path]
        return {"hops": hops, "path_len": len(path) - 1}

    def co_citation(self, paper_id: str, limit: int = 10) -> List[Dict]:
        pid = ("Paper", str(paper_id))
        citers = [u for u, _, d in self.g.in_edges(pid, data=True) if d["rel"] == "CITES"]
        counts: Dict[str, int] = defaultdict(int)
        for c in citers:
            for _, v, d in self.g.out_edges(c, data=True):
                if d["rel"] == "CITES" and v != pid:
                    counts[v[1]] += 1
        ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:limit]
        return [{"id": t, "title": self.g.nodes[("Paper", t)]["title"],
                 "co_cited_by": c} for t, c in ranked]

    def pagerank_authority(self, limit: int = 10) -> List[Dict]:
        DG = self.nx.DiGraph()
        for u, v, d in self.g.edges(data=True):
            if d["rel"] in ("SIMILAR_TO", "CITES"):
                w = float(d.get("score", 1.0))
                if DG.has_edge(u, v):
                    DG[u][v]["weight"] += w
                else:
                    DG.add_edge(u, v, weight=w)
        if DG.number_of_nodes() == 0:
            return []
        pr = self.nx.pagerank(DG, weight="weight")
        papers = [(n[1], s) for n, s in pr.items() if n[0] == "Paper"]
        ranked = sorted(papers, key=lambda x: -x[1])[:limit]
        return [{"id": t, "title": self.g.nodes[("Paper", t)]["title"],
                 "score": round(s, 5)} for t, s in ranked]

    # ---- carried-over D2 lookups (kept so the graph is a superset) ----------
    def papers_by_topic(self, topic: str, limit: int = 10):
        out = [(u[1], self.g.nodes[u]["title"])
               for u, v, d in self.g.in_edges(("Topic", topic), data=True) if d["rel"] == "ABOUT"]
        return sorted(out)[:limit]

    def related_via_topic(self, paper_id: str, limit: int = 10):
        pid = ("Paper", str(paper_id))
        if pid not in self.g:
            return []
        sibs = set()
        for t in self._topics_of(str(paper_id)):
            for u, _, d in self.g.in_edges(("Topic", t), data=True):
                if d["rel"] == "ABOUT" and u != pid:
                    sibs.add((u[1], self.g.nodes[u]["title"]))
        return sorted(sibs)[:limit]

    def stats(self) -> Dict:
        kinds = defaultdict(int)
        for _, d in self.g.nodes(data=True):
            kinds[d["kind"]] += 1
        rels = defaultdict(int)
        for _, _, d in self.g.edges(data=True):
            rels[d["rel"]] += 1
        return {"backend": self.backend, "nodes": self.g.number_of_nodes(),
                "edges": self.g.number_of_edges(),
                "node_kinds": dict(kinds), "edge_rels": dict(rels)}

    def close(self):
        pass


class Neo4jGraphStore:
    """Loads the enriched schema into Neo4j and runs the D3 Cypher. Used on a
    laptop with `docker compose up`; the NetworkX store is the sandbox/CI twin."""
    backend = "neo4j"

    def __init__(self, uri: str, user: str, password: str):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver.verify_connectivity()

    def load(self, papers, similar=None, cites=None):
        rows = [{"id": str(p["paper_id"]), "title": p["title"], "year": int(p.get("year", 0)),
                 "topic": p["topic"], "venue": p.get("venue", "arXiv"),
                 "authors": _authors(p.get("authors", ""))} for p in papers]
        sim_rows = [{"src": s, "tgt": t, "score": float(sc)}
                    for s, nbrs in (similar or {}).items() for t, sc in nbrs]
        cite_rows = [{"src": s, "tgt": t} for s, t in (cites or [])]
        with self.driver.session() as ses:
            ses.run("MATCH (n) DETACH DELETE n")
            for lbl, key in [("Paper", "id"), ("Author", "name"), ("Topic", "name"), ("Venue", "name")]:
                ses.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (x:{lbl}) REQUIRE x.{key} IS UNIQUE")
            ses.run("""
                UNWIND $rows AS row
                MERGE (p:Paper {id: row.id}) SET p.title=row.title, p.year=row.year
                MERGE (t:Topic {name: row.topic}) MERGE (p)-[:ABOUT]->(t)
                MERGE (v:Venue {name: row.venue}) MERGE (p)-[:PUBLISHED_IN]->(v)
                WITH p, row UNWIND row.authors AS an
                  MERGE (a:Author {name: an}) MERGE (a)-[:WROTE]->(p)
            """, rows=rows)
            ses.run("""UNWIND $rows AS r MATCH (a:Paper {id:r.src}),(b:Paper {id:r.tgt})
                       MERGE (a)-[s:SIMILAR_TO]->(b) SET s.score=r.score""", rows=sim_rows)
            ses.run("""UNWIND $rows AS r MATCH (a:Paper {id:r.src}),(b:Paper {id:r.tgt})
                       MERGE (a)-[:CITES]->(b)""", rows=cite_rows)
        return self.stats()

    def run_cypher(self, query: str, **params) -> List[Dict]:
        with self.driver.session() as s:
            return [r.data() for r in s.run(query, **params)]

    def weighted_subgraph(self, seed_ids, limit=10, w_sim=1.0, w_topic=0.5, w_author=0.5):
        from cypher_queries_d3 import WEIGHTED_SUBGRAPH
        return self.run_cypher(WEIGHTED_SUBGRAPH, seed_ids=[str(s) for s in seed_ids],
                               limit=limit, w_sim=w_sim, w_topic=w_topic, w_author=w_author)

    def semantic_neighbours(self, paper_id, limit=8, min_score=0.30):
        from cypher_queries_d3 import SEMANTIC_NEIGHBOURS
        return self.run_cypher(SEMANTIC_NEIGHBOURS, paper_id=str(paper_id), limit=limit, min_score=min_score)

    def stats(self) -> Dict:
        rec = self.run_cypher("MATCH (n) WITH count(n) AS nodes "
                              "MATCH ()-[r]->() RETURN nodes, count(r) AS edges")[0]
        return {"backend": self.backend, **rec}

    def close(self):
        self.driver.close()


def get_graph_store(settings=None):
    if settings is not None and getattr(settings, "using_real_neo4j", lambda: False)():
        return Neo4jGraphStore(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    return NetworkxGraphStore()
