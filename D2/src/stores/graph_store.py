"""Knowledge graph over the corpus.

Schema (matches the brief):
    (:Author)-[:WROTE]->(:Paper)
    (:Paper)-[:ABOUT]->(:Topic)
    (:Paper)-[:PUBLISHED_IN]->(:Venue)

Two interchangeable backends behind one interface:

* `Neo4jGraphStore`   — loads the graph into a real Neo4j (docker service) with
  MERGE/constraints and runs the example Cypher in `cypher_queries.py`.
* `NetworkxGraphStore` — an in-process MultiDiGraph implementing the *same*
  query methods, so we can exercise GraphRAG-style traversals and print real
  results in the sandbox (and in CI) with no database.

The query methods are deliberately the graph operations GraphRAG (D3) needs:
papers on a topic, an author's body of work, co-authors, venue/year slices, and
"related papers via a shared topic" (a 2-hop expansion).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple


def _authors(s: str) -> List[str]:
    return [a.strip() for a in str(s).split(",") if a.strip()]


# ----------------------------------------------------------------------------- #
# NetworkX backend (sandbox / CI)
# ----------------------------------------------------------------------------- #
class NetworkxGraphStore:
    backend = "networkx"

    def __init__(self):
        import networkx as nx
        self.nx = nx
        self.g = nx.MultiDiGraph()

    def load(self, papers: List[Dict]) -> Dict[str, int]:
        g = self.g
        g.clear()
        for p in papers:
            pid = p["paper_id"]
            g.add_node(("Paper", pid), kind="Paper",
                       title=p["title"], year=int(p.get("year", 0)))
            g.add_node(("Topic", p["topic"]), kind="Topic")
            g.add_edge(("Paper", pid), ("Topic", p["topic"]), rel="ABOUT")
            venue = p.get("venue", "arXiv")
            g.add_node(("Venue", venue), kind="Venue")
            g.add_edge(("Paper", pid), ("Venue", venue), rel="PUBLISHED_IN")
            for a in _authors(p.get("authors", "")):
                g.add_node(("Author", a), kind="Author")
                g.add_edge(("Author", a), ("Paper", pid), rel="WROTE")
        kinds = defaultdict(int)
        for _, d in g.nodes(data=True):
            kinds[d["kind"]] += 1
        return {"nodes": g.number_of_nodes(), "edges": g.number_of_edges(), **kinds}

    # --- query API (mirrors the Cypher in cypher_queries.py) ---
    def papers_by_topic(self, topic: str, limit: int = 10) -> List[Tuple[str, str]]:
        out = []
        for u, v, d in self.g.in_edges(("Topic", topic), data=True):
            if d["rel"] == "ABOUT":
                out.append((u[1], self.g.nodes[u]["title"]))
        return sorted(out)[:limit]

    def papers_by_author(self, author: str, limit: int = 10) -> List[Tuple[str, str]]:
        node = ("Author", author)
        if node not in self.g:
            return []
        out = [(v[1], self.g.nodes[v]["title"])
               for _, v, d in self.g.out_edges(node, data=True) if d["rel"] == "WROTE"]
        return sorted(out)[:limit]

    def coauthors(self, author: str, limit: int = 10) -> List[Tuple[str, int]]:
        node = ("Author", author)
        if node not in self.g:
            return []
        papers = {v for _, v, d in self.g.out_edges(node, data=True) if d["rel"] == "WROTE"}
        counts: Dict[str, int] = defaultdict(int)
        for paper in papers:
            for a, _, d in self.g.in_edges(paper, data=True):
                if d["rel"] == "WROTE" and a != node:
                    counts[a[1]] += 1
        return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:limit]

    def papers_by_venue_year(self, venue: str, year: int, limit: int = 10):
        out = []
        for u, _, d in self.g.in_edges(("Venue", venue), data=True):
            if d["rel"] == "PUBLISHED_IN" and self.g.nodes[u]["year"] == year:
                out.append((u[1], self.g.nodes[u]["title"]))
        return sorted(out)[:limit]

    def related_via_topic(self, paper_id: str, limit: int = 10):
        """2-hop: paper -> its topics -> sibling papers (GraphRAG expansion)."""
        node = ("Paper", paper_id)
        if node not in self.g:
            return []
        topics = [v for _, v, d in self.g.out_edges(node, data=True) if d["rel"] == "ABOUT"]
        sibs = set()
        for t in topics:
            for u, _, d in self.g.in_edges(t, data=True):
                if d["rel"] == "ABOUT" and u != node:
                    sibs.add((u[1], self.g.nodes[u]["title"]))
        return sorted(sibs)[:limit]

    def stats(self) -> Dict:
        kinds = defaultdict(int)
        for _, d in self.g.nodes(data=True):
            kinds[d["kind"]] += 1
        return {"backend": self.backend, "nodes": self.g.number_of_nodes(),
                "edges": self.g.number_of_edges(), **kinds}

    def close(self):
        pass


# ----------------------------------------------------------------------------- #
# Neo4j backend (docker service)
# ----------------------------------------------------------------------------- #
class Neo4jGraphStore:
    backend = "neo4j"

    def __init__(self, uri: str, user: str, password: str):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver.verify_connectivity()

    def load(self, papers: List[Dict]) -> Dict[str, int]:
        with self.driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")
            s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE")
            s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE")
            s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE")
            s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (v:Venue) REQUIRE v.name IS UNIQUE")
            rows = [{
                "id": p["paper_id"], "title": p["title"], "year": int(p.get("year", 0)),
                "topic": p["topic"], "venue": p.get("venue", "arXiv"),
                "authors": _authors(p.get("authors", "")),
            } for p in papers]
            s.run(
                """
                UNWIND $rows AS row
                MERGE (p:Paper {id: row.id})
                  SET p.title = row.title, p.year = row.year
                MERGE (t:Topic {name: row.topic})
                MERGE (p)-[:ABOUT]->(t)
                MERGE (v:Venue {name: row.venue})
                MERGE (p)-[:PUBLISHED_IN]->(v)
                WITH p, row
                UNWIND row.authors AS aname
                  MERGE (a:Author {name: aname})
                  MERGE (a)-[:WROTE]->(p)
                """,
                rows=rows,
            )
            rec = s.run(
                "MATCH (n) WITH count(n) AS nodes "
                "MATCH ()-[r]->() RETURN nodes, count(r) AS edges"
            ).single()
            return {"nodes": rec["nodes"], "edges": rec["edges"]}

    def run_cypher(self, query: str, **params) -> List[Dict]:
        with self.driver.session() as s:
            return [r.data() for r in s.run(query, **params)]

    # query API wrappers around cypher_queries.py
    def papers_by_topic(self, topic: str, limit: int = 10):
        from cypher_queries import PAPERS_BY_TOPIC
        return [(r["id"], r["title"]) for r in self.run_cypher(PAPERS_BY_TOPIC, topic=topic, limit=limit)]

    def papers_by_author(self, author: str, limit: int = 10):
        from cypher_queries import PAPERS_BY_AUTHOR
        return [(r["id"], r["title"]) for r in self.run_cypher(PAPERS_BY_AUTHOR, author=author, limit=limit)]

    def coauthors(self, author: str, limit: int = 10):
        from cypher_queries import COAUTHORS
        return [(r["coauthor"], r["shared"]) for r in self.run_cypher(COAUTHORS, author=author, limit=limit)]

    def papers_by_venue_year(self, venue: str, year: int, limit: int = 10):
        from cypher_queries import PAPERS_BY_VENUE_YEAR
        return [(r["id"], r["title"]) for r in self.run_cypher(PAPERS_BY_VENUE_YEAR, venue=venue, year=year, limit=limit)]

    def related_via_topic(self, paper_id: str, limit: int = 10):
        from cypher_queries import RELATED_VIA_TOPIC
        return [(r["id"], r["title"]) for r in self.run_cypher(RELATED_VIA_TOPIC, paper_id=paper_id, limit=limit)]

    def stats(self) -> Dict:
        rec = self.run_cypher(
            "MATCH (n) WITH count(n) AS nodes MATCH ()-[r]->() RETURN nodes, count(r) AS edges")[0]
        return {"backend": self.backend, **rec}

    def close(self):
        self.driver.close()


def get_graph_store(settings):
    if settings.using_real_neo4j():
        return Neo4jGraphStore(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    return NetworkxGraphStore()
