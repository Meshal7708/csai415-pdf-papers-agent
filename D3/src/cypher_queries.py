"""Example Cypher queries for the Neo4j knowledge graph (D2 deliverable: 3-5
example queries). Each is parameterised and returns rows the API/GraphRAG layer
consumes. Run them in Neo4j Browser at http://localhost:7474 after seeding.

Graph schema:
    (:Author {name})-[:WROTE]->(:Paper {id, title, year})
    (:Paper)-[:ABOUT]->(:Topic {name})
    (:Paper)-[:PUBLISHED_IN]->(:Venue {name})
"""

# 1) All papers on a given topic, newest first.
PAPERS_BY_TOPIC = """
MATCH (p:Paper)-[:ABOUT]->(t:Topic {name: $topic})
RETURN p.id AS id, p.title AS title, p.year AS year
ORDER BY p.year DESC
LIMIT $limit
"""

# 2) An author's body of work in the corpus.
PAPERS_BY_AUTHOR = """
MATCH (a:Author {name: $author})-[:WROTE]->(p:Paper)
RETURN p.id AS id, p.title AS title, p.year AS year
ORDER BY p.year DESC
LIMIT $limit
"""

# 3) Co-authors of an author, ranked by number of shared papers.
COAUTHORS = """
MATCH (a:Author {name: $author})-[:WROTE]->(p:Paper)<-[:WROTE]-(b:Author)
WHERE a <> b
RETURN b.name AS coauthor, count(p) AS shared
ORDER BY shared DESC, coauthor
LIMIT $limit
"""

# 4) Papers in a venue for a given year (provenance / slice queries).
PAPERS_BY_VENUE_YEAR = """
MATCH (p:Paper)-[:PUBLISHED_IN]->(v:Venue {name: $venue})
WHERE p.year = $year
RETURN p.id AS id, p.title AS title
ORDER BY p.title
LIMIT $limit
"""

# 5) GraphRAG expansion: papers related to a seed paper via a shared topic
#    (2-hop). This is exactly the subgraph the D3 executor walks to pull in
#    supporting chunks beyond the vector top-k.
RELATED_VIA_TOPIC = """
MATCH (seed:Paper {id: $paper_id})-[:ABOUT]->(t:Topic)<-[:ABOUT]-(p:Paper)
WHERE p <> seed
RETURN DISTINCT p.id AS id, p.title AS title, t.name AS via_topic
ORDER BY p.title
LIMIT $limit
"""

ALL = {
    "papers_by_topic": PAPERS_BY_TOPIC,
    "papers_by_author": PAPERS_BY_AUTHOR,
    "coauthors": COAUTHORS,
    "papers_by_venue_year": PAPERS_BY_VENUE_YEAR,
    "related_via_topic": RELATED_VIA_TOPIC,
}
