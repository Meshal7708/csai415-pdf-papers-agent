// Regenerates D2_Report.docx — the D2 deliverable report.
//   NODE_PATH="$HOME/.npm-global/lib/node_modules" node build_report.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType, ShadingType,
  ImageRun, Header, Footer, PageNumber, TableOfContents,
} = require("docx");

const DIR = __dirname;
const PNG = path.join(DIR, "diagram", "dataflow.png");

// ---------- helpers ----------
const FONT = "Arial";
const border = { style: BorderStyle.SINGLE, size: 1, color: "BFBFBF" };
const borders = { top: border, bottom: border, left: border, right: border };
const HEAD_FILL = "1F3864", ALT_FILL = "EEF2FB";

function P(text, opts = {}) {
  return new Paragraph({
    spacing: { after: opts.after ?? 120, before: opts.before ?? 0 },
    alignment: opts.align,
    children: [new TextRun({ text, bold: opts.bold, italics: opts.italics,
      size: opts.size, color: opts.color })],
  });
}
function runs(children, opts = {}) {
  return new Paragraph({ spacing: { after: opts.after ?? 120 }, children });
}
function H1(t) { return new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] }); }
function H2(t) { return new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] }); }
function bullet(t) {
  return new Paragraph({ numbering: { reference: "bul", level: 0 },
    spacing: { after: 60 }, children: [new TextRun(t)] });
}
function bulletRuns(children) {
  return new Paragraph({ numbering: { reference: "bul", level: 0 },
    spacing: { after: 60 }, children });
}

function cell(text, { w, fill, bold, align } = {}) {
  return new TableCell({
    borders, width: { size: w, type: WidthType.DXA },
    shading: fill ? { fill, type: ShadingType.CLEAR } : undefined,
    margins: { top: 60, bottom: 60, left: 110, right: 110 },
    children: [new Paragraph({ alignment: align,
      children: [new TextRun({ text, bold, color: fill === HEAD_FILL ? "FFFFFF" : undefined,
        size: 19 })] })],
  });
}
function table(widths, headers, rows) {
  const total = widths.reduce((a, b) => a + b, 0);
  const headRow = new TableRow({ tableHeader: true,
    children: headers.map((h, i) => cell(h, { w: widths[i], fill: HEAD_FILL, bold: true,
      align: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER })) });
  const bodyRows = rows.map((r, ri) => new TableRow({
    children: r.map((c, i) => cell(String(c), { w: widths[i],
      fill: ri % 2 ? ALT_FILL : undefined,
      bold: i === 0, align: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER })) }));
  return new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [headRow, ...bodyRows] });
}

const CW = 9360; // content width (US Letter, 1" margins)

// ---------- document ----------
const children = [];

// Title block
children.push(new Paragraph({ spacing: { after: 60 },
  children: [new TextRun({ text: "CSAI415 — Deliverable D2", bold: true, size: 22, color: "1F3864" })] }));
children.push(new Paragraph({ spacing: { after: 40 },
  children: [new TextRun({ text: "Retrieval Stack & Graph Build", bold: true, size: 40 })] }));
children.push(P("PDF-Papers AI Agent: Hybrid Retrieval + GraphRAG with Online Learning and AutoML",
  { italics: true, size: 22, after: 120 }));
children.push(runs([
  new TextRun({ text: "Team: ", bold: true, size: 20 }),
  new TextRun({ text: "Khalifa · Meshal · Mahmoud · Ahmed · Essam", size: 20 }),
], { after: 40 }));
children.push(P("Stack: Python · MongoDB · Qdrant · Neo4j · FastAPI · bge-small-en-v1.5 (fastembed/ONNX) · BM25",
  { size: 18, color: "595959", after: 200 }));

// 1. Overview
children.push(H1("1. Overview"));
children.push(P("D2 builds the retrieval backbone for the project: a real PDF → text → chunks → embeddings ingestion pipeline, three persistent stores (MongoDB, Qdrant, Neo4j), a hybrid BM25 + dense /search API that returns grounded citations with page ranges, and a knowledge graph with example Cypher queries. It continues directly from D1 — the hybrid fusion weight λ is the D1 Optuna winner, and the dense side is now the real bge-small-en-v1.5 encoder that D1 left stubbed."));
children.push(P("The corpus is 60 open-access arXiv PDFs (10 papers across each of 6 topics: transformers, RAG, online learning, vision, RL agents, AutoML), parsed to 4,155 chunks. Every component runs against the Dockerised services on a workstation, and against in-process fallbacks (mongomock, Qdrant in-memory/on-disk, NetworkX) where Docker is unavailable — the same source code, selected by environment variables."));

// 2. Architecture + diagram
children.push(H1("2. System architecture"));
children.push(P("A query fans out to two retrievers — BM25 over chunk text and dense approximate-nearest-neighbour (ANN) search over bge embeddings in Qdrant. Each side returns a candidate pool; the pools are unioned, each signal is min-max normalised, and the two are fused as λ·bm25 + (1−λ)·dense. The top-k chunks are returned with a citation assembled from MongoDB provenance (title, paper_id, page range). The Neo4j graph backs metadata queries today and the 2-hop subgraph expansion that GraphRAG will use in D3."));
if (fs.existsSync(PNG)) {
  children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 60 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(PNG),
      transformation: { width: 600, height: 281 },
      altText: { title: "D2 dataflow", description: "ingest to stores to retrieval to graph", name: "dataflow" } })] }));
  children.push(P("Figure 1. Dataflow: ingest → stores → retrieval → graph.",
    { italics: true, size: 18, align: AlignmentType.CENTER, after: 160 }));
}

// 3. Ingestion + schema
children.push(H1("3. Ingestion pipeline & schema"));
children.push(P("PDFs are parsed with pypdf into per-page text, recording each page’s character span so any character offset maps back to a 1-based page number (the page map). A sliding-window chunker (900 characters, 150 overlap, minimum 120) cuts chunks over the concatenated text and snaps the right edge to the nearest sentence boundary for readability; a chunk may legitimately span a page break, captured as page_start..page_end. Leading pages are capped at 20 per PDF (configurable) because arXiv papers’ substantive content is up front and the long tail of references/appendices bloats chunk counts with low-value text."));
children.push(P("MongoDB schema (provenance-first):", { bold: true, after: 80 }));
children.push(table([1700, 4260, 3400],
  ["Collection", "Key fields", "Purpose"],
  [
    ["documents", "_id=paper_id, title, authors, venue, year, topic, doi, pdf_path/url, n_pages, n_chunks, sha256, run_id", "One record per paper; bibliographic + ingestion provenance"],
    ["chunks", "_id, paper_id, chunk_index, text, page_start/end, char_start/end, n_tokens, sha256, run_id", "Citable unit; page range makes every snippet groundable"],
    ["runs", "run_id, n_docs, n_chunks, embedder, chunk params, ts", "Ingestion run card — trace any chunk to the run that made it"],
    ["cache", "query, payload, created_at (TTL 3600s)", "Short-lived query cache; TTL index auto-expires entries"],
  ]));
children.push(P("Qdrant stores one 384-d cosine vector per chunk with a payload (paper_id, topic, page range) and payload indexes on paper_id/topic for filtered search and source pinning (used by D3 safety). Point ids are a stable UUIDv5 of the chunk id, so an ANN hit always maps back to its Mongo record.",
  { before: 80 }));

// 4. Hybrid retrieval
children.push(H1("4. Hybrid retrieval"));
children.push(P("BM25 (rank_bm25) is built once over all chunk texts. For a query, the dense side embeds with bge (asymmetric: queries get the model’s search-instruction prefix) and retrieves an ANN pool from Qdrant; the lexical side takes the BM25 top pool. The two pools are unioned, each score min-max normalised, and fused. Fusing over the union of pools — rather than scoring the whole corpus densely — is how production hybrid retrieval works: ANN-retrieve, then re-score, keeping latency flat as the corpus grows. λ = 0.5 is frozen from the D1 Optuna study and is overridable per request — it is exactly the knob the D1 online learner adapts."));

// 5. Graph + cypher
children.push(H1("5. Knowledge graph & Cypher"));
children.push(P("The graph models (:Author)-[:WROTE]->(:Paper), (:Paper)-[:ABOUT]->(:Topic) and (:Paper)-[:PUBLISHED_IN]->(:Venue). Loaded over the 60-paper corpus it has 365 nodes and 425 edges (60 Papers, 298 Authors, 6 Topics, 1 Venue). Five parameterised example queries ship in src/cypher_queries.py; the NetworkX fallback implements the same query API so results are identical with or without a Neo4j server."));
children.push(table([3100, 6260],
  ["Cypher query", "Returns"],
  [
    ["papers_by_topic($topic)", "All papers on a topic, newest first"],
    ["papers_by_author($author)", "An author’s body of work in the corpus"],
    ["coauthors($author)", "Co-authors ranked by shared papers"],
    ["papers_by_venue_year($venue,$year)", "Papers in a venue for a given year"],
    ["related_via_topic($paper_id)", "GraphRAG 2-hop: papers sharing a topic with a seed paper"],
  ]));

// 6. Evaluation
children.push(H1("6. Evaluation"));
children.push(P("Evaluated on the 59 D1 gold queries whose relevant papers fall in this 60-paper subset, at k = 5. Chunk results are deduped to a ranked paper list before scoring (gold is paper-level). Recall@5 and nDCG@5 use the same binary-relevance definitions as D1 for comparability."));
children.push(table([3360, 1500, 1500, 1500, 1500],
  ["Retriever", "Recall@5", "nDCG@5", "mean ms", "p95 ms"],
  [
    ["BM25 only (λ=1.0)", "0.624", "0.860", "17.9", "28.0"],
    ["Dense only (λ=0.0)", "0.615", "0.829", "18.2", "29.2"],
    ["Hybrid (λ=0.5)", "0.611", "0.841", "18.1", "28.0"],
  ]));
children.push(P("Hybrid by query type (Recall@5 / nDCG@5):", { bold: true, before: 120, after: 80 }));
children.push(table([3360, 2000, 2000, 2000],
  ["Query type", "Recall@5", "nDCG@5", "n"],
  [
    ["title", "1.000", "1.000", "18"],
    ["targeted", "0.526", "0.761", "17"],
    ["broad", "0.379", "0.779", "24"],
  ]));
children.push(P("Recall@5 = 0.611 clears the ≥ 0.60 target and p95 latency ≈ 28 ms is far under the 2 s CPU target. Title queries are answered perfectly; broad queries are structurally capped because their gold set is a whole topic (up to 10 papers) while k = 5.", { before: 100 }));
children.push(P("Honest finding.", { bold: true, before: 80, after: 40 }));
children.push(P("At a fixed λ = 0.5, BM25 alone is marginally ahead of the hybrid on this corpus: bge clearly helps semantic/title queries but min-max fusion dilutes BM25’s edge on keyword-heavy ones. This is precisely the gap the later stages close — D1’s online learner already adapts λ per query, and D3 adds cross-encoder reranking and graph-guided expansion. We report it rather than tuning λ to flatter the hybrid."));
children.push(P("Example (query: “retrieval augmented generation with citations”, 17.5 ms):", { bold: true, before: 100, after: 40 }));
children.push(P("1.  CG-RAG: Research Question Answering by Citation Graph Retrieval-Augmented LLMs (2501.15067), p.1 — score 0.617", { size: 18, after: 30 }));
children.push(P("2.  Passage Segmentation of Documents for Extractive Question Answering (2501.09940), p.8 — score 0.598", { size: 18, after: 30 }));
children.push(P("3.  W-RAG: Weakly Supervised Dense Retrieval in RAG for Open-domain QA (2408.08444), p.11 — score 0.596", { size: 18, after: 120 }));

// 7. Engineering & reproducibility
children.push(H1("7. Engineering & reproducibility"));
children.push(bullet("One-command stores: docker compose up -d brings up Mongo (27017), Qdrant (6333) and Neo4j (7474/7687) with health checks and named volumes."));
children.push(bullet("FastAPI exposes /search, /ingest, /stats, /healthz and /graph/related; Swagger UI at /docs."));
children.push(bullet("Scripts: download_pdfs.py (fetch PDFs + write papers.csv), seed_stores.py (ingest to services), eval_search.py (metrics + examples); plus seed_resumable.py / eval_from_cache.py for the no-Docker path."));
children.push(bullet("Tests: pytest smoke tests build synthetic PDFs and exercise the full ingest → store → search → graph path with an offline embedder — no downloads, no services (4/4 passing)."));
children.push(bullet("Pinned: shared seed 42; data/papers.csv pins the corpus; .env.example documents every switch; run_card.yaml records the active config and headline results."));

// 8. Decisions & pitfalls
children.push(H1("8. Key decisions & pitfalls"));
children.push(bullet("Embedder = fastembed, not sentence-transformers. Same bge-small-en-v1.5 weights via ONNX runtime: ~10× smaller install, CPU-only, no torch. The HybridRetriever interface from D1 was preserved so this was a drop-in swap."));
children.push(bullet("paper_id must be read as a string. arXiv ids like 2410.05250 are parsed as floats by default and silently lose the trailing zero, which breaks gold-matching and graph node ids; all CSV reads pin dtype=str."));
children.push(bullet("Page cap (max_pages=20). Bounds chunk counts and ingest time; configurable, 0 = all pages."));
children.push(bullet("One code path, two deployments. Each store auto-selects a real client or an in-process fallback from environment variables, so tests and a laptop demo share the exact production code."));

// 9. Rubric + contributions
children.push(H1("9. Mapping to the D2 rubric"));
children.push(table([3000, 6360],
  ["Criterion", "Where it is met"],
  [
    ["Ingest & storage (3%)", "pypdf parsing + page map, overlap chunking, Mongo provenance schema + TTL, Qdrant payload indexes"],
    ["Hybrid retrieval (5%)", "BM25 + dense fusion, metrics table + ablation, examples with page-range citations"],
    ["Graph build (5%)", "Neo4j schema (Authors/Papers/Topics/Venues), 5 Cypher queries, dataflow diagram"],
    ["Engineering (2%)", "docker-compose, FastAPI, seed/eval scripts, pytest smoke tests"],
  ]));
children.push(H2("Team & contributions"));
children.push(table([2100, 3500, 3760],
  ["Member", "Owns", "Primary files"],
  [
    ["Khalifa", "Hybrid search + API", "src/hybrid_search.py, api/main.py"],
    ["Meshal", "Ingestion + chunking/provenance", "src/ingest.py, src/pipeline.py"],
    ["Mahmoud", "Stores + embedder", "src/stores/, src/embedder.py"],
    ["Ahmed", "Graph + Cypher + diagram", "src/cypher_queries.py, src/stores/graph_store.py, diagram/"],
    ["Essam", "Corpus, eval, infra, report", "scripts/, docker-compose.yml, run_card.yaml, tests/"],
  ]));

// ---------- assemble ----------
const doc = new Document({
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: FONT, color: "1F3864" },
        paragraph: { spacing: { before: 260, after: 120 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 23, bold: true, font: FONT, color: "2E5496" },
        paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
      alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 460, hanging: 260 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 },
      margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "CSAI415 D2 — Retrieval Stack & Graph Build   |   Page ", size: 16, color: "808080" }),
        new TextRun({ children: [PageNumber.CURRENT], size: 16, color: "808080" })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(path.join(DIR, "D2_Report.docx"), buf);
  console.log("wrote D2_Report.docx");
});
