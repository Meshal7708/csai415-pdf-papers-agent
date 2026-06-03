# Knowledge graph — example queries (D2)

Backend: `networkx` · {"backend": "networkx", "nodes": 365, "edges": 425, "Paper": 60, "Topic": 6, "Venue": 1, "Author": 298}

These mirror the parameterised Cypher in `src/cypher_queries.py`.

**1. papers_by_topic('rag')** →
   - 2310.06626: Topic-DPR: Topic-based Prompts for Dense Passage Retrieval
   - 2402.11035: Dense Passage Retrieval: Is it Retrieving?
   - 2408.08444: W-RAG: Weakly Supervised Dense Retrieval in RAG for Open-domain Question Answering
   - 2412.14751: Query pipeline optimization for cancer patient question answering systems
   - 2501.09940: Passage Segmentation of Documents for Extractive Question Answering

**2. papers_by_author('Yueyang Cang')** →
   - 2501.17486: DINT Transformer

**3. coauthors('Yueyang Cang')** →
   - Erlu Zhao (1 shared)
   - Li Shi (1 shared)
   - Xiaoteng Zhang (1 shared)
   - Yuhang Liu (1 shared)

**4. papers_by_venue_year('arXiv', 2025)** →
   - 2501.09940: Passage Segmentation of Documents for Extractive Question Answering
   - 2501.15067: CG-RAG: Research Question Answering by Citation Graph Retrieval-Augmented LLMs
   - 2501.17486: DINT Transformer
   - 2502.10459: LLM4GNAS: A Large Language Model Based Toolkit for Graph Neural Architecture Search
   - 2502.20245: From Retrieval to Generation: Comparing Different Approaches

**5. related_via_topic('2501.17486')  [GraphRAG 2-hop]** →
   - 1906.09777: A Tensorized Transformer for Language Modeling
   - 1909.06639: Tree Transformer: Integrating Tree Structures into Self-Attention
   - 1910.11871: Towards Online End-to-end Transformer Automatic Speech Recognition
   - 1912.11637: Explicit Sparse Transformer: Concentrated Attention Through Explicit Selection
   - 2109.07152: Incorporating Residual and Normalization Layers into Analysis of Masked Language Models