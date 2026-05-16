"""Procedural corpus builder for D1.

it Generates ~150 plausible cs.AI / cs.CL paper records (title + abstract + metadata)
across 6 topic clusters. Output: data/corpus.parquet

 stands in for the real PDF ingestion that will be implemented in D2.
The structure mirrors the schema specified in the project brief
(paper_id, title, authors, venue, year, topic).
"""

# ---- standard-library imports ----------------------------------------------
from __future__ import annotations          # postpone type-hint evaluation (PEP 563), lets us use modern annotations on 3.10
import hashlib                              # SHA-256 to fingerprint the corpus for the run-card
import random                               # seeded RNG for reproducible procedural generation
from pathlib import Path                    # filesystem paths that work cross-platform

# ---- third-party imports ---------------------------------------------------
import pandas as pd                         # used to assemble the corpus into a DataFrame and write Parquet

# ---- paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]  # D1/  (one level up from D1/src/)
DATA = ROOT / "data"                        # D1/data/  — where corpus.parquet lands
DATA.mkdir(parents=True, exist_ok=True)     # create the folder if it doesn't exist (no error if it does)

SEED = 42                                   # single seed used for the whole D1 pipeline (run card records this)

# Six topic clusters. Each has: title fragments, methods, datasets, an abstract template
# pool. Combining them gives plausibly-distinct papers per topic.
TOPICS = {                                  # dict literal → ordered insertion → drives output order of papers
    "transformers": {                       # topic name doubles as the gold-set label later
        "title_heads": [                    # first half of paper titles
            "Efficient Attention", "Sparse Transformers", "Long-Context",
            "Mixture-of-Experts", "Token Pruning", "Linear Attention",
            "Rotary Embeddings", "Decoder-Only", "Pre-Norm", "FlashAttention",
        ],
        "title_tails": [                    # second half of paper titles
            "for Language Modeling", "with Memory Compression",
            "via Sliding Windows", "in Low-Resource Settings",
            "for Document Understanding", "with Context Distillation",
            "for Streaming Inference", "under Tight Memory Budgets",
        ],
        "methods": ["self-attention", "key-value cache", "rotary positional embedding",
                    "block-sparse attention", "FFN gating", "grouped-query attention"],
        "datasets": ["The Pile", "C4", "PG-19", "WikiText-103"],
    },
    "rag": {                                # topic 2: retrieval-augmented generation (the project's own area)
        "title_heads": [
            "Retrieval-Augmented Generation", "Dense Retrieval", "GraphRAG",
            "Hybrid Search", "Knowledge-Graph Reasoning", "Cross-Encoder Reranking",
            "Citation-Grounded Answering", "Provenance-Aware QA",
            "Multi-Hop Retrieval", "Vector Index Quantization",
        ],
        "title_tails": [
            "for Scientific Question Answering", "over PDF Corpora",
            "with BM25/Dense Fusion", "via Subgraph Expansion",
            "with Faithfulness Constraints", "for Long Documents",
            "for Open-Domain QA",
        ],
        "methods": ["BM25", "dense bi-encoder", "cross-encoder reranker",
                    "Cypher subgraph expansion", "hybrid score fusion",
                    "k-NN search", "HNSW index"],
        "datasets": ["NaturalQuestions", "TriviaQA", "ASQA", "HotpotQA"],
    },
    "online_learning": {                    # topic 3: streaming / online ML, drift detection
        "title_heads": [
            "Online Learning", "Concept Drift", "ADWIN-Based Detection",
            "Prequential Evaluation", "Adaptive Random Forests",
            "Hoeffding Trees", "Streaming Gradient Methods",
            "Drift-Aware Classification", "Incremental Learning",
            "Adaptive Hyperparameters",
        ],
        "title_tails": [
            "in Non-Stationary Streams", "for Evolving Click Logs",
            "under Distribution Shift", "with Sliding-Window Retraining",
            "for Real-Time Recommendation", "with Lightweight Detectors",
        ],
        "methods": ["ADWIN", "EDDM", "Hoeffding bound", "naive Bayes update",
                    "exponentially weighted average", "prequential metric"],
        "datasets": ["NYC Taxi", "Electricity", "Airlines", "SEA"],
    },
    "vision": {                             # topic 4: computer vision
        "title_heads": [
            "Vision Transformers", "Self-Supervised Vision", "Object Detection",
            "Segmentation", "Diffusion Models", "Multimodal Pretraining",
            "Open-Vocabulary Recognition", "3D Reconstruction",
            "Video Understanding", "Visual Grounding",
        ],
        "title_tails": [
            "via Masked Image Modeling", "with Contrastive Learning",
            "for Edge Devices", "in the Wild", "on Limited Compute",
            "for Medical Imaging", "with Synthetic Data",
        ],
        "methods": ["ViT", "MAE", "DINO", "DETR", "stable diffusion",
                    "CLIP-style alignment"],
        "datasets": ["ImageNet", "COCO", "ADE20K", "LVIS"],
    },
    "rl_agents": {                          # topic 5: RL / agents / tool use
        "title_heads": [
            "Reinforcement Learning", "Tool-Using Agents", "ReAct Planners",
            "LangGraph Workflows", "Offline RL", "Multi-Agent Coordination",
            "Reward Modeling", "Actor-Critic Methods", "World Models",
            "LLM Agents",
        ],
        "title_tails": [
            "for Browser Automation", "with Verifier Feedback",
            "for Code Generation", "via Self-Play", "in Open Environments",
            "with Sparse Rewards",
        ],
        "methods": ["PPO", "Q-learning", "ReAct loop", "tool-call planner",
                    "self-consistency voting", "tree-of-thoughts"],
        "datasets": ["MiniWoB", "WebShop", "ALFWorld", "GSM8K"],
    },
    "automl": {                             # topic 6: AutoML / hyperparameter tuning (D1's own area)
        "title_heads": [
            "AutoML", "Hyperparameter Optimization", "Bayesian Optimization",
            "FLAML", "Optuna Studies", "Hyperband Scheduling",
            "Multi-Fidelity Tuning", "Cost-Sensitive AutoML",
            "Pipeline Search", "kNN Hyperparameter Tuning",
        ],
        "title_tails": [
            "for Tabular Classification", "under Strict Latency Budgets",
            "with Early Stopping", "for Retrieval Systems",
            "via Population-Based Training", "with Warm Starting",
        ],
        "methods": ["TPE", "BOHB", "Hyperband", "Optuna pruner",
                    "FLAML estimator search", "median pruner"],
        "datasets": ["OpenML-CC18", "AutoML Benchmark", "TabularData"],
    },
}

# Abstract template — placeholder slots are filled per-paper from the topic config.
ABSTRACT_TEMPLATE = (
    "We study {head} {tail}. The proposed approach combines {m1} with {m2} "
    "to address {issue}. We evaluate on {ds1} and {ds2}, reporting "
    "{metric1} of {v1:.2f} and {metric2} of {v2:.2f}. Compared to a strong "
    "baseline using {m3}, our method achieves {delta:+.1f}% relative improvement "
    "while reducing {cost} by {cost_drop}%. Ablations isolate the contribution "
    "of {m1} and confirm robustness under {stress}. We release code, configs, "
    "and a reproducibility checklist."
)

# Sentence-level grab-bags shared across topics
ISSUES = [                                  # the "problem" the paper claims to address
    "limited memory budgets", "label scarcity", "distribution shift",
    "noisy supervision", "long-tail distributions", "non-stationary streams",
    "computational cost", "reranking latency", "retrieval recall ceilings",
]
METRICS_PAIRS = [                           # paper claims a result on two metrics
    ("accuracy", "F1"), ("Recall@5", "NDCG@5"), ("BLEU", "ROUGE-L"),
    ("AUC", "MRR"), ("Recall@10", "MAP"),
]
COSTS = ["GPU memory", "inference latency", "FLOPs", "wall-clock time"]
STRESSES = ["domain shift", "input perturbations", "subsampled training data",
            "noisy labels", "concept drift"]

# Tiny first/last name pool used to compose author lists
FIRST_NAMES = ["Avery", "Sasha", "Nadia", "Kenji", "Mira", "Tariq", "Zheng",
               "Lucia", "Owen", "Priya", "Hugo", "Iman", "Yuri", "Anya",
               "Diego", "Mei", "Lars", "Aiko", "Nour", "Felix"]
LAST_NAMES = ["Patel", "Garcia", "Nguyen", "Kim", "Smith", "Hassan", "Ivanov",
              "Müller", "Tanaka", "Okafor", "Ricci", "Sokolov", "Chen",
              "Yamada", "Khoury", "Rossi", "Park", "Levin", "Singh", "Costa"]
VENUES = ["NeurIPS", "ICML", "ICLR", "ACL", "EMNLP", "KDD", "AAAI",
          "SIGIR", "CVPR"]


def build_corpus(target_size: int = 150) -> pd.DataFrame:
    """Generate `target_size` papers split evenly across topics."""
    rng = random.Random(SEED)               # local RNG seeded with the global SEED — keeps generation reproducible
    rows = []                               # accumulator for the per-paper records
    per_topic = target_size // len(TOPICS)  # 150 / 6 = 25 papers per topic
    seq = 0                                 # running id for paper_id (P0000, P0001, ...)
    for topic, cfg in TOPICS.items():       # iterate topics in insertion order
        for _ in range(per_topic):          # generate `per_topic` papers in this topic
            head = rng.choice(cfg["title_heads"])  # pick a title fragment
            tail = rng.choice(cfg["title_tails"])  # pick the second fragment
            title = f"{head} {tail}"        # assemble the title
            # Sample 3 distinct methods if the pool is large enough; otherwise tile the pool
            m1, m2, m3 = rng.sample(cfg["methods"], k=3) if len(cfg["methods"]) >= 3 \
                         else (cfg["methods"] * 3)[:3]
            ds1, ds2 = rng.sample(cfg["datasets"], k=min(2, len(cfg["datasets"])))  # two distinct datasets
            metric1, metric2 = rng.choice(METRICS_PAIRS)                            # one (metric1, metric2) pair
            abstract = ABSTRACT_TEMPLATE.format(   # plug all the variables into the template
                head=head.lower(), tail=tail.lower(),
                m1=m1, m2=m2, m3=m3,
                issue=rng.choice(ISSUES),
                ds1=ds1, ds2=ds2,
                metric1=metric1, metric2=metric2,
                v1=rng.uniform(0.55, 0.92),       # plausible scores in [0.55, 0.92]
                v2=rng.uniform(0.55, 0.92),
                delta=rng.uniform(1.5, 12.0),     # plausible relative improvement
                cost=rng.choice(COSTS),
                cost_drop=rng.randint(8, 45),
                stress=rng.choice(STRESSES),
            )
            n_authors = rng.randint(2, 5)         # 2..5 authors per paper
            authors = ", ".join(                  # join authors with ", " to match standard citation style
                f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
                for _ in range(n_authors)
            )
            paper_id = f"P{seq:04d}"              # zero-padded id; stable across runs because seq is deterministic
            rows.append({                         # add the record (matches the brief's required schema)
                "paper_id": paper_id,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "venue": rng.choice(VENUES),
                "year": rng.randint(2019, 2025),  # publication year window
                "topic": topic,
                "text": f"{title}. {abstract}",   # combined retrieval field used by both BM25 and TF-IDF
            })
            seq += 1                              # bump the id counter

    df = pd.DataFrame(rows)                       # rows → DataFrame
    # Stable hash of full corpus for the run card — concatenate texts and SHA-256 the bytes
    h = hashlib.sha256(
        ("\n".join(df["text"].tolist())).encode("utf-8")
    ).hexdigest()[:16]                            # short prefix is plenty for human inspection
    df.attrs["corpus_hash"] = h                   # attach as a DataFrame attribute (preserved in some serialisers)
    return df


if __name__ == "__main__":
    df = build_corpus()                           # produce the 150-paper corpus
    out_path = DATA / "corpus.parquet"            # canonical output location
    df.to_parquet(out_path, index=False)          # Parquet keeps types and is ~2x smaller than CSV
    # Also dump topic counts for sanity checking — should be 25 each
    print(f"Wrote {len(df)} papers to {out_path}")
    print(df["topic"].value_counts().to_string())
    print(f"corpus_hash={df.attrs['corpus_hash']}")
    # Save the hash separately so the run-card script can read it without reloading the parquet
    (DATA / "corpus_hash.txt").write_text(df.attrs["corpus_hash"])
