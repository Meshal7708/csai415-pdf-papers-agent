# GraphRAG executor — worked examples

Each shows the four stages: seed papers → weighted subgraph (Cypher) → expanded chunk pool → grounded answer with page-range citations.

## Q: *how is concept drift detected in streaming data?*
- **Seeds:** 2311.06396, 1703.06683, 2212.1472, 2509.08176, 2407.05375, 2305.11942
- **Weighted subgraph:** 1901.02052 (author+similar+topic, 9.6938); 2306.12574 (similar+topic, 7.6332); 1903.12483 (similar+topic, 6.7271); 1809.10388 (similar+topic, 5.7844)
- **Pinned set:** 10 papers · 57 candidate chunks · rerank=semantic-mmr
- **Latency:** 74.0 ms

**Answer.** However, data streams often do not conform to the same distribution over time, leading to a phenomenon called concept drift. [1] Since a fixed static model is unreliable for inferring concept-drifted data streams, es- tablishing an adaptive mechanism for detecting concept drift is crucial. [1] Benchmarks To explicitly assess the performance of classifiers and drift detectors in data streams featuring the concept drift categories outlined earlier, we introduce a set of drift di fficulties corresponding to each category within our proposed framework. [2] A key challenge in data stream learning is that the joint probability distribution of an application may change over time, i.e., there may be concept drift [3]. [3]

**Citations:** [1] Online Drift Detection with Maximum Concept Discrepancy (2407.05375), p.1; [2] A comprehensive analysis of concept drift locality in data streams (2311.06396), p.6; [3] MARLINE: Multi-Source Mapping Transfer Learning for Non-Stationary Environments (2509.08176), p.1

## Q: *what is the goal of neural architecture search in AutoML?*
- **Seeds:** 2311.0787, 2307.09099, 1908.00709, 2011.01507, 2105.01015, 2502.10459
- **Weighted subgraph:** 2505.16561 (author+similar+topic, 9.1633); 2101.10951 (similar+topic, 7.6867); 2505.05226 (similar+topic, 4.8842); 2410.09596 (similar+topic, 4.8623); 1903.12483 (similar, 3.7076)
- **Pinned set:** 12 papers · 59 candidate chunks · rerank=semantic-mmr
- **Latency:** 76.6 ms

**Answer.** In this work, we present a sampling- based AutoML search method that focuses on neural architecture search and hyperparameter joint optimization and discuss how our proposed method tackles real-world production challenges: •Industry-scale ranking model. [1] This is an error prone and time consuming task; so the Neural Architecture Search has been employed to automatically find a suitable architecture according to certain objectives. [2] Several review articles on the search for neural architecture have been written so far, but none Department of Computer Engineering, IRAN University of Science and Technology(IUST), Tehran, IRAN. [2] Hyperparameter optimization (HO) and neural architecture search (NAS) are common optimization methods used in AutoML. [3]

**Citations:** [1] AutoML for Large Capacity Modeling of Meta's Ranking Systems (2311.0787), p.1; [2] A Survey on Multi-Objective Neural Architecture Search (2307.09099), p.1; [3] AutoML for Large Capacity Modeling of Meta's Ranking Systems (2311.0787), p.2

## Q: *how do vision transformers reduce computation through token sampling?*
- **Seeds:** 2111.15667, 2210.03168, 2605.27458, 2504.04025, 2508.17858, 2412.16446
- **Weighted subgraph:** 2112.13528 (similar+topic, 4.7145); 2408.16859 (similar+topic, 3.8795); 2209.0604 (similar+topic, 3.7967); 2508.18387 (similar+topic, 3.3201); 2410.05258 (similar+topic, 3.2978)
- **Pinned set:** 12 papers · 58 candidate chunks · rerank=semantic-mmr
- **Latency:** 67.6 ms

**Answer.** Although vision transformers have a superior rep- resentation power, the high computational cost of their transformer blocks make them unsuitable for many edge devices. [1] IA-RED2[41] proposes an interpretability-aware redundancy reduction framework for vision transformers that discards less in- formative patches in the input data. [2] 3 Adaptive Token Sampler State-of-the-art vision transformers are computationally expensive since their computational costs grow quadratically with respect to the number of tokens, which is static at all stages of the network and corresponds to the number of 6 Fayyaz, Abbasi Koohpayegani, Rezaei Jafari et al. [3] In this work, we proposed a novel dif- ferentiable parameter-free module called Adaptive Token Sampler (ATS) to in- crease the efficiency of vision transformers for image and video classification. [4]

**Citations:** [1] Adaptive Token Sampling For Efficient Vision Transformers (2111.15667), p.2; [2] Adaptive Token Sampling For Efficient Vision Transformers (2111.15667), p.5; [3] Adaptive Token Sampling For Efficient Vision Transformers (2111.15667), pp.5-6; [4] Adaptive Token Sampling For Efficient Vision Transformers (2111.15667), p.14
