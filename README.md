# Personalized Hybrid Recommendation System | Amazon Video Games

A two-stage personalized recommender built using **Bayesian Personalized Ranking (BPR)**, **contrastively fine-tuned MiniLM item embeddings**, **recency-aware user profiles**, and **linear score fusion**.

This repository documents the actual development progression, from popularity and BPR baselines through metadata retrieval, contrastive domain adaptation, candidate union, recency modeling, final validation, and a one-time temporal TEST evaluation.

## Architecture

```text
Offline:
  TRAIN (+ VALID for the final TEST refit)
   ├── BPR user/item embeddings (64 dimensions)
   └── interaction-derived text pairs → MiniLM fine-tuning
          └── encode each game's metadata (384 dimensions)
                       ↓
Serving:
  user → BPR top-500 ───────────────┐
                                   ├── union (up to 1,000 items)
  user → MiniLM metadata top-500 ──┘
          (all-history 0.40 + recent-2 0.60)
                       ↓
         per-user normalized scores
         0.50 × BPR_z + 0.50 × metadata_z
                       ↓
                    Top-10
```

Online recommendations use precomputed embeddings, dot products, top-k retrieval, and score fusion; no Transformer text encoding is needed per request. BPR retrieves train-observed items; metadata can retrieve across the full catalog.

## Notebook sequence

| Stage | Notebook | Purpose |
| --- | --- | --- |
| M0–M1 | [01_M0_M1_popularity_BPR.ipynb](notebooks/01_M0_M1_popularity_BPR.ipynb) | Popularity benchmarks and accepted all-interaction BPR |
| M2 | [02_M2_metadata_minilm.ipynb](notebooks/02_M2_metadata_minilm.ipynb) | Construct item text and encode 25,612 game metadata records |
| M2.2 | [03_M2_2_finetune_minilm.ipynb](notebooks/03_M2_2_finetune_minilm.ipynb) | MiniLM domain fine-tuning with up to 60K chronological co-interaction pairs; frozen vs fine-tuned validation |
| M2.5 | [04_M2_5_candidate_union.ipynb](notebooks/04_M2_5_candidate_union.ipynb) | Combine BPR and MiniLM top-500 candidates to improve retrieval coverage |
| M2.6 | [05_M2_6_recency_metadata.ipynb](notebooks/05_M2_6_recency_metadata.ipynb) | Long-term vs recent-2 profiles; final rho=0.60 |
| M4 | [06_M4_final_hybrid_validation.ipynb](notebooks/06_M4_final_hybrid_validation.ipynb) | Validated frozen-MiniLM hybrid retrieval + 50/50 fusion, before fine-tuned MiniLM was accepted |
| M5 | [07_M5_final_test.ipynb](notebooks/07_M5_final_test.ipynb) | Refit BPR and MiniLM on TRAIN+VALID and evaluate TEST once |


## Dataset and protocol

- Source: [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/), Video Games 5-core subset.
- 94,762 users, 25,612 products, 814,586 total interactions.
- Temporal leave-last-two-out: 625,062 TRAIN, 94,762 VALID, and 94,762 TEST rows.
- One held-out relevant item per user per evaluation split, full-catalog recommendation (not sampled-negative ranking).
- Validation used TRAIN history only. Following architecture selection, final BPR and MiniLM were refit using TRAIN+VALID; TEST was scored once.
- Candidate Recall@500 measures whether the target appears in the union of the two top-500 retrieval lists. For single-positive evaluation, HR@10 equals Recall@10.

## Measured results

### Final one-time TEST evaluation

| Recommender | Candidate recall | HR@10 | MRR@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| BPR (final TRAIN+VALID refit) | 0.343545 | 0.042306 | 0.016353 | 0.022380 |
| Fine-tuned recency metadata | 0.252411 | 0.026846 | 0.010333 | 0.014161 |
| **Final hybrid** | **0.484667** | **0.067970** | **0.026642** | **0.036237** |

**Relative to the refit BPR baseline:** candidate recall **+41.1%**, HR@10 **+60.7%**, NDCG@10 **+61.9%**.

### Single-user serving benchmark

A one-user-at-a-time NumPy/CPU-side benchmark in a Colab GPU-enabled session reported P50 **11.96 ms**, P95 **25.67 ms**, and throughput **73.7 requests/second** (sampled ~5,000 users, 100 warm-ups). These are implementation- and hardware-specific measurements, **not** a matched production benchmark against SASRec/HSTU.

## Reproduction

1. Use Python 3.10+; install dependencies with `pip install -r requirements.txt`. GPU is recommended for MiniLM fine-tuning.
2. Obtain the original interaction and metadata parquet files from the upstream dataset and reproduce the 5-core temporal split described above. The processed parquet splits are **not included**.
3. Put these local files into the working data directory: `train_interactions.parquet`, `valid_interactions.parquet`, `test_interactions.parquet`, and `product_metadata.parquet`.
4. Run M0/M1 then M2 to create the BPR checkpoint, `metadata_minilm_embeddings.npy`, and `metadata_embedding_index.parquet`. M2.5/M2.6/M4 are development validation notebooks. M2.2 trains and validates the fine-tuned encoder.
5. Run M5 **only for the final refit and one-time TEST evaluation**, with its `DATA_DIR` pointing to the data directory. It expects the metadata embedding index from M2.

**Compute:** MiniLM fine-tuning and full-catalog evaluation are computationally expensive. Saved model weights, encoded vectors, intermediate matrices, datasets, and temporary training artifacts are intentionally not tracked. The source data split-generation script is not included, so recreating the exact data requires reconstructing the preprocessing separately.

## Fixed model settings

- BPR: dim=64, batch=2048, 10 epochs, Adam lr=1e-3, weight decay=1e-6.
- MiniLM: `sentence-transformers/all-MiniLM-L6-v2`, up to 60,000 pairs, MultipleNegativesRankingLoss, batch=64, one epoch, lr=2e-5.
- Retrieval: top 500 per branch; recency window=2; rho=0.60; linear fusion weights=0.50/0.50.
- Selection metric: NDCG@10. No tuning on TEST.
