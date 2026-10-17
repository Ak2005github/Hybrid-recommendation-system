# Dataset and Artifact Conventions

This repository intentionally does not ship Amazon review data, serialized embeddings, checkpoints, or Hugging Face model weights.

## Expected local inputs

- `train_interactions.parquet`, `valid_interactions.parquet`, `test_interactions.parquet`
  - Required fields: `user_id`, `parent_asin`, `timestamp`; `rating` used for the M0 positive-only popularity diagnostic.
- `product_metadata.parquet`
  - Required fields: `parent_asin`, `title`, `categories`, `features`, `description`.

The notebooks expect split files derived from the Amazon Reviews 2023 Video Games 5-core subset. Validation/test must not be used to train the development-stage models.

## Outputs created locally

- `bpr_m1a_all_interactions.pt` from M1.
- `metadata_minilm_embeddings.npy` and `metadata_embedding_index.parquet` from M2.
- `minilm_recommender_finetuned/` and `metadata_minilm_finetuned_embeddings.npy` from M2.2.
- Final M5 outputs: `bpr_final_train_valid.pt`, `minilm_recommender_finetuned_final/`, `metadata_minilm_finetuned_final_embeddings.npy`, `final_test_results.json`.

The original user mapping must be constructed with `train['user_id'].unique()`, and the item mapping with `pd.concat([train['parent_asin'], valid['parent_asin'], test['parent_asin']]).unique()`. Do not sort those IDs when restoring the M1 checkpoint.

For the final M5 refit, only TRAIN+VALID are permitted as interaction evidence; the TEST item IDs are used only for the fixed catalog mapping and final scoring. M5's pre-existing metadata index maps `parent_asin` to embedding rows and is required for alignment.

## Exact split reconstruction

The precise upstream filtering and split-generation implementation is not included in this repo. Existing parquet inputs are required for reproducing the numerical results exactly; applying a different five-core filter or temporal tie-breaking scheme can change results.
