# SemanticSASRec: Language-Model Semantics for Sequential Recommendation

Research code accompanying the draft *When Semantic Alignment Helps Sequential
Recommendation*. See the [paper draft](Paper%20Presentation/SemanticSASRec.pdf) and
[presentation](Paper%20Presentation/SemanticSASRec.pptx). The current validation
status and remaining publication checks are recorded in [§8](#8-validation-and-publication-status).

The research question is simple: **RLMRec's contrastive semantic-alignment recipe was published on graph-CF backbones (LightGCN, SGL, SimGCL, DCCF, etc.). Does that same recipe transfer to a sequential SASRec backbone?** To answer it we train three models head-to-head on the same data splits and evaluator:

1. **SASRec** — sequential baseline based on Kang & McAuley (ICDM 2018).
2. **RLMRec+LightGCN** — graph baseline adapting the RLMRec-Con alignment recipe (Ren et al., WWW 2024) to this project’s profiles and evaluation protocol.
3. **SemanticSASRec** — project-specific extension: SASRec backbone with RLMRec's two InfoNCE alignment heads (user-state ↔ user-profile-text, item-embedding ↔ item-profile-text). Not in the original RLMRec paper.

The training scripts support **MovieLens-1M**, **Yelp** (Open Dataset), and **Amazon-Books** (Reviews 2023), with full-catalog leave-one-out evaluation. This is a project implementation, not a claim to reproduce the original papers’ numerical results: dataset versions, preprocessing, templated profiles, encoder, and evaluation can differ. The original RLMRec paper and the separate graph baseline retain the name RLMRec; the sequential extension is consistently named **SemanticSASRec** (`semantic_sasrec` in CLI identifiers and output paths).

---

## Table of contents

1. [Repository layout](#1-repository-layout)
2. [What every file does](#2-what-every-file-does)
3. [Environment setup](#3-environment-setup)
4. [Getting the datasets](#4-getting-the-datasets)
5. [Reproducing the paper](#5-reproducing-the-paper)
6. [Per-script flag reference](#6-per-script-flag-reference)
7. [Output / artifact layout](#7-output--artifact-layout)
8. [Validation and publication status](#8-validation-and-publication-status)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Repository layout

```
repository/
├── README.md                                ← this file
│
├── Paper Presentation/
│   ├── SemanticSASRec.pdf                   ← paper draft
│   └── SemanticSASRec.pptx                  ← presentation
│
├── tests/
│   └── test_training_modes.py              ← CPU/MPS training/evaluation regression tests
│
├── sasrec_model.py                          ← SASRec architecture
├── semantic_sasrec_model.py                 ← SemanticSASRec wrapper + InfoNCE
├── lightgcn_model.py                        ← LightGCN + LightGCNRLMRec wrapper
│
├── sasrec_data.py                           ← MovieLens-1M loader (auto-download)
├── yelp_data.py                             ← Yelp loader (manual download)
├── amazon_data.py                           ← Amazon Reviews 2023 loader (auto-download)
├── rlmrec_data.py                           ← ML-1M-specific semantic-profile builder
├── eval_full_rank.py                        ← shared full-rank LOO evaluator
│
├── sasrec_train.py                          ← train pure SASRec
├── semantic_sasrec_train.py                 ← train SemanticSASRec
├── rlmrec_lightgcn_train.py                 ← train RLMRec+LightGCN
│
├── seed_aggregation.py                      ← multi-seed aggregation helpers
├── run_all.py                              ← orchestrator: ML-1M, 3 models × N seeds
├── run_all_yelp.py                          ← orchestrator: Yelp,   3 models × N seeds
├── run_all_amazon.py                        ← orchestrator: Amazon, 3 models × N seeds
├── run_everything.py                        ← top-level: runs all three orchestrators + make_overview
├── run_ablations.py                         ← ablation grid for SemanticSASRec (and RLMRec+LightGCN)
└── make_overview.py                         ← generate per-dataset results_overview.md
```

---

## 2. What every file does

### 2.1 Model files

| File | Purpose |
|------|---------|
| `sasrec_model.py` | Implements `SASRec`: an item-embedding table, learned positional embeddings, N causal self-attention blocks, and a final-position dot-product head. Exposes `forward(input_seq, pos_seq, neg_seq)` for BCE-style training and `score_all_items(input_seq) -> (B, num_items+1)` for full-rank scoring. Implementation defaults are `hidden=50, blocks=2, heads=1, dropout=0.2, max_len=200`. |
| `semantic_sasrec_model.py` | Defines `ProjectionHead` (linear or 2-layer GELU MLP), `SemanticSASRec` (wraps `SASRec` with two `ProjectionHead`s, one for user-state and one for item-embedding), and `info_nce(projected, target, temperature)` — symmetric cross-entropy on cosine-similarity logits. This is the InfoNCE alignment loss that ties CF embeddings to the frozen sentence-transformer space. |
| `lightgcn_model.py` | Implements `LightGCN` (He et al., 2020) using edge-based `index_add_` propagation (works on MPS without sparse matmul), `LightGCNRLMRec` (wraps `LightGCN` with the same `ProjectionHead`s), and `build_norm_adjacency(...)` which constructs the symmetric-normalized bipartite adjacency. |

### 2.2 Data + preprocessing modules

The semantic loaders are `rlmrec_data.load_ml1m_with_semantics`, `yelp_data.load_yelp_with_semantics`, and `amazon_data.load_amazon_with_semantics`. They accept data/cache directories, encoder name, device, and applicable dataset filters, and return a bundle with splits, ID mappings, counts, and user/item semantic tensors. The plain SASRec trainer uses the interaction loaders without encoding text.

| File | Purpose |
|------|---------|
| `sasrec_data.py` | MovieLens-1M loader: `download_ml1m` (auto-fetches the zip from `files.grouplens.org`), `build_sequences` (treats ratings as implicit events, sorts by timestamp, remaps to 1-indexed IDs, excludes user sequences shorter than five; no item-core filtering), `split_sequences` (leave-one-out), `SASRecTrainDataset` (yields `(input_seq, pos_seq, neg_seq)` for BCE), and `build_eval_inputs`. |
| `yelp_data.py` | Yelp Open Dataset loader. **Manual download** is required (Yelp's TOS). Streams the ~5 GB review JSON, filters by `start_year`, applies iterative k-core (`min_inter` default 5), builds business + user profile texts from metadata, encodes them with a frozen sentence-transformer, and caches the result as `.npy`. Function `load_yelp_with_semantics` is the one-stop loader used by the SemanticSASRec and RLMRec+LightGCN training scripts. |
| `amazon_data.py` | Amazon Reviews 2023 (McAuley Lab) loader. **Auto-downloads** the gzipped review and metadata JSONL for one category (default `Books`). Uses the 2023 release, not necessarily the same Amazon-Books data or split as the original RLMRec paper. Same start-year and iterative k-core approach as Yelp. |
| `rlmrec_data.py` | ML-1M-specific semantic-profile builder. Constructs item texts as `"Title (Year). Genres: A, B, C."` and user texts as `"A <age> <gender> <occupation> who enjoys <top genres>. Recently watched: <recent titles>."`. Encodes them with `sentence-transformers` and caches the result. |
| `eval_full_rank.py` | Shared full-rank leave-one-out evaluator. Computes HR@K and NDCG@K for K ∈ {1, 5, 10, 20} plus MRR. Applies the shared candidate masking and metrics described in §2.5, subject to the repeated-target limitation in §2.6. |

### 2.3 Training scripts

| File | Purpose |
|------|---------|
| `sasrec_train.py` | End-to-end pure-SASRec trainer. BCE-with-logits loss; auto-picks CUDA → MPS → CPU; selects the best epoch by val NDCG@10; writes `<out_dir>/{train_history.csv, sasrec_best.pt, test_metrics.csv, run_metadata.json, *.png}`. Supports `--dataset {ml1m,yelp,amazon}`. |
| `semantic_sasrec_train.py` | SemanticSASRec trainer. Same SASRec backbone, plus two InfoNCE alignment heads at every batch. Loss is `cf_loss + λ_user · user_align + λ_item · item_align`. |
| `rlmrec_lightgcn_train.py` | RLMRec+LightGCN trainer under the project protocol. BPR ranking loss + L2 reg on layer-0 embeddings + two InfoNCE heads, identical loss schema to the SASRec variant. |

### 2.4 Orchestration / aggregation

| File | Purpose |
|------|---------|
| `seed_aggregation.py` | Shared helpers: read per-seed `test_metrics.csv`, compute mean/std, write per-seed/aggregate/comparison CSVs, plot grouped bars with error bars and validation curves with ±1σ bands. |
| `run_all.py` | ML-1M orchestrator. For each of 3 models × N seeds, invokes the trainer as a subprocess into `artifacts/<model>/seed_<N>/`, then aggregates into `artifacts/comparison/`. |
| `run_all_yelp.py` | Same as `run_all.py`, but writes to `artifacts_yelp/` and forwards `--dataset yelp`. Adds Yelp-specific filter flags. |
| `run_all_amazon.py` | Same as `run_all.py`, but writes to `artifacts_amazon/` and forwards `--dataset amazon`. Adds Amazon-specific filter and category flags. |
| `run_everything.py` | Top-level: invokes `run_all.py`, `run_all_yelp.py`, `run_all_amazon.py`, then `make_overview.py` for each dataset. This is the one-shot reproduction entrypoint. |
| `run_ablations.py` | Separate ablation orchestrator for the SemanticSASRec ablation grid (λ sweep, τ sweep, MLP-vs-linear projection head, MiniLM-vs-BGE encoder, loss-component flips, plus optional E5-Mistral-7B). 50 epochs × 3 seeds × N variants × 3 datasets by default. Uses paired t-tests against an in-ablation reference variant (`sasrec_full` / `lightgcn_full`). |
| `make_overview.py` | Reads per-model `test_metrics.csv`, `test_metrics_aggregate.csv`, and `run_metadata.json` for one dataset and renders `<artifacts_root>/comparison/results_overview.md` with the headline table, run-config table, head-to-head observations, and a dataset-specific paper-comparison section. |

### 2.5 Data, training, and evaluation protocol

- **Interactions and splits:** rating values are ignored. Within each user’s chronological sequence, the final event is test, the preceding event is validation, and earlier events train. Yelp/Amazon break timestamp ties by raw item ID; MovieLens preserves file order for ties. Yelp/Amazon use iterative user/item filtering with `min_inter` (default 5), capped at ten passes, then discard user sequences shorter than five. Repeated user–item events are retained.
- **Semantic features:** user-history summaries use training events. Item/user text comes from templates and metadata, not generated LLM profiles. A frozen sentence transformer produces normalized float32 embeddings; trainable projection heads use symmetric cosine InfoNCE. At recommendation time, the models use backbone dot-product scores without semantic reranking.
- **Training and checkpoint selection:** each epoch starts with `model.train()`. Validation and final testing use `model.eval()` and no gradients, disabling dropout. Validation runs every `eval_every` epochs and on the final epoch. The checkpoint with the highest validation NDCG@10 is selected; equal scores retain the earlier checkpoint. Training always runs the requested epoch count; there is no early stopping.
- **Evaluation inputs:** SASRec and SemanticSASRec left-pad/truncate histories to `max_len` and append the validation item to the test history. LightGCN keeps its training-only graph. All three mask padding and training items from candidates; test evaluation also masks the validation item.
- **Ranking and metrics:** every remaining catalog item is scored, without sampled evaluation negatives. Rank is `1 + count(score > target_score)`, so ties receive the best tied rank. HR@K and NDCG@K use K = 1, 5, 10, 20; MRR is untruncated. Metrics are means across evaluated users. These values are not directly comparable to sampled-negative paper results.
- **Seeds and statistics:** headline results use seeds 42–46; ablations use 42–44. Aggregates report mean, sample standard deviation, and the number of available seed values (`n`), not confidence intervals. Ablation significance uses two-sided paired t-tests on common seed IDs against the matched reference, without multiple-comparison correction. Seeds are initialized, but deterministic kernels are not enforced across hardware/software versions.

### 2.6 Implementation limits relevant to the paper

**Temporal metadata:** training-only history summaries do not make all side information time-filtered. Yelp business stars and user rating/review aggregates, and Amazon item average ratings, come from the full release metadata without a split-time cutoff. These fields may contain information from after the held-out events. Disclose that assumption; do not describe the current pipeline as verified leakage-free.

**Repeated held-out items:** if a validation/test target already occurs in its exclusion history, the current evaluator masks that target too and does not restore its score. Repeated events are not removed during preprocessing. Measure how often this occurs and settle the intended repeated-item evaluation policy before treating the resulting metrics as publication-ready.

**Resource use:** Yelp/Amazon files are parsed line by line, but retained interactions occupy RAM. Evaluation allocates dense validation and test exclusion masks of users × (items + 1), in addition to batched scores. Reducing evaluation batch size does not remove the full mask allocation.

---

## 3. Environment setup

Run commands from the repository root using Python 3.12. A virtual environment is local to your checkout and **is not included in the Git repository**. Create it on a new machine:

```bash
cd /path/to/repository
python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch numpy scipy matplotlib sentence-transformers huggingface_hub tqdm transformers
python -m pip check
```

For an existing environment, activate it with `source venv/bin/activate`. You can also use `venv/bin/python` directly. The installation command above resolves available package versions; there is currently no committed dependency lockfile. Save the resolved environment for each experiment as described in §8.

The **local regression-test environment checked on 2026-10-19** was Python 3.12.7 with these packages. This is a development snapshot, not verified provenance for the manuscript’s numerical results:

| Package | Installed version |
|---------|-------------------|
| torch | 2.14.1 |
| numpy | 2.5.4 |
| scipy | 1.18.1 |
| matplotlib | 3.11.2 |
| sentence-transformers | 6.1.0 |
| huggingface-hub | 1.33.0 |
| transformers | 5.20.2 |
| tqdm | 4.70.1 |

### Device selection and evaluation mode

Every trainer calls `sasrec_train.pick_device`, which chooses **CUDA → MPS → CPU** according to availability. There is no `--device` CLI flag. CPU and MPS have regression coverage; CUDA is selected by the code but was not tested in this audit.

Evaluation must use `model.eval()`; `torch.no_grad()` alone does not turn off dropout. The trainers now do both, including after loading the best checkpoint. This fixes the observed MPS attention-dropout exception and ensures dropout is disabled for validation/test on every backend. Training dropout remains enabled at the next epoch. See §8 for the implications for older experiment outputs.

---

## 4. Getting the datasets

All three datasets should be placed under `data/`. **Two of them auto-download on first run** (MovieLens-1M and Amazon). Yelp requires a manual download because of Yelp's terms-of-service click-through.

The expected on-disk layout after all downloads (sizes are approximate and depend on the release):

```
data/
├── ml-1m/
│   ├── ratings.dat                                (24 MB)
│   ├── movies.dat                                 (171 KB)
│   ├── users.dat                                  (134 KB)
│   └── README                                     (5 KB, from GroupLens)
│
├── yelp/
│   ├── yelp_academic_dataset_review.json          (5.3 GB)
│   ├── yelp_academic_dataset_business.json        (119 MB)
│   ├── yelp_academic_dataset_user.json            (3.3 GB)
│   ├── yelp_academic_dataset_checkin.json         (287 MB, optional)
│   ├── yelp_academic_dataset_tip.json             (181 MB, optional)
│   └── Dataset_User_Agreement.pdf                 (80 KB)
│
├── amazon/
│   ├── Books.jsonl.gz                             (6.2 GB)
│   └── meta_Books.jsonl.gz                        (4.9 GB)
│
└── semantic_cache/                                (auto-populated *.npy cache, see §4.4)
```

### 4.1 MovieLens-1M (auto-downloaded)

The loader (`sasrec_data.download_ml1m`) fetches the zip from GroupLens and extracts it under `data/ml-1m/` on first run. **You don't have to do anything**. If you want to download it manually:

```bash
mkdir -p data
cd data
curl -O https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip
# This creates data/ml-1m/ containing ratings.dat, movies.dat, users.dat, README.
cd ..
```

Source URL: <https://files.grouplens.org/datasets/movielens/ml-1m.zip>
Project page: <https://grouplens.org/datasets/movielens/1m/>

### 4.2 Yelp Open Dataset (manual download required)

The Yelp Open Dataset is too large to auto-download anonymously, and Yelp requires you to accept their dataset terms before downloading. Steps:

1. Open <https://www.yelp.com/dataset> in a browser.
2. Click **Download Dataset**, accept the dataset agreement, and download the **JSON** archive (e.g. `yelp_dataset.tar` or `yelp_dataset.tgz`). The download is ~4 GB compressed.
3. Extract it. You should end up with files named `yelp_academic_dataset_business.json`, `yelp_academic_dataset_review.json`, `yelp_academic_dataset_user.json`, `yelp_academic_dataset_checkin.json`, `yelp_academic_dataset_tip.json`.
4. Move them into `data/yelp/`:

```bash
mkdir -p data/yelp
mv /path/to/extracted/yelp_academic_dataset_*.json data/yelp/
```

The loader requires `*_review.json`, `*_business.json`, and `*_user.json`; the semantic models use business/user metadata. Checkin and tip files are optional and are not read.

If the JSON files are missing, `yelp_data.ensure_yelp_files` will raise a `FileNotFoundError` with a clear message telling you where to put them.

### 4.3 Amazon Reviews 2023 — Books (auto-downloaded)

`amazon_data.ensure_amazon_files` will auto-fetch the gzipped Books review and metadata files from the McAuley Lab's public hosting on first run, no authentication needed. The download is ~11 GB total and can take a while.

The two URLs the loader hits:

- Reviews: <https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Books.jsonl.gz>
- Metadata: <https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Books.jsonl.gz>

To download manually instead (e.g. if your machine has a poor connection during training):

```bash
mkdir -p data/amazon
cd data/amazon
curl -O https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Books.jsonl.gz
curl -O https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Books.jsonl.gz
cd ../..
```

If you'd rather benchmark a different category (e.g. `Electronics`, `Movies_and_TV`), the URL template is the same — just swap `Books` for the category name listed on the [Amazon Reviews 2023 page](https://amazon-reviews-2023.github.io/). Pass `--amazon_category <Name>` to a trainer, `run_all_amazon.py`, or `run_everything.py`. The draft’s stated category is `Books`; another category is a different experimental setting.

### 4.4 Sentence-transformer embedding cache

The semantic-profile pipeline encodes user/item texts with a frozen sentence-transformer and caches the result as `.npy` under `data/semantic_cache/`. **You don't download this cache** — it's generated on first run.

Encoder-name slashes become `__` in cache filenames. MovieLens has no dataset prefix; Yelp uses `yelp_`; Amazon uses `amazon_<lowercase-category>_`. Example files generated by the default BGE-large encoder and MiniLM ablation are:

```
data/semantic_cache/
├── user_emb_BAAI__bge-large-en-v1.5.npy
├── item_emb_BAAI__bge-large-en-v1.5.npy
├── yelp_user_emb_BAAI__bge-large-en-v1.5.npy
├── yelp_item_emb_BAAI__bge-large-en-v1.5.npy
├── amazon_books_user_emb_BAAI__bge-large-en-v1.5.npy
├── amazon_books_item_emb_BAAI__bge-large-en-v1.5.npy
├── user_emb_sentence-transformers__all-MiniLM-L6-v2.npy              (ML-1M MiniLM)
├── item_emb_sentence-transformers__all-MiniLM-L6-v2.npy              (ML-1M MiniLM)
├── yelp_user_emb_sentence-transformers__all-MiniLM-L6-v2.npy
├── yelp_item_emb_sentence-transformers__all-MiniLM-L6-v2.npy
├── amazon_books_user_emb_sentence-transformers__all-MiniLM-L6-v2.npy
└── amazon_books_item_emb_sentence-transformers__all-MiniLM-L6-v2.npy
```

**Cache validity:** keys do not include data revisions, filter settings, ID mappings, profile templates, or encoder revisions. MovieLens checks only whether both files exist; Yelp/Amazon additionally check row counts, which cannot detect a changed mapping with the same size. Reuse caches only when all those inputs are unchanged. Otherwise use a fresh `--cache_dir` on the semantic trainers or regenerate the relevant files. Initial encoding may download encoder weights and can take substantial time; the regression tests bypass encoding and do not validate those downloads.

---

## 5. Reproducing the paper

### 5.1 One command (the full grid)

After preparing Yelp and the environment, train SASRec, RLMRec+LightGCN, and SemanticSASRec on all three datasets with the draft’s stated headline budget: 5 seeds × 75 epochs, or 45 training runs. Validation is every 20 epochs and on the final epoch. Read §8 before using existing results from older code.

```bash
source venv/bin/activate
python run_everything.py
```

After this finishes you'll have:

- `artifacts/comparison/comparison.csv` and `results_overview.md` (ML-1M)
- `artifacts_yelp/comparison/comparison.csv` and `results_overview.md` (Yelp)
- `artifacts_amazon/comparison/comparison.csv` and `results_overview.md` (Amazon-Books)

The headline comparison corresponds to Table 1 and Figure 2 in the current draft; Table 2 and Figure 3 concern the separate ablation runs. This mapping does not verify the draft’s numbers against regenerated results.

### 5.2 Single dataset

```bash
# ML-1M only
python run_everything.py --datasets ml1m

# Yelp only
python run_everything.py --datasets yelp

# Amazon-Books only
python run_everything.py --datasets amazon
```

To train and aggregate directly, invoke a per-dataset orchestrator. These commands do not create the Markdown overview by default; generate it explicitly:

```bash
python run_all.py
python make_overview.py --dataset ml1m

python run_all_yelp.py --also_make_overview
python run_all_amazon.py --also_make_overview
```

### 5.3 Single model on a single dataset, single seed

If you only want to sanity-check one variant:

```bash
python sasrec_train.py           --dataset ml1m   --seed 42 --out_dir artifacts/sasrec/seed_42
python semantic_sasrec_train.py  --dataset yelp   --seed 42 --out_dir artifacts_yelp/semantic_sasrec/seed_42
python rlmrec_lightgcn_train.py   --dataset amazon --seed 42 --out_dir artifacts_amazon/rlmrec_lightgcn/seed_42
```

### 5.4 Smoke test (a few epochs)

First run the offline regression tests in §8. For a small run using real MovieLens data (downloaded if absent):

```bash
python sasrec_train.py --dataset ml1m --seed 42 --epochs 5 --eval_every 5 \
    --out_dir /tmp/smoke_sasrec_ml1m
```

### 5.5 Aggregation-only (no retraining)

If completed runs already contain per-seed `test_metrics.csv` files, rebuild aggregates and comparison plots without training. The top-level command also regenerates overviews; direct dataset runners require the separate `make_overview.py` commands shown below:

```bash
python run_everything.py --compare_only
# or per-dataset:
python run_all.py        --compare_only
python run_all_yelp.py   --compare_only
python run_all_amazon.py --compare_only
```

Pass the same `--seeds` used for the original runs if they differ from the default list. Missing seed files are skipped; inspect each metric’s `n` before reporting it. This mode does not finish interrupted training, re-evaluate checkpoints, or regenerate per-seed training plots. It cannot correct old metrics after an evaluation change.

You can also rebuild a single dataset’s Markdown from its per-model metrics without invoking the runner:

```bash
python make_overview.py --dataset ml1m
python make_overview.py --dataset yelp
python make_overview.py --dataset amazon
```

### 5.6 Ablation grid

The default ablation grid uses **50 epochs × 3 seeds × 15 variants × 3 datasets**, or **135 training runs**, with validation every 10 epochs. There are 13 SemanticSASRec configurations and 2 LightGCN configurations, including their references. Paired t-tests compare variants with `sasrec_full` or `lightgcn_full` at the same budget. Optional E5 adds two configurations, giving 17 variants and 153 runs. The draft’s references to 14 variants need reconciliation with this catalog before publication.

```bash
# Full ablation grid
python run_ablations.py

# Smoke test: selected variant plus its automatically included reference
python run_ablations.py --datasets ml1m --variants sasrec_no_user_align \
    --seeds 42 --epochs 5

# Re-aggregate existing per-seed CSVs without retraining
python run_ablations.py --aggregate_only

# Add optional E5-Mistral-7B encoder variants for both backbones
python run_ablations.py --include_e5_mistral
```

Output goes to `artifacts_ablation/{ml1m,yelp,amazon}/comparison/ablation_overview.md`. Selecting a variant automatically includes its reference; `--skip` can remove that reference. E5 variants require `--include_e5_mistral` even when selected by name. A one-seed smoke run cannot support paired significance. The 75-epoch headline baselines may appear as reference rows, but are not used as the 50-epoch ablation significance baseline.

### 5.7 Reruns and custom encoders

There is no resume or skip-completed option. A training rerun starts at epoch 1 and writes to the same configured paths; checkpoints contain model state, without optimizer or RNG state. Preserve earlier outputs separately and use fresh run directories for changed settings. The orchestrators use fixed model directories under `artifacts*`; `--compare_dir` changes comparison output only.

`run_everything.py --model_name ...` currently forwards the encoder flag to every active trainer, but plain SASRec does not accept it. To change the encoder through this entry point, run the plain baseline separately and skip it in the encoder run:

```bash
# Plain baseline, all datasets
python run_everything.py --skip_models rlmrec_lightgcn semantic_sasrec

# Semantic models, all datasets, with a custom encoder
python run_everything.py --skip_models sasrec \
    --model_name sentence-transformers/all-MiniLM-L6-v2
```

Use fresh or separately archived output directories when comparing different encoders: encoder names are part of cache filenames, but not run-output paths. `--skip` / `--skip_models` prevent training; existing metrics for skipped models can still appear in aggregates.

---

## 6. Per-script flag reference

### 6.1 `sasrec_train.py`

Pure SASRec trainer.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--dataset` | str | `ml1m` | One of `ml1m`, `yelp`, `amazon`. |
| `--epochs` | int | `75` | Number of training epochs. |
| `--batch_size` | int | `128` | Mini-batch size. |
| `--max_len` | int | `200` | Maximum sequence length. |
| `--hidden` | int | `50` | Hidden dimension. |
| `--blocks` | int | `2` | Number of self-attention blocks. |
| `--heads` | int | `1` | Number of attention heads. |
| `--dropout` | float | `0.2` | Dropout rate. |
| `--lr` | float | `1e-3` | Adam learning rate. |
| `--l2` | float | `0.0` | Adam weight decay. |
| `--eval_every` | int | `20` | Run validation every N epochs (+ always on the last epoch). |
| `--seed` | int | `42` | RNG seed (PyTorch + NumPy + Python random). |
| `--num_workers` | int | `0` | DataLoader workers. |
| `--data_dir` | str | `data` | Where to find / download data. |
| `--out_dir` | str | `artifacts/sasrec` | Where to write checkpoints, history, metrics. |
| `--yelp_start_year` | int | `2018` | Drop Yelp reviews before YYYY-01-01. |
| `--yelp_min_inter` | int | `5` | K for Yelp k-core filter. |
| `--amazon_start_year` | int | `2018` | Drop Amazon reviews before YYYY-01-01. |
| `--amazon_min_inter` | int | `5` | K for Amazon k-core filter. |
| `--amazon_category` | str | `Books` | Amazon Reviews 2023 category. |

### 6.2 `semantic_sasrec_train.py`

Same flags as `sasrec_train.py`, with the differences/additions below. `--num_workers` is accepted but currently unused because this trainer batches manually. Sequential full-rank evaluation uses the evaluator’s fixed default batch size of 256; it has no `--eval_batch_size` CLI flag.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--cache_dir` | str | `data/semantic_cache` | Where to cache the `.npy` profile embeddings. |
| `--out_dir` | str | `artifacts/semantic_sasrec` | (Default differs from `sasrec_train.py`.) |
| `--model_name` | str | `BAAI/bge-large-en-v1.5` | Sentence-transformer / HF model used to encode user & item profile texts. |
| `--lambda_item` | float | `0.1` | Weight on the item-alignment InfoNCE loss. |
| `--lambda_user` | float | `0.1` | Weight on the user-alignment InfoNCE loss. |
| `--temperature` | float | `0.1` | InfoNCE temperature. |
| `--proj_head` | str | `mlp` | `mlp` (2-layer GELU) or `linear` (single nn.Linear). |
| `--items_per_batch` | int | `512` | Maximum distinct positive items from the current batch used for item-alignment InfoNCE. |

### 6.3 `rlmrec_lightgcn_train.py`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--dataset` | str | `ml1m` | One of `ml1m`, `yelp`, `amazon`. |
| `--epochs` | int | `75` | Number of training epochs. |
| `--batch_size` | int | `2048` | Mini-batch size (BPR pairs). |
| `--dim` | int | `64` | LightGCN embedding dimension. |
| `--num_layers` | int | `3` | LightGCN propagation depth. |
| `--lr` | float | `1e-3` | Adam learning rate. |
| `--l2` | float | `1e-4` | L2 reg on layer-0 embeddings. |
| `--eval_every` | int | `20` | Validation cadence. |
| `--eval_batch_size` | int | `256` | Eval-time user batch size. |
| `--seed` | int | `42` | RNG seed. |
| `--data_dir` | str | `data` | Data directory. |
| `--cache_dir` | str | `data/semantic_cache` | Embedding cache. |
| `--out_dir` | str | `artifacts/rlmrec_lightgcn` | Output directory. |
| `--model_name` | str | `BAAI/bge-large-en-v1.5` | Sentence-transformer model. |
| `--lambda_item` | float | `0.1` | Item-alignment InfoNCE weight. |
| `--lambda_user` | float | `0.1` | User-alignment InfoNCE weight. |
| `--temperature` | float | `0.1` | InfoNCE temperature. |
| `--yelp_start_year` / `--yelp_min_inter` | int | `2018` / `5` | Yelp filter knobs. |
| `--amazon_start_year` / `--amazon_min_inter` / `--amazon_category` | int / int / str | `2018` / `5` / `Books` | Amazon filter knobs. |

### 6.4 `run_all.py` (ML-1M orchestrator)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--skip` | list | `[]` | Models not to train: `sasrec`, `rlmrec_lightgcn`, `semantic_sasrec`. Existing metrics are still aggregated. |
| `--compare_only` | bool | `False` | Skip training; only re-aggregate existing per-seed CSVs. |
| `--compare_dir` | str | `artifacts/comparison` | Where to write the comparison files. |
| `--epochs` | int | _(trainer default 75)_ | Override `--epochs` for every training subprocess. |
| `--eval_every` | int | _(trainer default 20)_ | Override `--eval_every`. |
| `--seeds` | int+ | `42 43 44 45 46` | One or more seeds to train each model with. |
| `--seed` | int | _none_ | Legacy single-seed alias for `--seeds <N>`. |
| `extra` | REMAINDER | _none_ | Anything after `--` is forwarded verbatim to every training subprocess. |

`run_all_yelp.py` adds `--yelp_start_year` / `--yelp_min_inter`; `run_all_amazon.py` adds `--amazon_start_year` / `--amazon_min_inter` / `--amazon_category`. Unspecified filters use trainer defaults. Their default comparison directories are `artifacts_yelp/comparison` and `artifacts_amazon/comparison`. Both add `--also_make_overview` (default off); `run_all.py` has no such option.

Arguments after `--` go to **every active trainer**, so only use flags supported by all selected trainers. In particular, semantic-only flags require skipping plain SASRec. Do not override `--seed`, `--dataset`, or `--out_dir` through this tail: the runner’s aggregation settings will not follow those overrides. Supply either `--seed` or `--seeds`, not both.

### 6.5 `run_everything.py`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--datasets` | list | `ml1m yelp amazon` | Which datasets to run. |
| `--skip_models` | list | `[]` | Models to skip on every dataset. |
| `--skip_models_ml`, `--skip_models_yelp`, `--skip_models_amazon` | list | _none_ | Replace the global skip list for that dataset; lists are not merged. |
| `--compare_only` | bool | `False` | Skip training, just regenerate comparisons + overviews. |
| `--no_overview` | bool | `False` | Skip the markdown overview generation step. |
| `--epochs` | int | _(trainer default)_ | Set the number of epochs for every training run; no early stopping. |
| `--eval_every` | int | _(trainer default)_ | Validation cadence for every training run. |
| `--seeds` | int+ | _(runner default `42 43 44 45 46`)_ | Seeds for every training run. |
| `--seed` | int | _none_ | Single-seed legacy alias. |
| `--model_name` | str | _(trainer default BGE-large)_ | Encoder override. Currently requires skipping plain SASRec; see §5.7. |
| `--yelp_start_year` / `--yelp_min_inter` | int | _(trainer default)_ | Yelp filter knobs. |
| `--amazon_start_year` / `--amazon_min_inter` / `--amazon_category` | int / int / str | _(trainer default)_ | Amazon filter knobs. |

### 6.6 `run_ablations.py`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--datasets` | list | `ml1m yelp amazon` | Datasets to run. |
| `--variants` | list | _all_ | Select variants; their references are automatically included. See the module catalog or `--help` for names. |
| `--skip` | list | `[]` | Remove variants after reference inclusion; removing a reference disables its paired comparisons. |
| `--seeds` | int+ | `42 43 44` | Seeds per variant. |
| `--epochs` | int | `50` | Epochs per training run. |
| `--eval_every` | int | `10` | Validation cadence. |
| `--include_e5_mistral` | bool | `False` | Add the heavy E5-Mistral-7B encoder variants. |
| `--aggregate_only` | bool | `False` | Skip training; rebuild aggregates from existing per-seed CSVs. |
| `extra` | REMAINDER | _none_ | Forwarded to every selected trainer after the variant arguments; overriding alignment weights, temperature, or encoder here changes the declared ablation. |

### 6.7 `make_overview.py`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--dataset` | str (required) | _none_ | `ml1m`, `yelp`, or `amazon`. |
| `--artifacts_root` | str | _(depends on `--dataset`: `artifacts`, `artifacts_yelp`, or `artifacts_amazon`)_ | Override artifacts root. |
| `--out` | str | `<root>/comparison/results_overview.md` | Override the output path. |

---

## 7. Output / artifact layout

After `run_everything.py` completes, the structure is:

```
artifacts/                                 ← ML-1M
├── sasrec/
│   ├── seed_42/
│   │   ├── sasrec_best.pt                 ← best-by-val-NDCG@10 checkpoint
│   │   ├── train_history.csv              ← per-epoch loss + val metrics
│   │   ├── test_metrics.csv               ← final test metrics (this seed)
│   │   ├── run_metadata.json              ← all flags + best epoch + final metrics
│   │   ├── loss_curve.png
│   │   ├── val_HR@{1,5,10,20}.png
│   │   ├── val_NDCG@{1,5,10,20}.png
│   │   ├── val_MRR.png
│   │   ├── val_metrics_all.png
│   │   └── test_metrics.png
│   ├── seed_43/...
│   ├── seed_44/...
│   ├── seed_45/...
│   ├── seed_46/...
│   ├── test_metrics.csv                   ← mean over seeds (legacy schema)
│   ├── test_metrics_per_seed.csv          ← rows: seed, cols: metrics
│   ├── test_metrics_aggregate.csv         ← rows: metric, cols: mean,std,n
│   └── run_metadata.json                  ← copied from one representative seed
├── semantic_sasrec/                       ← same layout (checkpoint: semantic_sasrec_best.pt)
├── rlmrec_lightgcn/                       ← same layout (checkpoints: rlmrec_lightgcn_best.pt)
└── comparison/
    ├── comparison.csv                     ← metric × model (mean only)
    ├── comparison_per_seed.csv            ← long format
    ├── comparison_aggregate.csv           ← per-model mean / std / n
    ├── comparison_bars.png                ← grouped bars with error bars
    ├── val_NDCG@10_curves.png             ← per-model mean curve + ±1σ band
    ├── summary.txt                        ← formatted mean ± std table + per-seed tables
    └── results_overview.md                ← human-readable narrative
```

`artifacts_yelp/` and `artifacts_amazon/` use the same structure. Both semantic trainers additionally write `loss_components.png`. Model-root `run_metadata.json` is copied from the first available seed, so its seed, best epoch, and test metrics are not aggregated values; use the CSV aggregates for means/standard deviations. Per-seed metadata records arguments, device, dataset counts, model parameter count, best epoch, and final metrics; semantic trainers also record encoder name and dimension.

`artifacts_ablation/{ml1m,yelp,amazon}/` follows a similar layout keyed by variant name (e.g. `sasrec_lambda_0.5/seed_42/...`). Its comparison directory contains `ablation.csv`, `ablation_aggregate.csv`, `ablation_per_seed.csv`, `ablation_bars.png`, and `ablation_overview.md`. `ablation_significance.csv` is written only when there are eligible comparisons with at least two common seeds and reference data.

No generated experiment outputs or checkpoints are tracked in this repository. `make_overview.py` uses templated explanatory text that can assume default settings; check it against per-seed metadata before transferring it into the paper.

---

## 8. Validation and publication status

### 8.1 Checks you can run locally

From the repository root, using the environment from §3:

```bash
source venv/bin/activate
python -m pip check
python -m compileall -q -l . tests
python -m unittest discover -s tests -v
```

`tests/test_training_modes.py` exercises all three trainers on CPU and, when available, MPS. Each case runs two epochs on synthetic interactions, performs real forward/backward passes and full-rank validation/test evaluation, writes and reloads the best checkpoint, and checks finite losses/metrics. The sequential models retain default dropout `0.2` and one attention head. Tests verify that evaluation disables dropout and that the next training epoch restores training mode. Dataset loading and semantic embeddings are supplied locally in the test; plotting is mocked, so the suite does not download data or encoder weights.

On **2026-10-19**, all six tests passed in the §3 environment, including the three MPS cases. All 19 Python source/test files compiled, and `pip check` reported no broken requirements. On a machine without MPS, the three MPS cases are skipped. This verifies the tested training/evaluation paths; it does not validate real-data preprocessing, encoder downloads, CUDA execution, or the manuscript’s performance claims.

### 8.2 Evaluation correction and older results

The **2026-10-19 evaluation-mode correction** adds `model.eval()` before validation and final testing in every trainer. Previously, the SASRec-based models could retain active dropout while scoring. On the tested MPS runtime this raised `NotImplementedError: scaled_dot_product_attention for MPS does not support dropout`; on backends that accepted the operation, active dropout could change validation/test scores and best-checkpoint selection.

Older metrics must not be treated as verified results of the corrected protocol. Re-run the headline and ablation experiments using the corrected code, then regenerate aggregates, tables, and figures. `--compare_only` and `--aggregate_only` merely combine saved CSV values; they do not fix historical evaluation or choose a new best checkpoint.

### 8.3 Before publishing numerical results

At this audit, no generated experiment CSVs, checkpoints, or run metadata were available in the checkout or tracked by Git. The included PDF is a draft; its numerical results have **not** been verified against a complete run of the corrected code.

Before publication:

1. Reconcile the protocol and manuscript: learned positional embeddings, actual MovieLens filtering, **15 default ablation configurations**, the temporal-metadata assumptions, and the repeated-held-out-item policy described in §2.6. Any protocol change requires corresponding result regeneration.
2. Include the current trainers, evaluator, and `tests/test_training_modes.py` in the release commit. Run the checks above from that revision.
3. Preserve earlier artifacts separately, then run the complete headline and ablation grids into clean output directories. Retain all per-seed histories, checkpoints, metrics, and metadata. Confirm every reported headline metric has `n=5` and every default ablation metric has `n=3`; do not infer completeness from a single summary or a zero standard deviation.
4. Archive the exact code revision, installed dependencies, hardware/backend, dataset release identifiers/checksums, preprocessing flags and resulting counts, encoder revision, and embedding-cache provenance. `run_metadata.json` does not record all of these. Record an encoder revision separately because `--model_name` alone is not a version pin.
5. Regenerate the tables/figures from those outputs and reconcile the PDF and presentation. Check headline Table 1/Figure 2 and ablation Table 2/Figure 3, sample-standard-deviation labels, seed counts, and the use of unadjusted paired tests. Review generated overview prose against actual run metadata before copying it into the paper.

For example, record a minimal environment/code snapshot alongside a run (also archive any uncommitted/untracked source changes):

```bash
mkdir -p reproduction_metadata
python --version > reproduction_metadata/python-version.txt
python -m pip freeze > reproduction_metadata/installed-packages.txt
git rev-parse HEAD > reproduction_metadata/source-commit.txt
git status --short > reproduction_metadata/source-status.txt
```

These commands do not capture dataset checksums, encoder revisions, hardware details, or the contents of uncommitted files; preserve those with the experiment archive too.

---

## 9. Troubleshooting

| Symptom | What to check |
|---------|---------------|
| `ModuleNotFoundError` for PyTorch, NumPy, or another dependency | Use the environment from §3 (`source venv/bin/activate` or `venv/bin/python`) and run `python -m pip check`. A fresh clone does not contain `venv/`. |
| MPS error: `scaled_dot_product_attention ... does not support dropout` during evaluation | Ensure the trainers contain the evaluation-mode correction and run the regression suite. Custom scoring callers must call `model.eval()` before `evaluate_full_rank`; `torch.no_grad()` alone is insufficient. Do not disable training dropout to hide this evaluation bug. |
| Yelp `FileNotFoundError` | Put review, business, and user JSON files directly under `data/yelp/`, or the corresponding directory under `--data_dir`. The archive is not downloaded by the loader. |
| First semantic run is slow or encoder loading fails | The first run may download model weights and encode all profiles. Check connectivity/model access, available memory, and the selected encoder. Cached embeddings are reusable only with unchanged inputs (§4.4). |
| Encoder/cache results look inconsistent after changing preprocessing | Use a fresh semantic `--cache_dir` or regenerate that dataset/encoder’s cache. Matching row counts alone do not establish matching user/item mappings. |
| `unrecognized arguments: --model_name` | Plain SASRec has no encoder flag. For `run_everything.py`, use the two-stage example in §5.7 or invoke a semantic trainer directly. Arguments forwarded after `--` must be supported by every active trainer. |
| Out-of-memory during loading or full-rank evaluation | Retained interactions and dense exclusion masks can dominate memory (§2.6). Training `--batch_size` does not control sequential evaluation batching. Only the LightGCN trainer exposes `--eval_batch_size`; reducing it reduces batched scores, not full exclusion masks. Changing filters changes the experiment and requires fresh cache/provenance. |
| Missing models/seeds or no significance CSV | Check completed per-seed `test_metrics.csv` files and the selected seed list. Aggregation skips missing files. Paired tests need reference results and at least two shared seeds; `--skip` can remove a reference. |
| `make_overview.py` reports no metrics | First finish training and aggregate the selected seeds. The generator reads per-model root metrics, not checkpoints or only the cross-model comparison CSV. Use `--artifacts_root` when those roots are elsewhere. |
| Restarting a run does not resume from its checkpoint | Resume is not implemented. Training starts from epoch 1, and checkpoints contain model state without optimizer or RNG state. Archive old outputs before reusing their paths. |
