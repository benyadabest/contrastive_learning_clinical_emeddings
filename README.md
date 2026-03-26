# Contrastive Learning for Clinical Embeddings

Fine-tuning [EmbeddingGemma-300m](https://huggingface.co/google/embeddinggemma-300m) on MIMIC-III clinical notes using temporal and hierarchical contrastive learning to produce embeddings that capture clinical semantics over stylistic differences.

Based on [Radical Health AI's approach](https://radicalhealth.ai/blog/training-a-model-that-understands-your-notes-7x-better-than-openai), which achieved 0.934 AUROC on diagnosis prediction vs 0.809 for OpenAI.

## Approach

1. **Temporal contrastive learning (InfoNCE):** Anchor = patient note at time *t*, positive = same patient's note at *t+1*, in-batch negatives from other patients. Forces embeddings to capture patient trajectory rather than writing style.

2. **Hierarchical contrastive learning (HiMulCon-style):** Extends temporal loss with soft targets from ICD-9 code hierarchy — notes sharing diagnosis chapters get partial positive weight, producing embeddings that reflect clinical similarity at multiple granularities.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

*Formerly* required MIMIC-III data in `mimic-iii-clinical-database-demo-1.4/`. The demo dataset has structured data but no clinical notes — the full dataset is located at [PhysioNet](https://physionet.org/content/mimiciii/1.4/).

Now requires MIMIC-III data in `MIMIC -III (10000 patients)/`, which can be downloaded from [Kaggle](https://www.kaggle.com/datasets/bilal1907/mimic-iii-10k). This dataset is a subset of the full dataset with all tables included.

## Usage

```bash
# 1. Preprocess MIMIC-III data
python src/preprocess.py

# 2. Generate baseline embeddings
python src/embed.py --mode pairs --model library-model-embeddinggemma
python src/embed.py --mode pairs --model text-embedding-3-small

# 3. Fine-tune with contrastive loss
python src/train_contrastive.py --loss infonce --epochs 5
python src/train_contrastive.py --loss hierarchical --epochs 10

# 4. Evaluate
python src/evaluate.py --task compare
python src/evaluate.py --task umap --embeddings embeddings/<file>.npy
```

## Evaluation

| Metric | Task |
|--------|------|
| Top-5 recall accuracy | Retrieving next patient note from embeddings |
| Macro AUROC | Multi-label ICD-9 diagnosis prediction (logistic regression on frozen embeddings) |
| UMAP visualization | Embedding clusters colored by ICD chapter |

## Model Comparison

| Model | Type |
|-------|------|
| OpenAI text-embedding-3-small | General-purpose baseline |
| OpenAI text-embedding-3-large | General-purpose baseline |
| EmbeddingGemma (temporal contrastive) | Fine-tuned baseline |
| EmbeddingGemma (hierarchical contrastive) | Our extension |

## Project Structure

```
src/
├── preprocess.py          # MIMIC-III data preprocessing + temporal pair construction
├── embed.py               # Embedding generation (EmbeddingGemma + OpenAI)
├── train_contrastive.py   # Contrastive fine-tuning (InfoNCE + hierarchical)
└── evaluate.py            # Note recall, diagnosis prediction, UMAP
```

## Authors

Benjamin Shvartsman, Timothy Lin, Gaspard Loeillot
