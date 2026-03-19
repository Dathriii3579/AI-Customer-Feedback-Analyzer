# AI-Driven Customer Feedback Analysis System
### CSAT & CES Prediction with Multi-Task Transformer Learning

---

## 📌 Overview

This project builds an end-to-end AI pipeline that automatically predicts two key customer experience metrics directly from raw review text:

- **CSAT (Customer Satisfaction Score)** — *Dissatisfied / Neutral / Satisfied*
- **CES (Customer Effort Score)** — *Easy / Difficult*

A single **multi-task BERT model** handles both predictions simultaneously, sharing a transformer encoder and branching into two independent classification heads. This avoids training two separate models while allowing each task to learn complementary signals.

---

## 📂 Project Structure

```
csat proj 2/
├── 7817_1.csv                            # Primary Amazon Reviews dataset
├── raw_data/
│   ├── trust_pilot_reviews_data_2022_06.csv  # TrustPilot reviews (merged for balance)
│   ├── twitter_training.csv                   # Twitter sentiment (exploratory only)
│   └── twitter_validation.csv
├── sentiment-analysis.csv                # Supplementary sentiment reference
├── multitask_model/
│   ├── data_prep.py                      # Data loading, merging & label derivation
│   ├── model.py                          # Multi-task model architecture
│   ├── train.py                          # Training loop with class-weighted loss
│   ├── evaluate.py                       # Evaluation & confusion matrix plots
│   ├── inference.py                      # Single-review inference CLI
│   ├── multitask_pipeline.ipynb          # End-to-end Jupyter notebook
│   ├── requirements.txt
│   └── checkpoints/
│       ├── best_model.pt                 # Best saved checkpoint
│       └── training_history.json         # Full training + test metrics
├── CES (2).ipynb                         # Standalone CES exploration
├── CSAT3 (2).ipynb                       # Standalone CSAT exploration
└── final_inference_dashboard (3).ipynb   # Interactive inference dashboard
```

---

## 📊 Datasets Used

### 1. Amazon Consumer Reviews — `7817_1.csv` *(Primary)*
**~1,596 product reviews** from Amazon. On its own, this dataset was heavily skewed — 88% Satisfied — which caused poor CSAT boundary predictions.

| Column | Usage |
|---|---|
| `reviews.text` | Main review body → model input |
| `reviews.title` | Prepended to body text |
| `reviews.rating` | 1–5 star → CSAT label |
| `reviews.doRecommend` | Boolean → refines 3-star boundary |

### 2. TrustPilot Reviews — `trust_pilot_reviews_data_2022_06.csv` *(Merged — key fix)*
**~3,698 reviews** from real UK businesses across multiple industries. Added to fix the severe CSAT class imbalance from Amazon alone — TrustPilot has a much healthier distribution of 1–3 star (negative/neutral) reviews.

| Column | Usage |
|---|---|
| `review_title` | Prepended to body |
| `review_text` | Body text → model input |
| `rating` | 1–5 star → CSAT label |

**Combined dataset: 5,294 rows**
| Class | Count |
|---|---|
| CSAT Dissatisfied | 752 |
| CSAT Neutral | 163 |
| CSAT Satisfied | 4,379 |

### 3. Twitter Sentiment — `twitter_training/validation.csv` *(Exploratory only)*
Used during early research to understand short-form complaint patterns. Not included in the final training pipeline.

---

## 🏷️ Label Derivation

Since no ground-truth CSAT/CES labels exist, both are **algorithmically derived** from review text and ratings.

### CSAT (3 classes)

| Condition | Label |
|---|---|
| Rating 1–2 | `0` Dissatisfied |
| Rating 3 + doRecommend = True | `2` Satisfied |
| Rating 3 + doRecommend = False | `0` Dissatisfied |
| Rating 3 (ambiguous) | `1` Neutral |
| Rating 4–5 | `2` Satisfied |

### CES (2 classes)
Binary — derived from effort/friction keyword matching on review text:

| Condition | Label |
|---|---|
| Any effort keyword found | `1` Difficult |
| No effort keywords | `0` Easy |

**Effort keywords** (30+ patterns): `late`, `delayed`, `refund`, `return`, `customer service`, `complaint`, `damaged`, `broken`, `not working`, `frustrated`, `confused`, `poor quality`, `disappointed`, `wrong item`, `overcharged`, `terrible`, `unacceptable`, etc. — compiled as regex with inflection wildcards (`frustrat\w*`, `disappoint\w*`).

---

## 🧠 Model Architecture

**`MultiTaskReviewModel`** — `multitask_model/model.py`

```
Input Text
    ↓
[BERT Tokeniser]  →  token IDs + attention mask (max_len=256)
    ↓
[Shared Encoder]  —  bert-base-uncased  (110M params)
    ↓
[CLS] token embedding  (768-dim)
    ↓
        ┌──────────────────┬────────────────┐
        ▼                  ▼
  [CSAT Head]          [CES Head]
  768 → 256 → 3        768 → 128 → 2
  (GELU + Dropout)     (GELU + Dropout)
        ↓                  ↓
  Dissatisfied /       Easy /
  Neutral /            Difficult
  Satisfied
```

| Component | Detail |
|---|---|
| Base encoder | `bert-base-uncased` (HuggingFace) |
| Shared representation | `[CLS]` token embedding |
| CSAT head | Linear(768→256) → GELU → Dropout → Linear(256→3) |
| CES head | Linear(768→128) → GELU → Dropout → Linear(128→2) |
| Dropout | 0.3 on encoder output and within each head |
| Loss | `α·CrossEntropy(CSAT) + β·CrossEntropy(CES)` with class weights |

---

## ⚙️ Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 5 |
| Batch size | 16 |
| Max sequence length | 256 tokens |
| Optimiser | AdamW |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| LR schedule | Linear warmup + decay (warmup = 10%) |
| Gradient clipping | 1.0 |
| CSAT loss weight (α) | 1.0 (+ class weights) |
| CES loss weight (β) | 1.0 (+ class weights) |
| Train / Val / Test split | 70% / 15% / 15% (stratified) |
| Seed | 42 |

**Class weights** (computed from training split, `sklearn` balanced):

| CSAT Class | Weight |
|---|---|
| Dissatisfied | 2.35× |
| Neutral | **10.83×** |
| Satisfied | 0.40× |

---

## 📈 Training Results

*(Amazon + TrustPilot combined, 5,294 rows, class-weighted loss)*

| Epoch | Train Loss | Val Loss | Val CSAT F1 | Val CES F1 |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 1.4298 | 0.8667 | 0.9415 | 0.8449 |
| 2 | 0.8410 | 0.7150 | 0.9431 | 0.8693 |
| **3** | **0.5324** | **0.5973 ✓** | **0.9410** | **0.9093** |
| 4 | 0.3146 | 0.7201 | 0.9474 | 0.9423 |
| 5 | 0.2272 | 0.6987 | 0.9508 | 0.9423 |

> ✓ Best checkpoint saved at **Epoch 3** (lowest val loss = 0.5973)

### Final Test Results

| Metric | Score |
|---|---|
| **CSAT Accuracy** | **93.08%** |
| **CSAT Weighted F1** | **93.83%** |
| **CES Accuracy** | **90.44%** |
| **CES Weighted F1** | **90.77%** |

### Inference Example

```
Review : The product is great but the return process was frustrating.
CSAT   : Neutral       confidence=71.39%
CES    : Difficult     confidence=98.42%
```
> Correctly identifies mixed sentiment as **Neutral** (not Dissatisfied), and high-confidence **Difficult** CES from friction keywords.

---

## 🔧 Improvements Made

### 1. Label Engineering
- Combined `reviews.rating` + `reviews.doRecommend` to handle ambiguous 3-star reviews, reducing boundary label noise.

### 2. Dataset Merging to Fix Class Imbalance *(biggest fix)*
- Amazon alone: 88% Satisfied → model predicted Dissatisfied on mixed reviews
- Merged TrustPilot (3,698 rows) → 5,294 total, balanced CSAT distribution
- Result: mixed reviews now correctly predict **Neutral** instead of incorrectly predicting Dissatisfied

### 3. Class-Weighted CrossEntropyLoss
- `sklearn.compute_class_weight("balanced")` on training split
- Neutral class weighted 10.83× — strongly penalises ignoring the hardest boundary class
- Passed directly to `nn.CrossEntropyLoss(weight=...)`

### 4. Multi-Task Learning
- Single shared encoder for both CSAT + CES instead of two separate models
- Joint gradient flow improves both tasks simultaneously

### 5. CES Keyword Expansion
- 30+ regex patterns covering all major e-commerce friction types
- Inflection wildcards (`frustrat\w*`) reduce false negatives

### 6. Training Stability
- Linear LR warmup (10% of steps), gradient clipping (1.0), AdamW weight decay, best-checkpoint saving

### 7. Text Preprocessing
- Review title concatenated with body before tokenisation for richer input signal

---

## 🚀 How to Run

```bash
# Install dependencies
cd multitask_model
pip install -r requirements.txt

# Train
python train.py

# Evaluate + generate plots
python evaluate.py

# Inference (single review)
python inference.py "Your review text here."
```

---

## 📦 Dependencies

| Package | Version |
|---|---|
| `torch` | ≥ 2.0.0 |
| `transformers` | ≥ 4.35.0 |
| `pandas` | ≥ 1.5.0 |
| `scikit-learn` | ≥ 1.2.0 |
| `numpy` | ≥ 1.24.0 |
| `matplotlib` | ≥ 3.7.0 |
| `seaborn` | ≥ 0.12.0 |

---

## 📋 Summary

| Item | Detail |
|---|---|
| **Primary dataset** | Amazon Consumer Reviews (1,596 rows) |
| **Supplementary dataset** | TrustPilot Reviews (3,698 rows, merged) |
| **Combined training data** | 5,294 rows |
| **Model** | BERT-base-uncased + dual classification heads |
| **Tasks** | CSAT (3-class) + CES (2-class), trained jointly |
| **Loss** | Class-weighted CrossEntropyLoss |
| **Best epoch** | Epoch 3 (val loss = 0.5973) |
| **CSAT test accuracy** | **93.1%** |
| **CES test accuracy** | **90.4%** |
| **Framework** | PyTorch + HuggingFace Transformers |
