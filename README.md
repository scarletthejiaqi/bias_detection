# 😂🔍 Humor Bias Detection

**Multi-label classification of bias, stereotypes, and rhetorical devices in humorous text using transformer-based NLP models.**

> Can AI learn to detect when a joke crosses the line? This project builds and evaluates transformer models to automatically classify jokes along three axes: **bias type** (stereotype, offensive, targeted, normal), **target group** (ethnicity, gender, religion, etc.), and **rhetorical device** (sarcasm, wordplay, dark humor, etc.).

---

## 📌 Table of Contents

- [Motivation](#-motivation)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Models & Results](#-models--results)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Annotation Guidelines](#-annotation-guidelines)
- [License](#license)

---

## 💡 Motivation

Humor is one of the most nuanced and culturally loaded forms of human communication. While jokes can be harmless fun, they often serve as vehicles for stereotypes, prejudice, and targeted attacks against marginalized groups. Automated detection of bias in humor is a critical step toward:

- **Content moderation** — helping platforms flag harmful humorous content at scale
- **Bias auditing** — understanding how stereotypes propagate through comedy
- **NLP research** — advancing multi-label text classification in subjective, context-dependent domains

This project explores whether state-of-the-art transformer models can reliably distinguish between normal humor, stereotypical jokes, offensive content, and targeted attacks — and simultaneously identify *who* is being targeted and *how* the humor is constructed.

---

## 📊 Dataset

The dataset combines two curated sources of annotated jokes:

| Source | Description | Samples |
|--------|-------------|---------|
| `gold_labels_deduplicated.csv` | Expert-annotated jokes with gold-standard labels | 1,921 |
| `ACL_toAnnotate_IAA.csv` | Additional jokes annotated for inter-annotator agreement | 2,538 |

After merging, deduplication, and cleaning, the combined dataset is split **80/20** into:

- **Training set**: 2,417 samples
- **Validation set**: 605 samples

### Label Taxonomy

Each joke is annotated along **three dimensions**:

**1. Bias Label** (`label`) — *How* biased is the joke?

| Label | Description |
|-------|-------------|
| `normal` | Harmless humor with no discernible bias |
| `stereotype` | Reinforces group-level stereotypes |
| `offensive/provocative` | Deliberately shocking or provocative content |
| `targeted` | Directly attacks a specific person or group |
| `not a joke` | Non-humorous content (filtered out during training) |

**2. Target Group** (`target`) — *Who* is being targeted?

| Target | Target | Target |
|--------|--------|--------|
| Ethnicity / National origin | Gender | Religious beliefs |
| Sexual orientation | Disability / Health | Body image |
| Political beliefs | Education | Other / None |

**3. Rhetorical Device** (`rhetoric`) — *What technique* does the joke use?

| Device | Device | Device |
|--------|--------|--------|
| Wordplay | Sarcasm | Dark humor |
| Satire | Irony | Hyperbole |
| Cultural reference | Vulgarity | Self-deprecation |
| Understatement | Puns | Other / None |

---

## 🔬 Methodology

### Data Pipeline

```
gold_labels_deduplicated.csv ──┐
                                ├── merge & clean ──► train.csv (80%)
ACL_toAnnotate_IAA.csv ────────┘                  └► validation.csv (20%)
```

Key preprocessing steps (in `data_preprocessing_training.ipynb`):
1. Column renaming for consistency (`gold_label` → `label`, `stereotype_SSB` → `label`, etc.)
2. Dropping unlabeled rows and removing unnamed columns
3. Case normalization of labels and joke text
4. Random shuffle with seed (`random.seed(888)`) for reproducibility
5. **Undersampling** of majority classes (threshold: 60 samples per class) to mitigate class imbalance

### Model Training

Four pretrained transformer models are fine-tuned for **sequence classification** on the rhetoric label (13 classes):

| Model | Pretrained Checkpoint |
|-------|----------------------|
| **DistilBERT** | `distilbert/distilbert-base-uncased` |
| **DeBERTa v3** | `microsoft/deberta-v3-base` |
| **ELECTRA** | `google/electra-base-discriminator` |
| **RoBERTa** | `FacebookAI/roberta-base` |

Training details:
- **Framework**: Hugging Face `transformers` + PyTorch
- **Tokenization**: Model-specific tokenizers with `max_length=40`, padding, and truncation
- **Hyperparameter tuning**: Grid search over learning rate, batch size, and number of epochs
- **Optimizer**: AdamW with linear learning rate schedule and warmup
- **Loss**: Cross-entropy with class weights to handle imbalance

Two separate training pipelines exist:
- `01_train_transformer_models.ipynb` — Classifies **rhetorical devices** (e.g., wordplay, sarcasm, dark humor)
- `02_train_transformer_target.ipynb` — Classifies **target groups** (e.g., ethnicity, gender, religion)

---

## 📈 Models & Results

### F1 Scores by Rhetorical Device (per model)

| Rhetorical Device | DistilBERT | DeBERTa | ELECTRA | RoBERTa |
|:---|:---:|:---:|:---:|:---:|
| Wordplay | 0.44 | **0.52** | 0.51 | 0.49 |
| Vulgarity | **0.50** | 0.31 | 0.41 | **0.50** |
| Cultural reference | 0.34 | **0.42** | 0.15 | 0.27 |
| Dark humor | **0.32** | 0.21 | 0.07 | 0.26 |
| Sarcasm | **0.23** | 0.21 | 0.10 | 0.21 |
| Hyperbole | 0.23 | 0.17 | 0.19 | **0.25** |
| Satire | 0.15 | **0.22** | 0.12 | 0.19 |
| Irony | 0.14 | 0.00 | 0.18 | **0.20** |
| None | 0.14 | 0.00 | 0.00 | 0.00 |
| Self-deprecation | 0.00 | 0.00 | 0.00 | 0.00 |
| Other | 0.00 | 0.00 | 0.00 | 0.00 |
| Puns | 0.00 | 0.00 | 0.00 | 0.00 |

### Key Findings

- **DeBERTa v3** achieves the highest F1 on the most categories (wordplay, cultural reference, satire)
- **DistilBERT** and **RoBERTa** lead on vulgarity detection (F1 = 0.50)
- Underrepresented classes (self-deprecation, puns, other) remain challenging across all models
- Rhetorical devices with clearer lexical signals (wordplay, vulgarity) are easier to detect than those requiring contextual understanding (irony, satire)

---

## 📁 Repository Structure

```
humor_bias_detection/
│
├── README.md                          # This file
├── annotation_guidelines.docx         # Guidelines used by human annotators
├── reading_list.docx                  # Related literature and references
│
├── code/
│   ├── data_preprocessing_training.ipynb  # Data merging, cleaning, train/val split
│   ├── 01_train_transformer_models.ipynb  # Fine-tuning models on rhetoric labels
│   ├── 02_train_transformer_target.ipynb  # Fine-tuning models on target labels
│   ├── eda_visualizations.ipynb           # Exploratory data analysis & model evaluation plots
│   │
│   ├── gold_labels_deduplicated.csv       # Primary annotated dataset
│   ├── ACL_toAnnotate_IAA.csv             # Additional annotated dataset (IAA study)
│   ├── train.csv                          # Generated training split (80%)
│   ├── validation.csv                     # Generated validation split (20%)
│   └── model_f1_comparison.csv            # F1 scores across all models & labels
│
├── data/
│   ├── gold_labels_deduplicated.csv       # Raw data copy
│   ├── ACL_toAnnotate_IAA.csv             # Raw data copy
│   └── gold_labels_deduplicated_copy.csv  # Backup copy
│
└── presentation/
    ├── Humor Bias DSSI Poster.pdf                     # Research poster
    ├── Presentation - AI Humor Classification.pptx    # Final presentation
    └── Humour Bias Lightning Talk *.pptx              # Lightning talk slides
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- CUDA-capable GPU recommended (CPU training is slow)

### Installation

```bash
git clone https://github.com/scarletthejiaqi/humor_bias_detection.git
cd humor_bias_detection

pip install pandas numpy matplotlib seaborn
pip install torch transformers datasets evaluate sentencepiece accelerate scikit-learn
```

### Workflow

Follow the numbered notebooks in order:

```
1. data_preprocessing_training.ipynb   →  Prepare train.csv & validation.csv
2. 01_train_transformer_models.ipynb   →  Train & evaluate on rhetoric labels
3. 02_train_transformer_target.ipynb   →  Train & evaluate on target labels
4. eda_visualizations.ipynb            →  Visualize distributions & model performance
```

> **Note**: The `train.csv` and `validation.csv` files are already provided. Re-running the preprocessing notebook will regenerate them (the random seed is set to 888 for reproducibility).

---

## 📝 Annotation Guidelines

Detailed annotation guidelines are available in `annotation_guidelines.docx`. Annotators were asked to label each joke along three independent axes:

1. **Bias type** — Is the joke reinforcing stereotypes, being offensive, or targeting someone?
2. **Target group** — Which demographic group (if any) is the subject of the joke?
3. **Rhetorical device** — What linguistic or comedic technique is being used?

Inter-annotator agreement was measured using samples from `ACL_toAnnotate_IAA.csv`.

---

## License

This project was developed as part of the **Data Science Summer Institute (DSSI)** research program. Please cite appropriately if using this dataset or methodology in your work.
