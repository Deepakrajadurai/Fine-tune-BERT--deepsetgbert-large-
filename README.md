# 🤖 G-BERT: German AI Text Detector

A state-of-the-art AI text detection system for German language, built by fine-tuning [`deepset/gbert-large`](https://huggingface.co/deepset/gbert-large) on a curated multi-domain corpus of ~57,000 balanced paragraphs.

> **Overall Test Accuracy: 99.77% | Macro F1: 99.77%**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [AI Model Coverage](#ai-model-coverage)
- [License](#license)

---

## Overview

G-BERT is a binary classifier that distinguishes **human-written** from **AI-generated** German text across three domains:

| Domain | Human Source | AI Generation |
|--------|-------------|---------------|
| **Politics** | Bundestag parliamentary speeches | Multi-model synthetic speeches |
| **News** | GNAD (German News Articles Dataset) | Multi-model synthetic news |
| **Casual** | GermEval 2018 (social media, blogs, forums) | Multi-model synthetic casual text |

The system uses **grammar-preserving placeholder masking** (replacing dates, party names, legal references, person names with tokens like `[DATUM]`, `[PARTEI]`, `[PERSON]`) to prevent the model from relying on domain-specific shortcuts and ensure robust generalization.

---

## Key Features

- 🎯 **99.77% accuracy** on held-out test data
- 🌐 **Multi-domain**: Politics, News, and Casual German text
- 🤖 **Multi-model robust**: Trained against 8 diverse AI text generators
- 📊 **Streamlit Web App** with single text + batch CSV analysis
- 🔧 **Calibrated decision threshold** with adjustable sensitivity
- ⚡ **FP16 inference** for fast GPU predictions
- 📑 **Chunked long-document support** with overlapping window aggregation

---

## Architecture

```
deepset/gbert-large (340M params)
        │
  Fine-tuned on ~57K paragraphs
  (50% human / 50% AI, 3 domains)
        │
  Binary Classification Head
        │
  ┌─────┴─────┐
  │           │
Human (0)   AI (1)
```

**Training Configuration:**
- Optimizer: AdamW (lr=2e-5, weight_decay=0.01)
- Warmup: 10% of training steps
- Epochs: 3 with early stopping (patience=2)
- Max sequence length: 256 tokens
- Precision: BF16/FP16 (automatic)
- Batch size: 16
- Best checkpoint selected by **external validation F1** (not in-distribution)

---

## Project Structure

```
├── prepare_dataset.py          # Step 1: Data loading, cleaning, balancing
├── generate_synthetic_data.py  # Synthetic AI text generation
├── generate_diverse_ai.py      # Multi-model diverse AI text generation
├── train.py                    # Step 2: Fine-tuning with generalization tracking
├── evaluate .py                # Step 3: Evaluation & holdout verification
├── predict .py                 # Step 4: Inference pipeline (CLI + API)
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
├── results/
│   └── threshold.txt           # Calibrated decision threshold
├── Data/                       # (not tracked — see Data section)
│   ├── Human_model_ready_dataset.csv
│   ├── gnad_articles.csv
│   ├── germeval2018.txt
│   ├── ai_generated_sentences_500k.csv
│   ├── ai_generated_news.csv
│   ├── ai_generated_casual.csv
│   ├── train.csv / val.csv / test.csv
│   ├── external_val.csv
│   └── final_holdout.csv
└── models/                     # (not tracked — >1GB model weights)
    └── best_model/
```

> **Note:** The `Data/` and `models/` directories are excluded from version control due to file size constraints (>100MB). See the [Setup](#setup--installation) section for instructions.

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (recommended; CPU works but is slower)
- ~2GB disk space for model weights

### 1. Clone the Repository

```bash
git clone https://github.com/Deepakrajadurai/Fine-tune-BERT--deepsetgbert-large-.git
cd Fine-tune-BERT--deepsetgbert-large-
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# or
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install streamlit            # For the web app
```

### 4. Prepare Data & Train (if needed)

```bash
# Step 1: Prepare the dataset (requires raw data files in Data/)
python prepare_dataset.py

# Step 2: Train the model
python train.py --epochs 3 --batch_size 16 --lr 2e-5

# Step 3: Evaluate
python "evaluate .py"
```

---

## Usage

### 🖥️ Web Application (Streamlit)

```bash
streamlit run app.py
```

The web app provides:
- **Single Text Analysis**: Paste any German text and get an instant AI/Human verdict with confidence scores
- **Batch File Upload**: Upload a CSV file for high-throughput classification
- **Adjustable Threshold**: Fine-tune the decision boundary via the sidebar slider
- **Pre-loaded Examples**: Try built-in human and AI text samples

### 🔧 Command-Line Interface

```bash
# Classify a single text
python "predict .py" --text "Die Bundesregierung hat beschlossen..."

# Classify from a file
python "predict .py" --file my_document.txt

# Batch CSV prediction
python "predict .py" --csv input.csv --text_col text --out predictions.csv

# Override threshold
python "predict .py" --text "..." --threshold 0.5
```

### 🐍 Python API

```python
from importlib.util import spec_from_file_location, module_from_spec

spec = spec_from_file_location("predict", "predict .py")
predict = module_from_spec(spec)
spec.loader.exec_module(predict)

detector = predict.AITextDetector(threshold=0.30)
result = detector.predict("Ihr deutscher Text hier...")

print(result)
# {
#   "label": 0,          # 0 = Human, 1 = AI
#   "confidence": 0.98,
#   "ai_prob": 0.02,
#   "human_prob": 0.98,
#   "verdict": "Menschlich verfasst (sehr hohe Konfidenz)",
#   "threshold": 0.30,
#   "n_chunks": 1
# }
```

---

## Model Performance

### In-Distribution Test Set

| Metric | Score |
|--------|-------|
| **Accuracy** | 99.77% |
| **Macro F1** | 99.77% |
| **Validation Loss** | 0.02066 |

### Preprocessing Pipeline

The model applies domain-aware preprocessing to both training and inference:

| Pattern | Replacement | Purpose |
|---------|-------------|---------|
| `§ 18 Abs. 3` | `[PARAGRAPH]` | Legal references |
| `15.06.2026` | `[DATUM]` | Dates |
| `CDU`, `SPD`, etc. | `[PARTEI]` | Political parties |
| Person names | `[PERSON]` | Named entities |
| `Plenarsitzung` | `[PLENARSITZUNG]` | Parliamentary sessions |
| `Drucksache` | `[DRUCKSACHE]` | Parliamentary documents |

This prevents the model from memorizing surface-level domain markers and forces it to learn genuine stylistic differences between human and AI text.

---

## AI Model Coverage

The detector was trained against text generated by **8 diverse AI models**:

| Model | Type |
|-------|------|
| `gemini-1.5-flash` | Google Gemini |
| `mistralai/Mistral-7B-Instruct-v0.3` | Mistral AI |
| `llama3-70b-8192` | Meta LLaMA 3 (70B) |
| `gemma2-9b-it` | Google Gemma 2 |
| `mixtral-8x7b-32768` | Mistral MoE |
| `phi3` | Microsoft Phi-3 |
| `mistral` | Mistral (base) |
| `llama3` | Meta LLaMA 3 |

---

## Hardware

Trained on **NVIDIA GeForce RTX 4080** (16GB VRAM) with mixed-precision (BF16/FP16).

---

## License

This project is for academic and research purposes.

---

<p align="center">
  Built with 🇩🇪 <a href="https://huggingface.co/deepset/gbert-large">deepset/gbert-large</a> · <a href="https://huggingface.co/docs/transformers">🤗 Transformers</a> · <a href="https://streamlit.io">Streamlit</a>
</p>
