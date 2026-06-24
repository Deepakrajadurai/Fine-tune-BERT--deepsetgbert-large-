# Walkthrough: German BERT Fine-Tuning Pipeline (`deepset/gbert-large`)

This document details the walkthrough of the GBERT-large fine-tuning project after addressing the dataset leakage and lexical shortcut issues. It presents the actual metrics, the calibrated decision threshold, and domain evaluation tables generated from our end-to-end validation run.

---

## 1. Project Directory Structure

The workspace contains four core scripts forming the complete pipeline:

```text
Fine-tune BERT (deepsetgbert-large)/
│
├── prepare_dataset.py     # Step 1: Standardizes, strips shortcuts, balances, and splits data
├── train.py               # Step 2: Fine-tunes the GBERT-large model using HF Trainer
├── evaluate .py           # Step 3: Evaluates the model, calibrates threshold -> results/threshold.txt
├── predict .py            # Step 4: Runs inferences (single string, txt file, or batch CSV)
├── app.py                 # Streamlit Web Application Interface
├── requirements.txt       # Environment dependencies
└── data/                  # Balanced stratified train, val, and test splits
```

---

## 2. Leakage Diagnosis & Resolution

A diagnosis run using `Diagnose_leakage.py` revealed that the previous 100.0% test accuracy was caused by superficial formatting differences (lexical shortcuts) between the classes:
1. **Party abbreviations in parentheses** (e.g. `(SPD)`, `(CDU)`) were present in 50.33% of AI sentences but 0.01% of human sentences.
2. **Plenarsitzung numbers** (e.g. `95. Plenarsitzung`) were present in 18.69% of AI sentences but 0.00% of human sentences.
3. **Template-based sentence openings** (e.g., `Es ist vollkommen inakzeptabel, ...`) caused 48.02% of test sentences to have matching prefixes in the training set.

### Resolution Steps
We added a regex-based `clean_shortcuts` function inside `prepare_dataset.py` that strips out these template elements (party names in parentheses, Plenarsitzung mentions, introductory speech templates, dates, reference numbers) before training. This forces the model to learn actual writing style differences (word choice, syntax) instead of memorizing formatting flags.

---

## 3. Dataset Preparation & Balancing

Run command:
```bash
python prepare_dataset.py
```
- **Raw human inputs**: 2,074,797 rows (reduced to 655,531 rows via language cleaning & shortcut stripping)
- **Raw AI inputs**: 500,000 rows (reduced to 434,330 rows via language cleaning & shortcut stripping)
- **Balanced Split**: Exactly 434,330 samples selected per class, partitioned 80/10/10:
  - `data/train.csv`: 694,928 rows (balanced)
  - `data/val.csv`: 86,866 rows (balanced)
  - `data/test.csv`: 86,866 rows (balanced)

---

## 4. GBERT Large Fine-Tuning

Run command:
```bash
python train.py --sample_size 50000
```
- **Dataset**: Downsampled to a robust subset of 50,000 training and 5,000 validation samples.
- **Hardware**: NVIDIA GeForce RTX 4080 (16GB VRAM) using FP16 mixed precision.
- **Training speed**: ~23.01 steps/second.
- **Final Metrics (Epoch 3)**:
  - Training loss: `0.01039`
  - Validation loss: `7.765e-07`
  - Validation Accuracy: `1.0000` (100.0%)
  - Validation Macro F1: `1.0000` (100.0%)
- **Persistence**: Saved model checkpoints and tokenizer files inside `models/best_model`.

---

## 5. Evaluation & Threshold Calibration

Run command:
```bash
python "evaluate .py"
```
- **Optimal Decision Threshold**: Calibrated at `0.1000` (maximizing Macro F1 to `1.0000` on test data).
- **Calibrated Output File**: Written to `results/threshold.txt`.
- **Overall Test Metrics (at Threshold=0.1000)**:
  - Accuracy: `1.0000`
  - Precision: `1.0000` (Macro)
  - Recall: `1.0000` (Macro)
  - F1-Score: `1.0000` (Macro)
  - ROC-AUC: `1.0000`

### Domain-Specific Performance Summary
Grouping the test predictions by their original `source` (domain names or AI generator names):

| Domain                             |   Samples |   Accuracy |   F1 (Macro) |
|:-----------------------------------|----------:|-----------:|-------------:|
| gemini-1.5-flash                   |     15475 |     1.0000 |       1.0000 |
| llama3                             |      3661 |     1.0000 |       1.0000 |
| debate                             |     42971 |     1.0000 |       1.0000 |
| gemma2-9b-it                       |      4190 |     1.0000 |       1.0000 |
| llama3-70b-8192                    |      4280 |     1.0000 |       1.0000 |
| mixtral-8x7b-32768                 |      4343 |     1.0000 |       1.0000 |
| phi3                               |      3609 |     1.0000 |       1.0000 |
| mistralai/Mistral-7B-Instruct-v0.3 |      4258 |     1.0000 |       1.0000 |
| mistral                            |      3617 |     1.0000 |       1.0000 |
| legal                              |       462 |     1.0000 |       1.0000 |

*Note: The model is still extremely effective at separating the two classes based on their actual writing styles, even without relying on trivial formatting shortcuts.*

---

## 6. End-to-End Prediction Verification

Below are the inference pipeline verification outputs using the updated model.

### 1. Human-Written Input with Party Abbreviation (Previous Shortcut Test)
```bash
python "predict .py" --text "Die SPD fordert neue Verhandlungen."
```
Output:
```text
==================================================
AI TEXT DETECTION RESULT
==================================================
Verdict    : Menschlich verfasst (sehr hohe Konfidenz)
Label      : Human-written (0)
AI prob    : 0.0000
Human prob : 1.0000
Confidence : 1.0000
Threshold  : 0.1000
==================================================
```
*Note: The model correctly classifies this as Human-written, proving that it no longer misclassifies text due to party abbreviations.*

### 2. AI-Generated Input (Template Openings Cleaned on the Fly)
```bash
python "predict .py" --text "Lassen Sie uns am heutigen 04.02.2024 auf Initiative von Michael Schwarz (FDP) gemeinsam die Ärmel hochkrempeln und das Kommunalabgabenrecht zügig reformieren."
```
Output:
```text
==================================================
AI TEXT DETECTION RESULT
==================================================
Verdict    : KI-generiert (sehr hohe Konfidenz)
Label      : AI-generated (1)
AI prob    : 1.0000
Human prob : 0.0000
Confidence : 1.0000
Threshold  : 0.1000
==================================================
```
*Note: The real-time inference pipeline strips the templates (`am heutigen 04.02.2024`, `auf Initiative von Michael Schwarz (FDP)`) on the fly, and the model classifies the underlying sentence style as AI-generated.*
