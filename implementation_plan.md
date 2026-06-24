# Implementation Plan: German BERT Fine-Tuning Pipeline Improvements

This plan addresses the dataset preparation and modeling flaws identified in the initial walkthrough.

---

## Proposed Changes

### 1. Dependencies (`requirements.txt`)
#### [MODIFY] [requirements.txt](file:///c:/Users/vijayakr/Downloads/Fine-tune%20BERT%20%28deepsetgbert-large%29/requirements.txt)
* Add `langdetect` for robust language identification.
* Add `accelerate` (required by Hugging Face `Trainer` for training configuration).
* Add `evaluate` (Hugging Face tool for computing metrics).
* Add `scikit-learn` and `pandas` updates.

---

### 2. Dataset Preparation (`prepare_dataset.py`)
#### [MODIFY] [prepare_dataset.py](file:///c:/Users/vijayakr/Downloads/Fine-tune%20BERT%20%28deepsetgbert-large%29/prepare_dataset.py)
* **Problem 1 (German Filter)**: Replace regex `[äöüÄÖÜß]` with `langdetect.detect(text) == 'de'`. Handle any potential detection exceptions gracefully.
* **Problem 2 (Class Imbalance)**: Change default config sampling parameters to `HUMAN_SAMPLE = 250_000` and `AI_SAMPLE = 250_000` to create a 1:1 balanced dataset.
* **Problem 4 (Deduplication)**: Apply deduplication (`drop_duplicates(subset=["text"])`) on the combined dataset before splitting, preventing training-validation-test leakage.

---

### 3. Model Skeletons and Pipelines
To incorporate the training suggestions, we will write fully-implemented skeleton structures inside the files:

#### [MODIFY] [train.py](file:///c:/Users/vijayakr/Downloads/Fine-tune%20BERT%20%28deepsetgbert-large%29/train.py)
* Implement a training script using Hugging Face `Trainer` rather than a manual PyTorch training loop.
* Set default `max_length = 256`.
* Integrate hyperparameters for `deepset/gbert-large`:
  - `per_device_train_batch_size = 16`
  - `learning_rate = 2e-5`
  - `num_train_epochs = 3`
  - `weight_decay = 0.01`
  - `warmup_ratio = 0.1`
  - FP16/BF16 mixed precision training.
  - Early stopping based on validation loss/F1.

#### [MODIFY] [evaluate .py](file:///c:/Users/vijayakr/Downloads/Fine-tune%20BERT%20%28deepsetgbert-large%29/evaluate%20.py)
* Write evaluation routines computing Accuracy, Precision, Recall, Macro F1, and ROC-AUC.
* Group evaluation results by the `source` column to report domain-specific metrics (e.g., Politics, Law, Administration) as requested.

#### [MODIFY] [predict .py](file:///c:/Users/vijayakr/Downloads/Fine-tune%20BERT%20%28deepsetgbert-large%29/predict%20.py)
* Update prediction script to use `langdetect` logic and load the fine-tuned `gbert-large` tokenizer and model weights for inference.

---

### 4. Walkthrough Update
#### [MODIFY] [walkthrough.md](file:///c:/Users/vijayakr/Downloads/Fine-tune%20BERT%20%28deepsetgbert-large%29/walkthrough.md)
* Update the codebase walkthrough to describe the new balanced dataset pipeline, language detection strategy, Hugging Face `Trainer` setup, evaluation metrics, and domain-specific tests.

---

## Open Questions

> [!NOTE]
> 1. **Langdetect Speed**: `langdetect` is written in Python and can be slow when processing 500k+ sentences. Would you prefer a faster pre-filter (like keeping the basic word lengths/character length filters) combined with `langdetect`, or utilizing `fasttext` if a model binary is available? (For safety, we will default to `langdetect` wrapped in try/except, but with a warning in the code).
> 2. **Execution**: Do you want me to write the complete implementation of these scripts now, or just update the skeleton plans and walkthrough in the codebase?

---

## Verification Plan

### Automated Verification
* Run syntax checks on all updated Python files:
  `python -m py_compile prepare_dataset.py train.py "evaluate .py" "predict .py"`
