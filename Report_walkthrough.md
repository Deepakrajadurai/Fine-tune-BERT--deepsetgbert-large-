# Walkthrough: German Text Dataset Quality Assurance & Split

This document details the completed environment setup, dataset preparation, split, and the quality assurance metrics for the German Human vs. AI text dataset.

---

## 1. Project Directory Structure

The processed datasets and validation report are generated in the `Data/processed/` directory:

```text
E:/16-06-26/Dataset_Quality_Testing/
├── Data/
│   ├── HUman_model_ready_dataset.csv     # Raw Human text dataset (2.07M rows)
│   ├── ai_generated_sentences.csv        # Raw AI text dataset (250k rows)
│   └── processed/                        # Output folder
│       ├── combined_dataset.csv          # Merged standardized dataset (2.32M rows)
│       ├── train.csv                     # 80% Train split (1,859,837 rows)
│       ├── test.csv                      # 10% Test split (232,480 rows)
│       ├── val.csv                       # 10% Validation split (232,480 rows)
│       ├── combined_sample.csv           # Stratified combined sample (10,000 rows)
│       ├── train_sample.csv              # Stratified train sample (8,000 rows)
│       ├── test_sample.csv               # Stratified test sample (1,000 rows)
│       ├── val_sample.csv                # Stratified validation sample (1,000 rows)
│       └── dataset_quality_report.html   # Standalone HTML validation report
├── prepare_dataset.py                    # Script to standardize, split, and sample
└── validate_dataset.py                   # Script to run QA validations and plot charts
```

---

## 2. Environment Setup

* **Python version**: Python 3.12.7 (solving the previous `MemoryError` and compilation compatibility issues present under Python 3.14.3).
* **Deep Learning & Hardware**: CUDA 12.4 support successfully activated on the **NVIDIA GeForce RTX 4080 (16GB)**. Sentence-BERT embeddings are generated using the GPU, dramatically reducing computation times.

---

## 3. Dataset Preparation & Stratified Split

Run Command:
```powershell
.\venv\Scripts\python.exe prepare_dataset.py
```
* **Raw Human samples**: 2,074,797 rows (labeled `0`)
* **Raw AI samples**: 250,000 rows (labeled `1`)
* **Combined Dataset**: 2,324,797 rows total
* **Stratified Split (80-10-10)**:
  * `Data/processed/train.csv`: 1,859,837 rows (80.0%)
  * `Data/processed/test.csv`: 232,480 rows (10.0%)
  * `Data/processed/val.csv`: 232,480 rows (10.0%)
* **Stratified Sample Generation (for CPU/GPU QA checks)**:
  * Combined Sample: 10,000 rows
  * Train Sample: 8,000 rows
  * Test Sample: 1,000 rows
  * Validation Sample: 1,000 rows

---

## 4. Dataset Quality & Assurance Test Results

Run Command:
```powershell
.\venv\Scripts\python.exe validate_dataset.py `
  --dataset Data/processed/combined_sample.csv `
  --reference dataset_v1.csv `
  --train Data/processed/train_sample.csv `
  --test Data/processed/test_sample.csv `
  --output Data/processed/dataset_quality_report.html
```

The validation suite completed successfully with the following metrics:

| Metric Category | Result / Finding | Status / Grade |
| :--- | :--- | :--- |
| **Final Quality Score** | **90.49 / 100** | **Grade: A** |
| **Integrity** | 0 missing texts, 0 missing labels, 0 empty text rows | Excellent |
| **Class Separation Audit** | Accuracy: **99.95%**, ROC-AUC: **1.0000** | Trivial Separability Warning |
| **Cleanlab Label Issues** | **0** potential label noise cases identified | Extremely clean labeling |
| **Near-Duplicates (LSH)** | **610** near-duplicate text pairs identified | Duplication found |
| **Train-Test Leakage** | **46** leaked pairs (Cosine Similarity >= 0.95) | Leakage warning |
| **Drift Detection (JSD)** | Label JSD: 0.0900, Length JSD: 0.0040, Language JSD: 0.4105 | Language share shift |

### Key Observations from QA Report

1. **AI/Human Separation (ROC-AUC = 1.0000)**: The separation audit classifier gets near 100% accuracy. This indicates that AI vs. Human classes are extremely easy to separate using basic word TF-IDF features (e.g. style/formatting indicators, like markdown symbols, formal prepositions, or typical LLM intro phrases).
2. **Train-Test Leakage**: 46 semantic duplicate pairs were leaked between train and test splits (Jaccard/Cosine similarity >= 0.95). These should ideally be deduplicated before final training to prevent validation leakage.
3. **Language Drift**: The language distribution JSD (0.4105) shows a significant shift relative to the reference `dataset_v1.csv`. The current combined dataset is primarily German Bundestag speech/debate patterns, whereas `dataset_v1.csv` had a higher proportion of English texts.

---

## 5. Standalone Quality Report

The interactive dashboard report has been generated at:
*   [dataset_quality_report.html](file:///E:/16-06-26/Dataset_Quality_Testing/Data/processed/dataset_quality_report.html)

You can open this HTML file directly in any web browser to view the **Label Distribution Analysis**, **Text Length distribution chart**, **Domain Space Classification**, **Cleanlab label noise lists**, and **MinHash LSH duplicate logs**.
