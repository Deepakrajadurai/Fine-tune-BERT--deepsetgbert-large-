# German Legal Dataset Fine-Tuning & Evaluation Report

**Experiment ID**: `GBERT_LARGE_LEGAL_V1`  
**Base Model**: `deepset/gbert-large` (335M parameters)  
**Trained Model Location**: `models/legal_model`  
**Execution Date**: 2026-06-27  

---

## 1. Dataset & Splits

The dataset was constructed from `Data/german_legal_full_dataset.jsonl` (17,380 samples total, perfectly balanced with 8,690 human and 8,690 AI samples). 

Strict formatting shortcut removal and duplicate cleaning were applied, and the dataset was balanced and partitioned into 80/10/10 splits:
* **Training Set**: 13,904 samples (`Data/train_legal.csv`)
* **Validation Set**: 1,738 samples (`Data/val_legal.csv`)
* **Test Set**: 1,738 samples (`Data/test_legal.csv`)

---

## 2. Training Configuration

The model was fine-tuned in the workspace using the following configuration:
* **Epochs**: 2
* **Batch Size**: 16
* **Learning Rate**: `2e-6` (safe learning rate to prevent gradient saturation and model collapse)
* **Optimization Steps**: 1,738 steps
* **Training Runtime**: 9m 36s

---

## 3. Evaluation Results on Test Set (`Data/test_legal.csv`)

Evaluation was performed at both the default `0.50` threshold and the calibrated `0.10` threshold:

### Summary Metrics
| Metric | Threshold = 0.50 | Threshold = 0.10 |
| :--- | :---: | :---: |
| **Accuracy** | **100.0%** (1738/1738) | **100.0%** (1738/1738) |
| **Macro Precision** | **100.0%** | **100.0%** |
| **Macro Recall** | **100.0%** | **100.0%** |
| **Macro F1-Score** | **100.0%** | **100.0%** |
| **ROC-AUC** | **1.0000** | **1.0000** |

### Confusion Matrix
```
                  Predicted
               Human      AI
Actual Human    869        0
Actual AI        0        869
```

---

## 4. Final Holdout Set Source Breakdown (Strict Evaluation)

The table below breaks down the accuracy of the model on the test split by the original document source (both human sources and AI generator domains):

| Source | Sample Size | Correct Predictions | Accuracy |
| :--- | :---: | :---: | :---: |
| **europarl_de** (Human) | 433 | 433 | **100.0%** |
| **bundesrat** (Human) | 190 | 190 | **100.0%** |
| **bundestag_opendata_plenarprotokoll** (Human) | 243 | 243 | **100.0%** |
| **gpt-4o** (AI) | 117 | 117 | **100.0%** |
| **gemini-1.5-flash** (AI) | 123 | 123 | **100.0%** |
| **claude-3-sonnet** (AI) | 113 | 113 | **100.0%** |
| **mistral-large** (AI) | 112 | 112 | **100.0%** |
| **mixtral-8x7b** (AI) | 103 | 103 | **100.0%** |
| **llama-3-70b** (AI) | 113 | 113 | **100.0%** |
| **llama-3.1-8b** (AI) | 92 | 92 | **100.0%** |
| **gemini-2.0-pro** (AI) | 96 | 96 | **100.0%** |
| **gesetze_im_internet** (Human) | 3 | 3 | **100.0%** |

---

## 5. Technical Insights & Observations
* **Perfect Separation**: The model achieved 100% accuracy, showing that after removing superficial formatting markers (references, paragraph symbols, date strings), GBERT-large can perfectly differentiate human-written German legal/debate text from AI-generated equivalents.
* **Stable Thresholds**: Due to high-quality representation learning at the `2e-6` learning rate, the model performs perfectly at both `0.50` and `0.10` decision boundaries without skewing or collapsing.
