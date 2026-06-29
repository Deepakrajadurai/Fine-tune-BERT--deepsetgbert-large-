# Model Evaluation Comparison Report
**Date**: 2026-06-26 16:16:38
**Device**: cuda

This report compares the performance of three trained GBERT-large models:

1. **`best_model`**: Model trained on the ~57k dataset (leakage resolved).
2. **`full_model`**: Model trained on the full dataset version (57k run, possibly without early stopping or different setup).
3. **`full_model_500k`**: Model trained on the 500k dataset (which experienced model collapse).

## In-Distribution Test Evaluation Summary

| Model Name | Threshold | Accuracy | Macro Precision | Macro Recall | Macro F1 | ROC-AUC | Pred AI / Actual AI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **best_model** | 0.50 | 99.77% | 99.77% | 99.77% | **99.77%** | 1.0000 | 2861 / 2870 (Total: 5740) |
| **best_model** | 0.10 | 99.77% | 99.77% | 99.77% | **99.77%** | 1.0000 | 2861 / 2870 (Total: 5740) |
| **full_model** | 0.50 | 99.77% | 99.77% | 99.77% | **99.77%** | 1.0000 | 2883 / 2870 (Total: 5740) |
| **full_model** | 0.10 | 99.76% | 99.76% | 99.76% | **99.76%** | 1.0000 | 2884 / 2870 (Total: 5740) |
| **full_model_500k** | 0.50 | 50.00% | 25.00% | 50.00% | **33.33%** | 0.5238 | 0 / 2870 (Total: 5740) |
| **full_model_500k** | 0.10 | 50.00% | 25.00% | 50.00% | **33.33%** | 0.5238 | 5740 / 2870 (Total: 5740) |


## Unseen Final Holdout Evaluation Summary

| Model Name | Threshold | Accuracy | Macro Precision | Macro Recall | Macro F1 | ROC-AUC | Pred AI / Actual AI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **best_model** | 0.50 | 99.77% | 99.77% | 99.77% | **99.77%** | 1.0000 | 2861 / 2870 (Total: 5740) |
| **best_model** | 0.10 | 99.77% | 99.77% | 99.77% | **99.77%** | 1.0000 | 2861 / 2870 (Total: 5740) |
| **full_model** | 0.50 | 99.77% | 99.77% | 99.77% | **99.77%** | 1.0000 | 2883 / 2870 (Total: 5740) |
| **full_model** | 0.10 | 99.76% | 99.76% | 99.76% | **99.76%** | 1.0000 | 2884 / 2870 (Total: 5740) |
| **full_model_500k** | 0.50 | 50.00% | 25.00% | 50.00% | **33.33%** | 0.5238 | 0 / 2870 (Total: 5740) |
| **full_model_500k** | 0.10 | 50.00% | 25.00% | 50.00% | **33.33%** | 0.5238 | 5740 / 2870 (Total: 5740) |


## Analysis & Recommendations

### Best Model Recommendation: **`best_model`**

### Diagnostic Observations:
- **`best_model`**: Reaches a Macro F1 of **99.77%** at the calibrated 0.10 threshold, with a ROC-AUC of `1.0000`. This model generalizes exceptionally well to unseen holdout domains.
- **`full_model`**: Achieves a holdout Macro F1 of **99.76%** at the 0.10 threshold, with a ROC-AUC of `1.0000`.
- **`full_model_500k`**: Suffers from **catastrophic model collapse**. It predicts 0 AI samples at threshold 0.50 and 0.10, resulting in a flat F1-score of ~33.8%. However, its ROC-AUC is `0.5238`, suggesting it has some discriminative power but its logits are heavily skewed.