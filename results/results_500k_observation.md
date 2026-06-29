# GBERT-Large 500k Training Run Observation Report

## 1. Experiment Overview
* **Model Base**: `deepset/gbert-large` (335M parameters)
* **Dataset Size**: ~978,234 samples total (782,587 Train / 97,823 Val / 97,824 Test)
* **Training Settings**: 3 Epochs, per-device Batch Size = 16, Learning Rate = `2e-5`, Warmup Ratio = 10% (14,673 steps).
* **Hardware**: NVIDIA GeForce RTX 4080 (16 GB VRAM)
* **Execution Time**: 4h 18m 20s

---

## 2. Proof of Logs (Key Milestones)

Below is the summary of the training progression extracted from the training state:

| Epoch | Step | Train Loss (Avg) | Val Loss | Val Accuracy | Val Macro F1 | Learning Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0.01** | 500 | 0.0865 | — | — | — | 6.80e-7 (Warmup) |
| **0.02** | 1000 | 0.0082 | — | — | — | 1.36e-6 (Warmup) |
| **0.20** | 10000 | 0.0496 | — | — | — | 1.36e-5 |
| **1.00** | 48912 | 0.3541 (avg) | 1.0776 | 51.11% | 33.82% | 1.36e-5 |
| **2.00** | 97824 | 0.3421 (avg) | 1.5260 | 51.11% | 33.82% | 6.80e-6 |
| **3.00** | 146736 | 0.3392 (avg) | 1.8904 | 51.11% | 33.82% | 0.00 |

*Note: Full step-by-step telemetry has been exported to `gbert_500k_training_export.json` and `gbert_500k_training_logs.csv`.*

---

## 3. Evaluation Results on In-Distribution Test Set
Evaluation was performed at the default `0.50` decision threshold:

* **Test Accuracy**: 51.11%
* **Macro F1-Score**: 33.82%
* **Macro Precision**: 25.56%
* **Macro Recall**: 50.00%
* **ROC-AUC**: **0.6829**

### Confusion Matrix
```
                  Predicted
               Human      AI
Actual Human   50,000      0
Actual AI      47,824      0
```

---

## 4. Observations and Technical Diagnosis

### A. Catastrophic Model Collapse
The model predicted the majority class `Human` (0) for **100% of all validation and test sentences**. This resulted in an accuracy of exactly `51.11%` (corresponding to the ratio of Human samples in the split) and a recall of `50.00%` (100% for Human, 0% for AI), which is equivalent to random guessing or a majority-class classifier.

### B. High Training Volatility
1. **Early Overfitting/Saturating**: In the first 1,000 steps, the training loss dropped from `0.668` to `0.0082`, indicating that the model rapidly fit the initial batches.
2. **Gradient Saturation**: As the learning rate reached its peak (`2e-5`) and update steps continued, the large number of updates (`146,736` steps) caused the model's weights to experience gradient over-saturation. The average training loss rose back to a flat `0.34 - 0.35` and the validation loss skyrocketed to `1.89` by the end of training.

### C. The Significance of ROC-AUC = 0.6829
Despite the model predicting 100% `Human` at the default threshold, the **ROC-AUC of 0.6829** indicates that the model's underlying representations still successfully order the classes (i.e., AI samples generally receive higher probability scores than Human samples). 
However, due to the loss collapse, the logits shifted and scaled entirely below the $0.50$ decision boundary. If we calibrate the decision threshold (e.g., setting it to a much lower value), we would recover a much higher classification performance.

---

## 5. Concrete Recommendations for Stabilization
To prevent model collapse on large datasets (500k+), we recommend:

1. **Lower the Learning Rate**: Decrease the learning rate from `2e-5` to `2e-6` or `1e-6` to preserve pre-trained language weights over hundreds of thousands of updates.
2. **Increase Batch Size via Gradient Accumulation**: Use `gradient_accumulation_steps=16` (effective batch size = 256) to smooth gradients and reduce parameter updates from `146,736` down to `9,171` steps.
3. **Reduce Epochs**: Train for only `1` epoch (or even `0.5` epochs), which is more than sufficient for a dataset of this size.
