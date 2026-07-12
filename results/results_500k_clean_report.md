# GBERT-Large Cleaned 500k Sentence Training Experiment Report

**Experiment Identifier**: `GBERT_LARGE_500K_CLEAN_STABILIZED`  
**Model Base**: `deepset/gbert-large` (335M parameters)  
**Execution Date**: 2026-07-09  
**Hardware**: NVIDIA GeForce RTX 4080 (16 GB VRAM)  
**Precision**: Bfloat16 (`bf16`) Mixed Precision  

---

## 1. Dataset Characteristics & Splits
The dataset was constructed by cleaning and balance-sampling the massive 500k splits (`Data/Human_model_ready_dataset.csv` and `Data/ai_generated_sentences_500k.csv`).
* **Deduplication**: Exact duplicates and cross-split leaks were strictly removed.
* **Cleaning**: Specific LLM political boilerplates, party references, disclaimers, and dates were stripped out to limit shortcut memorization.

### Final Balanced Split Distribution (80/10/10)
* **Training Split (`train_500k_clean.csv`)**: 663,840 rows (331,920 Human / 331,920 AI)
* **Validation Split (`val_500k_clean.csv`)**: 82,980 rows (41,490 Human / 41,490 AI)
* **Test Split (`test_500k_clean.csv`)**: 82,982 rows (41,491 Human / 41,491 AI)

---

## 2. Training Configuration
The model was fine-tuned using:
```bash
python train.py \
  --train_csv Data/train_500k_clean.csv \
  --val_csv Data/val_500k_clean.csv \
  --ext_val_csv Data/test_500k_clean.csv \
  --epochs 1 \
  --batch_size 16 \
  --gradient_accumulation_steps 16 \
  --lr 5e-6 \
  --eval_steps 500 \
  --log_every_n_steps 100 \
  --output_dir models/full_model_500k_clean \
  --bf16
```

### Hyperparameters & System Telemetry
* **Effective Batch Size**: 256 (16 batch size * 16 gradient accumulation steps)
* **Optimizer**: AdamW (weight decay = 0.01)
* **Learning Rate Schedule**: Linear decay with 6% Warmup Steps
* **Total Update Steps**: 2,594 planned steps
* **Training Runtime**: `3h 22m` (stopped early at step 2,000)
* **Peak GPU Memory Usage**: ~12.2 GB VRAM

---

## 3. Evaluation Logs & Metrics
The evaluation ran every 500 update steps. Due to the high stability of the gradient accumulation updates and optimized learning rate, the model trained successfully without collapse.

| Step | Epoch | Training Loss (avg) | Validation Loss | Validation Accuracy | Validation Macro F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **500** | 0.19 | 0.0382 | 6.161e-06 | 100.0% | 1.0 |
| **1000** | 0.39 | 0.0031 | 6.210e-06 | 100.0% | 1.0 |
| **1500** | 0.58 | 0.0009 | 6.250e-06 | 100.0% | 1.0 |
| **2000** | 0.77 | 0.0001 | 6.302e-06 | 100.0% | 1.0 |

* **Early Stopping**: The trainer detected that the validation Macro F1 score had reached its maximum of `1.0` at step 500 and showed no further change. Early stopping was triggered at step 2,000, terminating training early to prevent unnecessary overhead.

---

## 4. Final Evaluation Results on test_500k_clean.csv
The best checkpoint (`models/full_model_500k_clean`) was evaluated on the held-out test split of 82,982 samples.

| Metric | Test Value |
| :--- | :--- |
| **Accuracy** | **100.0%** (82,982 / 82,982) |
| **Macro F1** | **1.0** |
| **Test Loss** | `6.30e-06` |
| **Predicted Distribution** | **50.0% Human** (41,491) / **50.0% AI** (41,491) |

---

## 5. Summary & Key Takeaways
1. **Model Collapse Remedied:** By increasing the effective batch size to `256` and decreasing the learning rate to `5e-6`, we completely resolved the representation collapse (where the model previously predicted 100% human). The predictions are perfectly balanced.
2. **Linguistic template shortcuts:** The 100.0% test accuracy confirms that the synthetic template generation model `ai_generated_sentences_500k.csv` contains highly predictable syntactic patterns that GBERT easily memorizes.
3. **Deployment Recommendation:** For general German AI text detection, use the `models/organic_gbert_large` checkpoint (trained on free-form news and casual datasets), as it generalized to writing style. The `models/full_model_500k_clean` checkpoint is highly robust specifically for identifying templates and structured outputs.
