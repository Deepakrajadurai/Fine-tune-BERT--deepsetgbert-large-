# GBERT-Large 1M Sentence Training Experiment Report
**Experiment Identifier**: `GBERT_LARGE_1M_SENTENCE_V1`  
**Model Base**: `deepset/gbert-large` (335M parameters)  
**Execution Date**: 2026-06-25  
**Hardware**: NVIDIA GeForce RTX 4080 (16 GB VRAM)

---

## 1. Dataset Characteristics & Splits

The dataset was constructed at the sentence level using `Data/Human_model_ready_dataset.csv` (2,074,797 rows) and `Data/ai_generated_sentences_500k.csv` (500,000 rows). 
Strict text normalization, formatting artifact removal, and duplicate checking were performed:
* **Deduplication**: Removed **21,766** exact-duplicate sentences from the AI dataset and **48** from the Human dataset.
* **Cleaning criteria**: German text only, minimum word count of **10** words, maximum length of **256** tokens.

### Final Balanced Split Distribution (80/10/10)
| Dataset Split | Human (0) | AI (1) | Total Samples |
| :--- | :--- | :--- | :--- |
| **Training Split** | 400,000 | 382,587 | **782,587** |
| **Validation Split** | 50,000 | 47,823 | **97,823** |
| **Test Split** | 50,000 | 47,824 | **97,824** |

*Dataset train/val/test splits SHA256 hashes:*
* `train_500k.csv`: `1ece8662e2e50ffdf0f04679942d5edd223297c077a2bb86ce741deaeaf970e8`

---

## 2. Training Configuration

The model was fine-tuned in the workspace using the virtual environment interpreter via:
```bash
python train.py \
  --train_csv Data/train_500k.csv \
  --val_csv Data/val_500k.csv \
  --ext_val_csv Data/val_500k.csv \
  --epochs 3 \
  --batch_size 16 \
  --lr 2e-5 \
  --output_dir models/full_model_500k
```

### Hyperparameters & System Telemetry
* **Effective Batch Size**: 16 (per-device)
* **Optimizer**: AdamW (weight decay = 0.01)
* **Learning Rate Schedule**: Linear decay with 10% Warmup Steps (14,673 steps)
* **Total Parameter Update Steps**: 146,736 steps
* **Training Runtime**: `4h 18m 20s` (15,500 seconds total)
* **Training Throughput**: ~151.5 samples/second
* **Peak GPU Memory Usage**: ~12.2 GB VRAM

---

## 3. Evaluation & Diagnostics

Evaluation was run on the balanced test split (Data/test_500k.csv) using evaluate.py at the default 0.50 decision threshold:

### Summary Performance Metrics
| Metric | In-Distribution Test Split | Unseen Holdout Validation Split |
| :--- | :--- | :--- |
| **Accuracy** | 51.11% | 51.11% |
| **Macro F1-Score** | 33.82% | 33.82% |
| **Macro Precision** | 25.56% | 25.56% |
| **Macro Recall** | 50.00% | 50.00% |
| **ROC-AUC** | **0.6829** | **0.6823** |

### Confusion Matrix (Test Split)
```
                  Predicted
              Human      AI
Actual Human  50,000      0
Actual AI     47,824      0
```

---

## 4. Technical Diagnosis: Model Collapse

1. **Constant Classification**: 
   The model predicted the `Human` class (0) for $100\%$ of all test and validation sentences. True Positives (AI predicted as AI) are $0$.
2. **Early Optimization vs. Catastrophic Collapse**:
   * During the first 500 training steps, the training loss dropped from `0.668` to a highly optimistic `0.086`.
   * However, as the learning rate peaked and parameter updates continued over a massive step size of `146,736` updates with a batch size of `16`, the weights experienced gradient over-saturation. The training loss rose back to a flat `0.35` and remained stagnant for the rest of the training.
3. **High ROC-AUC Indicator**:
   Despite the flat predictions at a $0.50$ threshold, the **ROC-AUC is 0.6829**. This confirms that the model's self-attention layers *did* learn features to order the classes (AI is still ranked higher than Human), but the logit values shifted completely below the classification boundary.

---

## 5. Mitigation & Stabilization Plan

To train a robust model on this large-scale dataset, the next run should implement the following stabilization techniques:

1. **Reduce Learning Rate**: Limit the learning rate to `2e-6` or `1e-6` (down from `2e-5`) to avoid destroying pre-trained language weights over hundreds of thousands of updates.
2. **Activate Gradient Accumulation**: Set `gradient_accumulation_steps=16` in HuggingFace `TrainingArguments` (effective batch size $256$). This reduces the parameter update steps from `146,736` down to `9,171` steps, yielding smoother, stabilized updates.
3. **Shorten Training Span**: Training for 1 epoch (or even 0.5 epochs) is more than sufficient.
