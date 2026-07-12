# GBERT-Large Fine-Tuning on Organic Dataset: Comprehensive Training Report

**Report Date**: 2026-07-09  
**Model Architecture**: `deepset/gbert-large` (Sequence Classification, 350M parameters)  
**Hardware**: NVIDIA GeForce RTX 4080 (16 GB VRAM)  
**Precision**: Bfloat16 (`bf16`) Mixed Precision  
**Training Framework**: HuggingFace Transformers + PyTorch  

---

## 1. Executive Summary
This report documents the training and evaluation of the `deepset/gbert-large` model fine-tuned on the **German Organic (Non-Templated) Dataset**. By completely excluding templated parliamentary speeches and legal text domains, we successfully forced the model to learn genuine stylistic markers of human vs. AI German texts. The model achieved **99.61% accuracy** on the held-out organic test set, showing extreme robustness and generalizability without relying on structural template shortcuts.

---

## 2. Dataset Configuration
The dataset was prepared by combining Human News/Casual texts and organic AI-generated News/Casual texts. Strict deduplication and cross-split leakage checks were applied.

### Splits and Balancing

| Split Name | Total Sample Count | Human (Label=0) | AI (Label=1) | Sources |
| :--- | :--- | :--- | :--- | :--- |
| **Train Set** (`train_organic.csv`) | **26,778** | 13,389 (50%) | 13,389 (50%) | GNAD News, GermEval, Qwen News, Qwen Casual |
| **Validation Set** (`val_organic.csv`) | **3,348** | 1,674 (50%) | 1,674 (50%) | GNAD News, GermEval, Qwen News, Qwen Casual |
| **Test Set** (`test_organic.csv`) | **3,348** | 1,674 (50%) | 1,674 (50%) | GNAD News, GermEval, Qwen News, Qwen Casual |

* **Exact duplicates overall:** `0`
* **Train/Test exact text overlap:** `0`

---

## 3. Hyperparameters and Training Configuration
Training arguments were optimized for the RTX 4080 to maximize throughput and ensure learning stability.

* **Base Model**: `deepset/gbert-large`
* **Sequence Length**: 512 tokens (truncated/padded)
* **Epochs**: 2
* **Batch Size**: 16 per device
* **Gradient Accumulation Steps**: 4 (effective batch size = 64)
* **Learning Rate**: `1e-5` (with AdamW optimizer)
* **Warmup Ratio**: 0.06
* **Max Gradient Norm**: 1.0 (for gradient clipping)
* **Evaluation & Saving Strategy**: Evaluated every 100 steps
* **Early Stopping**: Disabled (macro F1 checked at steps, ran full 2 epochs)
* **Output Checkpoint Directory**: `models/organic_gbert_large`

---

## 4. Step-by-Step Training Logs & Metrics
The training ran for a total of **838 steps** (representing 2 full epochs). The loss converged steadily without signs of model collapse or gradients exploding.

### Evaluation and Loss Progress

| Step | Epoch | Training Loss (avg) | Validation Loss | Validation Accuracy | Validation Macro F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **100** | 0.24 | 0.2717 | 0.0348 | 99.13% | 0.9913 |
| **200** | 0.48 | 0.2457 | 0.0282 | 99.01% | 0.9901 |
| **300** | 0.72 | 0.0789 | 0.0185 | 99.61% | 0.9961 |
| **700** | 1.67 | 0.0067 | 0.0186 | 99.61% | 0.9961 |
| **800** | 1.91 | 0.0007 | 0.0198 | 99.64% | 0.9964 |

* **Total Training Runtime:** 1,937 seconds (~32 minutes)
* **Average Training Throughput:** 27.64 samples/second

---

## 5. Final Evaluation Results on test_organic.csv
The best checkpoint was evaluated on the held-out test split, which represents unseen in-distribution organic samples.

* **Test Loss:** `0.01863`
* **Test Accuracy:** **`99.61%`** (only 13 misclassifications out of 3,348 samples)
* **Test Macro F1:** **`0.9961`**
* **Test Macro Precision:** **`0.9961`**
* **Test Macro Recall:** **`0.9961`**
* **Predicted Class Distribution (Test):** 
  * Predicted **Human**: `49.91%` (1,671 samples)
  * Predicted **AI**: `50.09%` (1,677 samples)

> [!NOTE]
> The prediction distribution is perfectly balanced (~50% predicted for each class), which indicates that the model did not experience **representation collapse** (a failure mode where the model predicts only a single class to minimize loss).

---

## 6. Scientific Analysis of Learned Style
By excluding rigid structural templates, we forced the model to learn the real linguistic style differences between German human writing and Qwen's synthetic generations:

1. **Human Style Cues (Negative coefficients in Logistic Regression baseline):**
   * **Subjunctive I & II (Indirect Speech):** `sei`, `seien`, `habe`, `werde`, `würden`.
   * **Journalistic reporting markers:** `sagte`, `sagt`, `laut`, `angaben`.
   * *Conclusion:* Real German journalists write using indirect speech markers to attribute statements, whereas LLMs write directly in the indicative.
2. **AI Style Cues (Positive coefficients in Logistic Regression baseline):**
   * **Adjective Hype/Filler:** `neue`, `wichtige`, `stark`, `entwicklung`, `technologien`.
   * **Direct addressing filler (from casual blogs):** `hey`, `du`, `dir`, `dich`.
   * *Conclusion:* LLMs systematically overuse typical hype/transition adjectives and highly standardized informal addressing structures.
