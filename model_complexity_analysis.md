# G-BERT Model Complexity, Bias & Variance Analysis

This document provides a rigorous analysis of the model complexity, fitting behavior (overfitting vs. underfitting), bias-variance trade-offs, and underlying causal mechanisms for all models trained across this project.

---

## 1. Conceptual Framework: The Bias-Variance Tradeoff

In machine learning and deep learning, **Bias** and **Variance** represent two sources of error that define a model's ability to generalize to unseen datasets:
*   **Bias (Underfitting)**: Error caused by erroneous, overly simple assumptions in the model. High bias leads to **underfitting**—the model fails to capture the true underlying patterns in both training and test sets.
*   **Variance (Overfitting)**: Error caused by extreme sensitivity to small fluctuations in the training set. High variance leads to **overfitting**—the model memorizes noise, layout artifacts, or specific sentence templates from the training set, causing in-distribution performance to be near-perfect, while failing on unseen out-of-distribution (OOD) sets.

The chart below shows where the key models from the G-BERT suite sit along this tradeoff spectrum:

![Bias-Variance Tradeoff Chart](results/images/bias_variance_tradeoff.png)

---

## 2. Per-Model Complexity & Error Decomposition

Below is a detailed breakdown of all models trained in this suite, organized from underfitting (collapsed) to overfitting (memorizing), and ending with optimal generalization.

### Group A: Underfitting & Diverged Models (High Bias, Low Variance)

#### 1. `full_model_500k` (Massive v3 Split)
*   **Fitting Status**: ❌ **Severe Underfitting (Total Gradient Collapse)**
*   **Metrics**: In-Distribution Macro F1: **33.33%** | Validation Accuracy: **51.11%** | OOD Macro F1: **N/A**
*   **Bias / Variance Profile**:
    *   **Bias**: 🔴 **Extremely High**. The model assumed all texts belong to a single class (Human) and failed to learn any representation boundaries.
    *   **Variance**: 🟢 **Zero**. The prediction distribution is static and insensitive to input variation.
*   **Causes of Collapse**:
    *   **Learning Rate Mismatch**: Using `lr = 2e-5` on a massive 782k dataset without gradient accumulation led to massive, fluctuating gradients that saturated the neural network weights in early epochs.
    *   **Absence of Warmup & Decay**: The learning rate decayed to absolute zero too quickly, locking model weights in their collapsed state.

---

### Group B: In-Distribution Overfit Models (Low Bias In-Dist, High Variance OOD)

#### 2. `v5_best_model` (v5 Baseline)
*   **Fitting Status**: ⚠️ **Severe Generalization Overfitting (Template Memorization)**
*   **Metrics**: In-Distribution Macro F1: **100.00%** | OOD Macro F1: **35.44%** (Recall on AI class dropped to **1.6%**)
*   **Bias / Variance Profile**:
    *   **Bias**: 🟢 **Very Low (In-Distribution)**. The model captured the training target perfectly.
    *   **Variance**: 🔴 **Extremely High**. The model is highly sensitive to the presence of training templates. When evaluated on the OOD dataset where these templates are absent, it predicted almost all texts as Human.
*   **Causes**:
    *   **Template Collapse**: The synthetic AI class was generated using exactly 568 repeating sentence templates. The 335M-parameter G-BERT model found it mathematically easier to memorize these specific word combinations than to learn stylistic signatures.

#### 3. `legal_model` (v4 Legal Domain)
*   **Fitting Status**: ⚠️ **Domain Overfitting / Under-generalizing**
*   **Metrics**: In-Distribution Macro F1: **98.40%** | OOD Macro F1: **58.30%**
*   **Bias / Variance Profile**:
    *   **Bias**: 🟢 **Low (In-Distribution)**.
    *   **Variance**: 🟡 **Moderately High**. The model memorized legislative sentence structures and parliamentary vocabulary unique to the Bundestag/Bundesrat, leading to higher error rates on news and casual prose.
*   **Causes**:
    *   **Narrow Training Domain**: Fine-tuned exclusively on a highly structured legal corpus (n = 17,380), causing representation shift on non-legal texts.

#### 4. `full_model_500k_clean` (Stabilized 500k Split)
*   **Fitting Status**: ⚠️ **In-Distribution Overfitting (Template Memorization)**
*   **Metrics**: In-Distribution Macro F1: **100.00%** | OOD Macro F1: **N/A**
*   **Bias / Variance Profile**:
    *   **Bias**: 🟢 **Very Low (In-Distribution)**.
    *   **Variance**: 🔴 **High**. While hyperparameter optimization stabilized the training, the model quickly memorized the remaining templates by step 500.
*   **Causes**:
    *   **Scale Without Stripping Templates**: Stabilizing training hyperparameters allowed the model to converge perfectly, but because templates were not removed, the network still fell victim to template memorization.

---

### Group C: Moderately Generalizing Baselines (Balanced Bias / Variance with Shortcuts)

#### 5. `best_model` / `full_model` (v1, 57k) & `model_100k` (v2, 70k)
*   **Fitting Status**: 🟡 **Moderate Generalization (Partially Overfit)**
*   **Metrics**: In-Distribution Macro F1: **99.77%** / **99.05%** | OOD Macro F1: **65.81%** / **61.20%**
*   **Bias / Variance Profile**:
    *   **Bias**: 🟢 **Low**.
    *   **Variance**: 🟡 **Moderate**.
*   **Causes**:
    *   These models retained formatting and length shortcuts. The 65.81% OOD performance was achieved because the model utilized the text length shortcut present in the OOD benchmark (Human average length: 19.1 words, AI: 9.6 words).

---

### Group D: Optimal Generalizing Models (Optimized Bias & Variance)

#### 6. `v5_best_model_clean` (Primary Recommendation)
*   **Fitting Status**: 🏆 **Optimal Fit (Robust Stylistic Generalization)**
*   **Metrics**: In-Distribution Macro F1: **99.99%** | OOD Macro F1: **65.81%** (Calibrated threshold: `0.18`)
*   **Bias / Variance Profile**:
    *   **Bias**: 🟢 **Low**. Captures the stylistic features of both human and template-free AI writing.
    *   **Variance**: 🟢 **Low**. Insensitive to layout changes, template presence, or document length differences.
*   **Causal Factors**:
    *   **Double-Layered Cleaning**: Staged formatting stripping and removing the 568 sentence templates forced the model to learn grammatical markers instead of layout/template shortcuts.
    *   **Length-Stratified Matching**: Downsampling classes to match word length bins prevented the model from using length as a shortcut.

#### 7. `organic_gbert_large` (General Prose Focus)
*   **Fitting Status**: 🏆 **Optimal Fit for Non-Legislative German**
*   **Metrics**: In-Distribution Macro F1: **99.61%** | OOD Macro F1: **N/A** (OOD dataset is legislative)
*   **Bias / Variance Profile**:
    *   **Bias**: 🟢 **Low**.
    *   **Variance**: 🟢 **Low**.
*   **Causes**:
    *   Trained on news/casual prose splits where political parliamentary speeches were excluded, preventing domain bias.

#### 8. `XGBoost Classifier` (Explainable Baseline)
*   **Fitting Status**: 🏆 **High Explainability, Stable Fit**
*   **Metrics**: In-Distribution Macro F1: **99.99%** | OOD Macro F1: **N/A**
*   **Bias / Variance Profile**:
    *   **Bias**: 🟡 **Moderate**. A feature-engineered model lacks the deep contextual understanding of G-BERT's attention layers.
    *   **Variance**: 🟢 **Low**. Built on character n-grams and dense stylometric features (entropy, type-token ratio), making it highly robust against vocabulary shifts.

---

## 3. Training & Validation Loss Curves

The actual training and validation loss profiles for the three primary fitting behaviors are illustrated below:

![G-BERT Model Learning Curves](results/images/learning_curves.png)

### Metric Comparison Table

| Model Identifier | Training Loss (Final) | Validation Loss (Final) | In-Distribution F1 | Out-of-Distribution F1 | Fitting Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `full_model_500k` | 0.3393 | 1.8905 | 33.33% | N/A | **Underfitting / Diverged** |
| `v5_best_model` | ~0.0000 | ~0.0000 | 100.00% | 35.44% | **In-Distribution Overfitting** |
| `v5_best_model_clean` | 0.0001 | 0.0002 | 99.99% | 65.81% | **Optimal Fit** |

---

## 4. Key Takeaways & Mitigation Strategies

1.  **Generalization Gap**: The massive gap between in-distribution F1 (100.00%) and OOD F1 (~35%) in uncleaned models is a hallmark of high variance due to dataset artifacts.
2.  **Brevity Bias**: The primary limitation of `v5_best_model_clean` is on very short text blocks (<30 words), where OOD F1 drops to ~65%. This is a fundamental limitation of stylistic classifiers: short texts lack sufficient grammatical and stylometric signals for reliable feature mapping.
3.  **Regularization via Data Engineering**: Traditional model regularization (dropout, weight decay) was insufficient to prevent template memorization. True generalization was only achieved through data engineering (template removal and length-stratification).
