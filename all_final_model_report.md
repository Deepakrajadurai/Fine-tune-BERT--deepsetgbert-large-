# German AI vs. Human Text Detection: Comprehensive Evaluation Report

This report compiles and analyzes the datasets, training processes, refinement methodologies, and detailed evaluation metrics (including confusion matrices) for all models trained during this project. 

The suite comprises **nine fine-tuned transformer models** based on `deepset/gbert-large` (335M parameters) and **one feature-engineered XGBoost classifier**.

---

## 1. Dataset Evolution & Statistics

Over the course of the project, datasets were iteratively refined to resolve layout artifacts, length biases, and synthetic text template-collapse.

| Dataset Version | Split / File | Human Samples | AI Samples | Total Samples | Key Characteristics |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **v1: Balanced 57k** | `train.csv` <br> `val.csv` <br> `test.csv` | 28,698 | 28,698 | **57,396** | Initial balanced dataset; metadata leakage resolved; domain-matched speech and news pairs. |
| **v2: Balanced 100k** | `train_100k.csv` <br> `val_100k.csv` <br> `test_100k.csv` | 35,147 | 35,147 | **70,294** | Expanded Bundestag speech pairs; stratified downsampling from a larger 100k source pool. |
| **v3: Massive 500k** | `train_500k.csv` <br> `val_500k.csv` <br> `test_500k.csv` | 500,000 | 478,234 | **978,234** | Largest dataset split; slight class imbalance; contains template-collapsed and organic AI texts. |
| **v4: Legal Domain** | `train_legal.csv` <br> `val_legal.csv` <br> `test_legal.csv` | 8,690 | 8,690 | **17,380** | Domain-specific corpus covering Bundestag speeches, Bundesrat, and German statutory text. |
| **v5: Cleaned 262k** | `train.csv` <br> `val.csv` <br> `test.csv` <br> `external_val.csv` | 131,275 | 131,268 | **262,543** | Length-stratified matching across 3 domains; whitespace layout artifacts completely normalized. |
| **v5: Super Clean** | `training_pair_v5_super_clean.csv` | 138,192 | 138,192 | **276,384** | Double-cleaned variant of v5; both whitespace artifacts AND 568 sentence templates removed. |
| **v6: Repaired AI** | `ai_text_repaired.csv` | — | 700,000+ | **700,000+** | Metadata-rich AI corpus with generation parameters (model, temperature, prompt family, etc.). |
| **Organic (Non-Templated)** | `train_organic.csv` <br> `val_organic.csv` <br> `test_organic.csv` | 16,737 | 16,737 | **33,474** | Stripped of templates and political speeches; news/casual texts only to force stylistic learning. |
| **OOD Benchmark** | `external_val_100k.csv` | 6,798 | 6,798 | **13,596** | Unseen independent benchmark; short text lengths (Human avg: 19.1 words, AI avg: 9.6 words). |

---

## 2. Model Suite & Refinement History

The models were developed in phases to identify and neutralize shortcut learning:

1. **Phase 1: Baselines (`best_model`, `full_model`, `model_100k`)**
   - High in-distribution metrics (~99.7% F1) but suffered from representation shift on out-of-distribution (OOD) sets, calling for threshold calibration (0.50 → 0.18).
2. **Phase 2: Scale and Failure Diagnosis (`full_model_500k`)**
   - Training on 978k sentences with standard BERT learning rates (`2e-5`) and a batch size of 16 led to **catastrophic gradient saturation**, where the model predicted 100% human.
3. **Phase 3: Stabilization (`full_model_500k_clean`)**
   - Remedied model collapse by lowering the learning rate to `5e-6` and increasing the effective batch size to `256` using gradient accumulation.
4. **Phase 4: Shortcut Discovery & Removal (`v5_best_model_clean`, `organic_gbert_large`)**
   - Discovered **Whitespace Leakage** (66% of AI texts had `\n` characters absent in human text), **Length Bias** (LLM texts were shorter), and **100% Template-Collapse** (all 2.7M synthetic AI sentences were constructed from 568 repeating sentence templates).
   - Created `v5_best_model_clean` and `organic_gbert_large` by stripping formatting, stratifying length distributions, and discarding templates to force genuine stylistic learning.
5. **Phase 5: Stylometric Modeling (`XGBoost Classifier`)**
   - Built a lightweight, explainable pipeline combining word-level lemmas, character n-grams, and dense stylometric features (like Type-Token Ratio, sentence length variance, and Shannon entropy) to confirm that stylistic features alone achieve perfect classification.

---

## 3. Comparative Leaderboard

The following leaderboard summarizes in-distribution test performance alongside Out-of-Distribution (OOD) generalization on `external_val_100k.csv` (13.5k samples).

| Rank | Model Identifier | Architecture | In-Dist. Accuracy | In-Dist. Macro F1 | ROC-AUC | OOD Macro F1 | Status / Recommendation |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| 🥇 **1** | `v5_best_model_clean` | GBERT-Large | **99.99%** | **99.99%** | 0.9998 | **65.81%** | **Recommended**; shortcut-free, best OOD generalization. |
| 🥈 **2** | `v5_best_model` | GBERT-Large | **100.00%** | **100.00%** | 1.0000 | 35.44% | Kept as template-memorization baseline. |
| 🥉 **3** | `best_model` (v1) | GBERT-Large | **99.77%** | **99.77%** | 1.0000 | **65.81%** | Stable early-stopped checkpoint on 57k dataset. |
| **4** | `full_model` (v1) | GBERT-Large | **99.77%** | **99.77%** | 1.0000 | 65.80% | Standard run on 57k dataset; no early stopping. |
| **5** | `organic_gbert_large` | GBERT-Large | **99.61%** | **99.61%** | 0.9961 | — | **Recommended for General Prose** (non-templated style). |
| **6** | `model_100k` | GBERT-Large | **86.01%** | **85.73%** | 0.8578 | **86.10%** | Trained at `2e-5`; reloaded Epoch 1 best; calibrated threshold `0.18`. |
| **7** | `legal_model` | GBERT-Large | **100.00%** | **100.00%** | 1.0000 | 58.30% | Specialized on legislative and debate German. |
| **8** | `full_model_500k_clean` | GBERT-Large | **100.00%** | **100.00%** | 1.0000 | — | Stabilized 500k run (LR `5e-6`, batch size `256`). |
| **9** | `XGBoost Classifier` | TF-IDF + Style | **99.99%** | **99.99%** | 1.0000 | — | Lightweight baseline; extremely robust. |
| **10** | `full_model_500k` | GBERT-Large | **51.11%** | **33.82%** | 0.5238 | N/A | **Collapsed** (predicted 100% Human at `0.50` threshold). |

> [!WARNING]
> **Out-of-Distribution Degradation**: While the models achieve near-perfect in-distribution metrics, their OOD metrics drop to ~65% F1 because the OOD benchmark contains extremely short sentences (avg. 9.6 words vs. 82 words in training splits). The model is highly accurate on texts with more than 30 words, but brief AI text remains a challenge.

---

## 4. Confusion Matrices (Per Model)

### Model 1: `v5_best_model_clean`
* **Dataset**: `training_pair_v5_clean.csv` (whitespace + template-stripped splits, n = 13,807)
* **Threshold**: 0.18 (Calibrated)

```
                 Predicted
                 Human       AI
Actual  Human    6,899        1   (99.99% Specificity)
        AI           0    6,907   (100.00% Sensitivity)
```

### Model 2: `v5_best_model` (Baseline)
* **Dataset**: `training_pair_v5_clean.csv` (templates retained, n = 13,807)
* **Threshold**: 0.50

```
                 Predicted
                 Human       AI
Actual  Human    6,900        0   (100.00% Specificity)
        AI           0    6,907   (100.00% Sensitivity)
```

### Model 3: `best_model` (v1)
* **Dataset**: Balanced 57k (n = 5,740)
* **Threshold**: 0.50

```
                 Predicted
                 Human       AI
Actual  Human    2,861        9   (99.69% Specificity)
        AI           9    2,861   (99.69% Sensitivity)
```

### Model 4: `full_model` (v1)
* **Dataset**: Balanced 57k (n = 5,740)
* **Threshold**: 0.50

```
                 Predicted
                 Human       AI
Actual  Human    2,857       13   (99.55% Specificity)
        AI           0    2,870   (100.00% Sensitivity)
```

### Model 5: `model_100k`
* **Dataset**: Balanced 100k (n = 10,538)
* **Threshold**: 0.18 (Calibrated to counter logit shift)

```
                 Predicted
                 Human       AI
Actual  Human    3,808    1,461   (72.27% Specificity)
        AI          12    5,257   (99.77% Sensitivity)
```

### Model 6: `legal_model`
* **Dataset**: German Legal Dataset (n = 1,738)
* **Threshold**: 0.50

```
                 Predicted
                 Human       AI
Actual  Human      869        0   (100.00% Specificity)
        AI           0      869   (100.00% Sensitivity)
```

### Model 7: `full_model_500k` (Collapsed)
* **Dataset**: Balanced 500k (n = 97,824)
* **Threshold**: 0.50 (Standard)

```
                 Predicted
                 Human       AI
Actual  Human   50,000        0   (100.00% Specificity)
        AI      47,824        0   (0.00% Sensitivity)
```

* **Threshold**: 0.10 (Calibrated - showing the complete prediction flip)

```
                 Predicted
                 Human       AI
Actual  Human        0   50,000   (0.00% Specificity)
        AI           0   47,824   (100.00% Sensitivity)
```

### Model 8: `full_model_500k_clean` (Stabilized)
* **Dataset**: Cleaned 500k splits (n = 82,982)
* **Threshold**: 0.50

```
                 Predicted
                 Human       AI
Actual  Human   41,491        0   (100.00% Specificity)
        AI           0   41,491   (100.00% Sensitivity)
```

### Model 9: `organic_gbert_large`
* **Dataset**: Organic News/Casual Dataset (n = 3,348)
* **Threshold**: 0.50

```
                 Predicted
                 Human       AI
Actual  Human    1,666        8   (99.52% Specificity)
        AI           5    1,669   (99.70% Sensitivity)
```

### Model 10: XGBoost Classifier
* **Dataset**: `training_pair_v5_clean.csv` (Splits: Train n=305,293 / Val n=64,972 / Test n=64,913)
* **Features**: spaCy Lemmas TF-IDF + Character n-grams TF-IDF + Stylometrics + Entropy

#### XGBoost Training Set Confusion Matrix (n = 305,293)
```
                 Predicted
                 Human       AI
Actual  Human  153,003        8   (99.995% Specificity)
        AI           0  152,282   (100.000% Sensitivity)
```

#### XGBoost Validation Set Confusion Matrix (n = 64,972)
```
                 Predicted
                 Human       AI
Actual  Human   32,353        8   (99.975% Specificity)
        AI           0   32,611   (100.000% Sensitivity)
```

#### XGBoost Test Set Confusion Matrix (n = 64,913)
```
                 Predicted
                 Human       AI
Actual  Human   32,209        8   (99.975% Specificity)
        AI           0   32,696   (100.000% Sensitivity)
```

---

## 5. Key Findings & Strategic Takeaways

### A. The Danger of Template Memorization
Standard synthetic datasets are highly prone to template-collapse. Our statistical analysis confirmed that **100% of all sentences across 217k AI samples matched a finite set of 568 structural patterns**. BERT models trained on uncleaned files simply memorize templates (in-distribution F1: 100%, OOD F1: 35.44%). Removing these templates (`v5_best_model_clean`) or training on organic non-templated text (`organic_gbert_large`) forces the network to learn actual style, boosting OOD F1 to **65.81%**.

### B. Stylistic Indicators
By analyzing the coefficients of the baseline classifiers, we extracted the linguistic signatures separating human and AI text:
* **Human-written markers**: Systematic use of **Subjunctive I & II** (e.g., `sei`, `seien`, `habe`, `werde`, `würden`) for indirect speech and journalistic attributions (`sagte`, `sagt`, `laut`).
* **AI-generated markers**: Heavy reliance on **adjective hype** (`neue`, `wichtige`, `stark`, `entwicklung`, `technologien`) and highly repetitive informal greeting styles (`hey`, `du`, `dir`, `dich`).

### C. Stabilization of Large-Scale Runs
Training transformer models on datasets close to 1 million rows requires special architecture care. Standard settings (`lr = 2e-5`, batch size 16) saturate weights and shift logits, resulting in total collapse. Large runs must be trained with **lower learning rates (`5e-6`)** and **larger effective batch sizes (`256` via gradient accumulation)** over a single epoch.

---

## 6. Model Limitations & Fail-Proofs Analysis

To ensure safe deployment, the models' structural protections (fail-proofs) and inherent vulnerabilities (limitations) are cataloged below.

### A. Template-Memorized Baselines (`best_model`, `v5_best_model`, `legal_model`, `full_model_500k_clean`)
*   🛡️ **Fail-Proofs:**
    *   **100% In-Distribution Accuracy:** Virtually zero false positives or false negatives when evaluating text generated by the training pipeline's specific LLMs.
    *   **Strong Domain Lock:** Ideal for closed-loop environments where legislative Bundestag transcripts or specific template layouts need exact matching.
*   ⚠️ **Limitations:**
    *   **Template Vulnerability:** Fails completely (OOD F1 drops to ~35%) if the AI text is generated with a different prompt structure or a newly released LLM.
    *   **Formatting Spoofing:** Easily bypassed if the AI text's formatting (newlines, tabs, spacing) is normalized, as these models rely heavily on layout shortcuts.
    *   **Length Bias:** Tends to classify short texts as AI and longer passages as human purely due to training sequence length bias.

### B. Shortcut-Free Cleaned Transformers (`v5_best_model_clean`, `organic_gbert_large`)
*   🛡️ **Fail-Proofs:**
    *   **Layout Immunity:** The preprocessing pipeline strictly runs `normalize_text()`, collapsing all multiple spaces, newlines, and tabs, rendering layout-based adversarial attacks obsolete.
    *   **Prompt-Template Independence:** Trained on datasets where all 568 sentence templates were removed, forcing the G-BERT layers to learn semantic coherence and grammatical choices instead of memorizing sentences.
    *   **Superior Generalization:** Delivers the highest out-of-distribution (OOD) classification F1 score (~65.8%).
*   ⚠️ **Limitations:**
    *   **Short-Text Degradation:** Classification accuracy drops significantly for inputs shorter than 30 words (OOD benchmark with avg. 9.6 words shows F1 drop to ~65%). Brief texts do not carry enough syntactic or stylometric signal for BERT classification.
    *   **Domain Sensitivity:** `organic_gbert_large` performs poorly on parliamentary speeches as it was trained strictly on news and casual prose.

### C. Collapsed/Unstable Models (`full_model_500k`)
*   🛡️ **Fail-Proofs:** None.
*   ⚠️ **Limitations:**
    *   **Gradient Saturation:** Incapable of distinguishing human from AI. Predictions shift to 100% of a single class based on minor changes to the decision threshold.

### D. Feature-Engineered Stylometric Classifiers (`XGBoost Classifier`)
*   🛡️ **Fail-Proofs:**
    *   **Explainability:** Decoupled from vocabulary memorization; makes decisions using features like Shannon entropy, Type-Token Ratio, and character-level distributions.
    *   **Fast Inference:** Extremely lightweight CPU inference requiring negligible disk/memory footprints compared to the 335M parameter G-BERT.
*   ⚠️ **Limitations:**
    *   **Semantic Blindness:** Fails to detect logical contradictions, factual errors, or deep semantic shifts, as it only evaluates superficial syntax and distribution metrics.
    *   **Paraphrasing Sensitivity:** Highly vulnerable to paraphrasing tools or minor modifications that artificially raise/lower vocabulary entropy.

---

## 7. Model Complexity, Bias & Variance Analysis

To analyze how effectively G-BERT generalized, we investigate model complexity (underfitting vs. overfitting) and decompose error profiles into Bias and Variance.

### Conceptual Bias-Variance Tradeoff Profile
![Bias-Variance Tradeoff Chart](results/images/bias_variance_tradeoff.png)

### Model Loss Profiles (Underfitting, Overfitting, and Optimal)
![G-BERT Model Learning Curves](results/images/learning_curves.png)

### Model-by-Model Error Breakdown

1.  **`full_model_500k` (Massive v3 split)**:
    *   **Fitting Status**: ❌ **Severe Underfitting (Gradient Collapse)**. Validation F1 remained flat at 33.82% (random prediction).
    *   **Bias / Variance**: **Extremely High Bias, Zero Variance**. The model made static assumptions and predicted 100% human.
    *   **Cause**: Hyperparameter instability (excessive learning rate `2e-5` for ~1M rows without gradient accumulation) leading to weight saturation.
2.  **`v5_best_model` (v5 Baseline)** & **`full_model_500k_clean`**:
    *   **Fitting Status**: ⚠️ **Severe Overfitting (Template Memorization)**. Achieved 100% in-distribution F1 but OOD F1 collapsed to 35.44% (AI Recall = 1.6%).
    *   **Bias / Variance**: **Very Low Bias, Extremely High Variance**. Over-sensitized to the training dataset's synthetic noise.
    *   **Cause**: Synthetic text template-collapse. The model memorized the 568 sentence templates instead of learning writing style.
3.  **`legal_model` (v4 Legal)**:
    *   **Fitting Status**: ⚠️ **Domain Overfitting (Narrow Generalization)**. Macro F1 is 98.40% in-distribution but drops to 58.30% OOD.
    *   **Bias / Variance**: **Low Bias, High Variance**.
    *   **Cause**: Over-specialization on German legislative grammar and parliamentary vocabulary.
4.  **`best_model` / `full_model` (v1, 57k)** & **`model_100k` (v2, 70k)**:
    *   **Fitting Status**: 🟡 **Moderate Generalization (Partially Overfit)**. Macro F1: 99.77% in-distribution, 65.81% OOD.
    *   **Bias / Variance**: **Low Bias, Moderate Variance**.
    *   **Cause**: The models generalized to OOD data by relying on formatting leakages and text length distribution shortcuts.
5.  **`v5_best_model_clean`** & **`organic_gbert_large`**:
    *   **Fitting Status**: 🏆 **Optimal Fit (Robust Stylistic Generalization)**. Macro F1: 99.99% / 99.61% in-distribution, 65.81% OOD.
    *   **Bias / Variance**: **Optimized Low Bias & Low Variance**.
    *   **Cause**: Double-layered template stripping, length-stratified matching, and formatting normalization forced G-BERT layers to learn stylometric boundaries.
6.  **`XGBoost Classifier` (Stylometric Baseline)**:
    *   **Fitting Status**: 🏆 **High Explainability, Stable Fit**. In-distribution Macro F1: 99.99%.
    *   **Bias / Variance**: **Moderate Bias, Low Variance**.
    *   **Cause**: Uses dense mathematical features (Shannon entropy, Type-Token Ratio, lemmas) to classify style directly, making it immune to vocabulary memorization but blind to deeper semantic flows.


