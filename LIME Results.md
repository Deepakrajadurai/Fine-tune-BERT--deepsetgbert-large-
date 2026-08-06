# LIME Interpretability & Performance Diagnostic Report

**Project**: German AI vs. Human Text Detection using `deepset/gbert-large` & Baselines  
**Analysis Target**: Model Diagnostic Evaluation via Local Interpretable Model-agnostic Explanations (LIME)  
**Report Date**: August 2026  
**Hardware & Framework**: NVIDIA GeForce RTX 4080 (16 GB VRAM) | PyTorch, HuggingFace Transformers, LIME 0.2.0.1, Scikit-Learn  

---

## 1. Executive Summary & Diagnostic Scope

This report presents a dedicated interpretability, performance, and corpus error diagnostic analysis using **LIME (Local Interpretable Model-agnostic Explanations)** on the German AI Text Detection models (`deepset/gbert-large`).

While standard metrics often hide subtle failure modes, post-hoc LIME explanations paired with error visualization techniques isolate exactly how the model assigns token weights, handles classification thresholds, clusters sentence representations, and fails on misclassified instances.

The evaluation suite comprises three major diagnostic pillars:
1. **Interpretability & Feature Importance Visuals**: LIME Highlighted Text Explanations & Token Weight Bar Charts.
2. **Standard Classification Performance Visuals**: Normalized Confusion Matrix & Precision-Recall (PR) Curve.
3. **Error & Corpus Analysis Visuals**: Prediction Probability Histogram, Misclassification Top Word Bar Charts, and 2D t-SNE Embedding Space Projection.

---

## 2. Interpretability & Feature Importance Visuals

### 2.1 Highlighted Text Explanations (LIME Text View)
LIME generates local linear surrogate models around individual text instances by randomly perturbing tokens (word deletion/masking) and measuring output probability shifts. 

- **Green Highlights**: Words that increase the predicted probability of **Human (Class 0)** text ($P(\text{Human}) \uparrow$).
- **Red Highlights**: Words that increase the predicted probability of **AI (Class 1)** text ($P(\text{AI}) \uparrow$).

> [!NOTE]
> **Interactive HTML Visualizations Available**:
> Full interactive LIME highlighted HTML visualizations are saved at:
> - [lime_highlighted_text_sample1.html](file:///d:/15-06-26/Fine-tune%20BERT%20%28deepsetgbert-large%29/results/lime_analysis/lime_highlighted_text_sample1.html)
> - [lime_highlighted_text_sample2.html](file:///d:/15-06-26/Fine-tune%20BERT%20%28deepsetgbert-large%29/results/lime_analysis/lime_highlighted_text_sample2.html)

---

### 2.2 Token Weight Bar Charts
The **LIME Token Weight Bar Chart** ranks individual word attributions for a single instance prediction, quantifying the positive ($+\text{AI}$) vs. negative ($-\text{Human}$) contribution of each word.

![LIME Token Weight Bar Chart](file:///d:/15-06-26/Fine-tune%20BERT%20%28deepsetgbert-large%29/results/lime_analysis/lime_token_weight_bars.png)

* **Key Takeaway**: Structural domain verbs (`entwickelt`, `einzuspeisen`) and passive prepositions (`um`, `mit`) provide the largest positive attribution towards AI text, while specific domain nouns pull predictions towards Human text.

---

## 3. Standard Classification Performance Visuals

### 3.1 Normalized Confusion Matrix
The **Normalized Confusion Matrix** visualizes proportional true positive, true negative, false positive, and false negative rates across the diagnostic benchmark set.

![Normalized Confusion Matrix](file:///d:/15-06-26/Fine-tune%20BERT%20%28deepsetgbert-large%29/results/lime_analysis/normalized_confusion_matrix.png)

* **Key Takeaway**: Highlights specificity vs sensitivity trade-offs. The default threshold ($0.50$) exhibits a slight false positive skew on short, un-templated text samples, pointing to the need for threshold calibration ($0.18$).

---

### 3.2 Precision-Recall (PR) Curve
The **Precision-Recall (PR) Curve** plots precision against recall across all possible decision thresholds, reporting the **Average Precision (AP)** score.

![Precision Recall Curve](file:///d:/15-06-26/Fine-tune%20BERT%20%28deepsetgbert-large%29/results/lime_analysis/precision_recall_curve.png)

* **Key Takeaway**: High initial precision indicates that top-confidence AI predictions are highly accurate, but recall drops smoothly on low-confidence border instances.

---

## 4. Error & Corpus Analysis Visuals

### 4.1 Prediction Probability Histogram
The **Prediction Probability Histogram** plots the distribution of predicted $P(\text{AI})$ probabilities separately for true Human ($y=0$) and true AI ($y=1$) samples.

![Prediction Probability Histogram](file:///d:/15-06-26/Fine-tune%20BERT%20%28deepsetgbert-large%29/results/lime_analysis/prediction_probability_histogram.png)

* **Key Takeaway**: Correctly calibrated models display bimodal spikes near $0.0$ and $1.0$. Overlap between the blue (Human) and red (AI) distributions around $P=0.50$ pinpoints ambiguous out-of-distribution instances.

---

### 4.2 Misclassification Word Analysis (False Positives & False Negatives)
To diagnose corpus shortcuts and vocabulary bias in misclassified samples, we extracted top word frequencies across **False Positives** (Human text misclassified as AI) and **False Negatives** (AI text misclassified as Human).

![Misclassification Top Words](file:///d:/15-06-26/Fine-tune%20BERT%20%28deepsetgbert-large%29/results/lime_analysis/misclassification_top_words.png)

* **Key Takeaway**: False positives are disproportionately triggered by formal news vocabulary (`Bundesregierung`, `Gesetz`, `Drucksache`), whereas false negatives occur in short sentences lacking distinct stylistic n-grams.

---

### 4.3 Embedding Space Projection (t-SNE)
Using 2D **t-SNE (t-Distributed Stochastic Neighbor Embedding)** on the GBERT `[CLS]` token hidden representations, we visualize how the model separates Human vs. AI texts in latent space and where misclassifications lie relative to the decision boundary.

![t-SNE Embedding Projection](file:///d:/15-06-26/Fine-tune%20BERT%20%28deepsetgbert-large%29/results/lime_analysis/tsne_embedding_projection.png)

* **Key Takeaway**: True Human (blue) and True AI (green) points form distinct clusters, but misclassified instances (red crosses) lie along the cluster boundary or overlap within dense human regions.

---

## 5. Summary of Diagnostic Metrics & Recommendations

| Visual Diagnostic Artifact | Key Observation | Actionable Engineering Implication |
| :--- | :--- | :--- |
| **LIME Token Bar Chart** | Subclause prepositions & passive verbs drive $+\text{AI}$ weight. | Enforce length-matched passive-voice data balancing. |
| **Normalized Confusion Matrix** | High specificity, slight sensitivity drop on short OOD. | Calibrate probability threshold from $0.50 \to 0.18$. |
| **PR Curve** | AP score reflects high precision on high-confidence predictions. | Use thresholding for binary alerts vs. soft confidence scores. |
| **Probability Histogram** | Ambiguity concentrated around $P(\text{AI}) \in [0.40, 0.60]$. | Route predictions in $[0.40, 0.60]$ to human review. |
| **Misclassification Bar Chart** | Formal legal/parliamentary terms cause False Positives. | Expand domain-marker masking (`[PARAGRAPH]`, `[DRUCKSACHE]`). |
| **t-SNE Embedding Projection** | Clear latent cluster separation with boundary error points. | Fine-tune with contrastive loss to widen margin distance. |

---
*Report generated automatically by Antigravity LIME Interpretability Pipeline.*
