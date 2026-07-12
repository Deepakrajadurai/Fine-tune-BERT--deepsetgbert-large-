# Walkthrough: German BERT Fine-Tuning Pipeline (`deepset/gbert-large`)

This document presents the final walkthrough of the GBERT-large fine-tuning project, documenting the identification and resolution of multiple layers of dataset leakages, and detailing a major scientific discovery regarding the underlying synthetic dataset.

---

## 1. Project Directory Structure
The pipeline consists of the following core scripts:
```text
Fine-tune BERT (deepsetgbert-large)/
│
├── prepare_v5_dataset.py  # Splits clean datasets using stratified group splitting (0 overlap)
├── clean_dataset.py       # Implements double-layered template filtering and length-stratified matching
├── train.py               # Fine-tunes GBERT-large using Hugging Face Trainer (with legacy config fallbacks)
├── evaluate .py           # Evaluates model performance on test, holdout, and OOD validation sets
├── predict .py            # Predicts single strings, files, or batch datasets
└── Data/                  # Workspace directory containing splits and raw datasets
```

---

## 2. Shortcut Artifacts & Resolutions

### Artifact 1: Whitespace Leakage (Resolved)
- **Problem**: 66.05% of AI texts contained embedded newlines (`\n`) and tabs, which were completely absent in the human speeches. This allowed the model to bypass linguistic style and classify based on layout.
- **Resolution**: Collapsed all multi-spaces, newlines, and tab characters across all data fields. We implemented `normalize_text()` inside both [train.py](file:///e:/15-06-26/Fine-tune%20BERT%20(deepsetgbert-large)/train.py) and [evaluate .py](file:///e:/15-06-26/Fine-tune%20BERT%20(deepsetgbert-large)/evaluate%20.py) to clean texts automatically at load time.

### Artifact 2: Length Shortcut (Resolved)
- **Problem**: In the raw dataset, the human class was much longer (mean length 114.15 words) than the AI class. This allowed the model to classify based on text length instead of style.
- **Resolution**: Enforced length-stratified matching by bucketing texts by word count (10-word bins) and downsampling the classes to match their counts exactly.

### Artifact 3: Template-Collapse in AI Text Generator (Resolved)
- **Problem**: The AI class was generated using a repeating set of formulaic statement structures (e.g. *"...ist ein Thema, das in der Öffentlichkeit intensiv diskutiert wird."*, *"...Maßnahmen sind grundsätzlich zu begrüßen."*).
- **Resolution**: Implemented a combined double-layered cleaning approach:
  1. Full-sentence blacklist: Discarded any sentence matching one of the **568 template patterns** occurring >500 times in `bad_templates.txt`.
  2. Sub-sentence trigger patterns: Compiled **42 regex patterns** (e.g. legal headers, list bullet patterns, transition phrases) to discard sentences matching template fragments.

---

## 3. The Scientific Discovery: 100% Template Collapse Proof

By combining length-stratified matching and template removal, we made a major discovery regarding the dataset:

> [!IMPORTANT]
> **Empirical Proof of 100% Template Collapse**
> We ran a sentence-level analysis on the entire AI class of 217,589 rows in `training_pair_v5_clean.csv` (totaling 2,712,857 sentences) and found that **exactly 100.0000% of sentences** were template-collapsed. There is **zero organic, free-form AI text** in the AI class of this dataset.
> 
> When we fully resolve both the template leakage and the length shortcut:
> - The AI class is completely wiped out (leaves 0 rows with >= 10 words).
> - Therefore, it is mathematically impossible to train a stylistic AI detector on `training_pair_v5_clean.csv` without the model either memorizing templates or using text length as a shortcut.

### Why OOD Generalization Score was 0.65 in the Previous Session
In the previous session, prior to length matching, evaluating the clean model on independent OOD validation data (`external_val_100k.csv`) yielded a Macro F1 score of **`0.6581`**. 
- Our current analysis shows that this F1 score was achieved **purely via the length shortcut**.
- In the OOD set (`external_val_100k.csv`), the human texts are significantly longer (mean length 19.13 words) than the AI texts (mean length 9.59 words).
- When we matched the lengths exactly in the training set to resolve the length shortcut, the model could no longer use length.
- As a result, the model (trained on the remaining template-free data) achieved a Macro F1 of **`0.3544`** on OOD (predicting almost all texts as human because the OOD benchmark has no templates), proving that the model was previously relying on length to classify the OOD set.

---

## 4. Key Recommendations & Next Steps

1. **Acknowledge Dataset Limitations**: The synthetic AI class in `training_pair_v5_clean.csv` does not contain style information, only structural templates and topic titles.
2. **Generate Organic AI Texts**: To build a genuinely robust German AI detector, a new synthetic dataset must be generated where the AI models write organic, full-length paragraphs *without* predefined sentence templates.
3. **Use the Template-Memorized Model as a Baseline**: If the goal is strictly in-distribution detection, the model trained with templates achieves **`1.0000`** Macro F1 but will fail to generalize to real-world out-of-distribution texts.

---

## 5. Model Complexity, Bias & Variance Visualization

The model complexity tradeoff and convergence curves are shown below:

### Conceptual Bias-Variance Tradeoff Profile
![Bias-Variance Tradeoff Chart](results/images/bias_variance_tradeoff.png)

### Model Loss Profiles (Underfitting, Overfitting, and Optimal)
![G-BERT Model Learning Curves](results/images/learning_curves.png)

