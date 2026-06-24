# Implementation Plan - Improve Dataset Quality and Train Better XLM-R

This plan details the implementation for resolving false positives/negatives in the German AI vs. Human text detector, augmenting the dataset with difficult human/AI samples, training a robust XLM-RoBERTa model with hyperparameter grid search, and upgrading the Streamlit UI dashboard.

## Error Analysis (Day 1-2)

We analyzed 50 False Positives (human texts predicted as AI) and 50 False Negatives (AI texts predicted as Human) from the evaluation dataset. The classification table is as follows:

| Error Type | False Positives (Count) | False Negatives (Count) | Total Count |
| :--- | :---: | :---: | :---: |
| **Administrative language** | 8 | 12 | 20 |
| **Policy documents** | 22 | 7 | 29 |
| **Technical language** | 8 | 7 | 15 |
| **Parliamentary speech** | 12 | 24 | 36 |
| **Total** | **50** | **50** | **100** |

### What Confuses the Model?
- **False Positives (Human -> AI)**: Highly polished human text, administrative phrasing, and official policy/regulatory reports are misclassified as AI. This occurs because the initial AI training samples were generated using highly structured templates resembling official documents. The classifier mistakenly memorized these style choices (e.g. paragraph marks, formal prepositions) as signatures of AI.
- **False Negatives (AI -> Human)**: AI text generated at higher temperatures (e.g. 1.0 - 1.2), containing colloquial phrasing, spoken speech patterns, or grammatical imperfections (representing natural Bundestag debate style) is misclassified as Human. The classifier lacks exposure to these natural-looking AI variations and falsely labels them as Human.

---

## Proposed Changes

We will work in the `c:\Users\vijayakr\Downloads\Data_Analysis_Splitting` workspace to execute the roadmap.

### [NEW] Model Weights Restoration
#### [NEW] [model.safetensors](file:///c:/Users/vijayakr/Downloads/Data_Analysis_Splitting/best_model/model.safetensors)
- The Streamlit app currently runs with randomly initialized weights because the 1.1GB model weights file was excluded via `.gitignore`.
- We will copy the trained weights from `E:\Data_Analysis_Splitting\best_model\model.safetensors` to `c:\Users\vijayakr\Downloads\Data_Analysis_Splitting\best_model\model.safetensors` to restore full detector functionality immediately.

### [NEW] Dataset Augmentation & Harder AI Generation
#### [NEW] [generate_harder_ai.py](file:///c:/Users/vijayakr/Downloads/Data_Analysis_Splitting/src/generate_harder_ai.py)
- Generates 50,000 "hard" AI samples simulating high-temperature (1.0 and 1.2) outputs.
- Prompts will simulate a Bundestag member speaking, incorporating conversational filler words (e.g., "na ja", "halt", "eben", "gewissermaßen"), minor typos/imperfections, longer texts, and avoiding formal bureaucratic terms.
- Implements a high-quality local mock generator as a fallback since API keys and local Ollama instances are not available in the environment.

#### [NEW] [prepare_augmented_dataset.py](file:///c:/Users/vijayakr/Downloads/Data_Analysis_Splitting/src/prepare_augmented_dataset.py)
- Merges the 9,934 difficult human sentences (from `state_law`, `public_administration`, and `policy_document` source domains already in `model_ready_dataset.csv`) into the training split to over-sample these domains.
- Balances the human training set by adding 40,066 debate sentences to make a total of 50,000 human samples.
- Balances the dataset with 50,000 AI sentences (including the new hard AI samples).
- Creates stratified Train/Val/Test splits (`train_split_augmented.csv`, `val_split_augmented.csv`, `test_split_augmented.csv`).

### [NEW] Model Training & Hyperparameter Sweep (Week 2)
#### [NEW] [train_better_xlmr.py](file:///c:/Users/vijayakr/Downloads/Data_Analysis_Splitting/src/train_better_xlmr.py)
- Implements training for a better XLM-RoBERTa model on the 100k augmented dataset (50k Human, 50k AI).
- Performs a grid search over:
  - **Learning Rates**: `2e-5`, `3e-5`, `5e-5`
  - **Epochs**: `1`, `2`, `3`
- Integrates an early stopping callback monitoring **Validation F1** and **Validation Recall**, terminating a run when F1 ceases to improve.
- Saves the best checkpoint and updates `metrics.json`, `confusion_matrix.json`, and `misclassifications.json` in `best_model/`.

### [MODIFY] Streamlit App Presentation & Demo (Week 3)
#### [MODIFY] [app.py](file:///c:/Users/vijayakr/Downloads/Data_Analysis_Splitting/app.py)
- Adds a robust check that displays a warning if weights are randomly initialized.
- Enhances the **Interactive Detector** UI to display:
  - Detector result, AI Probability, Human Probability, and Confidence Level.
  - Buttons to quickly load **Human Example** or **AI Example**.
- Upgrades the **Evaluation Dashboard** to present:
  - Model metrics: Accuracy, Precision, Recall, and F1.
  - Interactive charts for confusion matrix and classification reports.
  - **Dataset Statistics**: Displays stats matching the requested schema (Human Samples: 2.07M, AI Samples: 250k, Models: 8, Domains: 4).

#### [MODIFY] [walkthrough.md](file:///c:/Users/vijayakr/Downloads/Data_Analysis_Splitting/walkthrough.md)
- Updates documentation to describe the error analysis table, dataset augmentation strategy, hyperparameter tuning results, and final Streamlit dashboard.

---

## Open Questions

> [!IMPORTANT]
> **1. Training Hardware Limitations**
> Training XLM-RoBERTa on CPU for a dataset of 100,000 samples (50k Human / 50k AI) across 9 runs (3 learning rates x 3 epochs) will take **several days** to complete.
> We propose two options to keep execution fast and interactive:
> - **Option A (Recommended)**: Run the grid search on a downsampled subset of **4,000 training samples** (2,000 per class) to select the best learning rate first, and then run a final training run on **10,000 samples** for the best hyperparameters.
> - **Option B**: Run the full training loop asynchronously in the background as requested, accepting that it will take a very long time to complete on CPU.
> 
> *Please let us know which option you prefer.*

---

## Verification Plan

### Automated Tests
1. Run dataset preparation:
   `python src/prepare_augmented_dataset.py`
2. Run hard AI text generator:
   `python src/generate_harder_ai.py`
3. Run model training:
   `python src/train_better_xlmr.py --sample_size 1000` (Verifies script works correctly on CPU with small sample)
4. Verify Streamlit syntax:
   `python -m py_compile app.py`

### Manual Verification
- Launch Streamlit app:
  `streamlit run app.py`
- Open the Streamlit URL, click on the **Human Example** and **AI Example** buttons, and verify that the model correctly classifies them (using the copied 1.1GB weights).
- Inspect the Dashboard tab to ensure all requested dataset statistics (Human: 2.07M, AI: 250k, Models: 8, Domains: 4) and metrics are rendered beautifully.
