# Walkthrough - Baseline Training on Organic (Non-Templated) Dataset

We have successfully prepared the organic dataset splits (news and casual domains only, completely omitting the templated Bundestag speech and law domains) and retrained the baseline TF-IDF + Logistic Regression model.

## 1. Organic Dataset Construction

We created a new dataset builder script [prepare_organic_dataset.py](file:///e:/15-06-26/Fine-tune%20BERT%20%28deepsetgbert-large%29/prepare_organic_dataset.py) to load and split the following:
* **Human Data (News + Casual):**
  * 15,000 News paragraphs from GNAD (`gnad_articles.csv`).
  * 5,000 Casual texts from GermEval (`germeval2018.txt`).
* **AI Data (News + Casual - Organic Qwen-generated):**
  * 15,000 News paragraphs (`ai_generated_news.csv`).
  * 5,000 Casual texts (`ai_generated_casual.csv`).

### Split Summary (Balanced)
* **Train Set (`train_organic.csv`):** `26,778` rows (13,389 Human / 13,389 AI)
* **Val Set (`val_organic.csv`):** `3,348` rows (1,674 Human / 1,674 AI)
* **Test Set (`test_organic.csv`):** `3,348` rows (1,674 Human / 1,674 AI)
* **Cross-class exact duplicate matches:** `0`
* **Cross-split duplicate leakage:** `0`

---

## 2. Verification & Leakage Diagnostics

We ran `leakage_diagnostic.py` on `Data/train_organic.csv`:
* **Exact duplicates:** `0`
* **Train/test exact overlap:** `0`
* **Top Predictive Coefficients:**
  * **AI Class (LLM style markers):** `diese` (6.43), `hat` (6.01), `eine` (5.86), `wurde` (5.73), `neue` (5.41), `von` (5.36). Includes casual address tags: `hey` (4.43), `du` (4.32), `dir` (2.90), `dich` (2.72) from casual comments.
  * **Human Class (Journalistic reporting style):** Subjunctive reporting markers: `sei` (-4.91), `seien` (-3.10), `habe` (-3.66), `werde` (-2.44), `würden` (-2.35). Reporting verbs: `sagte` (-4.52), `sagt` (-3.98), `laut` (-3.72), `angaben` (-2.21).

> [!NOTE]
> This confirms that the model is no longer memorizing structural template phrases. Instead, it is learning genuine stylistic and grammatical differences (such as human journalists' heavy use of indirect speech subjunctive markers vs. LLM direct statements).

---

## 3. Retrained Baseline Performance

We retrained the **TF-IDF + Logistic Regression** model on `train_organic.csv` and evaluated it on the held-out `test_organic.csv` split.

### Metrics (on 3,348 test samples)
* **Accuracy:** `0.9588` (95.88%)
* **Macro F1:** `0.9588`
* **Macro Precision:** `0.9596`
* **Macro Recall:** `0.9588`
* **ROC-AUC:** `0.9937`

### Confusion Matrix
```
               Predicted Human    Predicted AI
Actual Human       1,641              33
Actual AI            105           1,569
```

> [!TIP]
> The model achieves a very high classification performance (95.88% accuracy, 0.9937 ROC-AUC), but is no longer overfitted to a trivial 100%. The smooth probability stats (min=0.0015, max=0.9981) confirm a realistic learning signal.

---

## 4. Conclusion & Takeaway

By training on organic, non-templated text, we have built a baseline that is realistic and generalizable. 
This balanced organic dataset (`train_organic.csv` / `val_organic.csv` / `test_organic.csv`) is now the ideal corpus for training the GBERT model to learn real stylistic patterns of AI text rather than exploiting template shortcuts.
