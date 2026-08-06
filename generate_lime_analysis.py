import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
import time
import re
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    f1_score
)
from sklearn.manifold import TSNE
from lime.lime_text import LimeTextExplainer

# ---------------------------------------------------------------------------
# Setup Output Directory
# ---------------------------------------------------------------------------
RESULTS_DIR = Path("results/lime_analysis")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {DEVICE}")

# ---------------------------------------------------------------------------
# 1. Load Model & Tokenizer
# ---------------------------------------------------------------------------
MODEL_PATH = "models/v5_best_model_clean"
if not Path(MODEL_PATH).exists():
    MODEL_PATH = "models/best_model"

print(f"[INFO] Loading model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

def predict_proba(texts):
    if isinstance(texts, str):
        texts = [texts]
    elif isinstance(texts, (np.ndarray, list, tuple)):
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        texts = [str(t) for t in texts]
    else:
        texts = [str(texts)]
    
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()
    return probs

def extract_cls_embeddings(texts):
    if isinstance(texts, str):
        texts = [texts]
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        if hasattr(model, "bert"):
            outputs = model.bert(**inputs)
            cls_embeds = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        else:
            outputs = model(**inputs, output_hidden_states=True)
            cls_embeds = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
    return cls_embeds

# ---------------------------------------------------------------------------
# 2. Load Evaluation Dataset
# ---------------------------------------------------------------------------
data_file = Path("Data/test_cleaned.csv")
if not data_file.exists():
    data_file = Path("Data/test_100k.csv")
if not data_file.exists():
    data_file = Path("Data/test.csv")

print(f"[INFO] Loading test data from {data_file}...")
df = pd.read_csv(data_file, nrows=300).dropna()
text_col = "text" if "text" in df.columns else df.columns[0]
label_col = "label" if "label" in df.columns else df.columns[1]

texts = df[text_col].astype(str).tolist()
y_true = df[label_col].astype(int).values

# ---------------------------------------------------------------------------
# 3. Batch Evaluation & Predictions
# ---------------------------------------------------------------------------
print("[INFO] Computing predictions and probabilities...")
probs_all = []
batch_size = 32
for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size]
    p = predict_proba(batch_texts)
    probs_all.append(p)

probs_all = np.vstack(probs_all)
y_prob_ai = probs_all[:, 1]
y_pred = (y_prob_ai >= 0.5).astype(int)

acc = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro")
print(f"[INFO] Model Evaluation -> Accuracy: {acc*100:.2f}%, Macro F1: {macro_f1*100:.2f}%")

# ---------------------------------------------------------------------------
# 4. Standard Classification Performance Visuals
# ---------------------------------------------------------------------------
# Plot 1: Normalized Confusion Matrix
print("[INFO] Generating Normalized Confusion Matrix...")
cm = confusion_matrix(y_true, y_pred, normalize="true")
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues", xticklabels=["Human (0)", "AI (1)"], yticklabels=["Human (0)", "AI (1)"])
plt.title("Normalized Confusion Matrix (LIME Diagnostic Bench)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "normalized_confusion_matrix.png", dpi=200)
plt.close()

# Plot 2: Precision-Recall Curve
print("[INFO] Generating Precision-Recall Curve...")
precision, recall, thresholds = precision_recall_curve(y_true, y_prob_ai)
ap_score = average_precision_score(y_true, y_prob_ai)

plt.figure(figsize=(7, 5))
plt.plot(recall, precision, color="#2ecc71", lw=2, label=f"PR Curve (AP = {ap_score:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall (PR) Curve — AI Detection")
plt.legend(loc="lower left")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "precision_recall_curve.png", dpi=200)
plt.close()

# ---------------------------------------------------------------------------
# 5. Error & Corpus Analysis Visuals
# ---------------------------------------------------------------------------
# Plot 3: Prediction Probability Histogram
print("[INFO] Generating Prediction Probability Histogram...")
plt.figure(figsize=(8, 5))
plt.hist(y_prob_ai[y_true == 0], bins=30, alpha=0.6, color="#3498db", label="True Human (y=0)", density=True)
plt.hist(y_prob_ai[y_true == 1], bins=30, alpha=0.6, color="#e74c3c", label="True AI (y=1)", density=True)
plt.xlabel("Predicted P(AI)")
plt.ylabel("Density")
plt.title("Prediction Probability Histogram P(AI) Distribution")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "prediction_probability_histogram.png", dpi=200)
plt.close()

# Plot 4: Misclassification Analysis & Top Words
print("[INFO] Analyzing Misclassifications...")
fp_indices = np.where((y_true == 0) & (y_pred == 1))[0]
fn_indices = np.where((y_true == 1) & (y_pred == 0))[0]

def get_top_words(text_list, top_n=10):
    words = []
    for t in text_list:
        tokens = re.findall(r'\b[A-Za-zÄöüß1-9]{3,}\b', t.lower())
        words.extend(tokens)
    return Counter(words).most_common(top_n)

fp_words = get_top_words([texts[i] for i in fp_indices]) if len(fp_indices) > 0 else [("no_fp", 1)]
fn_words = get_top_words([texts[i] for i in fn_indices]) if len(fn_indices) > 0 else [("no_fn", 1)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
w_fp, c_fp = zip(*fp_words)
ax1.barh(w_fp[::-1], c_fp[::-1], color="#e74c3c")
ax1.set_title(f"Top Words in False Positives (n={len(fp_indices)})")
ax1.set_xlabel("Frequency")

w_fn, c_fn = zip(*fn_words)
ax2.barh(w_fn[::-1], c_fn[::-1], color="#3498db")
ax2.set_title(f"Top Words in False Negatives (n={len(fn_indices)})")
ax2.set_xlabel("Frequency")

plt.tight_layout()
plt.savefig(RESULTS_DIR / "misclassification_top_words.png", dpi=200)
plt.close()

# Plot 5: Embedding Space Projection (t-SNE)
print("[INFO] Generating t-SNE Embedding Projection...")
embeds_all = []
for i in range(0, len(texts), batch_size):
    b_embeds = extract_cls_embeddings(texts[i:i+batch_size])
    embeds_all.append(b_embeds)
embeds_all = np.vstack(embeds_all)

tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(texts)-1))
embeds_2d = tsne.fit_transform(embeds_all)

plt.figure(figsize=(9, 7))
correct_mask = (y_true == y_pred)
plt.scatter(embeds_2d[correct_mask & (y_true == 0), 0], embeds_2d[correct_mask & (y_true == 0), 1], c="#3498db", label="Human (Correct)", alpha=0.6, s=30)
plt.scatter(embeds_2d[correct_mask & (y_true == 1), 0], embeds_2d[correct_mask & (y_true == 1), 1], c="#2ecc71", label="AI (Correct)", alpha=0.6, s=30)
plt.scatter(embeds_2d[~correct_mask, 0], embeds_2d[~correct_mask, 1], c="#e74c3c", label="Misclassified", marker="x", s=80, lw=2)

plt.title("t-SNE Projection of BERT CLS Embeddings")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "tsne_embedding_projection.png", dpi=200)
plt.close()

# ---------------------------------------------------------------------------
# 6. LIME Explanations & Visuals
# ---------------------------------------------------------------------------
print("[INFO] Generating LIME Highlighted Explanations & Token Bar Charts...")
explainer = LimeTextExplainer(class_names=["Human", "AI"], random_state=42)

# Select 2 sample instances for detailed LIME explanations
sample_idx1 = 0
sample_idx2 = 1 if len(texts) > 1 else 0

text_sample1 = texts[sample_idx1][:256]
text_sample2 = texts[sample_idx2][:256]

exp1 = explainer.explain_instance(text_sample1, predict_proba, num_features=10, num_samples=150, labels=(1,))
exp2 = explainer.explain_instance(text_sample2, predict_proba, num_features=10, num_samples=150, labels=(1,))

# Save LIME HTML Highlighted Text files
html1_path = RESULTS_DIR / "lime_highlighted_text_sample1.html"
html2_path = RESULTS_DIR / "lime_highlighted_text_sample2.html"

exp1.save_to_file(str(html1_path))
exp2.save_to_file(str(html2_path))

# Plot LIME Token Weight Bar Chart
weights1 = exp1.as_list(label=1)
words1, scores1 = zip(*weights1)

plt.figure(figsize=(9, 5))
colors = ["#e74c3c" if s > 0 else "#3498db" for s in scores1[::-1]]
plt.barh(words1[::-1], scores1[::-1], color=colors)
plt.xlabel("LIME Attribution Weight (+AI / -Human)")
plt.title(f"LIME Token Weight Bar Chart — Sample 1\nP(AI) = {y_prob_ai[sample_idx1]:.3f}")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "lime_token_weight_bars.png", dpi=200)
plt.close()

print(f"[SUCCESS] LIME analysis visuals complete! All artifacts saved in {RESULTS_DIR}")
