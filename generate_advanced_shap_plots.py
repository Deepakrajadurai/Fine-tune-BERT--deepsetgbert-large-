import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
import time
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

import shap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

RESULTS_DIR = Path("results/explainability/advanced_shap")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {DEVICE}")

# ---------------------------------------------------------------------------
# 1. Load Model and Prepare Dataset Samples
# ---------------------------------------------------------------------------
MODEL_PATH = "models/v5_best_model_clean"
if not Path(MODEL_PATH).exists():
    MODEL_PATH = "models/best_model"

print(f"[INFO] Loading Transformer Model from {MODEL_PATH}...")
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

# Load sample sentences for text SHAP
sample_texts = [
    "Die Bundesregierung beschließt ein neues Maßnahmepaket zur Digitalisierung.",
    "Wissenschaftler der Universität München haben eine neue Methode entwickelt, um erneuerbare Energien einzuspeisen.",
    "Auf Initiative von Abgeordnetem Dr. Müller wird in dieser Plenarsitzung die Drucksache 19/4820 verabschiedet.",
    "Das Gesetz tritt am folgenden Tag in Kraft.",
    "Im Rahmen der heutigen Debatte forderten Oppositionspolitiker eine umfassende Überprüfung der Ausgaben.",
    "Die Entwicklung künstlicher Intelligenz schreitet in Deutschland rasant voran.",
    "Der Bundesrat hat den Entwurf zum Haushaltsgesetz einstimmig abgelehnt.",
    "Neue Forschungsergebnisse zeigen signifikante Verbesserungen bei der Speicherung wissenschaftlicher Daten.",
    "In der morgigen Sitzung stehen wichtige Beschlüsse zur Energiewende auf der Tagesordnung.",
    "Bürgerinnen und Bürger erwarten eine transparente politische Entscheidung bei diesen Gesetzesvorhaben."
]

print(f"[INFO] Running SHAP Text Explainer on {len(sample_texts)} sample texts...")
explainer_text = shap.Explainer(predict_proba, shap.maskers.Text(tokenizer))
shap_text_values = explainer_text(sample_texts)

# ---------------------------------------------------------------------------
# 2. Text SHAP Plots
# ---------------------------------------------------------------------------

# Plot 1: SHAP Bar Plot (Mean |SHAP| per token across texts)
print("[INFO] Generating SHAP Bar Plot...")
plt.figure(figsize=(10, 6))
shap.plots.bar(shap_text_values[:, :, 1], max_display=15, show=False)
plt.title("SHAP Global Feature Importance (Bar Plot) — Class: AI")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "shap_bar_plot.png", dpi=200, bbox_inches="tight")
plt.close()

# Plot 2: SHAP Waterfall Plot (Local prediction breakdown for text 0)
print("[INFO] Generating SHAP Waterfall Plot...")
plt.figure(figsize=(10, 7))
shap.plots.waterfall(shap_text_values[0, :, 1], max_display=12, show=False)
plt.title("SHAP Local Waterfall Plot (Prediction Decomposition)")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "shap_waterfall_local.png", dpi=200, bbox_inches="tight")
plt.close()

# Plot 3: SHAP Force Plot (Local prediction push/pull for text 0)
print("[INFO] Generating SHAP Force Plot...")
plt.figure(figsize=(12, 4))
try:
    shap.plots.force(
        shap_text_values[0, :, 1],
        matplotlib=True,
        show=False
    )
except Exception as e:
    print(f"[WARNING] Native force plot fallback: {e}")
    base_val = shap_text_values.base_values[0, 1] if hasattr(shap_text_values, "base_values") else 0.5
    vals = shap_text_values.values[0, :, 1]
    tokens = shap_text_values.data[0]
    shap.plots.force(
        base_val,
        vals,
        feature_names=tokens,
        matplotlib=True,
        show=False
    )

plt.title("SHAP Local Force Plot — Token Contributions to P(AI)")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "shap_force_local.png", dpi=200, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 3. Tabular / TF-IDF Stylometric Model for Beeswarm & Dependence Plots
# ---------------------------------------------------------------------------
print("[INFO] Training TF-IDF + LogisticRegression model for Beeswarm & Dependence Plots...")
# Prepare dataset sample
data_file = Path("Data/test_cleaned.csv")
if not data_file.exists():
    data_file = Path("Data/test_100k.csv")

if data_file.exists():
    df = pd.read_csv(data_file, nrows=300).dropna()
    text_col = "text" if "text" in df.columns else df.columns[0]
    label_col = "label" if "label" in df.columns else df.columns[1]
    X_raw = df[text_col].astype(str).tolist()
    y_raw = df[label_col].astype(int).values
else:
    X_raw = sample_texts * 30
    y_raw = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 30)

vectorizer = TfidfVectorizer(max_features=50, ngram_range=(1, 2))
X_matrix = vectorizer.fit_transform(X_raw).toarray()
feature_names = vectorizer.get_feature_names_out()

lr_model = LogisticRegression()
lr_model.fit(X_matrix, y_raw)

# Linear / LinearExplainer or Explainer for tabular TF-IDF
explainer_tabular = shap.Explainer(lr_model, X_matrix)
shap_tabular_values = explainer_tabular(X_matrix)

# Plot 4: SHAP Heatmap Plot (Instances x Features)
print("[INFO] Generating SHAP Heatmap Plot...")
plt.figure(figsize=(12, 6))
try:
    shap.plots.heatmap(shap_tabular_values, max_display=15, show=False)
except Exception as e:
    print(f"[WARNING] Tabular heatmap fallback: {e}")
    sns.heatmap(shap_tabular_values.values[:20, :15], cmap="coolwarm", center=0)
    plt.xlabel("Features")
    plt.ylabel("Instances")

plt.title("SHAP Heatmap (Feature Attributions across Instances)")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "shap_heatmap_tokens.png", dpi=200, bbox_inches="tight")
plt.close()

# Plot 5: SHAP Beeswarm Plot (Global Summary Plot)
print("[INFO] Generating SHAP Global Beeswarm Plot...")
plt.figure(figsize=(10, 8))
shap.plots.beeswarm(shap_tabular_values, max_display=15, show=False)
plt.title("SHAP Global Summary (Beeswarm) Plot — Feature Values vs SHAP Attributions")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "shap_beeswarm_plot.png", dpi=200, bbox_inches="tight")
plt.close()

# Plot 6 & 7: SHAP Feature Dependence Plots
print("[INFO] Generating SHAP Feature Dependence Plots...")
# Select top 2 features by mean absolute SHAP value
mean_abs_shap = np.abs(shap_tabular_values.values).mean(axis=0)
top_idx = np.argsort(mean_abs_shap)[::-1]
feat1 = feature_names[top_idx[0]]
feat2 = feature_names[top_idx[1]]

# Dependence Plot 1
plt.figure(figsize=(8, 6))
shap.dependence_plot(top_idx[0], shap_tabular_values.values, X_matrix, feature_names=feature_names, show=False)
plt.title(f"SHAP Feature Dependence Plot — '{feat1}'")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "shap_dependence_feature1.png", dpi=200, bbox_inches="tight")
plt.close()

# Dependence Plot 2
plt.figure(figsize=(8, 6))
shap.dependence_plot(top_idx[1], shap_tabular_values.values, X_matrix, feature_names=feature_names, show=False)
plt.title(f"SHAP Feature Dependence Plot — '{feat2}'")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "shap_dependence_feature2.png", dpi=200, bbox_inches="tight")
plt.close()

print(f"[SUCCESS] All 5 advanced SHAP plots generated successfully in {RESULTS_DIR}!")
