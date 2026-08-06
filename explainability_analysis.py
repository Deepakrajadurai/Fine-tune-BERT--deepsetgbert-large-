import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
import time
import re
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
from lime.lime_text import LimeTextExplainer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Directories & Settings
# ---------------------------------------------------------------------------
RESULTS_DIR = Path("results/explainability")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {DEVICE}")

# Sample test cases representing different model failure modes and regimes
TEST_CASES = {
    "whitespace_shortcut": {
        "text": "Die Bundesregierung beschließt ein neues Maßnahmepaket zur Digitalisierung.\n\n \nAbgeordnete fordern schnelle Umsetzungen in allen Bundesländern.\n ",
        "label": "AI (with whitespace leakage)",
        "desc": "Contains newline \\n and trailing space artifacts present in 66% of raw synthetic AI texts."
    },
    "template_collapse": {
        "text": "Auf Initiative von Abgeordnetem Dr. Müller wird in dieser Plenarsitzung die Drucksache 19/4820 eingehend beraten und verabschiedet.",
        "label": "AI (templated synthetic)",
        "desc": "Contains repeating synthetic parliamentary template boilerplate."
    },
    "clean_organic": {
        "text": "Wissenschaftler der Universität München haben eine neue Methode entwickelt, um erneuerbare Energien effizienter in das bestehende Stromnetz einzuspeisen.",
        "label": "Organic (Clean prose)",
        "desc": "Natural German news article without synthetic templates or whitespace leakage."
    },
    "short_ood": {
        "text": "Das Gesetz tritt am folgenden Tag in Kraft.",
        "label": "Short OOD (<10 words)",
        "desc": "Brief sentence typical of OOD benchmark failure cases."
    }
}

# ---------------------------------------------------------------------------
# Helper Model Class for Transformer Prediction
# ---------------------------------------------------------------------------
class TransformerPredictor:
    def __init__(self, model_path: str, device: str = DEVICE):
        print(f"[INFO] Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        self.model.eval()
        self.device = device

    def predict_proba(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, (np.ndarray, list, tuple)):
            if isinstance(texts, np.ndarray):
                texts = texts.tolist()
            texts = [str(t) for t in texts]
        else:
            texts = [str(texts)]
        
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()
        return probs

    def predict_ai_prob(self, texts):
        return self.predict_proba(texts)[:, 1]

# ---------------------------------------------------------------------------
# Baseline TF-IDF Model
# ---------------------------------------------------------------------------
def build_tfidf_baseline():
    print("[INFO] Building diagnostic TF-IDF + LogisticRegression baseline...")
    # Load small sample of train data or build synthetic fallback
    train_path = Path("Data/train_cleaned.csv")
    if not train_path.exists():
        train_path = Path("Data/train_100k.csv")
    
    if train_path.exists():
        df = pd.read_csv(train_path, nrows=5000)
        text_col = "text" if "text" in df.columns else df.columns[0]
        label_col = "label" if "label" in df.columns else df.columns[1]
        df = df.dropna(subset=[text_col, label_col])
        X = df[text_col].astype(str)
        y = df[label_col].astype(int)
    else:
        # Fallback synthetic training data
        X = [
            "Plenarsitzung Drucksache Auf Initiative von Abgeordnetem",
            "Wissenschaftler haben neues Experiment durchgeführt",
            "Der Ausschuss empfiehlt die Zustimmung zur Drucksache",
            "Heute schien die Sonne über Berlin ganz überraschend",
            "Gemäß § 18 Abs. 3 Absatz 4 der Ordnung",
            "Ein schöner Sommertag im Park mit Freunden"
        ] * 20
        y = [1, 0, 1, 0, 1, 0] * 20

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
    X_tfidf = vectorizer.fit_transform(X)
    model = LogisticRegression()
    model.fit(X_tfidf, y)

    def predict_proba(texts):
        if isinstance(texts, str):
            texts = [texts]
        vecs = vectorizer.transform(texts)
        return model.predict_proba(vecs)

    return predict_proba, vectorizer, model

# ---------------------------------------------------------------------------
# Interpretability Runner (SHAP & LIME)
# ---------------------------------------------------------------------------
def run_explainability():
    results_summary = {}

    # Identify existing model paths
    candidate_models = {
        "v5_best_model_clean": "models/v5_best_model_clean",
        "best_model_v1": "models/best_model",
        "organic_gbert": "models/organic_gbert_large",
    }

    loaded_models = {}
    for name, path in candidate_models.items():
        if Path(path).exists():
            try:
                loaded_models[name] = TransformerPredictor(path)
            except Exception as e:
                print(f"[WARNING] Could not load {name} from {path}: {e}")

    tfidf_pred_fn, vectorizer, tfidf_model = build_tfidf_baseline()

    lime_explainer = LimeTextExplainer(class_names=["Human", "AI"], random_state=42)

    # We will evaluate each test case on the available models
    for case_id, case_info in TEST_CASES.items():
        print(f"\n========================================================")
        print(f"[RUNNING CASE] {case_id}: {case_info['label']}")
        print(f"Text snippet: {case_info['text'][:80]}...")
        print(f"========================================================")

        text = case_info["text"]
        case_metrics = {}

        # 1. Evaluate Transformer Models
        for model_name, predictor in loaded_models.items():
            print(f"\n--- Analyzing {model_name} on {case_id} ---")

            probs = predictor.predict_proba(text)[0]
            p_ai = float(probs[1])
            print(f"[{model_name}] Predicted P(AI) = {p_ai:.4f}")

            # --- LIME Explanation ---
            lime_exp = lime_explainer.explain_instance(
                text,
                predictor.predict_proba,
                num_features=10,
                num_samples=250,
                labels=(1,)
            )
            lime_weights = dict(lime_exp.as_list(label=1))

            # --- SHAP Explanation ---
            shap_explainer = shap.Explainer(predictor.predict_proba, shap.maskers.Text(predictor.tokenizer))
            shap_values = shap_explainer([text])
            
            # Extract SHAP values for class 1 (AI)
            # shap_values[0].values shape: (seq_len, 2)
            shap_tokens = shap_values.data[0]
            shap_vals_class1 = shap_values.values[0][:, 1]

            shap_dict = {}
            for tok, val in zip(shap_tokens, shap_vals_class1):
                tok_clean = tok.strip()
                if tok_clean:
                    shap_dict[tok_clean] = shap_dict.get(tok_clean, 0.0) + float(val)

            # --- Quantitative Comparison between SHAP and LIME ---
            # Compare top words present in both
            common_words = set(lime_weights.keys()).intersection(set(shap_dict.keys()))
            if len(common_words) >= 3:
                lime_vec = [lime_weights[w] for w in common_words]
                shap_vec = [shap_dict[w] for w in common_words]
                corr, pval = spearmanr(lime_vec, shap_vec)
                corr = float(corr) if not np.isnan(corr) else 0.0
            else:
                corr = 0.0

            print(f"[{model_name}] Top LIME features: {list(lime_weights.items())[:5]}")
            print(f"[{model_name}] Top SHAP features: {sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]}")
            print(f"[{model_name}] Spearman correlation (SHAP vs LIME): {corr:.4f}")

            # Plot LIME & SHAP comparative bar chart
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # LIME plot
            lime_words = list(lime_weights.keys())[:8]
            lime_scores = [lime_weights[w] for w in lime_words]
            ax1.barh(lime_words[::-1], lime_scores[::-1], color=['#e74c3c' if s > 0 else '#3498db' for s in lime_scores[::-1]])
            ax1.set_title(f"LIME Top Token Attributions ({model_name})\nP(AI) = {p_ai:.3f}")
            ax1.set_xlabel("Attribution Weight (+AI / -Human)")

            # SHAP plot
            top_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
            shap_words = [x[0] for x in top_shap]
            shap_scores = [x[1] for x in top_shap]
            ax2.barh(shap_words[::-1], shap_scores[::-1], color=['#e74c3c' if s > 0 else '#3498db' for s in shap_scores[::-1]])
            ax2.set_title(f"SHAP Top Token Attributions ({model_name})\nSpearman Correlation = {corr:.2f}")
            ax2.set_xlabel("SHAP Value (+AI / -Human)")

            plt.tight_layout()
            plot_path = RESULTS_DIR / f"{case_id}_{model_name}_shap_lime.png"
            plt.savefig(plot_path, dpi=200)
            plt.close()

            case_metrics[model_name] = {
                "p_ai": p_ai,
                "top_lime": list(lime_weights.items())[:5],
                "top_shap": sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5],
                "spearman_corr": corr,
                "plot_file": str(plot_path)
            }

        # 2. Evaluate TF-IDF Baseline
        print(f"\n--- Analyzing TF-IDF Baseline on {case_id} ---")
        probs_tfidf = tfidf_pred_fn(text)[0]
        p_ai_tfidf = float(probs_tfidf[1])

        lime_exp_tfidf = lime_explainer.explain_instance(
            text,
            tfidf_pred_fn,
            num_features=10,
            num_samples=250,
            labels=(1,)
        )
        lime_weights_tfidf = dict(lime_exp_tfidf.as_list(label=1))

        case_metrics["tfidf_baseline"] = {
            "p_ai": p_ai_tfidf,
            "top_lime": list(lime_weights_tfidf.items())[:5],
        }

        results_summary[case_id] = case_metrics

    # Save summary json
    summary_path = RESULTS_DIR / "metrics_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print(f"\n[SUCCESS] Explainability analysis complete! Results saved to {RESULTS_DIR}")

if __name__ == "__main__":
    run_explainability()
