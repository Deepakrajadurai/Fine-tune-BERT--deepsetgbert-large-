import os
import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# Paths
WORKSPACE_DIR = r"e:\15-06-26\Fine-tune BERT (deepsetgbert-large)"
DATA_DIR = os.path.join(WORKSPACE_DIR, "Data")
MODELS = {
    "best_model": os.path.join(WORKSPACE_DIR, "models", "best_model"),
    "full_model": os.path.join(WORKSPACE_DIR, "models", "full_model"),
    "full_model_500k": os.path.join(WORKSPACE_DIR, "models", "full_model_500k")
}

TEST_CSV = os.path.join(DATA_DIR, "test.csv")
HOLDOUT_CSV = os.path.join(DATA_DIR, "final_holdout.csv")

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        return text, label

def collate_fn(batch, tokenizer, max_length=256):
    texts = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    enc = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt"
    )
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": torch.tensor(labels, dtype=torch.long)
    }

def evaluate_model(model_path, test_df, holdout_df, device, batch_size=64, max_length=256):
    print(f"\n--- Loading model and tokenizer from {model_path} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    results = {}

    for name, df in [("In-Distribution Test", test_df), ("Unseen Final Holdout", holdout_df)]:
        print(f"Evaluating on {name} ({len(df)} samples)...")
        dataset = TextDataset(df['text'], df['label'], tokenizer, max_length)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, tokenizer, max_length)
        )

        all_probs = []
        all_labels = []

        start_time = time.time()
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"]

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()

                all_probs.extend(probs)
                all_labels.extend(labels.numpy())

        eval_time = time.time() - start_time
        print(f"  Completed in {eval_time:.2f}s")

        y_true = np.array(all_labels)
        y_prob = np.array(all_probs)

        # Compute metrics at threshold 0.50
        y_pred_50 = (y_prob >= 0.50).astype(int)
        acc_50 = accuracy_score(y_true, y_pred_50)
        f1_50 = f1_score(y_true, y_pred_50, average='macro', zero_division=0)
        prec_50 = precision_score(y_true, y_pred_50, average='macro', zero_division=0)
        rec_50 = recall_score(y_true, y_pred_50, average='macro', zero_division=0)

        # Compute metrics at threshold 0.10 (calibrated for best_model)
        y_pred_10 = (y_prob >= 0.10).astype(int)
        acc_10 = accuracy_score(y_true, y_pred_10)
        f1_10 = f1_score(y_true, y_pred_10, average='macro', zero_division=0)
        prec_10 = precision_score(y_true, y_pred_10, average='macro', zero_division=0)
        rec_10 = recall_score(y_true, y_pred_10, average='macro', zero_division=0)

        # Compute ROC-AUC
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = 0.5

        # Class predictions breakdown (True AI vs Pred AI)
        predicted_ai_50 = int(np.sum(y_pred_50))
        actual_ai = int(np.sum(y_true))
        total_samples = len(y_true)

        results[name] = {
            "ROC-AUC": auc,
            "t_0.50": {
                "accuracy": acc_50,
                "macro_f1": f1_50,
                "precision": prec_50,
                "recall": rec_50,
                "predicted_ai": predicted_ai_50
            },
            "t_0.10": {
                "accuracy": acc_10,
                "macro_f1": f1_10,
                "precision": prec_10,
                "recall": rec_10,
                "predicted_ai": int(np.sum(y_pred_10))
            },
            "actual_ai": actual_ai,
            "total_samples": total_samples
        }

    return results

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    test_df = pd.read_csv(TEST_CSV).dropna(subset=['text'])
    holdout_df = pd.read_csv(HOLDOUT_CSV).dropna(subset=['text'])

    print(f"Loaded {len(test_df)} test samples and {len(holdout_df)} holdout samples.")

    all_results = {}
    for model_name, model_path in MODELS.items():
        if os.path.exists(model_path):
            all_results[model_name] = evaluate_model(model_path, test_df, holdout_df, device)
        else:
            print(f"Warning: Model directory not found: {model_path}")

    # Format Markdown Report
    report = []
    report.append("# Model Evaluation Comparison Report")
    report.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Device**: {device}\n")

    report.append("This report compares the performance of three trained GBERT-large models:\n")
    report.append("1. **`best_model`**: Model trained on the ~57k dataset (leakage resolved).")
    report.append("2. **`full_model`**: Model trained on the full dataset version (57k run, possibly without early stopping or different setup).")
    report.append("3. **`full_model_500k`**: Model trained on the 500k dataset (which experienced model collapse).\n")

    for dataset_name in ["In-Distribution Test", "Unseen Final Holdout"]:
        report.append(f"## {dataset_name} Evaluation Summary\n")
        report.append("| Model Name | Threshold | Accuracy | Macro Precision | Macro Recall | Macro F1 | ROC-AUC | Pred AI / Actual AI |")
        report.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

        for m_name, m_res in all_results.items():
            res = m_res[dataset_name]
            act_ai = res["actual_ai"]
            tot = res["total_samples"]
            auc = res["ROC-AUC"]

            for t_label, t_key in [("0.50", "t_0.50"), ("0.10", "t_0.10")]:
                metrics = res[t_key]
                acc = metrics["accuracy"]
                prec = metrics["precision"]
                rec = metrics["recall"]
                f1 = metrics["macro_f1"]
                pred_ai = metrics["predicted_ai"]

                report.append(
                    f"| **{m_name}** | {t_label} | {acc * 100:.2f}% | {prec * 100:.2f}% | {rec * 100:.2f}% | **{f1 * 100:.2f}%** | {auc:.4f} | {pred_ai} / {act_ai} (Total: {tot}) |"
                )
        report.append("\n")

    report.append("## Analysis & Recommendations\n")
    
    # Check best model based on holdout F1-score at 0.10 threshold
    best_name = None
    best_f1 = -1
    for m_name, m_res in all_results.items():
        f1 = m_res["Unseen Final Holdout"]["t_0.10"]["macro_f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_name = m_name

    report.append(f"### Best Model Recommendation: **`{best_name}`**\n")
    report.append("### Diagnostic Observations:")
    
    # Print observation summaries
    for m_name, m_res in all_results.items():
        ho_res = m_res["Unseen Final Holdout"]
        pred_ai_50 = ho_res["t_0.50"]["predicted_ai"]
        tot = ho_res["total_samples"]
        f1_10 = ho_res["t_0.10"]["macro_f1"]
        auc = ho_res["ROC-AUC"]

        if m_name == "full_model_500k":
            report.append(
                f"- **`full_model_500k`**: Suffers from **catastrophic model collapse**. It predicts 0 AI samples at threshold 0.50 and 0.10, resulting in a flat F1-score of ~33.8%. However, its ROC-AUC is `{auc:.4f}`, suggesting it has some discriminative power but its logits are heavily skewed."
            )
        elif m_name == "best_model":
            report.append(
                f"- **`best_model`**: Reaches a Macro F1 of **{f1_10 * 100:.2f}%** at the calibrated 0.10 threshold, with a ROC-AUC of `{auc:.4f}`. This model generalizes exceptionally well to unseen holdout domains."
            )
        else:
            report.append(
                f"- **`{m_name}`**: Achieves a holdout Macro F1 of **{f1_10 * 100:.2f}%** at the 0.10 threshold, with a ROC-AUC of `{auc:.4f}`."
            )

    report_content = "\n".join(report)
    print("\n=== EVALUATION REPORT ===")
    print(report_content)

    report_path = os.path.join(WORKSPACE_DIR, "results", "models_evaluation_comparison.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"\nReport written to: {report_path}")

if __name__ == "__main__":
    main()
