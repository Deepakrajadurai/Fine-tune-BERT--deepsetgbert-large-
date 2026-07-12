# """
# Step 3 (FIXED) — Model Evaluation and Holdout Verification
# ===========================================================
# - Loads the fine-tuned BERT model and datasets
# - Computes standard metrics on the in-distribution test split
# - Executes a strict, single-run evaluation on the final holdout split (unseen domains & models)
# - Generates a source-aware accuracy breakdown table for final verification
# """

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# import argparse
# import logging
# import numpy as np
# import pandas as pd
# import torch
# from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     roc_auc_score,
#     classification_report
# )

# logging.basicConfig(level=logging.INFO,
#                     format="%(asctime)s [%(levelname)s] %(message)s")
# log = logging.getLogger(__name__)

# def evaluate_on_dataframe(df, model, tokenizer, device, max_length, threshold=0.10):
#     texts = df['text'].tolist()
#     labels = df['label'].tolist()
    
#     preds = []
#     probs = []
    
#     model.eval()
#     with torch.no_grad():
#         for text in texts:
#             enc = tokenizer(
#                 str(text),
#                 truncation=True,
#                 max_length=max_length,
#                 return_tensors='pt'
#             )
#             input_ids = enc['input_ids'].to(device)
#             attention_mask = enc['attention_mask'].to(device)
            
#             outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#             logits = outputs.logits
#             prob = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            
#             ai_prob = prob[1]
#             pred = 1 if ai_prob >= threshold else 0
            
#             preds.append(pred)
#             probs.append(ai_prob)
            
#     return np.array(labels), np.array(preds), np.array(probs)

# def print_metrics(y_true, y_pred, y_prob, title="Evaluation Results"):
#     acc = accuracy_score(y_true, y_pred)
#     prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
#     rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
#     f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
#     try:
#         auc = roc_auc_score(y_true, y_prob)
#         auc_str = f"{auc:.4f}"
#     except ValueError:
#         auc_str = "N/A"
        
#     print(f"\n=== {title} ===")
#     print(f"Accuracy:  {acc:.4f}")
#     print(f"Precision: {prec_macro:.4f} (Macro)")
#     print(f"Recall:    {rec_macro:.4f} (Macro)")
#     print(f"F1-Score:  {f1_macro:.4f} (Macro)")
#     print(f"ROC-AUC:   {auc_str}")
#     return acc, f1_macro

# def main():
#     parser = argparse.ArgumentParser(description="Evaluate Fine-tuned GBERT Model")
#     parser.add_argument('--model_dir', type=str, default='models/best_model')
#     parser.add_argument('--test_csv', type=str, default='Data/test.csv')
#     parser.add_argument('--holdout_csv', type=str, default='Data/final_holdout.csv')
#     parser.add_argument('--max_length', type=int, default=256)
#     parser.add_argument('--threshold', type=float, default=None, help="Decision threshold override")
#     args = parser.parse_args()

#     if not os.path.exists(args.model_dir):
#         raise FileNotFoundError(f"Saved model directory not found at {args.model_dir}")

#     # 1. Load model and tokenizer
#     log.info(f"Loading tokenizer and model from {args.model_dir}...")
#     is_bert = os.path.exists(os.path.join(args.model_dir, "vocab.txt")) or 'gbert' in args.model_dir
#     if is_bert:
#         tokenizer = BertTokenizer.from_pretrained(args.model_dir)
#         model = BertForSequenceClassification.from_pretrained(args.model_dir)
#     else:
#         tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
#         model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     log.info(f"Using device: {device.type.upper()}")
#     model.to(device)

#     # 2. Get calibrated threshold
#     if args.threshold is not None:
#         threshold = args.threshold
#     elif os.path.exists("results/threshold.txt"):
#         with open("results/threshold.txt") as f:
#             threshold = float(f.read().strip())
#     else:
#         threshold = 0.10
#     log.info(f"Using decision threshold: {threshold}")

#     # 3. Evaluate in-distribution test split
#     if os.path.exists(args.test_csv):
#         log.info(f"Evaluating in-distribution test split from {args.test_csv}...")
#         test_df = pd.read_csv(args.test_csv).dropna(subset=['text'])
#         y_true, y_pred, y_prob = evaluate_on_dataframe(test_df, model, tokenizer, device, args.max_length, threshold)
#         print_metrics(y_true, y_pred, y_prob, title="IN-DISTRIBUTION TEST SPLIT METRICS")
#         print("\nClassification Report (In-Distribution):")
#         print(classification_report(y_true, y_pred, target_names=['Human', 'AI'], zero_division=0))
#     else:
#         log.warning(f"Test split not found at {args.test_csv}")

#     # 4. Evaluate strict final holdout split
#     if os.path.exists(args.holdout_csv):
#         log.info(f"Evaluating unseen final holdout split from {args.holdout_csv}...")
#         holdout_df = pd.read_csv(args.holdout_csv).dropna(subset=['text'])
        
#         y_true_ho, y_pred_ho, y_prob_ho = evaluate_on_dataframe(holdout_df, model, tokenizer, device, args.max_length, threshold)
#         print_metrics(y_true_ho, y_pred_ho, y_prob_ho, title="UNSEEN FINAL HOLDOUT SPLIT METRICS")
        
#         print("\n" + "=" * 70)
#         print("FINAL HOLDOUT SET SOURCE BREAKDOWN (Strict Evaluation)")
#         print("=" * 70)
#         print(f"{'Source':<35} | {'Samples':<8} | {'Correct':<8} | {'Accuracy':<8}")
#         print("-" * 70)
        
#         sources = holdout_df["source"].unique()
#         for source in sources:
#             sub_df = holdout_df[holdout_df["source"] == source]
#             sub_true, sub_pred, sub_prob = evaluate_on_dataframe(sub_df, model, tokenizer, device, args.max_length, threshold)
#             correct = sum(1 for t, p in zip(sub_true, sub_pred) if t == p)
#             acc = correct / len(sub_df) if len(sub_df) > 0 else 0
#             print(f"{source:<35} | {len(sub_df):<8} | {correct:<8} | {acc * 100:.1f}%")
#         print("=" * 70 + "\n")
#     else:
#         log.warning(f"Holdout split not found at {args.holdout_csv}")

# if __name__ == '__main__':
#     main()

"""
evaluate.py

Full evaluation script implementing all Stage 5 checks from the training
pipeline: aggregate metrics, per-domain breakdown, per-generator
breakdown (the direct re-test of the original bimodal-overfitting
failure mode), confidence calibration histogram, length-bucket sanity
check, human-source breakdown, and a test-vs-holdout cross-check.

Usage:
    python evaluate.py \
        --model_dir models/best_model \
        --test_csv Data/test.csv \
        --holdout_csv Data/final_holdout.csv \
        --output_dir eval_report \
        --max_length 512
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Defense-in-depth: collapse embedded newlines/tabs/repeated spaces
    to a single space, applied uniformly to both classes. See train.py
    for the full rationale -- this closes the same potential shortcut at
    inference time in case a future dataset re-introduces the
    AI-vs-human whitespace asymmetry found in training_pair_v5.csv."""
    return _WHITESPACE_RE.sub(" ", str(text)).strip()


LENGTH_BUCKET_EDGES = [0, 500, 650, 800, 950, 1100, 1300, 1500, 1750, 10**7]
LENGTH_BUCKET_LABELS = [
    f"{LENGTH_BUCKET_EDGES[i]}-{LENGTH_BUCKET_EDGES[i+1]}"
    for i in range(len(LENGTH_BUCKET_EDGES) - 1)
]


def run_inference(model, tokenizer, texts, max_length=512, batch_size=32, device="cpu"):
    model.eval()
    all_probs = []
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(list(batch), truncation=True, max_length=max_length,
                             padding=True, return_tensors="pt").to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()  # P(AI)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            if (i // batch_size) % 20 == 0:
                print(f"  inference: {i}/{len(texts)}")
    return np.array(all_preds), np.array(all_probs)


def evaluate_split(df, preds, probs, split_name, output_dir):
    y_true = df["label"].astype(int).values
    y_pred = preds
    df = df.copy()
    df["pred"] = y_pred
    df["prob_ai"] = probs
    df["correct"] = (df["pred"] == df["label"])

    report = {}
    lines = [f"# Evaluation Report: {split_name}\n"]

    # 5a. Aggregate metrics
    lines.append("## 5a. Aggregate metrics\n")
    cls_report = classification_report(y_true, y_pred, target_names=["human", "ai"],
                                        output_dict=True)
    lines.append("```\n" + classification_report(y_true, y_pred,
                                                    target_names=["human", "ai"]) + "\n```\n")
    cm = confusion_matrix(y_true, y_pred)
    lines.append(f"Confusion matrix (rows=true, cols=pred, order=[human, ai]):\n```\n{cm}\n```\n")
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report["macro_f1"] = macro_f1
    report["classification_report"] = cls_report

    # 5b. Per-domain breakdown
    lines.append("## 5b. Per-domain Macro F1\n")
    per_domain = {}
    for domain, sub in df.groupby("domain"):
        f1 = f1_score(sub["label"], sub["pred"], average="macro")
        per_domain[domain] = {"macro_f1": f1, "n": len(sub)}
        lines.append(f"- **{domain}**: macro_f1={f1:.4f} (n={len(sub)})")
    report["per_domain"] = per_domain
    lines.append("")

    # 5c. Per-generator breakdown (AI class only, via 'meta' column)
    lines.append("## 5c. Per-generator recall (AI class only, via `meta` column)\n")
    ai_rows = df[df["label"] == 1]
    per_generator = ai_rows.groupby("meta")["correct"].agg(["mean", "count"])
    per_generator = per_generator.sort_values("mean")
    per_generator_dict = per_generator.to_dict(orient="index")
    for gen, row in per_generator_dict.items():
        lines.append(f"- **{gen}**: recall={row['mean']:.4f} (n={int(row['count'])})")
    report["per_generator_ai_recall"] = per_generator_dict
    spread = per_generator["mean"].max() - per_generator["mean"].min() if len(per_generator) else None
    if spread is not None:
        lines.append(f"\nSpread across generators: {spread:.4f} "
                      f"({'wide -- expected, generators differ in detectability' if spread > 0.15 else 'narrow -- check for uniform bimodal pattern (5d)'})\n")

    # 5d. Confidence calibration / bimodality check
    lines.append("## 5d. Confidence distribution (P(AI))\n")
    hist, bin_edges = np.histogram(df["prob_ai"], bins=20, range=(0, 1))
    hist_lines = "\n".join(
        f"  [{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}): {'#' * int(hist[i] / max(hist.max(),1) * 50)} ({hist[i]})"
        for i in range(len(hist))
    )
    lines.append("```\n" + hist_lines + "\n```\n")
    frac_extreme = ((df["prob_ai"] < 0.05) | (df["prob_ai"] > 0.95)).mean()
    frac_extreme_wrong = (((df["prob_ai"] < 0.05) | (df["prob_ai"] > 0.95)) & ~df["correct"]).mean()
    lines.append(f"- Fraction with extreme confidence (<0.05 or >0.95): {frac_extreme:.2%}\n"
                  f"- Fraction confidently WRONG (extreme confidence but incorrect): {frac_extreme_wrong:.2%}\n")
    report["frac_extreme_confidence"] = frac_extreme
    report["frac_confidently_wrong"] = frac_extreme_wrong

    # 5e. Length-bucket sanity check
    lines.append("## 5e. Accuracy by length bucket and label\n")
    df["len_bucket"] = pd.cut(df["text"].str.len(), bins=LENGTH_BUCKET_EDGES,
                               labels=LENGTH_BUCKET_LABELS)
    per_bucket = df.groupby(["len_bucket", "label"], observed=True)["correct"].agg(["mean", "count"])
    per_bucket_dict = {}
    for (bucket, label), row in per_bucket.iterrows():
        key = f"{bucket}_label{label}"
        per_bucket_dict[key] = {"accuracy": row["mean"], "n": int(row["count"])}
        lines.append(f"- {bucket}, label={label} ({'ai' if label==1 else 'human'}): "
                      f"acc={row['mean']:.4f} (n={int(row['count'])})")
    report["per_length_bucket"] = per_bucket_dict
    lines.append("")

    # 5f. Human-source breakdown
    lines.append("## 5f. Per human-source recall (human class only, via `meta` column)\n")
    human_rows = df[df["label"] == 0]
    per_source = human_rows.groupby("meta")["correct"].agg(["mean", "count"])
    per_source = per_source.sort_values("mean")
    per_source_dict = per_source.to_dict(orient="index")
    for src, row in per_source_dict.items():
        lines.append(f"- **{src}**: recall={row['mean']:.4f} (n={int(row['count'])})")
    report["per_human_source_recall"] = per_source_dict

    os.makedirs(output_dir, exist_ok=True)
    md_path = os.path.join(output_dir, f"{split_name}_report.md")
    json_path = os.path.join(output_dir, f"{split_name}_report.json")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"[{split_name}] macro_f1={macro_f1:.4f} -- report written to {md_path}")
    return report


def cross_check(test_report, holdout_report, output_dir):
    lines = ["# 5g. Test vs. Final Holdout Cross-Check\n"]
    test_f1 = test_report["macro_f1"]
    holdout_f1 = holdout_report["macro_f1"]
    diff = test_f1 - holdout_f1
    lines.append(f"- test.csv macro_f1: {test_f1:.4f}")
    lines.append(f"- final_holdout.csv macro_f1: {holdout_f1:.4f}")
    lines.append(f"- difference (test - holdout): {diff:+.4f}")
    if abs(diff) > 0.03:
        lines.append(f"\n**WARNING**: difference exceeds 0.03 -- possible subtle "
                      f"overfitting to decisions made while iterating against "
                      f"test/val, even with group-disjoint splits. Report both "
                      f"numbers in the thesis, not just the better one.")
    else:
        lines.append(f"\nDifference is small -- results are consistent across both "
                      f"independent splits.")

    path = os.path.join(output_dir, "cross_check_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Cross-check written to {path}")
    print(f"test={test_f1:.4f}, holdout={holdout_f1:.4f}, diff={diff:+.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--holdout_csv", required=True)
    parser.add_argument("--output_dir", default="eval_report")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {args.model_dir} (device={device})")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    except Exception as e:
        print(f"AutoTokenizer failed ({e}). Falling back to BertTokenizer...")
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(args.model_dir)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    except Exception as e:
        print(f"AutoModel failed ({e}). Falling back to BertForSequenceClassification...")
        from transformers import BertForSequenceClassification
        model = BertForSequenceClassification.from_pretrained(args.model_dir).to(device)

    reports = {}
    for split_name, csv_path in [("test", args.test_csv), ("final_holdout", args.holdout_csv)]:
        print(f"\n=== Evaluating {split_name}: {csv_path} ===")
        df = pd.read_csv(csv_path)
        if "domain" not in df.columns:
            df["domain"] = df["source"] if "source" in df.columns else "unknown"
        if "meta" not in df.columns:
            df["meta"] = "unknown"
        df["text"] = df["text"].apply(normalize_text)
        preds, probs = run_inference(model, tokenizer, df["text"].astype(str).tolist(),
                                      max_length=args.max_length, batch_size=args.batch_size,
                                      device=device)
        reports[split_name] = evaluate_split(df, preds, probs, split_name, args.output_dir)

    print("\n=== Cross-check ===")
    cross_check(reports["test"], reports["final_holdout"], args.output_dir)

    print(f"\nAll reports written to {args.output_dir}/")


if __name__ == "__main__":
    main()
