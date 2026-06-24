"""
Step 3 (FIXED) — Model Evaluation and Holdout Verification
===========================================================
- Loads the fine-tuned BERT model and datasets
- Computes standard metrics on the in-distribution test split
- Executes a strict, single-run evaluation on the final holdout split (unseen domains & models)
- Generates a source-aware accuracy breakdown table for final verification
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import logging
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def evaluate_on_dataframe(df, model, tokenizer, device, max_length, threshold=0.10):
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    preds = []
    probs = []
    
    model.eval()
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                str(text),
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            input_ids = enc['input_ids'].to(device)
            attention_mask = enc['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            prob = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            
            ai_prob = prob[1]
            pred = 1 if ai_prob >= threshold else 0
            
            preds.append(pred)
            probs.append(ai_prob)
            
    return np.array(labels), np.array(preds), np.array(probs)

def print_metrics(y_true, y_pred, y_prob, title="Evaluation Results"):
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, y_prob)
        auc_str = f"{auc:.4f}"
    except ValueError:
        auc_str = "N/A"
        
    print(f"\n=== {title} ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec_macro:.4f} (Macro)")
    print(f"Recall:    {rec_macro:.4f} (Macro)")
    print(f"F1-Score:  {f1_macro:.4f} (Macro)")
    print(f"ROC-AUC:   {auc_str}")
    return acc, f1_macro

def main():
    parser = argparse.ArgumentParser(description="Evaluate Fine-tuned GBERT Model")
    parser.add_argument('--model_dir', type=str, default='models/best_model')
    parser.add_argument('--test_csv', type=str, default='Data/test.csv')
    parser.add_argument('--holdout_csv', type=str, default='Data/final_holdout.csv')
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--threshold', type=float, default=None, help="Decision threshold override")
    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"Saved model directory not found at {args.model_dir}")

    # 1. Load model and tokenizer
    log.info(f"Loading tokenizer and model from {args.model_dir}...")
    is_bert = os.path.exists(os.path.join(args.model_dir, "vocab.txt")) or 'gbert' in args.model_dir
    if is_bert:
        tokenizer = BertTokenizer.from_pretrained(args.model_dir)
        model = BertForSequenceClassification.from_pretrained(args.model_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Using device: {device.type.upper()}")
    model.to(device)

    # 2. Get calibrated threshold
    if args.threshold is not None:
        threshold = args.threshold
    elif os.path.exists("results/threshold.txt"):
        with open("results/threshold.txt") as f:
            threshold = float(f.read().strip())
    else:
        threshold = 0.10
    log.info(f"Using decision threshold: {threshold}")

    # 3. Evaluate in-distribution test split
    if os.path.exists(args.test_csv):
        log.info(f"Evaluating in-distribution test split from {args.test_csv}...")
        test_df = pd.read_csv(args.test_csv).dropna(subset=['text'])
        y_true, y_pred, y_prob = evaluate_on_dataframe(test_df, model, tokenizer, device, args.max_length, threshold)
        print_metrics(y_true, y_pred, y_prob, title="IN-DISTRIBUTION TEST SPLIT METRICS")
        print("\nClassification Report (In-Distribution):")
        print(classification_report(y_true, y_pred, target_names=['Human', 'AI'], zero_division=0))
    else:
        log.warning(f"Test split not found at {args.test_csv}")

    # 4. Evaluate strict final holdout split
    if os.path.exists(args.holdout_csv):
        log.info(f"Evaluating unseen final holdout split from {args.holdout_csv}...")
        holdout_df = pd.read_csv(args.holdout_csv).dropna(subset=['text'])
        
        y_true_ho, y_pred_ho, y_prob_ho = evaluate_on_dataframe(holdout_df, model, tokenizer, device, args.max_length, threshold)
        print_metrics(y_true_ho, y_pred_ho, y_prob_ho, title="UNSEEN FINAL HOLDOUT SPLIT METRICS")
        
        print("\n" + "=" * 70)
        print("FINAL HOLDOUT SET SOURCE BREAKDOWN (Strict Evaluation)")
        print("=" * 70)
        print(f"{'Source':<35} | {'Samples':<8} | {'Correct':<8} | {'Accuracy':<8}")
        print("-" * 70)
        
        sources = holdout_df["source"].unique()
        for source in sources:
            sub_df = holdout_df[holdout_df["source"] == source]
            sub_true, sub_pred, sub_prob = evaluate_on_dataframe(sub_df, model, tokenizer, device, args.max_length, threshold)
            correct = sum(1 for t, p in zip(sub_true, sub_pred) if t == p)
            acc = correct / len(sub_df) if len(sub_df) > 0 else 0
            print(f"{source:<35} | {len(sub_df):<8} | {correct:<8} | {acc * 100:.1f}%")
        print("=" * 70 + "\n")
    else:
        log.warning(f"Holdout split not found at {args.holdout_csv}")

if __name__ == '__main__':
    main()
