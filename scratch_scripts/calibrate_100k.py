import os
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_on_dataframe(df, model, tokenizer, device, max_length, batch_size=64):
    texts = df['text'].tolist()
    probs = []
    model.eval()
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = [str(t) for t in texts[i:i+batch_size]]
            enc = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors='pt'
            )
            input_ids = enc['input_ids'].to(device)
            attention_mask = enc['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[:, 1]
            probs.extend(batch_probs)
            
    return np.array(df['label'].tolist()), np.array(probs)

def main():
    model_dir = 'models/model_100k'
    val_csv = 'Data/val_100k.csv'
    test_csv = 'Data/test_100k.csv'
    holdout_csv = 'Data/external_val_100k.csv'
    
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print("Evaluating on validation set...")
    val_df = pd.read_csv(val_csv).dropna(subset=['text'])
    y_val, y_prob_val = evaluate_on_dataframe(val_df, model, tokenizer, device, 256)
    
    # Calibrate threshold to find the one maximizing Macro F1
    best_threshold = 0.50
    best_f1 = 0.0
    thresholds = np.linspace(0.01, 0.99, 99)
    for t in thresholds:
        y_pred = (y_prob_val >= t).astype(int)
        f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            
    print(f"Optimal Threshold: {best_threshold:.4f} (Validation Macro F1: {best_f1:.4f})")
    
    # Save optimal threshold
    with open("results/threshold.txt", "w") as f:
        f.write(f"{best_threshold:.4f}")
    print("Saved optimal threshold to results/threshold.txt")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_df = pd.read_csv(test_csv).dropna(subset=['text'])
    y_test, y_prob_test = evaluate_on_dataframe(test_df, model, tokenizer, device, 256)
    
    # Evaluate at default 0.50
    y_pred_50 = (y_prob_test >= 0.50).astype(int)
    acc_50 = accuracy_score(y_test, y_pred_50)
    f1_50 = f1_score(y_test, y_pred_50, average='macro', zero_division=0)
    
    # Evaluate at optimal
    y_pred_opt = (y_prob_test >= best_threshold).astype(int)
    acc_opt = accuracy_score(y_test, y_pred_opt)
    f1_opt = f1_score(y_test, y_pred_opt, average='macro', zero_division=0)
    prec_opt = precision_score(y_test, y_pred_opt, average='macro', zero_division=0)
    rec_opt = recall_score(y_test, y_pred_opt, average='macro', zero_division=0)
    auc_test = roc_auc_score(y_test, y_prob_test)
    
    print(f"--- Test Split (Threshold = 0.50) ---")
    print(f"  Accuracy:  {acc_50:.4f}")
    print(f"  Macro F1:  {f1_50:.4f}")
    print(f"--- Test Split (Threshold = {best_threshold:.4f}) ---")
    print(f"  Accuracy:  {acc_opt:.4f}")
    print(f"  Precision: {prec_opt:.4f}")
    print(f"  Recall:    {rec_opt:.4f}")
    print(f"  Macro F1:  {f1_opt:.4f}")
    print(f"  ROC-AUC:   {auc_test:.4f}")
    
    # Evaluate on holdout set
    print("\nEvaluating on external holdout set...")
    holdout_df = pd.read_csv(holdout_csv).dropna(subset=['text'])
    y_holdout, y_prob_holdout = evaluate_on_dataframe(holdout_df, model, tokenizer, device, 256)
    
    y_pred_ho = (y_prob_holdout >= best_threshold).astype(int)
    acc_ho = accuracy_score(y_holdout, y_pred_ho)
    f1_ho = f1_score(y_holdout, y_pred_ho, average='macro', zero_division=0)
    prec_ho = precision_score(y_holdout, y_pred_ho, average='macro', zero_division=0)
    rec_ho = recall_score(y_holdout, y_pred_ho, average='macro', zero_division=0)
    auc_ho = roc_auc_score(y_holdout, y_prob_holdout)
    
    print(f"--- Holdout Split (Threshold = {best_threshold:.4f}) ---")
    print(f"  Accuracy:  {acc_ho:.4f}")
    print(f"  Precision: {prec_ho:.4f}")
    print(f"  Recall:    {rec_ho:.4f}")
    print(f"  Macro F1:  {f1_ho:.4f}")
    print(f"  ROC-AUC:   {auc_ho:.4f}")

if __name__ == "__main__":
    main()
