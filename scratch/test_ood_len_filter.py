import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import f1_score, accuracy_score, classification_report
import numpy as np
from tqdm import tqdm

print("Loading model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = BertTokenizer.from_pretrained("models/v5_best_model_clean")
model = BertForSequenceClassification.from_pretrained("models/v5_best_model_clean").to(device)
model.eval()

print("Loading Data/external_val_100k.csv...")
df = pd.read_csv("Data/external_val_100k.csv")
df["word_count"] = df["text"].astype(str).apply(lambda t: len(t.split()))

# Filter to >= 10 words
df_subset = df[df["word_count"] >= 10].copy()
print(f"Subset >= 10 words: {len(df_subset):,} rows (out of {len(df):,})")
print(f"Label distribution:\n{df_subset['label'].value_counts()}")

texts = df_subset["text"].tolist()
labels = df_subset["label"].tolist()

preds = []
batch_size = 64
for i in tqdm(range(0, len(texts), batch_size), desc="Running inference"):
    batch_texts = texts[i:i+batch_size]
    enc = tokenizer(batch_texts, max_length=256, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device)).logits
    batch_preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
    preds.extend(batch_preds)

df_subset["pred"] = preds
print("\nAggregate metrics for >= 10 words:")
print(classification_report(labels, preds, target_names=["human", "ai"]))
print(f"Macro F1: {f1_score(labels, preds, average='macro'):.4f}")
