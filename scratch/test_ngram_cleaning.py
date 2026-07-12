import pandas as pd
import re
import numpy as np

print("Loading Data/Final/training_pair_v5_clean.csv...")
df = pd.read_csv("Data/Final/training_pair_v5_clean.csv", encoding="utf-8")
print(f"Loaded {len(df):,} rows.")

# Load bad ngrams
print("Loading bad ngrams...")
with open("bad_ngrams.txt", "r", encoding="utf-8") as f:
    bad_ngrams = set(line.strip() for line in f if line.strip())
print(f"Loaded {len(bad_ngrams)} bad n-grams.")

sentence_end = re.compile(r'(?<=[.!?])\s+')

def clean_text(text):
    if not isinstance(text, str):
        return ""
    sentences = sentence_end.split(text)
    cleaned = []
    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        s_lower = s_clean.lower()
        words = s_lower.split()
        
        # Extract ngrams of length 1 to 4 from this sentence
        s_ngrams = set()
        for n in range(1, 5):
            for i in range(len(words) - n + 1):
                s_ngrams.add(" ".join(words[i:i+n]))
                
        if s_ngrams.intersection(bad_ngrams):
            continue
            
        cleaned.append(s_clean)
    return " ".join(cleaned)

print("Cleaning texts...")
df["text_cleaned"] = df["text"].apply(clean_text)
df["word_count"] = df["text_cleaned"].apply(lambda t: len(str(t).split()))

# Filter out rows with fewer than 10 words
df_filtered = df[df["word_count"] >= 10].copy()
print(f"Rows after filtering length < 10: {len(df_filtered):,}")

# Align length distributions
bins = list(range(10, 150, 10)) + [150, 200, 10000]
df_filtered["len_bin"] = pd.cut(df_filtered["word_count"], bins=bins)

grouped = df_filtered.groupby(["len_bin", "label"], observed=False).size().unstack(fill_value=0)
print("\n--- Rows per length bucket by class before matching ---")
print(grouped)

matched_indices = []
for len_bin, group in df_filtered.groupby("len_bin", observed=False):
    human_subset = group[group["label"] == 0]
    ai_subset = group[group["label"] == 1]
    
    n_match = min(len(human_subset), len(ai_subset))
    if n_match > 0:
        matched_indices.extend(human_subset.sample(n=n_match, random_state=42).index)
        matched_indices.extend(ai_subset.sample(n=n_match, random_state=42).index)

df_matched = df_filtered.loc[matched_indices].reset_index(drop=True)
print(f"\nTotal matched rows: {len(df_matched):,}")

matched_grouped = df_matched.groupby(["len_bin", "label"], observed=False).size().unstack(fill_value=0)
print("\n--- Rows per length bucket after matching ---")
print(matched_grouped)
