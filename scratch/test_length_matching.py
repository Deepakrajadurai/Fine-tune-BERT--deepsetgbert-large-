import pandas as pd
import re
import numpy as np

print("Loading Data/Final/training_pair_v5_clean.csv...")
df = pd.read_csv("Data/Final/training_pair_v5_clean.csv")
print(f"Loaded {len(df):,} rows.")

# Load bad templates
print("Loading bad templates...")
with open("bad_templates.txt", "r", encoding="utf-8") as f:
    bad_templates = set(line.strip() for line in f if line.strip())
print(f"Loaded {len(bad_templates)} bad templates.")

# Abstraction function
def abstract_sentence(s):
    s_abs = re.sub(r'^[A-ZÄÖÜ][a-zäöüß]+(?:\s+und\s+[A-ZÄÖÜ][a-zäöüß]+)?\s+ist\s+ein\s+Thema', '[TOPIC] ist ein Thema', s)
    s_abs = re.sub(r'^[A-ZÄÖÜ][a-zäöüß]+(?:\s+und\s+[A-ZÄÖÜ][a-zäöüß]+)?\s+sind\s+ein\s+Thema', '[TOPIC] sind ein Thema', s_abs)
    s_abs = re.sub(r'^[A-ZÄÖÜ][a-zäöüß]+(?:\s+und\s+[A-ZÄÖÜ][a-zäöüß]+)?\s+ist\s+ein\s+ Politikfeld', '[TOPIC] ist ein Politikfeld', s_abs)
    s_abs = re.sub(r'^[A-ZÄÖÜ][a-zäöüß]+(?:\s+und\s+[A-ZÄÖÜ][a-zäöüß]+)?\s+sind\s+ein\s+ Politikfeld', '[TOPIC] sind ein Politikfeld', s_abs)
    return s_abs

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
        s_abs = abstract_sentence(s_clean)
        if s_abs in bad_templates:
            continue
        cleaned.append(s_clean)
    return " ".join(cleaned)

print("Cleaning texts...")
df["text_cleaned"] = df["text"].apply(clean_text)
df["word_count"] = df["text_cleaned"].apply(lambda t: len(str(t).split()))

# Filter out rows with fewer than 10 words
df_filtered = df[df["word_count"] >= 10].copy()
print(f"Rows after filtering length < 10: {len(df_filtered):,}")

# Let's define word count bins (buckets)
# We can use bins of size 10 words up to 150 words, then 150-200, 200+
bins = list(range(10, 150, 10)) + [150, 200, 10000]
df_filtered["len_bin"] = pd.cut(df_filtered["word_count"], bins=bins)

# Group by label and len_bin
grouped = df_filtered.groupby(["len_bin", "label"], observed=False).size().unstack(fill_value=0)
print("\n--- Rows per length bucket by class before matching ---")
print(grouped)

# Perform length-stratified downsampling
matched_indices = []
for len_bin, group in df_filtered.groupby("len_bin", observed=False):
    human_subset = group[group["label"] == 0]
    ai_subset = group[group["label"] == 1]
    
    n_match = min(len(human_subset), len(ai_subset))
    if n_match > 0:
        matched_indices.extend(human_subset.sample(n=n_match, random_state=42).index)
        matched_indices.extend(ai_subset.sample(n=n_match, random_state=42).index)

df_matched = df_filtered.loc[matched_indices].reset_index(drop=True)
print(f"\nTotal rows after length-stratified matching: {len(df_matched):,}")

matched_grouped = df_matched.groupby(["len_bin", "label"], observed=False).size().unstack(fill_value=0)
print("\n--- Rows per length bucket after matching ---")
print(matched_grouped)
