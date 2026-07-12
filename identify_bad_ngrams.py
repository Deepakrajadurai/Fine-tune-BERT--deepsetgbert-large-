import pandas as pd
import re
from collections import Counter
from tqdm import tqdm

print("Loading Data/Final/training_pair_v5_clean.csv...")
df = pd.read_csv("Data/Final/training_pair_v5_clean.csv", encoding="utf-8")
print(f"Loaded {len(df):,} rows.")

human_texts = df[df["label"] == 0]["text"].astype(str).tolist()
ai_texts = df[df["label"] == 1]["text"].astype(str).tolist()

def get_ngram_doc_freq(texts, max_n=4):
    # Track document frequency (number of docs containing the ngram)
    counts = Counter()
    for text in tqdm(texts[:100000], desc="Extracting ngrams"):  # sample 100k for speed
        words = str(text).lower().split()
        seen = set()
        
        # 1-grams
        for w in words:
            seen.add(w)
            
        # 2, 3, 4-grams
        for n in range(2, max_n + 1):
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i:i+n])
                seen.add(ngram)
                
        for ngram in seen:
            counts[ngram] += 1
            
    return counts

print("\nProcessing Human texts...")
h_ngrams = get_ngram_doc_freq(human_texts)
print("\nProcessing AI texts...")
a_ngrams = get_ngram_doc_freq(ai_texts)

print("\nFiltering for shortcut n-grams...")
sample_size = min(len(human_texts), 100000)

bad_ngrams = []
for ngram, a_count in a_ngrams.items():
    a_pct = a_count / sample_size * 100
    h_count = h_ngrams.get(ngram, 0)
    h_pct = h_count / sample_size * 100
    
    # If the ngram appears in > 1.5% of AI texts but < 0.1% of human texts
    if a_pct > 1.5 and h_pct < 0.1:
        bad_ngrams.append((ngram, a_pct, h_pct))

# Sort by AI frequency descending
bad_ngrams.sort(key=lambda x: x[1], reverse=True)

print(f"\nFound {len(bad_ngrams)} bad n-grams.")
print("\nTop 50 bad n-grams:")
for ngram, ap, hp in bad_ngrams[:50]:
    print(f"AI: {ap:5.2f}% | Human: {hp:5.2f}% | {ngram}")

# Save bad ngrams
with open("bad_ngrams.txt", "w", encoding="utf-8") as f:
    for ngram, ap, hp in bad_ngrams:
        f.write(f"{ngram}\n")
print("\nSaved bad n-grams to bad_ngrams.txt")
