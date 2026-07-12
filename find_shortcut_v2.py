import pandas as pd
import re
from collections import Counter

print("Loading Data/train.csv...")
df = pd.read_csv("Data/train.csv")
print(f"Loaded {len(df):,} rows.")

human_texts = df[df["label"] == 0]["text"].astype(str).tolist()
ai_texts = df[df["label"] == 1]["text"].astype(str).tolist()

print(f"Human texts: {len(human_texts):,}")
print(f"AI texts: {len(ai_texts):,}")

# 1. Check quote styles and other characters
def get_char_stats(texts):
    chars = ["„", "“", "”", "«", "»", "\"", "'", "`", "’", "–", "—", "[", "]", "(", ")", "§", "%", "&"]
    counts = Counter()
    for text in texts:
        for c in chars:
            if c in text:
                counts[c] += 1
    return counts

print("\n--- Special Character Prevalence (percentage of rows containing char) ---")
human_chars = get_char_stats(human_texts)
ai_chars = get_char_stats(ai_texts)

for c in sorted(set(list(human_chars.keys()) + list(ai_chars.keys()))):
    h_pct = human_chars[c] / len(human_texts) * 100
    a_pct = ai_chars[c] / len(ai_texts) * 100
    print(f"Char: {c:5s} | Human: {h_pct:6.2f}% | AI: {a_pct:6.2f}% | Diff: {abs(h_pct-a_pct):6.2f}%")

# 2. Analyze word n-grams (1-gram and 2-gram)
def get_ngram_stats(texts, n=1):
    counts = Counter()
    for text in texts[:50000]:  # Sample for speed
        words = str(text).lower().split()
        if n == 1:
            for w in set(words):
                counts[w] += 1
        elif n == 2:
            ngrams = set(zip(words[:-1], words[1:]))
            for ng in ngrams:
                counts[" ".join(ng)] += 1
    return counts

sample_size = min(len(human_texts), 50000)

print("\n--- Word Unigrams (Document Frequency in first 50k rows) ---")
h_unigrams = get_ngram_stats(human_texts, n=1)
a_unigrams = get_ngram_stats(ai_texts, n=1)

all_unigrams = set(list(h_unigrams.keys()) + list(a_unigrams.keys()))
unigram_diffs = []

for ug in all_unigrams:
    h_count = h_unigrams[ug]
    a_count = a_unigrams[ug]
    h_pct = h_count / sample_size * 100
    a_pct = a_count / sample_size * 100
    diff = abs(h_pct - a_pct)
    if h_count + a_count > 100:  # Ignore rare words
        unigram_diffs.append((ug, h_pct, a_pct, diff))

unigram_diffs.sort(key=lambda x: x[3], reverse=True)
print(f"{'Unigram':30s} | {'Human %':8s} | {'AI %':8s} | {'Diff %':8s}")
print("-" * 65)
for ug, hp, ap, d in unigram_diffs[:30]:
    print(f"{ug:30s} | {hp:7.2f}% | {ap:7.2f}% | {d:7.2f}%")

print("\n--- Word Bigrams (Document Frequency in first 50k rows) ---")
h_bigrams = get_ngram_stats(human_texts, n=2)
a_bigrams = get_ngram_stats(ai_texts, n=2)

all_bigrams = set(list(h_bigrams.keys()) + list(a_bigrams.keys()))
bigram_diffs = []

for bg in all_bigrams:
    h_count = h_bigrams[bg]
    a_count = a_bigrams[bg]
    h_pct = h_count / sample_size * 100
    a_pct = a_count / sample_size * 100
    diff = abs(h_pct - a_pct)
    if h_count + a_count > 100:
        bigram_diffs.append((bg, h_pct, a_pct, diff))

bigram_diffs.sort(key=lambda x: x[3], reverse=True)
print(f"{'Bigram':30s} | {'Human %':8s} | {'AI %':8s} | {'Diff %':8s}")
print("-" * 65)
for bg, hp, ap, d in bigram_diffs[:30]:
    print(f"{bg:30s} | {hp:7.2f}% | {ap:7.2f}% | {d:7.2f}%")
