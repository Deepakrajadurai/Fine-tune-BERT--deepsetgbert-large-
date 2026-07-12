import pandas as pd
from collections import Counter
import re

print("Loading Data/Final/training_pair_v5_clean.csv...")
df = pd.read_csv("Data/Final/training_pair_v5_clean.csv", encoding="utf-8")
print(f"Loaded {len(df):,} rows.")

words_with_replacement = Counter()

for text in df["text"].astype(str):
    if "\ufffd" in text:
        words = text.split()
        for w in words:
            if "\ufffd" in w:
                w_clean = re.sub(r'^[^\w\ufffd]+|[^\w\ufffd]+$', '', w.lower())
                if "\ufffd" in w_clean:
                    words_with_replacement[w_clean] += 1

print(f"Found {len(words_with_replacement)} unique corrupted words.")

# Save to a file
with open("scratch/corrupted_words.txt", "w", encoding="utf-8") as f:
    f.write(f"Total unique corrupted words: {len(words_with_replacement)}\n\n")
    for w, count in words_with_replacement.most_common(500):
        # Escape the replacement character so it's readable
        w_escaped = w.replace("\ufffd", "[U+FFFD]")
        f.write(f"Count: {count:5d} | {w_escaped}\n")

print("Saved report to scratch/corrupted_words.txt")
