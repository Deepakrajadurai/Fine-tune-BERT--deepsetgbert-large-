import pandas as pd
import re
from collections import Counter

print("Loading Data/Final/training_pair_v5_clean.csv with UTF-8 encoding...")
df = pd.read_csv("Data/Final/training_pair_v5_clean.csv", encoding="utf-8")
print(f"Loaded {len(df):,} rows.")

ai_texts = df[df["label"] == 1]["text"].astype(str).tolist()

print("Splitting AI texts into sentences...")
sentence_end = re.compile(r'(?<=[.!?])\s+')
sentence_counter = Counter()

for text in ai_texts:
    sentences = sentence_end.split(text)
    for s in sentences:
        s_clean = s.strip()
        if s_clean:
            # abstract proper noun topics or name variations
            s_abs = re.sub(r'^[A-ZÄÖÜ][a-zäöüß]+(?:\s+und\s+[A-ZÄÖÜ][a-zäöüß]+)?\s+ist\s+ein\s+Thema', '[TOPIC] ist ein Thema', s_clean)
            s_abs = re.sub(r'^[A-ZÄÖÜ][a-zäöüß]+(?:\s+und\s+[A-ZÄÖÜ][a-zäöüß]+)?\s+sind\s+ein\s+Thema', '[TOPIC] sind ein Thema', s_abs)
            s_abs = re.sub(r'^[A-ZÄÖÜ][a-zäöüß]+(?:\s+und\s+[A-ZÄÖÜ][a-zäöüß]+)?\s+ist\s+ein\s+ Politikfeld', '[TOPIC] ist ein Politikfeld', s_abs)
            s_abs = re.sub(r'^[A-ZÄÖÜ][a-zäöüß]+(?:\s+und\s+[A-ZÄÖÜ][a-zäöüß]+)?\s+sind\s+ein\s+ Politikfeld', '[TOPIC] sind ein Politikfeld', s_abs)
            sentence_counter[s_abs] += 1

print("\nIdentifying templates occurring > 500 times...")
bad_templates = []
for sent, count in sentence_counter.most_common():
    if count > 500:
        bad_templates.append((sent, count))

print(f"Found {len(bad_templates)} bad templates.")
print("\nTop 30 bad templates:")
for sent, count in bad_templates[:30]:
    print(f"Count: {count:5d} | {sent}")

# Save the templates to a text file for cleaning
with open("bad_templates.txt", "w", encoding="utf-8") as f:
    for sent, count in bad_templates:
        f.write(sent + "\n")
print("\nSaved bad templates to bad_templates.txt")
