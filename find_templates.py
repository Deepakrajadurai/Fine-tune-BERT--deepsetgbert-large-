import pandas as pd
import re
from collections import Counter

print("Loading Data/train.csv...")
df = pd.read_csv("Data/train.csv")
ai_texts = df[df["label"] == 1]["text"].astype(str).tolist()

print(f"Loaded {len(ai_texts):,} AI texts.")

sentence_counter = Counter()

# Regex to split into sentences
sentence_end = re.compile(r'(?<=[.!?])\s+')

for text in ai_texts[:100000]:  # Look at first 100k rows
    sentences = sentence_end.split(text)
    for s in sentences:
        s_clean = s.strip()
        if s_clean:
            # abstract out specific topics or names to find the general templates
            # e.g., replace known topics with [TOPIC]
            s_templated = s_clean
            # We can abstract out anything that looks like "X ist ein Thema" or starts with a topic
            # For a basic analysis, let's look at exact sentences first
            sentence_counter[s_clean] += 1

print("\n--- Top 50 Exact AI Sentences (Most Frequent) ---")
for sent, count in sentence_counter.most_common(50):
    print(f"Count: {count:5d} | {sent}")

# Now, let's find general patterns by replacing words that vary
abstract_counter = Counter()
for sent, count in sentence_counter.items():
    # Replace uppercase nouns at start of "ist ein Thema"
    # e.g. "Klimaschutzgesetz ist ein Thema" -> "[PROPER_NOUN] ist ein Thema"
    abstracted = re.sub(r'^[A-ZÄÖÜ][a-zäöüß]+(?:\s+und\s+[A-ZÄÖÜ][a-zäöüß]+)?\s+ist\s+ein\s+Thema', '[TOPIC] ist ein Thema', sent)
    abstracted = re.sub(r'^[A-ZÄÖÜ][a-zäöüß]+(?:\s+und\s+[A-ZÄÖÜ][a-zäöüß]+)?\s+sind\s+ein\s+Thema', '[TOPIC] sind ein Thema', abstracted)
    abstract_counter[abstracted] += count

print("\n--- Top 50 Abstracted AI Sentences ---")
for sent, count in abstract_counter.most_common(50):
    print(f"Count: {count:5d} | {sent}")
