import pandas as pd
import re
from collections import Counter

print("Loading Data/Final/training_pair_v5_clean.csv...")
df = pd.read_csv("Data/Final/training_pair_v5_clean.csv", encoding="utf-8")
print(f"Loaded {len(df):,} rows.")

# Specific template phrases to trigger entire sentence discard
template_phrases = [
    "Defizite auf",
    "grundsätzlich zu begrüßen", "grundsätzlich zu begrüssen",
    "differenzierte Betrachtung",
    "ist ein Thema", "sind ein Thema", "war ein Thema", "wäre ein Thema",
    "ist ein Politikfeld", "sind ein Politikfeld",
    "kritisch anzumerken",
    "hätte jedoch zur Folge", "hätte jedoch zur folge",
    "Als Alternative kommt", "als alternative kommt",
    "Träger der Maßnahme", "Träger der Massnahme",
    "pflichtgemäßem Ermessen", "pflichtgemässem Ermessen",
    "Grenzen des Ermessens",
    "vor der Entscheidung anzuhören",
    "obliegt den zuständigen Landesbehörden",
    "Vorschriften dieses Gesetzes sind zu beachten",
    "Ermessensentscheidungen sind zu begründen",
    "Kosten trägt der Veranlasser",
    "im Bundeshaushalt vorgesehen",
    "Einwand greift zu kurz",
    "Begründung zum Gesetzentwurf",
    "Allgemeiner Teil",
    "Besonderer Teil",
    "zeitgemäß zu gestalten", "zeitgemäss zu gestalten",
    "Finanzielle Auswirkungen",
    "Erfüllungsaufwand",
    "Vollzug obliegt den Ländern",
    "Ich danke Ihnen für die Aussprache",
    "bitte um Zustimmung",
    "Koalition hat sich viel vorgenommen",
    "ob sie Wort hält",
    "vorgeschlagenen", "vorgelegten",
    "hoffe auf", "ich hoffe",
    "aufmerksamkeit",
    "werde einwenden", "werden einwenden",
    "möchte mich", "möchte betonen",
    
    # Structural headers
    "Analyse:", "Empfehlung", "Ausgangslage", "Alternativen", "Bewertung"
]

compiled_phrases = [p.lower() for p in template_phrases]
sentence_end = re.compile(r'(?<=[.!?])\s+')

# List header patterns
list_header_re = re.compile(r'^\d+\.\s*(?:[A-ZÄÖÜ]|$)', re.IGNORECASE)

def clean_text_by_sentence(text):
    if not isinstance(text, str):
        return ""
    sentences = sentence_end.split(text)
    cleaned = []
    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
            
        # Check if sentence is just a list marker or header
        if list_header_re.match(s_clean):
            continue
            
        # Check if sentence is very short list numbering like "2." or "B."
        if re.match(r'^[A-Za-z0-9]\.$', s_clean):
            continue
        
        # Check if this sentence contains any of the bad phrases (case insensitive)
        s_lower = s_clean.lower()
        is_bad = False
        for p in compiled_phrases:
            if p in s_lower:
                is_bad = True
                break
                
        if is_bad:
            continue
            
        cleaned.append(s_clean)
        
    return " ".join(cleaned)

print("Applying sentence discarding...")
df["text_cleaned"] = df["text"].apply(clean_text_by_sentence)
df["word_count"] = df["text_cleaned"].apply(lambda t: len(str(t).split()))

# Filter out short rows
df_filtered = df[df["word_count"] >= 10].copy()
print(f"Rows after filtering < 10 words: {len(df_filtered):,}")

# Match length distributions
bins = list(range(10, 150, 10)) + [150, 200, 10000]
df_filtered["len_bin"] = pd.cut(df_filtered["word_count"], bins=bins)

matched_indices = []
for len_bin, group in df_filtered.groupby("len_bin", observed=False):
    human_subset = group[group["label"] == 0]
    ai_subset = group[group["label"] == 1]
    
    n_match = min(len(human_subset), len(ai_subset))
    if n_match > 0:
        matched_indices.extend(human_subset.sample(n=n_match, random_state=42).index)
        matched_indices.extend(ai_subset.sample(n=n_match, random_state=42).index)

df_matched = df_filtered.loc[matched_indices].reset_index(drop=True)
print(f"Total matched rows: {len(df_matched):,}")

# Let's check ngram statistics of the matched output
human_texts = df_matched[df_matched["label"] == 0]["text_cleaned"].astype(str).tolist()
ai_texts = df_matched[df_matched["label"] == 1]["text_cleaned"].astype(str).tolist()

def get_ngram_stats(texts, n=2):
    counts = Counter()
    for text in texts[:30000]:
        words = str(text).lower().split()
        if n == 2:
            ngrams = set(zip(words[:-1], words[1:]))
            for ng in ngrams:
                counts[" ".join(ng)] += 1
    return counts

sample_size = min(len(human_texts), 30000)
h_bigrams = get_ngram_stats(human_texts, n=2)
a_bigrams = get_ngram_stats(ai_texts, n=2)

all_bigrams = set(list(h_bigrams.keys()) + list(a_bigrams.keys()))
bigram_diffs = []
for bg in all_bigrams:
    h_pct = h_bigrams.get(bg, 0) / sample_size * 100
    a_pct = a_bigrams.get(bg, 0) / sample_size * 100
    diff = abs(h_pct - a_pct)
    if h_bigrams.get(bg, 0) + a_bigrams.get(bg, 0) > 50:
        bigram_diffs.append((bg, h_pct, a_pct, diff))

bigram_diffs.sort(key=lambda x: x[3], reverse=True)
print("\n--- Top 30 Bigram Differences After Sentence Discarding (v3) ---")
print(f"{'Bigram':30s} | {'Human %':8s} | {'AI %':8s} | {'Diff %':8s}")
print("-" * 65)
for bg, hp, ap, d in bigram_diffs[:30]:
    print(f"{bg:30s} | {hp:7.2f}% | {ap:7.2f}% | {d:7.2f}%")
