import pandas as pd
import re
from collections import Counter

print("Loading Data/Final/training_pair_v5_clean.csv...")
df = pd.read_csv("Data/Final/training_pair_v5_clean.csv", encoding="utf-8")
print(f"Loaded {len(df):,} rows.")

# Define regex patterns for template shortcuts (expanded version)
patterns = [
    # 1. Topic/Thema template phrases
    r'\b(?:ist|sind|war|wäre)\s+(?:ein\s+)?Thema,\s*(?:das|zu\s+dem|über\s+das|welches)\s+(?:in\s+der\s+)?Öffentlichkeit\s+intensiv\s+diskutiert\s+wird',
    r'\b(?:ist|sind|war|wäre)\s+(?:ein\s+)?Thema,\s*(?:das\s+uns\s+alle\s+angeht|zu\s+dem\s+ich\s+mich\s+äußern\s+möchte)',
    r'\b(?:ist|sind|war|wäre)\s+(?:ein\s+)?Thema\b',
    r'\b(?:ist|sind|war|wäre)\s+(?:ein\s+)?Politikfeld\b',
    
    # 2. Proposed measures
    r'\b(?:die\s+)?vorgeschlagenen\s+Maßnahmen\b',
    r'\b(?:die\s+)?vorgeschlagenen\s+(?:Änderungen|Regelungen|Vorschläge)\b',
    r'\bder\s+einzelnen\s+Regelungsgegenstände\b',
    r'\bRegelungen\s+zur\b',
    
    # 3. "werden einwenden, dass"
    r'\bwerden\s+einwenden,\s+dass\b',
    r'\bwerden\s+einwenden\b',
    
    # 4. "dazu klar positioniert"
    r'\b(?:hat\s+sich\s+)?dazu\s+klar\s+positioniert\b',
    r'\böffentlichkeit\s+intensiv\s+diskutiert\b',
    
    # 5. Administrative/legal templates
    r'\bTräger\s+der\s+Maßnahme\b',
    r'\bnach\s+pflichtgemäßem\s+Ermessen\b',
    r'\bgesetzlichen\s+Grenzen\s+des\s+Ermessens\b',
    r'\bDie\s+Beteiligten\s+sind\s+vor\s+der\s+Entscheidung\s+anzuhören\b',
    r'\bDie\s+Durchführung\s+obliegt\s+den\s+zuständigen\s+Landesbehörden\b',
    r'\bDie\s+Vorschriften\s+dieses\s+Gesetzes\s+sind\s+zu\s+beachten\b',
    r'\bErmessensentscheidungen\s+sind\s+zu\s+begründen\b',
    r'\bDie\s+Kosten\s+trägt\s+der\s+Veranlasser\b',
    r'\bDie\s+Finanzierung\s+ist\s+im\s+Bundeshaushalt\s+vorgesehen\b',
    r'\bDoch\s+dieser\s+Einwand\s+greift\s+zu\s+kurz\b',
    r'\bWir\s+empfehlen\s+daher\s+eine\s+differenzierte\s+Betrachtung\b',
    
    # 6. Law intro templates
    r'\bBegründung\s+zum\s+Gesetzentwurf\s+zur\b',
    r'\bBegründung\s+zum\s+Gesetzentwurf\b',
    r'\bA\.\s+Allgemeiner\s+Teil\b',
    r'\bZiel\s+des\s+Gesetzentwurfs\s+ist\s+es,\s+die\s+Rechtslage\b',
    r'\bzeitgemäß\s+zu\s+gestalten\b',
    r'\bDie\s+derzeitige\s+Rechtslage\s+weist\b',
    r'\bDie\s+derzeitige\s+Rechtslage\b',
    
    # 7. List item patterns
    r'\b(?:1|2|3|4|5)\.\s+(?:Ausgangslage|Zielsetzung|Alternativen|Kosten|Finanzierung)\b',
    r'\bB\.\s+Besonderer\s+Teil\b',
    r'\bBesonderer\s+Teil\b',
    
    # 8. New legal sub-templates
    r'\bNotwendigkeit\s+ergibt\s+sich\s+aus\b',
    r'\b§\s+\d+:\s*(?:die|der|das)?\b',
    r'\b§+\s*\d+\b',
    r'\bder\s+Verantwortliche\b',
    r'\bder\s+Auftragnehmer\b',
    r'\bder\s+Antragsteller\b',
    r'\bbetroffene\s+Person\b',
    r'\bzuständige\s+Behörde\b',
    r'\bzu\s+beachten,\b',
    r'\b\(\d+\)\b'
]

compiled_patterns = [re.compile(p, re.IGNORECASE) for p in patterns]

def regex_clean(text):
    if not isinstance(text, str):
        return ""
    
    # Apply regex cleaning
    t = text
    for p in compiled_patterns:
        t = p.sub("", t)
        
    # Collapse double spaces
    t = re.sub(r'\s+', ' ', t)
    t = re.sub(r'\s+([.,;:!?])', r'\1', t)
    return t.strip()

print("Applying regex cleaning...")
df["text_cleaned"] = df["text"].apply(regex_clean)
df["word_count"] = df["text_cleaned"].apply(lambda t: len(str(t).split()))

# Filter out very short rows
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

# Let's check ngram statistics of the matched output to see if the top leaks are gone
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
print("\n--- Top 30 Bigram Differences After Expanded Regex Cleaning ---")
print(f"{'Bigram':30s} | {'Human %':8s} | {'AI %':8s} | {'Diff %':8s}")
print("-" * 65)
for bg, hp, ap, d in bigram_diffs[:30]:
    print(f"{bg:30s} | {hp:7.2f}% | {ap:7.2f}% | {d:7.2f}%")
