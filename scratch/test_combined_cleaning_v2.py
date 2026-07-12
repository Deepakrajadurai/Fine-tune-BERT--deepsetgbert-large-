import pandas as pd
import re
from collections import Counter

print("Loading Data/Final/training_pair_v5_clean.csv...")
df = pd.read_csv("Data/Final/training_pair_v5_clean.csv", encoding="utf-8")
print(f"Loaded {len(df):,} rows.")

# 1. Load full sentence templates from bad_templates.txt
print("Loading bad templates...")
bad_templates = set()
with open("bad_templates.txt", "r", encoding="utf-8") as f:
    for line in f:
        t_clean = line.strip()
        if t_clean:
            bad_templates.add(t_clean.lower())
print(f"Loaded {len(bad_templates)} bad templates.")

# 2. Define sub-sentence trigger patterns
template_phrases = [
    r"Defizite auf",
    r"grundsätzlich zu begrüßen", r"grundsätzlich zu begrüssen",
    r"differenzierte Betrachtung",
    r"ist ein Thema", r"sind ein Thema", r"war ein Thema", r"wäre ein Thema",
    r"ist ein Politikfeld", r"sind ein Politikfeld",
    r"kritisch anzumerken",
    r"hätte jedoch zur Folge",
    r"Als Alternative kommt",
    r"Träger der Maßnahme", r"Träger der Massnahme",
    r"pflichtgemäßem Ermessen", r"pflichtgemässem Ermessen",
    r"Grenzen des Ermessens",
    r"vor der Entscheidung anzuhören",
    r"obliegt den zuständigen Landesbehörden",
    r"Vorschriften dieses Gesetzes sind zu beachten",
    r"Ermessensentscheidungen sind zu begründen",
    r"Kosten trägt der Veranlasser",
    r"im Bundeshaushalt vorgesehen",
    r"Einwand greift zu kurz",
    r"Begründung zum Gesetzentwurf",
    r"Allgemeiner Teil",
    r"Besonderer Teil",
    r"zeitgemäß zu gestalten", r"zeitgemäss zu gestalten",
    r"Finanzielle Auswirkungen",
    r"Erfüllungsaufwand",
    r"Vollzug obliegt den Ländern",
    r"Ich danke Ihnen für die Aussprache",
    r"bitte um Zustimmung",
    r"Koalition hat sich viel vorgenommen",
    r"ob sie Wort hält",
    r"vorgeschlagenen", r"vorgelegten",
    r"hoffe auf", r"ich hoffe",
    r"aufmerksamkeit",
    r"werde einwenden", r"werden einwenden",
    r"möchte mich", r"möchte betonen",
    r"Maximum des politisch Machbaren",
    r"schwierigen Verhandlungen",
    r"deutlich gemacht",
    r"Anhörung überzeugende",
    r"Ressourcen scheitern",
    
    # Structural headers
    r"Analyse:", r"Empfehlung", r"Ausgangslage", r"Alternativen", r"Bewertung",
    
    # Regex legal templates
    r"Notwendigkeit\s+ergibt\s+sich\s+aus",
    r"§+\s*\d+:\s*(?:die|der|das)?",
    r"§+\s*\d+",
    r"der\s+Verantwortliche",
    r"der\s+Auftragnehmer",
    r"der\s+Antragsteller",
    r"betroffene\s+Person",
    r"zuständige\s+Behörde",
    r"zu\s+beachten,",
    r"\(\d+\)"
]

compiled_phrases = [re.compile(p, re.IGNORECASE) for p in template_phrases]
sentence_end = re.compile(r'(?<=[.!?])\s+')
list_header_re = re.compile(r'^\d+\.\s*(?:[A-ZÄÖÜ]|$)', re.IGNORECASE)

# Regex to match transition prefixes like "Wichtig ist in diesem Kontext: ", "Daraus folgt unmittelbar: ", etc.
prefix_re = re.compile(r'^[A-Za-z0-9ÄÖÜäöüß\s,-]+\s*:\s*')

def abstract_sentence(s):
    # Abstract proper nouns and names to template placeholders
    s_abs = re.sub(r'^[A-ZÄÖÜ][a-zäöüß]+(?:\s+und\s+[A-ZÄÖÜ][a-zäöüß]+)?\s+ist\s+ein\s+Thema', '[TOPIC] ist ein Thema', s)
    s_abs = re.sub(r'^[A-ZÄÖÜ][a-zäöüß]+(?:\s+und\s+[A-ZÄÖÜ][a-zäöüß]+)?\s+sind\s+ein\s+Thema', '[TOPIC] sind ein Thema', s_abs)
    s_abs = re.sub(r'^[A-ZÄÖÜ][a-zäöüß]+(?:\s+und\s+[A-ZÄÖÜ][a-zäöüß]+)?\s+ist\s+ein\s+Politikfeld', '[TOPIC] ist ein Politikfeld', s_abs)
    s_abs = re.sub(r'^[A-ZÄÖÜ][a-zäöüß]+(?:\s+und\s+[A-ZÄÖÜ][a-zäöüß]+)?\s+sind\s+ein\s+Politikfeld', '[TOPIC] sind ein Politikfeld', s_abs)
    return s_abs.lower()

def clean_text_combined(text):
    if not isinstance(text, str):
        return ""
    sentences = sentence_end.split(text)
    cleaned = []
    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
            
        # 1. Filter out list markers
        if list_header_re.match(s_clean):
            continue
        if re.match(r'^[A-Za-z0-9]\.$', s_clean):
            continue
            
        # 2. Strip colon transition prefixes if present
        # e.g., "Wichtig ist in diesem Kontext: Der vorliegende..." -> "Der vorliegende..."
        s_stripped = prefix_re.sub("", s_clean).strip()
        if not s_stripped:
            continue
            
        # 3. Check full-sentence template blacklist (on both original and stripped sentence)
        s_abs = abstract_sentence(s_stripped)
        s_abs_orig = abstract_sentence(s_clean)
        if s_abs in bad_templates or s_abs_orig in bad_templates:
            continue
            
        # 4. Check sub-sentence trigger phrases (on both original and stripped sentence)
        is_bad = False
        for p in compiled_phrases:
            if p.search(s_clean) or p.search(s_stripped):
                is_bad = True
                break
        if is_bad:
            continue
            
        cleaned.append(s_stripped)
    return " ".join(cleaned)

print("Applying combined cleaning...")
df["text_cleaned"] = df["text"].apply(clean_text_combined)
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

# Check remaining bigrams
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
print("\n--- Top 30 Bigram Differences After Combined Cleaning (v2) ---")
print(f"{'Bigram':30s} | {'Human %':8s} | {'AI %':8s} | {'Diff %':8s}")
print("-" * 65)
for bg, hp, ap, d in bigram_diffs[:30]:
    print(f"{bg:30s} | {hp:7.2f}% | {ap:7.2f}% | {d:7.2f}%")
