import pandas as pd
import re
import numpy as np

print("Loading Data/Final/training_pair_v5_clean.csv with UTF-8 encoding...")
df = pd.read_csv("Data/Final/training_pair_v5_clean.csv", encoding="utf-8")
print(f"Loaded {len(df):,} rows.")

# Specific template phrases to trigger entire sentence discard
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
    
    # Structural headers
    r"Analyse:", r"Empfehlung", r"Ausgangslage", r"Alternativen", r"Bewertung",
    
    # Regex legal templates (no \b near special characters)
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

# Compile all trigger patterns as regexes
compiled_phrases = [re.compile(p, re.IGNORECASE) for p in template_phrases]
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
        
        # Check if this sentence matches any of the bad patterns
        is_bad = False
        for p in compiled_phrases:
            if p.search(s_clean):
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
print("Filtering very short rows (< 10 words)...")
df_filtered = df[df["word_count"] >= 10].copy()

# Define length bins to align distributions
bins = list(range(10, 150, 10)) + [150, 200, 10000]
df_filtered["len_bin"] = pd.cut(df_filtered["word_count"], bins=bins)

# Perform length-stratified downsampling
print("Performing length-stratified downsampling...")
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

# Shuffle the matched dataset
df_export = df_matched.sample(frac=1, random_state=42).reset_index(drop=True)
df_export["text"] = df_export["text_cleaned"]

# Keep correct original columns
cols = ["id", "text", "label", "domain", "meta"]
df_export = df_export[[c for c in cols if c in df_export.columns]]

df_export.to_csv("Data/Final/training_pair_v5_super_clean.csv", index=False, encoding="utf-8")
print(f"Exported balanced clean dataset to Data/Final/training_pair_v5_super_clean.csv: {len(df_export):,} rows total.")
