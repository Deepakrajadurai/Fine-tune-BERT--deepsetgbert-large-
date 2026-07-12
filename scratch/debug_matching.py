import pandas as pd
import re

df = pd.read_csv("Data/Final/training_pair_v5_clean.csv", encoding="utf-8")

bad_templates = set()
with open("bad_templates.txt", "r", encoding="utf-8") as f:
    for line in f:
        t_clean = line.strip()
        if t_clean:
            bad_templates.add(t_clean.lower())

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
prefix_re = re.compile(r'^.*?\s*:\s*')

def abstract_sentence(s):
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
        if list_header_re.match(s_clean):
            continue
        if re.match(r'^[A-Za-z0-9]\.$', s_clean):
            continue
        s_stripped = prefix_re.sub("", s_clean).strip()
        if not s_stripped:
            continue
        s_abs = abstract_sentence(s_stripped)
        s_abs_orig = abstract_sentence(s_clean)
        if s_abs in bad_templates or s_abs_orig in bad_templates:
            continue
        is_bad = False
        for p in compiled_phrases:
            if p.search(s_clean) or p.search(s_stripped):
                is_bad = True
                break
        if is_bad:
            continue
        cleaned.append(s_stripped)
    return " ".join(cleaned)

print("Running cleaning on full dataset...")
df["text_cleaned"] = df["text"].apply(clean_text_combined)
df["word_count"] = df["text_cleaned"].apply(lambda t: len(str(t).split()))

df_filtered = df[df["word_count"] >= 10].copy()
print(f"Class distribution after filtering:\n{df_filtered['label'].value_counts()}")

# Check length bins
bins = list(range(10, 150, 10)) + [150, 200, 10000]
df_filtered["len_bin"] = pd.cut(df_filtered["word_count"], bins=bins)

grouped = df_filtered.groupby(["len_bin", "label"], observed=False).size().unstack(fill_value=0)
print("\n--- Rows per length bucket by class ---")
print(grouped)
