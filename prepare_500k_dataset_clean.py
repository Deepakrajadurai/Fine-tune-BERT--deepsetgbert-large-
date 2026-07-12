import os
import re
import hashlib
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="[%H:%M:%S]",
)
log = logging.getLogger(__name__)

# Config
MIN_WORDS = 10
RANDOM_SEED = 42
OUTPUT_DIR = Path("Data")

# Regex patterns to strip (case-insensitive).
# Replacing these with a space in both classes completely removes them as shortcuts.
CLEANING_PATTERNS = [
    # 1. Dates (e.g., am heutigen 11.10.2023 or am heutigen Tag)
    r"am\s+heutigen\s+\d{2}\.\d{2}\.\d{4}",
    r"am\s+heutigen\s+Tag",
    r"am\s+heutigen",
    r"heutigen\s+Tag",
    r"heutigen",
    r"aktuellen\s+Lage",
    r"aktuellen",

    # 2. Party Names (with or without parentheses, including compound names)
    r"\((?:CDU(?:/CSU)?|SPD|Grüne|FDP|AfD|Linke|BSW|CSU|Volt|ÖDP|Freie\s+Wähler)\)",
    r"\b(?:CDU/CSU|CDU|CSU|SPD|Grüne|FDP|AfD|Linke|BSW|Volt|ÖDP|Freie\s+Wähler)\b",

    # 3. Procedural / Legislative Boilerplate
    r"\b(?:\d+\.)?\s*Plenarsitzung\b",
    r"\bDrucksache\b",
    r"\bAktenzeichen\b",
    r"\bAz\.?\b",
    r"\bAbsatz\b",
    r"\bAbs\.?\b",
    r"\bParagraph\b",
    r"§+",
    r"\bRechtsverordnung\b",
    r"\bLandesgesetz\b",
    r"\bBundestag\b",
    r"\bLandtag\b",
    r"\bFraktion\b",
    r"\bAusschuss\b",
    r"\bGesetz(?:es)?\b",
    r"\bGesetzgebung\b",
    r"\bVorschrift(?:en)?\b",

    # 4. Target Names & Speaker Artifacts
    r"(?:auf\s+Initiative\s+von\s+|Abgeordnet(?:em|er|en)\s+)[A-ZÄÖÜ][a-zäöüß]+\s+[A-ZÄÖÜ][a-zäöüß]+",
    r"\b(?:Herr|Frau)\s+[A-ZÄÖÜ][a-zäöüß]+",
    r"\b(?:im\s+Namen\s+der|laut\s+der)\s+Fraktion\b",
    r"\bunter\s+Zeichnung\s+von\b",
    r"\blaut\s+[A-ZÄÖÜ][a-zäöüß]+\s+[A-ZÄÖÜ][a-zäöüß]+\b",
    r"\bvon\s+[A-ZÄÖÜ][a-zäöüß]+\s+[A-ZÄÖÜ][a-zäöüß]+\s+vertretenen\s+Bürger\b",

    # 5. LLM/Synthetic Template Boilerplate & Filler words
    r"\bdirekt\b",
    r"\bbezüglich\b",
    r"\baufgrund\b",
    r"\bAuskunftspflichten\b",
    r"\bim\s+Sinne\s+des\s+Gemeinwohls\b",
    r"\bim\s+internationalen\s+Vergleich\b",
    r"\bunter\s+keinen\s+Umständen\b",
    r"\bin\s+der\s+vorliegenden\s+Fassung\b",
    r"\bvorliegende\s+Entwurf\b",
    r"\bvorliegende\s+Fassung\b",
    r"\bin\s+eigenen\s+Worten\b",
    r"\bLassen\s+Sie\s+uns\b",
    r"\bgemeinsam\s+die\s+Ärmel\s+hochkrempeln\b",
    r"\bzügig\s+auf\s+den\s+Weg\s+bringen\b",
    r"\bverspielt\s+unsere\s+Zukunft\b",
    r"\bins\s+Hintertreffen\s+geraten\b",
    r"\bauf\s+die\s+lange\s+Bank\s+zu\s+schieben\b",
    r"\b Responsibility\s+zu\s+übertragen\b",
    r"\bVerantwortung\s+zu\s+übertragen\b",
    r"\bim\s+Bereich\b",
    r"\bbeim\s+Thema\b",
    r"\bzum\s+Thema\b",
    r"\bzum\s+Bereich\b",
    r"\büber\s+das\s+Thema\b",
    r"\büber\s+den\s+Bereich\b",
]

# Compile patterns
compiled_patterns = [re.compile(p, re.IGNORECASE) for p in CLEANING_PATTERNS]

def strip_artifacts_and_templates(text: str) -> str:
    for pattern in compiled_patterns:
        text = pattern.sub(" ", text)
    # Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def clean_text(text: str) -> str | None:
    if not isinstance(text, str) or not text.strip():
        return None
    
    # 1. Clean the text using template/artifact stripping
    text = strip_artifacts_and_templates(text)
    
    # 2. Strip HTTP links
    text = re.sub(r"http\S+", "", text)
    
    # 3. Collapse whitespace again
    text = re.sub(r"\s+", " ", text).strip()

    # 4. Length check
    words = text.split()
    if len(words) < MIN_WORDS:
        return None
    
    # 5. Ratio of digits check
    if sum(c.isdigit() for c in text) / len(text) > 0.25:
        return None

    # 6. Reject model meta-commentary
    bad_phrases = [
        "als ki", "als sprachmodell", "ich kann leider",
        "gerne helfe ich", "natürlich, hier", "hier sind die",
        "bitte beachten sie", "als assistent",
    ]
    lower = text.lower()
    if any(p in lower for p in bad_phrases):
        return None

    return text

def fingerprint(text: str) -> str:
    normalised = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.md5(normalised.encode()).hexdigest()

def load_and_clean(csv_path: Path, label: int) -> pd.DataFrame:
    log.info(f"Loading and cleaning {csv_path} (label={label})...")
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    df["label"] = label
    log.info(f"  Raw rows: {len(df):,}")
    
    # Clean texts
    df["text"] = df["text"].apply(clean_text)
    df.dropna(subset=["text"], inplace=True)
    log.info(f"  After cleaning/filtering: {len(df):,}")
    
    # Deduplicate within class using fingerprint
    df["_fp"] = df["text"].apply(fingerprint)
    before = len(df)
    df.drop_duplicates(subset=["_fp"], inplace=True)
    df.drop(columns=["_fp"], inplace=True)
    log.info(f"  After exact dedup: {len(df):,} (removed {before - len(df):,})")
    return df.reset_index(drop=True)

def main():
    np.random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Load and clean both datasets
    human_df = load_and_clean(Path("Data/Human_model_ready_dataset.csv"), label=0)
    ai_df = load_and_clean(Path("Data/ai_generated_sentences_500k.csv"), label=1)

    # 2. Cross-class exact duplicate deduplication (ensure AI sentences aren't exactly human sentences)
    log.info("Removing exact duplicates between Human and AI classes...")
    human_fps = set(human_df["text"].apply(fingerprint))
    ai_df["_fp"] = ai_df["text"].apply(fingerprint)
    before = len(ai_df)
    ai_df = ai_df[~ai_df["_fp"].isin(human_fps)]
    ai_df.drop(columns=["_fp"], inplace=True)
    log.info(f"  Removed {before - len(ai_df):,} cross-class duplicates from AI dataset")

    # 3. Perform train/val/test splits (80/10/10) on each class
    log.info("Splitting datasets into train/val/test splits...")
    h_train, h_temp = train_test_split(human_df, test_size=0.20, random_state=RANDOM_SEED)
    h_val, h_test = train_test_split(h_temp, test_size=0.50, random_state=RANDOM_SEED)

    a_train, a_temp = train_test_split(ai_df, test_size=0.20, random_state=RANDOM_SEED)
    a_val, a_test = train_test_split(a_temp, test_size=0.50, random_state=RANDOM_SEED)

    # 4. Subsample to balance the dataset:
    # Train: 400,000 of each class (or max available)
    # Val: 50,000 of each class
    # Test: 50,000 of each class
    train_size = 400_000
    val_size = 50_000
    test_size = 50_000

    actual_train_size = min(train_size, len(h_train), len(a_train))
    actual_val_size = min(val_size, len(h_val), len(a_val))
    actual_test_size = min(test_size, len(h_test), len(a_test))

    log.info(f"Subsampling to balance at {actual_train_size} train, {actual_val_size} val, {actual_test_size} test...")
    
    h_train_sampled = h_train.sample(n=actual_train_size, random_state=RANDOM_SEED)
    a_train_sampled = a_train.sample(n=actual_train_size, random_state=RANDOM_SEED)

    h_val_sampled = h_val.sample(n=actual_val_size, random_state=RANDOM_SEED)
    a_val_sampled = a_val.sample(n=actual_val_size, random_state=RANDOM_SEED)

    h_test_sampled = h_test.sample(n=actual_test_size, random_state=RANDOM_SEED)
    a_test_sampled = a_test.sample(n=actual_test_size, random_state=RANDOM_SEED)

    # 5. Assemble splits
    def assemble(h, a):
        cols = ["text", "label", "source"]
        extra = ["style", "model"]
        all_cols = cols + [c for c in extra if c in a.columns or c in h.columns]
        for df in [h, a]:
            for c in all_cols:
                if c not in df.columns:
                    df[c] = None
        merged = pd.concat([h[all_cols], a[all_cols]], ignore_index=True)
        return merged.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    train_df = assemble(h_train_sampled, a_train_sampled)
    val_df = assemble(h_val_sampled, a_val_sampled)
    test_df = assemble(h_test_sampled, a_test_sampled)

    # 6. Cross-split duplicate safety net
    log.info("Performing final cross-split leakage safety net deduplication...")
    train_fps = set(train_df["text"].apply(fingerprint))
    
    val_df["_fp"] = val_df["text"].apply(fingerprint)
    val_df = val_df[~val_df["_fp"].isin(train_fps)].drop(columns=["_fp"])
    
    test_df["_fp"] = test_df["text"].apply(fingerprint)
    test_df = test_df[~test_df["_fp"].isin(train_fps)].drop(columns=["_fp"])
    
    # Balance classes again if leakage check discarded rows
    for split_name, df_split in [("val", val_df), ("test", test_df)]:
        counts = df_split["label"].value_counts()
        min_c = counts.min()
        log.info(f"  {split_name} split after leakage safety net: class 0={counts.get(0, 0)}, class 1={counts.get(1, 0)}")
        # Optional: downsample to perfectly balance if desired, but minor mismatch is fine. Let's force balance.
        df_h = df_split[df_split["label"] == 0].sample(n=min_c, random_state=RANDOM_SEED)
        df_a = df_split[df_split["label"] == 1].sample(n=min_c, random_state=RANDOM_SEED)
        balanced = pd.concat([df_h, df_a]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        if split_name == "val":
            val_df = balanced
        else:
            test_df = balanced

    # 7. Save splits
    train_df.to_csv(OUTPUT_DIR / "train_500k_clean.csv", index=False, encoding="utf-8")
    val_df.to_csv(OUTPUT_DIR / "val_500k_clean.csv", index=False, encoding="utf-8")
    test_df.to_csv(OUTPUT_DIR / "test_500k_clean.csv", index=False, encoding="utf-8")

    log.info("=" * 60)
    log.info("FINAL CLEANED SPLITS SUMMARY")
    log.info("=" * 60)
    for name, split in [("train_500k_clean", train_df), ("val_500k_clean", val_df), ("test_500k_clean", test_df)]:
        counts = split["label"].value_counts()
        h_n = counts.get(0, 0)
        a_n = counts.get(1, 0)
        log.info(f"{name}: Total={len(split):,}, Human={h_n:,}, AI={a_n:,}")

if __name__ == "__main__":
    main()
