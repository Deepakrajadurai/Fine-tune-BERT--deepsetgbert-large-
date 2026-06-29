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
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Config
MIN_WORDS = 10
RANDOM_SEED = 42
OUTPUT_DIR = Path("Data")

ARTIFACT_PATTERNS = [
    (r"\d+\.\s*Plenarsitzung", "Plenarsitzung"),
    (r"\((?:CDU(?:/CSU)?|SPD|Grüne|FDP|AfD|Linke|BSW|CSU)\)", ""),
    (r"(?:auf Initiative von |Abgeordnet(?:em|er|en)\s+)[A-ZÄÖÜ][a-zäöüß]+\s+[A-ZÄÖÜ][a-zäöüß]+", ""),
    (r"\s{2,}", " "),
]

def strip_artifacts(text: str) -> str:
    for pattern, replacement in ARTIFACT_PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text.strip()

def clean_text(text: str) -> str | None:
    if not isinstance(text, str) or not text.strip():
        return None
    text = strip_artifacts(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    if len(words) < MIN_WORDS:
        return None
    
    # Reject if >25% digits
    if sum(c.isdigit() for c in text) / len(text) > 0.25:
        return None

    # Reject model meta-commentary
    bad_phrases = [
        "als ki ", "als sprachmodell", "ich kann leider",
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
    df["text"] = df["text"].apply(clean_text)
    df.dropna(subset=["text"], inplace=True)
    log.info(f"  After cleaning: {len(df):,}")
    
    df["_fp"] = df["text"].apply(fingerprint)
    before = len(df)
    df.drop_duplicates(subset=["_fp"], inplace=True)
    df.drop(columns=["_fp"], inplace=True)
    log.info(f"  After exact dedup: {len(df):,} (removed {before - len(df):,})")
    return df.reset_index(drop=True)

def main():
    np.random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Load and clean datasets
    human_df = load_and_clean(Path("Data/Human_model_ready_dataset.csv"), label=0)
    ai_df = load_and_clean(Path("Data/ai_generated_sentences_500k.csv"), label=1)

    # 2. Perform train/val/test splits (80/10/10) on each class
    log.info("Splitting datasets into train/val/test splits...")
    h_train, h_temp = train_test_split(human_df, test_size=0.20, random_state=RANDOM_SEED)
    h_val, h_test = train_test_split(h_temp, test_size=0.50, random_state=RANDOM_SEED)

    a_train, a_temp = train_test_split(ai_df, test_size=0.20, random_state=RANDOM_SEED)
    a_val, a_test = train_test_split(a_temp, test_size=0.50, random_state=RANDOM_SEED)

    # 3. Subsample to balance the dataset:
    # Train: 400,000 of each class
    # Val: 50,000 of each class
    # Test: 50,000 of each class
    train_size = 400_000
    val_size = 50_000
    test_size = 50_000

    log.info(f"Subsampling splits to balance at {train_size} train, {val_size} val, {test_size} test...")
    h_train_sampled = h_train.sample(n=min(train_size, len(h_train)), random_state=RANDOM_SEED)
    a_train_sampled = a_train.sample(n=min(train_size, len(a_train)), random_state=RANDOM_SEED)

    h_val_sampled = h_val.sample(n=min(val_size, len(h_val)), random_state=RANDOM_SEED)
    a_val_sampled = a_val.sample(n=min(val_size, len(a_val)), random_state=RANDOM_SEED)

    h_test_sampled = h_test.sample(n=min(test_size, len(h_test)), random_state=RANDOM_SEED)
    a_test_sampled = a_test.sample(n=min(test_size, len(a_test)), random_state=RANDOM_SEED)

    # 4. Assemble splits
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

    # 5. Cross-split dedup check (safety net)
    log.info("Cross-split leakage check...")
    train_fps = set(train_df["text"].apply(fingerprint))
    val_fps = set(val_df["text"].apply(fingerprint))
    test_fps = set(test_df["text"].apply(fingerprint))

    log.info(f"  Train ∩ Val  : {len(train_fps & val_fps)} duplicates")
    log.info(f"  Train ∩ Test : {len(train_fps & test_fps)} duplicates")
    
    if len(train_fps & test_fps) > 0:
        log.info("  Removing leaked rows from test set...")
        test_df = test_df[~test_df["text"].apply(fingerprint).isin(train_fps)].reset_index(drop=True)

    # 6. Save splits
    train_df.to_csv(OUTPUT_DIR / "train_500k.csv", index=False, encoding="utf-8")
    val_df.to_csv(OUTPUT_DIR / "val_500k.csv", index=False, encoding="utf-8")
    test_df.to_csv(OUTPUT_DIR / "test_500k.csv", index=False, encoding="utf-8")

    log.info("=" * 60)
    log.info("FINAL SPLITS SUMMARY")
    log.info("=" * 60)
    for name, split in [("train_500k", train_df), ("val_500k", val_df), ("test_500k", test_df)]:
        counts = split["label"].value_counts()
        h_n = counts.get(0, 0)
        a_n = counts.get(1, 0)
        log.info(f"{name}: Total={len(split):,}, Human={h_n:,}, AI={a_n:,}")

if __name__ == "__main__":
    main()
