import os
import re
import csv
import hashlib
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from langdetect import detect

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="[%H:%M:%S]"
)
log = logging.getLogger(__name__)

# Config
MIN_WORDS = 20
MAX_CHARS = 1024
RANDOM_SEED = 42
OUTPUT_DIR = Path("Data")

COMMON_GERMAN_WORDS = {"der", "die", "das", "und", "ist", "in", "zu", "den", "von", "mit", "sich", "des", "dem", "auf", "für"}

def replace_domain_markers(text: str) -> str:
    if not isinstance(text, str):
        return text
    
    # 1. Section/Law references (eg. § 18 Abs. 3, Absatz 4, Artikel 5)
    text = re.sub(r'§+\s*\d+(?:\s*(?:Abs\.|Absatz|Satz)\s*\d+)*', '[PARAGRAPH]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:Abs\.|Absatz)\s*\d+\b', '[PARAGRAPH]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:Art\.|Artikel)\s*\d+\b', '[PARAGRAPH]', text, flags=re.IGNORECASE)
    
    # 2. Reference numbers (Az. 32/93721)
    text = re.sub(r'\bAz\.\s*[A-Za-z0-9./-]+\b', '[AZ]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d{2,4}/\d{4,6}\b', '[AZ]', text)
    
    # 3. Dates (15.06.2026, 04.02.24)
    text = re.sub(r'\b\d{1,2}\.\d{1,2}\.\d{2,4}\b', '[DATUM]', text)
    
    # 4. Political party names
    parties = 'CDU|CSU|SPD|Grüne|Grünen|FDP|AfD|Linke|BSW|ÖDP|Volt|Freie Wähler|Freien Wähler'
    text = re.sub(rf'\b(?:{parties})\b', '[PARTEI]', text, flags=re.IGNORECASE)
    
    # 5. Template names
    text = re.sub(r'auf\s+Initiative\s+von\s+(?:Abgeordnet(?:em|er|en)\s+)?(?:[A-ZÄÖÜß][a-zäöüß]+)\s+(?:[A-ZÄÖÜß][a-zäöüß]+)', 'auf Initiative von [PERSON]', text)
    text = re.sub(r'unter\s+Aufsicht\s+von\s+(?:[A-ZÄÖÜß][a-zäöüß]+)\s+(?:[A-ZÄÖÜß][a-zäöüß]+)', 'unter Aufsicht von [PERSON]', text)
    text = re.sub(r'unter\s+Bezug(?:nahme)?\s+auf\s+(?:[A-ZÄÖÜß][a-zäöüß]+)\s+(?:[A-ZÄÖÜß][a-zäöüß]+)', 'unter Bezugnahme auf [PERSON]', text)
    text = re.sub(r'im\s+Namen\s+von\s+(?:[A-ZÄÖÜß][a-zäöüß]+)\s+(?:[A-ZÄÖÜß][a-zäöüß]+)', 'im Namen von [PERSON]', text)
    text = re.sub(r'unter\s+Leitung\s+von\s+(?:[A-ZÄÖÜß][a-zäöüß]+)\s+(?:[A-ZÄÖÜß][a-zäöüß]+)', 'unter Leitung von [PERSON]', text)
    text = re.sub(r'durch\s+(?:[A-ZÄÖÜß][a-zäöüß]+)\s+(?:[A-ZÄÖÜß][a-zäöüß]+)', 'durch [PERSON]', text)
    text = re.sub(r'gezeichnete\s+Antrag\s+von\s+(?:[A-ZÄÖÜß][a-zäöüß]+)\s+(?:[A-ZÄÖÜß][a-zäöüß]+)', 'gezeichnete Antrag von [PERSON]', text)
    text = re.sub(r'\((?:Abgeordnet(?:er|em|en)\s+)?(?:[A-ZÄÖÜß][a-zäöüß]+)\s+(?:[A-ZÄÖÜß][a-zäöüß]+)\)', '([PERSON])', text)
    
    # 6. Specific template keywords (Plenarsitzung, Drucksache)
    text = re.sub(r'in\s+(?:dieser|der\s+heutigen)\s+\d+\.\s*Plenarsitzung', 'in dieser [PLENARSITZUNG]', text, flags=re.IGNORECASE)
    text = re.sub(r'\bPlenarsitzung\b', '[PLENARSITZUNG]', text, flags=re.IGNORECASE)
    text = re.sub(r'\bDrucksache\b', '[DRUCKSACHE]', text, flags=re.IGNORECASE)
    
    # 7. Collapse spaces
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    return text.strip()

def clean_text(text: str) -> str | None:
    if not isinstance(text, str):
        return None
    text = replace_domain_markers(text)
    text = text.strip()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    words = text.split()
    if len(words) < MIN_WORDS:
        return None
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS].rsplit(" ", 1)[0]
        
    digit_ratio = sum(c.isdigit() for c in text) / len(text)
    if digit_ratio > 0.30:
        return None
        
    if re.search(r"[äöüÄÖÜß]", text):
        return text
    words_set = set(w.lower() for w in words)
    if words_set.intersection(COMMON_GERMAN_WORDS):
        return text
    try:
        if detect(text) != "de":
            return None
    except Exception:
        return None
    return text

def fingerprint(text: str) -> str:
    normalised = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.md5(normalised.encode()).hexdigest()

def load_human_news(n_sample=15000):
    log.info("Loading Human News paragraphs...")
    df = pd.read_csv("Data/gnad_articles.csv", sep=";", header=None, on_bad_lines="skip", dtype=str)
    raw_texts = df[1].dropna().tolist()
    
    paragraphs = []
    for text in raw_texts:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for i in range(0, len(sentences), 3):
            para = " ".join(sentences[i:i+3])
            paragraphs.append(para)
            
    df_para = pd.DataFrame({"text": paragraphs})
    df_para["text"] = df_para["text"].apply(clean_text)
    df_para.dropna(subset=["text"], inplace=True)
    df_para.drop_duplicates(subset=["text"], inplace=True)
    df_para = df_para.sample(n=min(n_sample, len(df_para)), random_state=RANDOM_SEED)
    df_para["label"] = 0
    df_para["source"] = "News"
    return df_para

def load_human_casual(n_sample=5000):
    log.info("Loading Human Casual texts (GermEval)...")
    df = pd.read_csv("Data/germeval2018.txt", sep="\t", header=None, on_bad_lines="skip", dtype=str)
    raw_texts = df[0].dropna().tolist()
    
    cleaned_texts = []
    for text in raw_texts:
        text = text.replace('|LBR|', ' ')
        text = re.sub(r'@[A-Za-z0-9_]+', '', text)
        text = re.sub(r'#\w+', '', text)
        cleaned_texts.append(text)
        
    df_casual = pd.DataFrame({"text": cleaned_texts})
    df_casual["text"] = df_casual["text"].apply(clean_text)
    df_casual.dropna(subset=["text"], inplace=True)
    df_casual.drop_duplicates(subset=["text"], inplace=True)
    df_casual = df_casual.sample(n=min(n_sample, len(df_casual)), random_state=RANDOM_SEED)
    df_casual["label"] = 0
    df_casual["source"] = "Casual"
    return df_casual

def load_ai_news(n_sample=15000):
    log.info("Loading AI News paragraphs...")
    df = pd.read_csv("Data/ai_generated_news.csv", usecols=["text"], dtype=str)
    df["text"] = df["text"].apply(clean_text)
    df.dropna(subset=["text"], inplace=True)
    df.drop_duplicates(subset=["text"], inplace=True)
    df = df.sample(n=min(n_sample, len(df)), random_state=RANDOM_SEED)
    df["label"] = 1
    df["source"] = "News"
    return df
    
def load_ai_casual(n_sample=5000):
    log.info("Loading AI Casual texts...")
    df = pd.read_csv("Data/ai_generated_casual.csv", usecols=["text"], dtype=str)
    df["text"] = df["text"].apply(clean_text)
    df.dropna(subset=["text"], inplace=True)
    df.drop_duplicates(subset=["text"], inplace=True)
    df = df.sample(n=min(n_sample, len(df)), random_state=RANDOM_SEED)
    df["label"] = 1
    df["source"] = "Casual"
    return df

def main():
    np.random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # 1. Load human and AI datasets
    h_news = load_human_news(15000)
    h_casual = load_human_casual(5000)
    human_df = pd.concat([h_news, h_casual], ignore_index=True)
    
    a_news = load_ai_news(15000)
    a_casual = load_ai_casual(5000)
    ai_df = pd.concat([a_news, a_casual], ignore_index=True)
    
    # 2. Strict fingerprint deduplication
    log.info("Deduplicating Human and AI classes...")
    human_df["_fp"] = human_df["text"].apply(fingerprint)
    before_h = len(human_df)
    human_df.drop_duplicates(subset=["_fp"], inplace=True)
    log.info(f"  Human dataset: {len(human_df):,} rows (removed {before_h - len(human_df):,} exact duplicates)")
    
    ai_df["_fp"] = ai_df["text"].apply(fingerprint)
    before_a = len(ai_df)
    ai_df.drop_duplicates(subset=["_fp"], inplace=True)
    log.info(f"  AI dataset: {len(ai_df):,} rows (removed {before_a - len(ai_df):,} exact duplicates)")
    
    # Remove cross-class exact duplicate matches
    common_fps = set(human_df["_fp"]).intersection(set(ai_df["_fp"]))
    if common_fps:
        log.info(f"  Removing {len(common_fps)} cross-class exact duplicate matches...")
        human_df = human_df[~human_df["_fp"].isin(common_fps)]
        ai_df = ai_df[~ai_df["_fp"].isin(common_fps)]
        
    human_df.drop(columns=["_fp"], inplace=True)
    ai_df.drop(columns=["_fp"], inplace=True)
    
    # 3. Splits (80/10/10) per class to guarantee balance
    log.info("Splitting datasets into train/val/test splits...")
    h_train, h_temp = train_test_split(human_df, test_size=0.20, random_state=RANDOM_SEED)
    h_val, h_test = train_test_split(h_temp, test_size=0.50, random_state=RANDOM_SEED)

    a_train, a_temp = train_test_split(ai_df, test_size=0.20, random_state=RANDOM_SEED)
    a_val, a_test = train_test_split(a_temp, test_size=0.50, random_state=RANDOM_SEED)
    
    # 4. Balanced subsampling
    train_size = min(len(h_train), len(a_train))
    val_size = min(len(h_val), len(a_val))
    test_size = min(len(h_test), len(a_test))
    
    log.info(f"Subsampling to balance splits: train={train_size}, val={val_size}, test={test_size}")
    
    h_train_s = h_train.sample(n=train_size, random_state=RANDOM_SEED)
    a_train_s = a_train.sample(n=train_size, random_state=RANDOM_SEED)
    
    h_val_s = h_val.sample(n=val_size, random_state=RANDOM_SEED)
    a_val_s = a_val.sample(n=val_size, random_state=RANDOM_SEED)
    
    h_test_s = h_test.sample(n=test_size, random_state=RANDOM_SEED)
    a_test_s = a_test.sample(n=test_size, random_state=RANDOM_SEED)
    
    # 5. Assemble and shuffle splits
    def assemble(h, a):
        merged = pd.concat([h, a], ignore_index=True)
        return merged.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        
    train_df = assemble(h_train_s, a_train_s)
    val_df = assemble(h_val_s, a_val_s)
    test_df = assemble(h_test_s, a_test_s)
    
    # 6. Final cross-split leakage check
    log.info("Performing final cross-split leakage check...")
    train_fps = set(train_df["text"].apply(fingerprint))
    
    val_df["_fp"] = val_df["text"].apply(fingerprint)
    val_df = val_df[~val_df["_fp"].isin(train_fps)].drop(columns=["_fp"])
    
    test_df["_fp"] = test_df["text"].apply(fingerprint)
    test_df = test_df[~test_df["_fp"].isin(train_fps)].drop(columns=["_fp"])
    
    # Downsample splits to remain perfectly balanced
    for split_name, df_split in [("val", val_df), ("test", test_df)]:
        counts = df_split["label"].value_counts()
        min_c = counts.min()
        df_h = df_split[df_split["label"] == 0].sample(n=min_c, random_state=RANDOM_SEED)
        df_a = df_split[df_split["label"] == 1].sample(n=min_c, random_state=RANDOM_SEED)
        balanced = pd.concat([df_h, df_a]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        if split_name == "val":
            val_df = balanced
        else:
            test_df = balanced
            
    # 7. Save splits
    train_df.to_csv(OUTPUT_DIR / "train_organic.csv", index=False, encoding="utf-8")
    val_df.to_csv(OUTPUT_DIR / "val_organic.csv", index=False, encoding="utf-8")
    test_df.to_csv(OUTPUT_DIR / "test_organic.csv", index=False, encoding="utf-8")
    
    log.info("=" * 60)
    log.info("FINAL ORGANIC SPLITS SUMMARY")
    log.info("=" * 60)
    for name, split in [("train_organic", train_df), ("val_organic", val_df), ("test_organic", test_df)]:
        counts = split["label"].value_counts()
        log.info(f"{name}: Total={len(split):,}, Human={counts.get(0, 0):,}, AI={counts.get(1, 0):,}")

if __name__ == "__main__":
    main()
