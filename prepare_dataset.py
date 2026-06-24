"""
Step 1 — Dataset Preparation (Updated)
======================================
- Loads Bundestag speeches, GNAD News, and GermEval casual texts
- Replaces domain shortcuts with grammar-preserving placeholders
- Performs length-matching stratification to align length distributions
- Creates train, val, test, external_val, and final_holdout splits
"""

import os
import re
import csv
import random
import logging
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from langdetect import detect

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Config
MIN_WORDS       = 20
MAX_CHARS       = 1_024      # BERT max token safety: ~512 tokens ≈ 1024 chars
RANDOM_SEED     = 42
OUTPUT_DIR      = Path("Data")

COMMON_GERMAN_WORDS = {"der", "die", "das", "und", "ist", "in", "zu", "den", "von", "mit", "sich", "des", "dem", "auf", "für"}

# ---------------------------------------------------------------------------
# GRAMMAR-PRESERVING PLACEHOLDER REPLACEMENTS
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# DOMAIN DATA LOADING
# ---------------------------------------------------------------------------
def load_human_bundestag(n_sample=15000):
    log.info("Loading Human Bundestag Speeches...")
    df = pd.read_csv("Data/Human_model_ready_dataset.csv", usecols=["text"], dtype=str)
    df["text"] = df["text"].apply(clean_text)
    df.dropna(subset=["text"], inplace=True)
    df.drop_duplicates(subset=["text"], inplace=True)
    df = df.sample(n=min(n_sample, len(df)), random_state=RANDOM_SEED)
    df["label"] = 0
    df["source"] = "Bundestag (Human)"
    return df

def load_human_news(n_sample=15000):
    log.info("Loading Human News paragraphs...")
    # GNAD is semicolon delimited
    df = pd.read_csv("Data/gnad_articles.csv", sep=";", header=None, on_bad_lines="skip", dtype=str)
    raw_texts = df[1].dropna().tolist()
    
    paragraphs = []
    for text in raw_texts:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Group sentences into paragraphs of about 3-4 sentences
        for i in range(0, len(sentences), 3):
            para = " ".join(sentences[i:i+3])
            paragraphs.append(para)
            
    df_para = pd.DataFrame({"text": paragraphs})
    df_para["text"] = df_para["text"].apply(clean_text)
    df_para.dropna(subset=["text"], inplace=True)
    df_para.drop_duplicates(subset=["text"], inplace=True)
    df_para = df_para.sample(n=min(n_sample, len(df_para)), random_state=RANDOM_SEED)
    df_para["label"] = 0
    df_para["source"] = "News (Human)"
    return df_para

def load_human_casual(n_sample=5000):
    log.info("Loading Human Casual texts (GermEval)...")
    # GermEval is tab-delimited
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
    df_casual["source"] = "Casual (Human)"
    return df_casual

def load_ai_bundestag(n_sample=15000):
    log.info("Loading AI Bundestag Speeches...")
    df = pd.read_csv("Data/ai_generated_sentences_500k.csv", usecols=["text"], dtype=str)
    df["text"] = df["text"].apply(clean_text)
    df.dropna(subset=["text"], inplace=True)
    df.drop_duplicates(subset=["text"], inplace=True)
    df = df.sample(n=min(n_sample, len(df)), random_state=RANDOM_SEED)
    df["label"] = 1
    df["source"] = "Bundestag (AI)"
    return df

def load_ai_news(n_sample=15000):
    log.info("Loading AI News paragraphs...")
    df = pd.read_csv("Data/ai_generated_news.csv", usecols=["text"], dtype=str)
    df["text"] = df["text"].apply(clean_text)
    df.dropna(subset=["text"], inplace=True)
    df.drop_duplicates(subset=["text"], inplace=True)
    df = df.sample(n=min(n_sample, len(df)), random_state=RANDOM_SEED)
    df["label"] = 1
    df["source"] = "News (AI)"
    return df

def load_ai_casual(n_sample=5000):
    log.info("Loading AI Casual texts...")
    df = pd.read_csv("Data/ai_generated_casual.csv", usecols=["text"], dtype=str)
    df["text"] = df["text"].apply(clean_text)
    df.dropna(subset=["text"], inplace=True)
    df.drop_duplicates(subset=["text"], inplace=True)
    df = df.sample(n=min(n_sample, len(df)), random_state=RANDOM_SEED)
    df["label"] = 1
    df["source"] = "Casual (AI)"
    return df

# ---------------------------------------------------------------------------
# LENGTH-MATCHING STRATIFICATION
# ---------------------------------------------------------------------------
def balance_by_length(human_df, ai_df):
    log.info("Balancing length distributions...")
    human_df["word_count"] = human_df["text"].str.split().apply(len)
    ai_df["word_count"] = ai_df["text"].str.split().apply(len)
    
    # Bins of size 10 words
    bins = list(range(20, 160, 10)) + [float('inf')]
    bin_labels = list(range(len(bins)-1))
    
    human_df["bin"] = pd.cut(human_df["word_count"], bins=bins, labels=bin_labels)
    ai_df["bin"] = pd.cut(ai_df["word_count"], bins=bins, labels=bin_labels)
    
    sampled_human = []
    sampled_ai = []
    
    for b in bin_labels:
        h_sub = human_df[human_df["bin"] == b]
        a_sub = ai_df[ai_df["bin"] == b]
        
        min_size = min(len(h_sub), len(a_sub))
        if min_size > 0:
            sampled_human.append(h_sub.sample(n=min_size, random_state=RANDOM_SEED))
            sampled_ai.append(a_sub.sample(n=min_size, random_state=RANDOM_SEED))
            
    final_human = pd.concat(sampled_human, ignore_index=True)
    final_ai = pd.concat(sampled_ai, ignore_index=True)
    
    final_human = final_human.drop(columns=["word_count", "bin"])
    final_ai = final_ai.drop(columns=["word_count", "bin"])
    
    return final_human, final_ai

# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Load all sets
    h_bt = load_human_bundestag(15000)
    h_news = load_human_news(15000)
    h_casual = load_human_casual(5000)
    human_df = pd.concat([h_bt, h_news, h_casual], ignore_index=True)
    
    a_bt = load_ai_bundestag(15000)
    a_news = load_ai_news(15000)
    a_casual = load_ai_casual(5000)
    ai_df = pd.concat([a_bt, a_news, a_casual], ignore_index=True)
    
    # cross-dedup
    common_texts = set(human_df["text"]).intersection(set(ai_df["text"]))
    if common_texts:
        log.info(f"Removing {len(common_texts)} cross-class duplicates...")
        human_df = human_df[~human_df["text"].isin(common_texts)]
        ai_df = ai_df[~ai_df["text"].isin(common_texts)]
        
    # Balance lengths
    human_bal, ai_bal = balance_by_length(human_df, ai_df)
    
    log.info(f"Balanced human size: {len(human_bal):,}, balanced AI size: {len(ai_bal):,}")
    
    df = pd.concat([human_bal, ai_bal], ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    # Splits (80/10/10)
    train_df, temp_df = train_test_split(df, test_size=0.20, stratify=df["label"], random_state=RANDOM_SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=RANDOM_SEED)
    
    # Save splits
    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False, encoding="utf-8")
    val_df.to_csv(OUTPUT_DIR / "val.csv", index=False, encoding="utf-8")
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False, encoding="utf-8")
    
    # Also save external_val and final_holdout for generalization tracking
    val_df.to_csv(OUTPUT_DIR / "external_val.csv", index=False, encoding="utf-8")
    test_df.to_csv(OUTPUT_DIR / "final_holdout.csv", index=False, encoding="utf-8")
    
    log.info(f"Splits saved successfully in {OUTPUT_DIR}:")
    log.info(f"  train: {len(train_df):,} rows")
    log.info(f"  val: {len(val_df):,} rows")
    log.info(f"  test: {len(test_df):,} rows")

if __name__ == "__main__":
    main()