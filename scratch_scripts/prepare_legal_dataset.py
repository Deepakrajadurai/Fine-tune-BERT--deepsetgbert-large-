import os
import json
import re
import pandas as pd
import random
from langdetect import detect

# Config
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

MIN_WORDS = 20
MAX_CHARS = 1024
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

def main():
    jsonl_path = r"e:\15-06-26\Fine-tune BERT (deepsetgbert-large)\Data\german_legal_full_dataset.jsonl"
    print(f"Loading legal dataset from {jsonl_path}...")
    
    human_data = []
    ai_data = []
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            text = item.get("text", "")
            label_str = item.get("label", "")
            source = item.get("source", "")
            
            cleaned = clean_text(text)
            if cleaned:
                label_val = 0 if label_str == "human" else 1
                row = {
                    "text": cleaned,
                    "label": label_val,
                    "source": source
                }
                if label_val == 0:
                    human_data.append(row)
                else:
                    ai_data.append(row)
                    
    print(f"Cleaned samples count: Human={len(human_data)}, AI={len(ai_data)}")
    
    # Balance dataset
    min_len = min(len(human_data), len(ai_data))
    print(f"Balancing dataset to {min_len} samples per class...")
    random.shuffle(human_data)
    random.shuffle(ai_data)
    human_data = human_data[:min_len]
    ai_data = ai_data[:min_len]
    
    # Split: 80% train, 10% val, 10% test
    train_size = int(min_len * 0.8)
    val_size = int(min_len * 0.1)
    
    train_list = human_data[:train_size] + ai_data[:train_size]
    val_list = human_data[train_size:train_size + val_size] + ai_data[train_size:train_size + val_size]
    test_list = human_data[train_size + val_size:] + ai_data[train_size + val_size:]
    
    # Shuffle splits
    random.shuffle(train_list)
    random.shuffle(val_list)
    random.shuffle(test_list)
    
    # Save to CSV
    train_df = pd.DataFrame(train_list)
    val_df = pd.DataFrame(val_list)
    test_df = pd.DataFrame(test_list)
    
    train_df.to_csv(r"e:\15-06-26\Fine-tune BERT (deepsetgbert-large)\Data\train_legal.csv", index=False)
    val_df.to_csv(r"e:\15-06-26\Fine-tune BERT (deepsetgbert-large)\Data\val_legal.csv", index=False)
    test_df.to_csv(r"e:\15-06-26\Fine-tune BERT (deepsetgbert-large)\Data\test_legal.csv", index=False)
    
    print("Splits created and saved:")
    print(f"  Train: {len(train_df)} rows")
    print(f"  Val:   {len(val_df)} rows")
    print(f"  Test:  {len(test_df)} rows")

if __name__ == "__main__":
    main()
