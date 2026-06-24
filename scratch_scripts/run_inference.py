import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import re
import csv
import random
import torch
import torch.nn.functional as F
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification
)

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "models/best_model"
BASE_MODEL = "deepset/gbert-large"
TOPICS = [
    "Fußball", "Gaming", "Smartphones", "Studentenleben", 
    "Stricken", "Imkerei", "Kryptowährung", "Vegetarismus", 
    "Origami", "Homeoffice", "Kochen"
]

# ---------------------------------------------------------------------------
# GRAMMAR-PRESERVING PREPROCESSING (Matching Training Pipeline)
# ---------------------------------------------------------------------------
def replace_domain_markers(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = re.sub(r'§+\s*\d+(?:\s*(?:Abs\.|Absatz|Satz)\s*\d+)*', '[PARAGRAPH]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:Abs\.|Absatz)\s*\d+\b', '[PARAGRAPH]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:Art\.|Artikel)\s*\d+\b', '[PARAGRAPH]', text, flags=re.IGNORECASE)
    text = re.sub(r'\bAz\.\s*[A-Za-z0-9./-]+\b', '[AZ]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d{2,4}/\d{4,6}\b', '[AZ]', text)
    text = re.sub(r'\b\d{1,2}\.\d{1,2}\.\d{2,4}\b', '[DATUM]', text)
    parties = 'CDU|CSU|SPD|Grüne|Grünen|FDP|AfD|Linke|BSW|ÖDP|Volt|Freie Wähler|Freien Wähler'
    text = re.sub(rf'\b(?:{parties})\b', '[PARTEI]', text, flags=re.IGNORECASE)
    text = re.sub(r'auf\s+Initiative\s+von\s+(?:Abgeordnet(?:em|er|en)\s+)?(?:[A-ZÄÖÜß][a-zäöüß]+)\s+(?:[A-ZÄÖÜß][a-zäöüß]+)', 'auf Initiative von [PERSON]', text)
    text = re.sub(r'unter\s+Aufsicht\s+von\s+(?:[A-ZÄÖÜß][a-zäöüß]+)\s+(?:[A-ZÄÖÜß][a-zäöüß]+)', 'unter Aufsicht von [PERSON]', text)
    text = re.sub(r'unter\s+Bezug(?:nahme)?\s+auf\s+(?:[A-ZÄÖÜß][a-zäöüß]+)\s+(?:[A-ZÄÖÜß][a-zäöüß]+)', 'unter Bezugnahme auf [PERSON]', text)
    text = re.sub(r'im\s+Namen\s+von\s+(?:[A-ZÄÖÜß][a-zäöüß]+)\s+(?:[A-ZÄÖÜß][a-zäöüß]+)', 'im Namen von [PERSON]', text)
    text = re.sub(r'unter\s+Leitung\s+von\s+(?:[A-ZÄÖÜß][a-zäöüß]+)\s+(?:[A-ZÄÖÜß][a-zäöüß]+)', 'unter Leitung von [PERSON]', text)
    text = re.sub(r'durch\s+(?:[A-ZÄÖÜß][a-zäöüß]+)\s+(?:[A-ZÄÖÜß][a-zäöüß]+)', 'durch [PERSON]', text)
    text = re.sub(r'gezeichnete\s+Antrag\s+von\s+(?:[A-ZÄÖÜß][a-zäöüß]+)\s+(?:[A-ZÄÖÜß][a-zäöüß]+)', 'gezeichnete Antrag von [PERSON]', text)
    text = re.sub(r'\((?:Abgeordnet(?:er|em|en)\s+)?(?:[A-ZÄÖÜß][a-zäöüß]+)\s+(?:[A-ZÄÖÜß][a-zäöüß]+)\)', '([PERSON])', text)
    text = re.sub(r'in\s+(?:dieser|der\s+heutigen)\s+\d+\.\s*Plenarsitzung', 'in dieser [PLENARSITZUNG]', text, flags=re.IGNORECASE)
    text = re.sub(r'\bPlenarsitzung\b', '[PLENARSITZUNG]', text, flags=re.IGNORECASE)
    text = re.sub(r'\bDrucksache\b', '[DRUCKSACHE]', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    return text.strip()

def preprocess_text(text: str) -> str:
    text = text.strip()
    text = replace_domain_markers(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 1024:
        text = text[:1024].rsplit(" ", 1)[0]
    return text

# ---------------------------------------------------------------------------
# CSV HELPERS
# ---------------------------------------------------------------------------
def read_csv_dict(path):
    if not os.path.exists(path):
        return []
    with open(path, mode="r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        return list(reader)

def write_csv_dict(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# ---------------------------------------------------------------------------
# EMBEDDING HELPERS FOR TOPIC LEAKAGE (Using GBERT-large base model)
# ---------------------------------------------------------------------------
tokenizer = None
classification_model = None

def get_embeddings(texts, batch_size=32):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_preprocessed = [preprocess_text(t) for t in batch]
        encoded = tokenizer(batch_preprocessed, padding=True, truncation=True, max_length=256, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            outputs = classification_model.bert(input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask'])
        # CLS token embeddings (first token)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        cls_embeddings = F.normalize(cls_embeddings, p=2, dim=1)
        all_embeddings.append(cls_embeddings)
    return torch.cat(all_embeddings, dim=0)

def clean_data_file(path_in, path_out, topic_embeddings):
    rows = read_csv_dict(path_in)
    if not rows:
        print(f"File {path_in} not found or empty.")
        return
    texts = [r["text"] for r in rows]
    print(f"Checking {len(texts)} texts in {path_in} for leakage...")
    
    # Batch process embeddings
    candidate_embeddings = get_embeddings(texts)
    similarities = torch.matmul(candidate_embeddings, topic_embeddings.T) # Shape: (N, num_topics)
    max_similarities, _ = torch.max(similarities, dim=1)
    
    keep_rows = []
    skipped = 0
    for idx, sim in enumerate(max_similarities.cpu().tolist()):
        if sim > 0.8:
            skipped += 1
        else:
            keep_rows.append(rows[idx])
            
    print(f"Filtered out {skipped} rows (leakage > 0.8) from {path_in}. Retained {len(keep_rows)}.")
    if rows:
        write_csv_dict(path_out, list(rows[0].keys()), keep_rows)

# ---------------------------------------------------------------------------
# MAIN FLOW
# ---------------------------------------------------------------------------
def main():
    global tokenizer, classification_model
    print("Loading GBERT-large detector model...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    classification_model = BertForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
    classification_model.eval()

    print("=== STARTING SEMANTIC TOPIC LEAKAGE CHECKS ===")
    topic_embeddings = get_embeddings(TOPICS)
    
    clean_data_file("Data/train.csv", "Data/train_cleaned.csv", topic_embeddings)
    clean_data_file("Data/val.csv", "Data/val_cleaned.csv", topic_embeddings)
    clean_data_file("Data/test.csv", "Data/test_cleaned.csv", topic_embeddings)
    
    # ---------------------------------------------------------------------------
    # GBERT INFERENCE ON RAW VALIDATION DATA
    # ---------------------------------------------------------------------------
    print("\n=== RUNNING BASE DETECTOR INFERENCE ===")
    raw_eval_rows = read_csv_dict("Data/external_eval_raw.csv")
    print(f"Loaded {len(raw_eval_rows)} validation samples.")
    
    inference_results = []
    batch_size = 32
    for i in range(0, len(raw_eval_rows), batch_size):
        batch_rows = raw_eval_rows[i:i+batch_size]
        preprocessed_texts = [preprocess_text(r["text"]) for r in batch_rows]
        
        enc = tokenizer(preprocessed_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = classification_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
        probs = torch.softmax(logits, dim=-1).cpu().tolist()
        
        for idx, row in enumerate(batch_rows):
            ai_prob = float(probs[idx][1])
            words = len(row["text"].split())
            inference_results.append({
                "text": row["text"],
                "source": row["source"],
                "label": row["label"],
                "prediction": 1 if ai_prob >= 0.5 else 0,
                "ai_probability": round(ai_prob, 4),
                "words": words
            })
            
    write_csv_dict("results/raw_predictions_original.csv", ["text", "source", "label", "prediction", "ai_probability", "words"], inference_results)
    print("Saved base predictions to results/raw_predictions_original.csv")
    
    # ---------------------------------------------------------------------------
    # OUT-OF-MODEL TRANSFER EXPERIMENTS
    # ---------------------------------------------------------------------------
    print("\n=== RUNNING OUT-OF-MODEL TRANSFER EXPERIMENTS ===")
    
    def collate_fn(batch_rows, tokenizer_obj, max_length=256):
        texts = [preprocess_text(r["text"]) for r in batch_rows]
        labels = [int(r["label"]) for r in batch_rows]
        enc = tokenizer_obj(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        input_ids = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)
        labels_tensor = torch.tensor(labels, dtype=torch.long).to(DEVICE)
        return input_ids, attention_mask, labels_tensor

    # Helper to train and evaluate
    def run_transfer_rotation(train_sources, test_sources, rotation_name):
        print(f"\n--- Running {rotation_name} ---")
        
        # Split rows
        train_rows = [r for r in raw_eval_rows if r["source"] in train_sources]
        test_rows = [r for r in raw_eval_rows if r["source"] in test_sources]
        
        print(f"Training on: {train_sources} ({len(train_rows)} samples)")
        print(f"Testing on: {test_sources} ({len(test_rows)} samples)")
        
        # Load raw GBERT model
        tokenizer_base = BertTokenizer.from_pretrained(BASE_MODEL)
        model_base = BertForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2).to(DEVICE)
        
        # Train setup
        optimizer = torch.optim.AdamW(model_base.parameters(), lr=2e-5, weight_decay=0.01)
        model_base.train()
        
        num_epochs = 3
        batch_size = 8
        
        # Seed for reproducibility
        random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
            
        print("Training model...")
        for epoch in range(num_epochs):
            random.shuffle(train_rows)
            epoch_loss = 0.0
            steps = 0
            for i in range(0, len(train_rows), batch_size):
                batch_batch = train_rows[i:i+batch_size]
                input_ids, attention_mask, labels_tensor = collate_fn(batch_batch, tokenizer_base)
                
                optimizer.zero_grad()
                outputs = model_base(input_ids=input_ids, attention_mask=attention_mask, labels=labels_tensor)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                steps += 1
            print(f"  Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss/steps:.4f}")
            
        model_base.eval()
        
        # Predict on test rows
        test_results = []
        for i in range(0, len(test_rows), batch_size):
            batch_batch = test_rows[i:i+batch_size]
            preprocessed_t = [preprocess_text(r["text"]) for r in batch_batch]
            enc = tokenizer_base(preprocessed_t, padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                logits = model_base(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
            probs = torch.softmax(logits, dim=-1).cpu().tolist()
            
            for idx, r in enumerate(batch_batch):
                ai_prob = float(probs[idx][1])
                test_results.append({
                    "text": r["text"],
                    "source": r["source"],
                    "label": r["label"],
                    "prediction": 1 if ai_prob >= 0.5 else 0,
                    "ai_probability": round(ai_prob, 4),
                    "words": len(r["text"].split())
                })
                
        write_csv_dict(f"results/raw_predictions_{rotation_name}.csv", ["text", "source", "label", "prediction", "ai_probability", "words"], test_results)
        print(f"Saved rotation predictions to results/raw_predictions_{rotation_name}.csv")
        
        # Cleanup memory
        del model_base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Define sources
    all_human_sources = ["Human Wiki", "Human News", "Human Casual", "Human Essay"]
    
    # Rotation 1: Train on Qwen + GPT-4 + Claude + Humans. Test on Gemini + Humans.
    train_sources_1 = all_human_sources + ["AI ChatGPT", "AI Claude", "AI Qwen"]
    test_sources_1 = all_human_sources + ["AI Gemini"]
    run_transfer_rotation(train_sources_1, test_sources_1, "rotation1")
    
    # Rotation 2: Train on GPT-4 + Claude + Gemini + Humans. Test on Qwen + Humans.
    train_sources_2 = all_human_sources + ["AI ChatGPT", "AI Claude", "AI Gemini"]
    test_sources_2 = all_human_sources + ["AI Qwen"]
    run_transfer_rotation(train_sources_2, test_sources_2, "rotation2")
    
    print("\n=== INFERENCE AND ROTATION EXPERIMENTS COMPLETE ===")

if __name__ == "__main__":
    main()
