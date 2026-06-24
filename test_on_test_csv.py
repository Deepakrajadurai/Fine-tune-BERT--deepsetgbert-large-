import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import csv
import traceback

try:
    import torch
    from transformers import BertTokenizer, BertForSequenceClassification

    MODEL_DIR = "models/best_model"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {DEVICE}")
    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    print("Loading model...")
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
    print("Setting eval mode...")
    model.eval()

    # Load test set using built-in csv module to avoid pandas DLL conflicts
    print("Loading Data/test.csv using csv module...")
    h_rows = []
    a_rows = []
    
    with open("Data/test.csv", mode="r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lbl = int(row["label"])
            text = row["text"]
            if lbl == 0 and len(h_rows) < 5:
                h_rows.append(text)
            elif lbl == 1 and len(a_rows) < 5:
                a_rows.append(text)
            if len(h_rows) >= 5 and len(a_rows) >= 5:
                break

    print(f"Loaded {len(h_rows)} human and {len(a_rows)} AI texts.")

    print("\n--- Testing Human rows from test.csv ---")
    for idx, text in enumerate(h_rows):
        enc = tokenizer(text, max_length=256, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(input_ids=enc["input_ids"].to(DEVICE), attention_mask=enc["attention_mask"].to(DEVICE)).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        print(f"Row {idx} | True Label: 0 | Pred probs: {probs} | Text: {text[:100]}...")

    print("\n--- Testing AI rows from test.csv ---")
    for idx, text in enumerate(a_rows):
        enc = tokenizer(text, max_length=256, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(input_ids=enc["input_ids"].to(DEVICE), attention_mask=enc["attention_mask"].to(DEVICE)).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        print(f"Row {idx} | True Label: 1 | Pred probs: {probs} | Text: {text[:100]}...")

except Exception as e:
    print("CRASHED:")
    traceback.print_exc()
