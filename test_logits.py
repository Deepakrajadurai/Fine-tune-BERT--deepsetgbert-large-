import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from transformers import BertTokenizer, BertForSequenceClassification

MODEL_DIR = "models/best_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

tests = [
    "Meine Damen und Herren, wir müssen uns in der heutigen Zeit fragen.",
    "Die Digitalisierung der Verwaltung ist ein zentraler Baustein der Modernisierung des öffentlichen Dienstes."
]

for text in tests:
    enc = tokenizer(text, max_length=256, padding=True, truncation=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    
    print(f"Text: {text}")
    print(f"Tokenized: {tokenizer.convert_ids_to_tokens(enc['input_ids'][0])}")
    print(f"Logits: {logits.cpu().numpy()[0]}")
    print(f"Probs: {probs}")
    print("-" * 50)
