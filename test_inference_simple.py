import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import torch
from transformers import BertTokenizer, BertForSequenceClassification

print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

MODEL_DIR = "models/best_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading tokenizer & model on {DEVICE}...")
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

print("Model loaded successfully.")

# Read threshold
with open("results/threshold.txt") as f:
    threshold = float(f.read().strip())
print(f"Threshold: {threshold}")

# Test texts
tests = [
    ("HUMAN (real Bundestag)", "Meine Damen und Herren, wir müssen uns in der heutigen Zeit fragen, wie wir den sozialen Wohnungsbau in unseren Städten nachhaltig stärken und bezahlbaren Wohnraum für alle Bürger garantieren können."),
    ("AI (generated)", "Die Digitalisierung der Verwaltung ist ein zentraler Baustein der Modernisierung des öffentlichen Dienstes. Durch den Einsatz innovativer Technologien können Prozesse effizienter gestaltet und Bürgerinnen und Bürger besser eingebunden werden. Es ist daher von großer Bedeutung, dass wir als Gesetzgeber die notwendigen Rahmenbedingungen schaffen, um diesen Transformationsprozess zu unterstützen und voranzutreiben."),
    ("AI (ChatGPT-style)", "In Anbetracht der aktuellen wirtschaftlichen Herausforderungen ist es von entscheidender Bedeutung, dass die Bundesregierung gezielte Maßnahmen ergreift, um die Wettbewerbsfähigkeit Deutschlands auf dem globalen Markt zu stärken. Hierbei spielen sowohl steuerliche Anreize als auch Investitionen in Forschung und Entwicklung eine zentrale Rolle."),
    ("AI (short)", "Die Bundesregierung hat beschlossen, die Förderung erneuerbarer Energien deutlich zu verstärken und gleichzeitig den Kohleausstieg zu beschleunigen."),
]

for name, text in tests:
    enc = tokenizer(text, max_length=256, padding=True, truncation=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)
    
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    ai_prob = probs[1]
    pred_label = "AI" if ai_prob >= threshold else "HUMAN"
    
    print(f"[{name}]")
    print(f"  Prediction: {pred_label}  |  AI prob: {ai_prob:.6f}  |  Human prob: {probs[0]:.6f}")
    print()
