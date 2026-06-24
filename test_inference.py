import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import traceback

try:
    import torch
    import numpy as np
    from transformers import BertTokenizer, BertForSequenceClassification
    import pandas as pd

    MODEL_DIR = "models/best_model"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {DEVICE}")

    # Load model
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()

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

    test_df = pd.read_csv("Data/test.csv", nrows=10)
    for i, row in test_df.iterrows():
        label = "HUMAN" if row["label"] == 0 else "AI"
        tests.append((f"TEST_CSV_{label}_row{i}", str(row["text"])[:500]))
        if i >= 5:
            break

    print(f"\nRunning {len(tests)} tests...\n")

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
        print(f"  Text: {text[:100]}...")
        print()

except Exception as e:
    print("CRASHED:")
    traceback.print_exc()
