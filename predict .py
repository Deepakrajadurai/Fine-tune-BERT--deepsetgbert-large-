"""
Step 4 — End-to-End Inference Pipeline
========================================
Input  : any German text (string or .txt file)
Output : { label, confidence, verdict }

Usage:
    # Single text
    python 04_predict.py --text "Die Bundesregierung hat beschlossen..."

    # From file
    python 04_predict.py --file my_text.txt

    # Batch CSV
    python 04_predict.py --csv input.csv --text_col text --out predictions.csv
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from sklearn.metrics import roc_curve
import argparse
import json
import logging
import re
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.cuda.amp import autocast
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODEL_DIR   = "models/best_model"
RESULTS_DIR = "results"
MAX_LENGTH  = 256
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16    = True and DEVICE == "cuda"


# ---------------------------------------------------------------------------
# PRE-PROCESSING (same rules as training pipeline)
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

def clean_shortcuts(text: str) -> str:
    return replace_domain_markers(text)

def preprocess(text: str) -> str:
    text = text.strip()
    text = clean_shortcuts(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 1024:
        text = text[:1024].rsplit(" ", 1)[0]
    return text


# ---------------------------------------------------------------------------
# POST-PROCESSING: multi-sentence consensus
# ---------------------------------------------------------------------------
def split_into_chunks(text: str, tokenizer, max_length: int) -> list[str]:
    """
    If a text is very long, split it into overlapping chunks and
    aggregate predictions. This improves accuracy on long documents.
    """
    words  = text.split()
    chunks = []
    step   = max_length // 2           # 50% overlap
    size   = max_length

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + size])
        if len(chunk.split()) < 10:    # skip tiny trailing chunks
            continue
        chunks.append(chunk)
        if i + size >= len(words):
            break

    return chunks if chunks else [text]


# ---------------------------------------------------------------------------
# PREDICTOR CLASS
# ---------------------------------------------------------------------------
class AITextDetector:
    def __init__(self, model_dir: str = MODEL_DIR, threshold: float | None = None):
        log.info(f"Loading model from {model_dir}...")
        is_bert = os.path.exists(os.path.join(model_dir, "vocab.txt")) or 'gbert' in model_dir
        if is_bert:
            from transformers import BertTokenizer, BertForSequenceClassification
            self.tokenizer = BertTokenizer.from_pretrained(model_dir)
            self.model     = BertForSequenceClassification.from_pretrained(model_dir).to(DEVICE)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model     = AutoModelForSequenceClassification.from_pretrained(
                                model_dir).to(DEVICE)
        self.model.eval()

        # Load calibrated threshold
        t_file = Path(RESULTS_DIR) / "threshold.txt"
        if threshold is not None:
            self.threshold = threshold
        elif t_file.exists():
            self.threshold = float(t_file.read_text().strip())
        else:
            self.threshold = 0.5
        log.info(f"Decision threshold: {self.threshold}")

    @torch.no_grad()
    def _predict_batch(self, texts: list[str]) -> np.ndarray:
        """Returns softmax probabilities shape (N, 2)."""
        enc = self.tokenizer(
            texts,
            max_length=MAX_LENGTH,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)

        with autocast(enabled=USE_FP16):
            logits = self.model(input_ids=input_ids,
                                attention_mask=attention_mask).logits
        return torch.softmax(logits, dim=-1).cpu().numpy()

    def predict(self, text: str) -> dict:
        """
        Predict a single text. Returns:
          {
            label      : 0 (human) or 1 (AI),
            confidence : float 0–1  (probability of predicted class),
            ai_prob    : float 0–1  (raw probability of AI class),
            verdict    : str,
            threshold  : float,
          }
        """
        text   = preprocess(text)
        chunks = split_into_chunks(text, self.tokenizer, MAX_LENGTH)

        # Predict each chunk
        probs  = self._predict_batch(chunks)          # shape (n_chunks, 2)
        # Aggregate: take the mean AI probability across chunks
        mean_ai_prob = float(probs[:, 1].mean())

        label  = 1 if mean_ai_prob >= self.threshold else 0
        conf   = mean_ai_prob if label == 1 else 1 - mean_ai_prob

        if label == 1:
            if mean_ai_prob >= 0.90:
                verdict = "KI-generiert (sehr hohe Konfidenz)"
            elif mean_ai_prob >= 0.75:
                verdict = "KI-generiert (hohe Konfidenz)"
            else:
                verdict = "KI-generiert (mittlere Konfidenz)"
        else:
            if mean_ai_prob <= 0.10:
                verdict = "Menschlich verfasst (sehr hohe Konfidenz)"
            elif mean_ai_prob <= 0.25:
                verdict = "Menschlich verfasst (hohe Konfidenz)"
            else:
                verdict = "Menschlich verfasst (mittlere Konfidenz) — grenzwertig"

        return {
            "label":      label,
            "confidence": round(conf, 4),
            "ai_prob":    round(mean_ai_prob, 4),
            "human_prob": round(1 - mean_ai_prob, 4),
            "verdict":    verdict,
            "threshold":  self.threshold,
            "n_chunks":   len(chunks),
        }

    def predict_batch_csv(self, csv_path: str, text_col: str,
                          out_path: str, batch_size: int = 64):
        """Batch-predict an entire CSV file."""
        df   = pd.read_csv(csv_path)
        texts = df[text_col].fillna("").tolist()
        log.info(f"Predicting {len(texts):,} rows from {csv_path}...")

        ai_probs, labels, verdicts = [], [], []

        for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
            batch_texts = texts[i:i + batch_size]
            # For batch inference we skip chunking (assume reasonable length)
            probs       = self._predict_batch(batch_texts)
            for p in probs:
                ai_p    = float(p[1])
                label   = 1 if ai_p >= self.threshold else 0
                ai_probs.append(round(ai_p, 4))
                labels.append(label)
                verdicts.append(
                    "AI" if label == 1 else "Human"
                )

        df["ai_probability"] = ai_probs
        df["predicted_label"] = labels
        df["verdict"]         = verdicts
        df.to_csv(out_path, index=False)
        log.info(f"Saved predictions to {out_path}")

        # Summary
        n_ai    = sum(l == 1 for l in labels)
        n_human = sum(l == 0 for l in labels)
        log.info(f"Results: {n_human:,} human, {n_ai:,} AI-generated")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Detect AI-generated German text")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text",  type=str, help="Text string to classify")
    group.add_argument("--file",  type=str, help="Path to .txt file")
    group.add_argument("--csv",   type=str, help="Path to CSV for batch prediction")

    parser.add_argument("--text_col",  default="text",           help="Text column in CSV")
    parser.add_argument("--out",       default="predictions.csv", help="Output CSV path")
    parser.add_argument("--threshold", type=float, default=None,  help="Override threshold")
    args = parser.parse_args()

    detector = AITextDetector(threshold=args.threshold)

    if args.text:
        result = detector.predict(args.text)
        print("\n" + "=" * 50)
        print("AI TEXT DETECTION RESULT")
        print("=" * 50)
        print(f"Verdict    : {result['verdict']}")
        print(f"Label      : {'AI-generated (1)' if result['label'] == 1 else 'Human-written (0)'}")
        print(f"AI prob    : {result['ai_prob']:.4f}")
        print(f"Human prob : {result['human_prob']:.4f}")
        print(f"Confidence : {result['confidence']:.4f}")
        print(f"Threshold  : {result['threshold']}")
        print("=" * 50)

    elif args.file:
        text   = Path(args.file).read_text(encoding="utf-8")
        result = detector.predict(text)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.csv:
        detector.predict_batch_csv(
            args.csv, args.text_col, args.out
        )


if __name__ == "__main__":
    main()