"""
Fully standalone FastAPI server for AI text detection.
Import order: torch/transformers FIRST, then fastapi/pydantic.
This avoids the DLL conflict that crashes torch when fastapi is loaded first.
"""
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH     = os.path.join(BASE_DIR, "models", "best_model")
THRESHOLD_PATH = os.path.join(BASE_DIR, "results", "threshold.txt")

# ── Step 1: Load torch and model FIRST ────────────────────────────────────────
logger.info("Importing torch …")
import torch
logger.info("torch OK – CUDA: %s", torch.cuda.is_available())

from transformers import AutoTokenizer, AutoModelForSequenceClassification
logger.info("transformers OK")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Loading tokenizer from %s …", MODEL_PATH)
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH)
logger.info("Tokenizer loaded.")

logger.info("Loading model …")
MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
MODEL.to(DEVICE)
MODEL.eval()
logger.info("Model loaded on %s", DEVICE)

THRESHOLD = 0.5
if os.path.isfile(THRESHOLD_PATH):
    with open(THRESHOLD_PATH) as f:
        THRESHOLD = float(f.read().strip())
logger.info("Threshold: %s", THRESHOLD)

# ── Step 2: Now import fastapi/pydantic ───────────────────────────────────────
logger.info("Importing FastAPI …")
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
import uvicorn
logger.info("FastAPI OK")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="AI Text Detection API – gbert-large")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    text: str = Field(..., description="German text to classify")


class PredictResponse(BaseModel):
    label:      str
    ai_prob:    float
    human_prob: float
    confidence: float
    threshold:  float


def _run_inference(text: str) -> dict:
    inputs = TOKENIZER(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        logits = MODEL(**inputs).logits.squeeze()
        probs  = torch.softmax(logits, dim=0).cpu().numpy()
    human_prob = float(probs[0])
    ai_prob    = float(probs[1])
    label      = "Human" if human_prob >= THRESHOLD else "AI"
    return {
        "label":      label,
        "ai_prob":    ai_prob,
        "human_prob": human_prob,
        "confidence": max(ai_prob, human_prob),
        "threshold":  THRESHOLD,
    }


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "threshold": THRESHOLD, "model": "gbert-large"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        return _run_inference(req.text)
    except Exception as exc:
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail=str(exc))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=== Starting uvicorn ===")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
