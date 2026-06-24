"""
FastAPI service for AI text detection using fine-tuned deepset/gbert-large.

Model is loaded lazily on the first /predict request so uvicorn starts cleanly
without any torch/CUDA initialisation in the startup path.
"""
from __future__ import annotations

import logging
import os
import threading

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH     = os.path.join(BASE_DIR, "models", "best_model")
THRESHOLD_PATH = os.path.join(BASE_DIR, "results", "threshold.txt")

# ─────────────────────────────────────────────────────────────────────────────
# Lazy model state
# ─────────────────────────────────────────────────────────────────────────────
_lock: threading.Lock = threading.Lock()

_tokenizer = None
_model     = None
_device    = "cpu"
_threshold = 0.5
_ready     = False

# Shared state dict – also writable by run_api.py pre-loader
_state: dict = {
    "tokenizer": None,
    "model":     None,
    "device":    "cpu",
    "threshold": 0.5,
    "ready":     False,
}


def _ensure_loaded() -> None:
    """Load model exactly once. If run_api.py already populated _state, this is a no-op."""
    global _tokenizer, _model, _device, _threshold, _ready

    # Fast path: pre-loader already filled _state
    if _state.get("ready"):
        _tokenizer = _state["tokenizer"]
        _model     = _state["model"]
        _device    = _state["device"]
        _threshold = _state["threshold"]
        _ready     = True
        return

    if _ready:
        return

    with _lock:
        if _ready:
            return
        if _state.get("ready"):   # re-check after acquiring lock
            _tokenizer = _state["tokenizer"]
            _model     = _state["model"]
            _device    = _state["device"]
            _threshold = _state["threshold"]
            _ready     = True
            return

        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _device = device
        logger.info("Loading model on %s from %s …", device, MODEL_PATH)

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()
        _model = model
        logger.info("Model ready.")

        if os.path.isfile(THRESHOLD_PATH):
            with open(THRESHOLD_PATH) as f:
                _threshold = float(f.read().strip())
        logger.info("Threshold: %s", _threshold)
        _ready = True


# ─────────────────────────────────────────────────────────────────────────────
# App  (no lifespan — avoids torch/asyncio conflict on Windows)
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="AI Text Detection API")

# Allow Swagger UI and local browser clients to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(..., description="Text to classify")


class PredictResponse(BaseModel):
    label:      str
    ai_prob:    float
    human_prob: float
    confidence: float
    threshold:  float


# ─────────────────────────────────────────────────────────────────────────────
# Inference helper
# ─────────────────────────────────────────────────────────────────────────────
def _predict(text: str) -> dict:
    import torch                       # already cached after first load

    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = _model(**inputs).logits.squeeze()
        probs  = torch.softmax(logits, dim=0).cpu().numpy()

    # label 0 = human, label 1 = AI
    human_prob = float(probs[0])
    ai_prob    = float(probs[1])
    label      = "Human" if human_prob >= _threshold else "AI"
    confidence = max(ai_prob, human_prob)

    return {
        "label":      label,
        "ai_prob":    ai_prob,
        "human_prob": human_prob,
        "confidence": confidence,
        "threshold":  _threshold,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    """Redirect homepage to the interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model_loaded": _ready or _state.get("ready", False),
        "device":       _state.get("device", _device),
        "threshold":    _state.get("threshold", _threshold),
    }


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest):
    try:
        _ensure_loaded()           # no-op after first call
    except Exception as exc:
        logger.exception("Model loading failed")
        raise HTTPException(status_code=503, detail=f"Model loading failed: {exc}")
    try:
        return _predict(req.text)
    except Exception as exc:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(exc))
