"""
FastAPI service for AI text detection using fine-tuned deepset/gbert-large.

Model is loaded lazily on the first /predict request or dynamically on switch.
"""
from __future__ import annotations

import logging
import os
import threading
import gc

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
THRESHOLD_PATH = os.path.join(BASE_DIR, "results", "threshold.txt")

# ── Available Models Configuration ───────────────────────────────────────────
AVAILABLE_MODELS = {
    "organic_gbert_large": {
        "name": "Organic GBERT-large (News & Casual)",
        "path": os.path.join(BASE_DIR, "models", "organic_gbert_large"),
        "description": "Trained exclusively on organic, non-templated News (GNAD) and Casual (GermEval) German texts. Generalizes to stylistic indicators (indirect speech subjunctive vs. LLM direct statements) rather than exploiting template shortcuts.",
        "accuracy": "99.61%",
        "f1": "99.61%",
        "threshold": 0.18
    },
    "v5_best_model_clean": {
        "name": "v5 GBERT-large (Clean / Robust)",
        "path": os.path.join(BASE_DIR, "models", "v5_best_model_clean"),
        "description": "Our most advanced model. Trained on length-balanced, template-stripped data to eliminate layout and phrasing shortcuts. Generalizes best to unseen out-of-distribution texts.",
        "accuracy": "99.99%",
        "f1": "99.99%",
        "threshold": 0.18
    },
    "v5_best_model": {
        "name": "v5 GBERT-large (Baseline)",
        "path": os.path.join(BASE_DIR, "models", "v5_best_model"),
        "description": "Baseline model for the v5 dataset, trained with repeating template statements.",
        "accuracy": "100.00%",
        "f1": "100.00%",
        "threshold": 0.18
    },
    "best_model": {
        "name": "v1 GBERT-large (Best Model)",
        "path": os.path.join(BASE_DIR, "models", "best_model"),
        "description": "The original baseline GBERT-large trained on the ~57k dataset version.",
        "accuracy": "99.77%",
        "f1": "99.77%",
        "threshold": 0.10
    },
    "full_model": {
        "name": "v1 GBERT-large (Full Model)",
        "path": os.path.join(BASE_DIR, "models", "full_model"),
        "description": "GBERT-large trained on the full 57k dataset version without early stopping.",
        "accuracy": "99.77%",
        "f1": "99.77%",
        "threshold": 0.10
    },
    "full_model_500k_clean": {
        "name": "500k GBERT-large (Clean / Robust)",
        "path": os.path.join(BASE_DIR, "models", "full_model_500k_clean"),
        "description": "Trained on a clean, balance-sampled 500k sentence corpus. Representation collapse resolved via larger effective batch size (256) and lower learning rate (5e-6). Highly robust for identifying templates and structured outputs.",
        "accuracy": "100.00%",
        "f1": "100.00%",
        "threshold": 0.18
    },
    "full_model_500k": {
        "name": "500k GBERT-large (Collapsed)",
        "path": os.path.join(BASE_DIR, "models", "full_model_500k"),
        "description": "Model trained on the massive 500k corpus. Experienced representation collapse during training, tending to predict a single class.",
        "accuracy": "50.00%",
        "f1": "33.33%",
        "threshold": 0.50
    },
    "legal_model": {
        "name": "Legal GBERT-large (Law Focus)",
        "path": os.path.join(BASE_DIR, "models", "legal_model"),
        "description": "Fine-tuned model with special emphasis on legal text and legislative sentence style.",
        "accuracy": "98.50%",
        "f1": "98.40%",
        "threshold": 0.18
    },
    "model_100k": {
        "name": "100k GBERT-large (Intermediate)",
        "path": os.path.join(BASE_DIR, "models", "model_100k"),
        "description": "Trained on the intermediate 100k corpus split.",
        "accuracy": "99.10%",
        "f1": "99.05%",
        "threshold": 0.18
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# Lazy model state
# ─────────────────────────────────────────────────────────────────────────────
_lock: threading.Lock = threading.Lock()

ACTIVE_MODEL_KEY: str | None = None
_tokenizer = None
_model     = None
_device    = "cpu"
_ready     = False

# Shared state dict – also writable by run_api.py pre-loader
# By default, run_api.py preloads "legal_model".
_state: dict = {
    "tokenizer": None,
    "model":     None,
    "device":    "cpu",
    "threshold": 0.5,
    "ready":     False,
}


def _ensure_loaded(model_key: str = "organic_gbert_large") -> None:
    """Load model exactly once, or switch if the active model key changed."""
    global ACTIVE_MODEL_KEY, _tokenizer, _model, _device, _ready

    # 1. Fast path: pre-loader already filled _state and we match that model (legal_model)
    if _state.get("ready") and ACTIVE_MODEL_KEY is None:
        with _lock:
            if ACTIVE_MODEL_KEY is None:
                logger.info("Using model pre-loaded in launcher state (assumed: legal_model)")
                _tokenizer = _state["tokenizer"]
                _model     = _state["model"]
                _device    = _state["device"]
                ACTIVE_MODEL_KEY = "legal_model"
                _ready     = True

    # 2. Check if already loaded correct model
    if _ready and ACTIVE_MODEL_KEY == model_key:
        return

    # 3. Load or Switch Model
    with _lock:
        if _ready and ACTIVE_MODEL_KEY == model_key:
            return

        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        logger.info("Switching API active model from %s to %s ...", ACTIVE_MODEL_KEY, model_key)

        # Unload old model
        if _model is not None:
            try:
                _model.cpu()
            except Exception:
                pass
            del _model
            _model = None
        if _tokenizer is not None:
            del _tokenizer
            _tokenizer = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _device = device

        m_info = AVAILABLE_MODELS[model_key]
        model_path = m_info["path"]

        logger.info("Loading tokenizer from %s ...", model_path)
        _tokenizer = AutoTokenizer.from_pretrained(model_path)

        logger.info("Loading model weights from %s ...", model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()
        _model = model

        ACTIVE_MODEL_KEY = model_key
        _ready = True
        logger.info("Model %s ready on API.", model_key)


# ─────────────────────────────────────────────────────────────────────────────
# App  (no lifespan — avoids torch/asyncio conflict on Windows)
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="AI Text Detection API")

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
    model: str = Field("organic_gbert_large", description="Model key to use for inference")
    threshold: float | None = Field(None, description="Optional override for decision threshold")


class PredictAllRequest(BaseModel):
    text: str = Field(..., description="Text to classify")
    threshold: float | None = Field(None, description="Optional override for decision threshold")


class PredictResponse(BaseModel):
    label:      str
    ai_prob:    float
    human_prob: float
    confidence: float
    threshold:  float
    model_name: str


# ─────────────────────────────────────────────────────────────────────────────
# Inference helper
# ─────────────────────────────────────────────────────────────────────────────
def _predict(text: str, model_key: str, custom_threshold: float | None) -> dict:
    import torch                       # already cached after first load

    t = custom_threshold
    if t is None:
        t = AVAILABLE_MODELS[model_key].get("threshold", 0.18)

    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = _model(**inputs).logits.squeeze()
        if logits.ndim == 0:
            logits = logits.unsqueeze(0)
        probs  = torch.softmax(logits, dim=0).cpu().numpy()

    # label 0 = human, label 1 = AI
    human_prob = float(probs[0])
    ai_prob    = float(probs[1])
    label      = "Human" if human_prob >= t else "AI"
    confidence = max(ai_prob, human_prob)

    return {
        "label":      label,
        "ai_prob":    ai_prob,
        "human_prob": human_prob,
        "confidence": confidence,
        "threshold":  t,
        "model_name": AVAILABLE_MODELS[model_key]["name"],
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
        "device":       _device,
        "active_model": ACTIVE_MODEL_KEY,
        "threshold":    AVAILABLE_MODELS[ACTIVE_MODEL_KEY].get("threshold", 0.18) if ACTIVE_MODEL_KEY else 0.18,
    }


@app.get("/models")
def list_models():
    """Return the list of all available models and their metadata."""
    return [
        {
            "key": k,
            "name": v["name"],
            "description": v["description"],
            "accuracy": v["accuracy"],
            "f1": v["f1"],
            "threshold": v.get("threshold", 0.18)
        }
        for k, v in AVAILABLE_MODELS.items()
    ]


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest):
    if req.model not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model category: {req.model}")
    try:
        _ensure_loaded(req.model)
    except Exception as exc:
        logger.exception("Model loading failed")
        raise HTTPException(status_code=503, detail=f"Model loading failed: {exc}")
    try:
        return _predict(req.text, req.model, req.threshold)
    except Exception as exc:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict_all", response_model=list[PredictResponse])
def predict_all_endpoint(req: PredictAllRequest):
    results = []
    for model_key in AVAILABLE_MODELS.keys():
        try:
            _ensure_loaded(model_key)
            res = _predict(req.text, model_key, req.threshold)
            results.append(res)
        except Exception as exc:
            logger.exception("Prediction failed for model %s", model_key)
            t = req.threshold or AVAILABLE_MODELS[model_key].get("threshold", 0.18)
            results.append({
                "label": "Error",
                "ai_prob": 0.0,
                "human_prob": 0.0,
                "confidence": 0.0,
                "threshold": t,
                "model_name": AVAILABLE_MODELS[model_key]["name"]
            })
    return results
