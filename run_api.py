"""
Standalone launcher – pre-loads the model BEFORE uvicorn starts its event loop,
avoiding the Windows CUDA crash that occurs when torch is first imported inside
the asyncio worker.
"""
import sys
import os
import traceback

os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(f"=== Launcher starting (CWD: {os.getcwd()}) ===", flush=True)

# ── Step 1: import uvicorn and the FastAPI app (no torch yet) ───────────────
try:
    import uvicorn
    from api import app, _state
    print("FastAPI app imported OK", flush=True)
except BaseException:
    traceback.print_exc()
    sys.exit(1)

# ── Step 2: pre-load model BEFORE uvicorn starts ─────────────────────────────
print("Pre-loading model …", flush=True)
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import logging

    logger = logging.getLogger("run_api")
    BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model")
    THRESHOLD_PATH = os.path.join(BASE_DIR, "results", "threshold.txt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}", flush=True)

    print("  Loading tokenizer …", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("  Loading model weights …", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()

    threshold = 0.5
    if os.path.isfile(THRESHOLD_PATH):
        with open(THRESHOLD_PATH) as f:
            threshold = float(f.read().strip())

    # Populate the shared state dict that api.py reads during inference
    _state["tokenizer"] = tokenizer
    _state["model"]     = model
    _state["device"]    = device
    _state["threshold"] = threshold
    _state["ready"]     = True

    print(f"  Model loaded on {device} | threshold={threshold}", flush=True)
    print("=== Model ready — starting uvicorn ===", flush=True)

except BaseException:
    print("ERROR during model pre-load:", flush=True)
    traceback.print_exc()
    sys.exit(1)

# ── Step 3: start uvicorn (model is already in _state, no lazy loading needed)
uvicorn.run(
    app,
    host="127.0.0.1",
    port=8000,
    reload=False,
    log_level="info",
)
