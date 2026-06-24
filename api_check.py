"""Step-by-step import test to find the exact crash point inside api.py"""
import sys, os, traceback
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("CWD:", os.getcwd(), flush=True)

# Step 1 – bare imports
print("Step 1: bare imports", flush=True)
try:
    from contextlib import asynccontextmanager
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import logging, os as _os
    print("  OK", flush=True)
except BaseException as e:
    print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

# Step 2 – pydantic model definitions
print("Step 2: pydantic models", flush=True)
try:
    class PredictRequest(BaseModel):
        text: str = Field(..., description="Text to classify")
    class PredictResponse(BaseModel):
        label: str
        ai_prob: float
        human_prob: float
        confidence: float
        threshold: float
    print("  OK", flush=True)
except BaseException as e:
    print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

# Step 3 – create FastAPI app
print("Step 3: FastAPI app creation", flush=True)
try:
    app = FastAPI(title="AI Text Detection API")
    print("  OK", flush=True)
except BaseException as e:
    print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

# Step 4 – lifespan decorator
print("Step 4: lifespan decorator", flush=True)
try:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        yield
    app2 = FastAPI(title="Test", lifespan=lifespan)
    print("  OK", flush=True)
except BaseException as e:
    print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

# Step 5 – route decorator
print("Step 5: route decorator", flush=True)
try:
    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/predict")
    def pred(req: PredictRequest):
        return {}
    print("  OK", flush=True)
except BaseException as e:
    print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

print("\n=== ALL STEPS PASSED ===", flush=True)
