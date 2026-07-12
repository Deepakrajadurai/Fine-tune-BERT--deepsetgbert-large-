import sys
import os

# Ensure the repository root is in PYTHONPATH
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(repo_root)

from fastapi.testclient import TestClient
from server import app, AVAILABLE_MODELS

client = TestClient(app)

def test_models_catalog_endpoint():
    """Verify that /models returns all 9 registered models."""
    response = client.get("/models")
    assert response.status_code == 200
    models = response.json()
    assert len(models) == 9, f"Expected 9 models, got {len(models)}"
    
    keys = {m["key"] for m in models}
    expected_keys = set(AVAILABLE_MODELS.keys())
    assert keys == expected_keys, f"Mismatch in model keys: {keys} vs {expected_keys}"

def test_predict_across_all_models():
    """Verify that predictions can run on each model, trigger dynamic switching, and return valid results."""
    test_text = "Die Bundesregierung plant neue Gesetze zur Förderung erneuerbarer Energien in ganz Deutschland."
    
    # Iterate over all registered models and run prediction
    for model_key in AVAILABLE_MODELS.keys():
        print(f"Testing model loading & inference for: {model_key}")
        payload = {
            "text": test_text,
            "model": model_key
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200, f"Failed prediction for {model_key}: status {response.status_code}"
        
        data = response.json()
        assert "label" in data
        assert data["label"] in ["Human", "AI"]
        assert "ai_prob" in data
        assert "human_prob" in data
        assert "confidence" in data
        assert "threshold" in data
        assert "model_name" in data
        
        assert 0.0 <= data["ai_prob"] <= 1.0
        assert 0.0 <= data["human_prob"] <= 1.0
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["model_name"] == AVAILABLE_MODELS[model_key]["name"]
        
        # Verify default thresholds match model configuration
        expected_t = AVAILABLE_MODELS[model_key].get("threshold", 0.18)
        assert abs(data["threshold"] - expected_t) < 1e-4, f"Threshold mismatch for {model_key}: {data['threshold']} vs {expected_t}"

def test_custom_threshold_override():
    """Verify that passing a custom threshold override works correctly."""
    test_text = "SPD und CDU beraten über Entwurf."
    payload = {
        "text": test_text,
        "model": "organic_gbert_large",
        "threshold": 0.85
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert abs(data["threshold"] - 0.85) < 1e-4

def test_predict_all_models_simultaneously():
    """Verify that /predict_all runs sequentially across all models and returns 9 valid results."""
    test_text = "Die Digitalisierung in Deutschland schreitet weiter voran."
    payload = {
        "text": test_text
    }
    response = client.post("/predict_all", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 9
    for item in data:
        assert "label" in item
        assert "ai_prob" in item
        assert "human_prob" in item
        assert "confidence" in item
        assert "threshold" in item
        assert "model_name" in item
        assert 0.0 <= item["ai_prob"] <= 1.0
        assert 0.0 <= item["human_prob"] <= 1.0
        assert 0.0 <= item["confidence"] <= 1.0

