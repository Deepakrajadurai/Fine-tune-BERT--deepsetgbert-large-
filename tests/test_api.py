import sys
import os

# Ensure the repository root is in PYTHONPATH
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(repo_root)

from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_predict_endpoint():
    payload = {"text": "Die SPD fordert neue Verhandlungen."}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, f"Status code {response.status_code}"
    data = response.json()
    # Expected keys
    for key in ["label", "ai_prob", "human_prob", "confidence", "threshold", "model_name"]:
        assert key in data, f"Missing key {key}"
    # Probabilities should be floats between 0 and 1
    assert 0.0 <= data["ai_prob"] <= 1.0
    assert 0.0 <= data["human_prob"] <= 1.0
    assert 0.0 <= data["confidence"] <= 1.0
    # Threshold should match organic_gbert_large threshold (default 0.18)
    assert abs(data["threshold"] - 0.18) < 1e-4
    assert data["model_name"] == "Organic GBERT-large (News & Casual)"

def test_predict_all_endpoint():
    payload = {"text": "Die Bundesregierung beschließt Digitalstrategie."}
    response = client.post("/predict_all", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 9
    for item in data:
        for key in ["label", "ai_prob", "human_prob", "confidence", "threshold", "model_name"]:
            assert key in item, f"Missing key {key}"
        assert 0.0 <= item["ai_prob"] <= 1.0
        assert 0.0 <= item["human_prob"] <= 1.0
        assert 0.0 <= item["confidence"] <= 1.0
