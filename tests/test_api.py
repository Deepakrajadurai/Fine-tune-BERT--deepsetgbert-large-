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
    for key in ["label", "ai_prob", "human_prob", "confidence", "threshold"]:
        assert key in data, f"Missing key {key}"
    # Probabilities should be floats between 0 and 1
    assert 0.0 <= data["ai_prob"] <= 1.0
    assert 0.0 <= data["human_prob"] <= 1.0
    assert 0.0 <= data["confidence"] <= 1.0
    # Threshold should match the file value (default 0.5)
    assert data["threshold"] == 0.5
