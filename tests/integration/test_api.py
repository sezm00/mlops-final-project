from fastapi.testclient import TestClient
from src.serving.app import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200

def test_predict():
    sample = {
        "tenure": 1,
        "MonthlyCharges": 50
    }

    response = client.post("/predict", json=sample)
    assert response.status_code == 200