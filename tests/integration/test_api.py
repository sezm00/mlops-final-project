import os
import pytest
import httpx
from unittest.mock import patch, MagicMock
import numpy as np

# Base URL from env (CI sets API_BASE_URL, local defaults to TestClient)
API_BASE_URL = os.getenv("API_BASE_URL", None)

# ── Fixtures ──────────────────────────────────────────────────────────────────

VALID_PAYLOAD = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 844.20,
}

INVALID_PAYLOAD = {
    "gender": "Unknown",          # invalid literal
    "SeniorCitizen": 5,           # out of range
    "tenure": -1,                 # negative
    "MonthlyCharges": "not_a_float",  # wrong type
}


@pytest.fixture(scope="module")
def mock_model():
    """Mock model that returns deterministic predictions."""
    model = MagicMock()
    model.predict.return_value = np.array([1])
    model.predict_proba.return_value = np.array([[0.25, 0.75]])
    return model


@pytest.fixture(scope="module")
def client(mock_model):
    """Return either a live httpx client (CI) or TestClient with mocked model."""
    if API_BASE_URL:
        # CI: real server running
        with httpx.Client(base_url=API_BASE_URL, timeout=10.0) as c:
            yield c
    else:
        # Local: use FastAPI TestClient with mocked model
        from fastapi.testclient import TestClient
        from src.serving.app import app, MODEL_STATE

        MODEL_STATE["model"] = mock_model
        MODEL_STATE["version"] = "test-v1"
        MODEL_STATE["run_id"] = "test-run-id"
        MODEL_STATE["loaded_at"] = __import__("time").time()

        with TestClient(app) as c:
            yield c


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_required_fields(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_version" in data

    def test_health_status_value(self, client):
        data = client.get("/health").json()
        assert data["status"] in ("healthy", "degraded")


class TestPredict:
    def test_predict_valid_input_returns_200(self, client):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 200

    def test_predict_response_schema(self, client):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert "churn" in data
        assert "churn_probability" in data
        assert "model_version" in data

    def test_predict_churn_is_bool(self, client):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert isinstance(data["churn"], bool)

    def test_predict_probability_in_range(self, client):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert 0.0 <= data["churn_probability"] <= 1.0

    def test_predict_invalid_gender_returns_422(self, client):
        payload = {**VALID_PAYLOAD, "gender": "Unknown"}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_predict_negative_tenure_returns_422(self, client):
        payload = {**VALID_PAYLOAD, "tenure": -1}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_predict_missing_field_returns_422(self, client):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "MonthlyCharges"}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_predict_invalid_contract_type_returns_422(self, client):
        payload = {**VALID_PAYLOAD, "Contract": "Weekly"}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422


class TestBatchPredict:
    def test_batch_predict_valid_input(self, client):
        payload = {"instances": [VALID_PAYLOAD, VALID_PAYLOAD]}
        resp = client.post("/predict/batch", json=payload)
        assert resp.status_code == 200

    def test_batch_predict_response_count(self, client):
        payload = {"instances": [VALID_PAYLOAD, VALID_PAYLOAD, VALID_PAYLOAD]}
        data = client.post("/predict/batch", json=payload).json()
        assert data["total"] == 3
        assert len(data["predictions"]) == 3

    def test_batch_predict_empty_list_returns_422(self, client):
        payload = {"instances": []}
        resp = client.post("/predict/batch", json=payload)
        assert resp.status_code == 422

    def test_batch_predict_each_has_probability(self, client):
        payload = {"instances": [VALID_PAYLOAD]}
        data = client.post("/predict/batch", json=payload).json()
        for pred in data["predictions"]:
            assert 0.0 <= pred["churn_probability"] <= 1.0


class TestMetrics:
    def test_metrics_endpoint_returns_200(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_contains_inference_counter(self, client):
        # Make a prediction first to generate metrics
        client.post("/predict", json=VALID_PAYLOAD)
        text = client.get("/metrics").text
        assert "model_inference_total" in text

    def test_metrics_contains_latency_histogram(self, client):
        text = client.get("/metrics").text
        assert "model_inference_latency_seconds" in text