from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.serving import app as serving_app

SAMPLE_CUSTOMER = {
    "tenure": 24,
    "MonthlyCharges": 65.5,
    "TotalCharges": 1572.0,
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "One year",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Credit card (automatic)",
}


class FakePreprocessor:
    _loaded = True

    def transform(self, customer_dict):
        return pd.DataFrame([customer_dict])

    def transform_batch(self, customer_dicts):
        return pd.DataFrame(customer_dicts)


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict.return_value = np.array([0])
    model.predict_proba.return_value = np.array([[0.72, 0.28]])
    return model


@pytest.fixture
def client(monkeypatch, mock_model):
    monkeypatch.setattr(serving_app, "MODEL", mock_model)
    monkeypatch.setattr(
        serving_app,
        "MODEL_INFO",
        {"source": "test", "name": "MockModel", "version": "test-1"},
    )
    monkeypatch.setattr(serving_app, "PREPROCESSOR", FakePreprocessor())
    return TestClient(serving_app.app)


def test_health_returns_loaded_model_metadata(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["preprocessor_loaded"] is True
    assert body["model_info"]["name"] == "MockModel"
    assert body["model_info"]["version"] == "test-1"


def test_predict_accepts_customer_record_and_returns_prediction(client):
    response = client.post("/predict", json={"customer": SAMPLE_CUSTOMER})

    assert response.status_code == 200
    body = response.json()
    assert body["churn"] == "No"
    assert body["churn_probability"] == 0.28
    assert body["model_version"] == "test-1"


def test_predict_validates_required_fields(client):
    invalid_customer = SAMPLE_CUSTOMER.copy()
    invalid_customer.pop("MonthlyCharges")

    response = client.post("/predict", json={"customer": invalid_customer})

    assert response.status_code == 422


def test_predict_validates_allowed_categories(client):
    invalid_customer = SAMPLE_CUSTOMER | {"Contract": "Weekly"}

    response = client.post("/predict", json={"customer": invalid_customer})

    assert response.status_code == 422


def test_batch_predict_returns_confidence_scores(client, mock_model):
    mock_model.predict.return_value = np.array([0, 1])
    mock_model.predict_proba.return_value = np.array([[0.72, 0.28], [0.2, 0.8]])

    response = client.post(
        "/predict/batch",
        json={
            "records": [{"customer": SAMPLE_CUSTOMER}, {"customer": SAMPLE_CUSTOMER}]
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert body["model_version"] == "test-1"
    assert body["predictions"] == [
        {"churn": "No", "churn_probability": 0.28},
        {"churn": "Yes", "churn_probability": 0.8},
    ]


def test_metrics_exposes_required_prometheus_metrics(client):
    client.post("/predict", json={"customer": SAMPLE_CUSTOMER})

    response = client.get("/metrics")

    assert response.status_code == 200
    metrics_text = response.text
    assert "churn_prediction_confidence" in metrics_text
    assert "churn_input_tenure_months" in metrics_text
    assert "churn_input_monthly_charges" in metrics_text
    assert "churn_model_version_info" in metrics_text
    assert "churn_prediction_label_total" in metrics_text

def test_predict_rejects_invalid_payload(client):
    response = client.post("/predict", json={})

    assert response.status_code in [400, 422]
