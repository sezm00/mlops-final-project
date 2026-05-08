import pandas as pd
import pytest

from src.serving.preprocessor import PreprocessingAdapter


@pytest.fixture
def sample_customer():
    return {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.5,
        "TotalCharges": 850.0,
    }


def test_manual_feature_engineering(sample_customer):
    adapter = PreprocessingAdapter()

    df = pd.DataFrame([sample_customer])

    result = adapter._manual_feature_engineering(df)

    assert "AvgMonthlyCharges" in result.columns
    assert "ChargeDeviation" in result.columns
    assert "TotalCharges_log" in result.columns
    assert "NumServices" in result.columns
    assert "IsHighValueCustomer" in result.columns


def test_transform_requires_load(sample_customer):
    adapter = PreprocessingAdapter()

    with pytest.raises(RuntimeError):
        adapter.transform(sample_customer)


def test_transform_batch_requires_load(sample_customer):
    adapter = PreprocessingAdapter()

    with pytest.raises(RuntimeError):
        adapter.transform_batch([sample_customer])