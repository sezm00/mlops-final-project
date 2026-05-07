import os

import numpy as np
import pandas as pd
import pytest
import yaml

# Import pipeline functions and classes directly from your source tree
from src.data.prepare import clean, encode_target
from src.data.preprocess import build_pipeline
from src.features.feature_engineer import FeatureEngineer


@pytest.fixture
def real_pipeline_params():
    """
    Dynamically loads your actual params.yaml file so tests match
    your production infrastructure configurations perfectly.
    """
    params_path = "configs/params.yaml"
    if not os.path.exists(params_path):
        raise FileNotFoundError(
            f"Critical configuration file missing at: {params_path}"
        )

    with open(params_path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Data Preparation & Cleaning (Covers src/data/prepare.py)
# ─────────────────────────────────────────────────────────────────────────────


def test_prepare_clean_and_target_encoding(real_pipeline_params):
    """
    Executes raw string cleaning, redundant label consolidation,
    and target variable mapping directly from prepare.py.
    """
    # Create a comprehensive raw data mockup containing all keys expected by clean() loops
    raw_df = pd.DataFrame(
        {
            "customerID": ["1234-AAAA", "5678-BBBB"],
            "gender": ["Female", "Male"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "Yes"],
            "tenure": [0, 10],
            "PhoneService": ["No", "Yes"],
            "MultipleLines": ["No phone service", "No"],
            "InternetService": ["DSL", "Fiber optic"],
            "OnlineSecurity": ["No internet service", "Yes"],
            "OnlineBackup": ["No internet service", "No"],
            "DeviceProtection": ["No internet service", "Yes"],
            "TechSupport": ["No internet service", "No"],
            "StreamingTV": ["No internet service", "Yes"],
            "StreamingMovies": ["No internet service", "No"],
            "Contract": ["Month-to-month", "Two year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check"],
            "MonthlyCharges": [29.85, 56.95],
            "TotalCharges": [
                " ",
                "1889.5",
            ],  # Testing numeric string whitespace coercion
            "Churn": ["No", "Yes"],
        }
    )

    # 1. Test target encoding function execution
    encoded_df = encode_target(raw_df.copy(), target="Churn")
    assert encoded_df["Churn"].tolist() == [0, 1]

    # 2. Test clean function execution
    cleaned_df = clean(encoded_df, real_pipeline_params)

    assert "customerID" not in cleaned_df.columns
    assert cleaned_df.loc[0, "TotalCharges"] == 0.0  # Verify tenure=0 fix handled it
    assert cleaned_df.loc[1, "TotalCharges"] == 1889.5
    assert (
        cleaned_df.loc[0, "OnlineSecurity"] == "No"
    )  # Verify category consolidation mapping


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Feature Engineering (Covers src/features/feature_engineer.py)
# ─────────────────────────────────────────────────────────────────────────────


def test_feature_engineering_transformer(real_pipeline_params):
    """
    Executes both the stateful fit and transform stages of FeatureEngineer
    to ensure math tracking and downstream row calculations cover the code.
    """
    engineered_input = pd.DataFrame(
        {
            "gender": ["Male", "Female", "Male"],
            "Partner": ["Yes", "No", "Yes"],
            "Dependents": ["No", "Yes", "No"],
            "SeniorCitizen": [0, 1, 0],
            "PhoneService": ["Yes", "Yes", "No"],
            "PaperlessBilling": ["Yes", "No", "Yes"],
            "tenure": [5, 24, 2],
            "MonthlyCharges": [20.0, 80.0, 100.0],
            "TotalCharges": [100.0, 1920.0, 200.0],
            "OnlineSecurity": ["No", "Yes", "No"],
            "OnlineBackup": ["No", "No", "No"],
            "DeviceProtection": ["No", "No", "No"],
            "TechSupport": ["No", "No", "No"],
            "StreamingTV": ["No", "No", "No"],
            "StreamingMovies": ["No", "No", "No"],
            "MultipleLines": ["No", "Yes", "No"],
            "InternetService": ["DSL", "Fiber optic", "DSL"],
            "Contract": ["Month-to-month", "One year", "Month-to-month"],
            "PaymentMethod": [
                "Electronic check",
                "Credit card (automatic)",
                "Electronic check",
            ],
        }
    )

    engineer = FeatureEngineer(params=real_pipeline_params)

    # Run fit to activate statistics calculations code paths
    engineer.fit(engineered_input)
    assert "high_value_threshold" in engineer.stats_

    # Run transform to check step execution coverage
    processed_df = engineer.transform(engineered_input)

    # Validate output columns were created by code blocks
    assert "AvgMonthlyCharges" in processed_df.columns
    assert "ChargeDeviation" in processed_df.columns
    assert "TotalCharges_log" in processed_df.columns
    assert "NumServices" in processed_df.columns
    assert "IsHighValueCustomer" in processed_df.columns


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Preprocessing Pipeline Object (Covers src/data/preprocess.py)
# ─────────────────────────────────────────────────────────────────────────────


def test_scikit_learn_preprocessing_pipeline(real_pipeline_params):
    """
    Instantiates and applies the custom structural ColumnTransformer
    built by preprocess.py to verify scaling and OHE encoding functionality.
    """
    # This payload must mirror exactly what emerges from the featurization stage,
    # including all engineered features and original string categorical properties.
    preproc_input = pd.DataFrame(
        {
            "tenure": [12, 24, 36],
            "MonthlyCharges": [30.5, 65.2, 99.1],
            "TotalCharges": [360.0, np.nan, 3564.0],  # Test numeric imputer fallback
            "AvgMonthlyCharges": [30.5, 65.2, 99.1],
            "ChargeDeviation": [0.0, 0.0, 0.0],
            "TotalCharges_log": [5.88, 7.0, 8.17],
            "NumServices": [1, 2, 4],
            "IsHighValueCustomer": [0, 0, 1],
            "MultipleLines": ["No", "Yes", "No"],
            "InternetService": ["DSL", "Fiber optic", "No"],
            "OnlineSecurity": ["No", "Yes", "No"],
            "OnlineBackup": ["Yes", "No", "No"],
            "DeviceProtection": ["No", "No", "Yes"],
            "TechSupport": ["No", "No", "No"],
            "StreamingTV": ["Yes", "No", "Yes"],
            "StreamingMovies": ["No", "Yes", "No"],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Credit card (automatic)",
            ],
            "gender": ["Male", "Female", "Male"],
            "Partner": ["Yes", "No", "Yes"],
            "Dependents": ["No", "Yes", "No"],
            "PhoneService": ["Yes", "Yes", "No"],
            "PaperlessBilling": ["Yes", "No", "Yes"],
            "SeniorCitizen": [0, 1, 0],
        }
    )

    # Instantiate the exact pipeline configuration structure used in production
    pipeline = build_pipeline(real_pipeline_params)

    # Fit and transform dataset payload
    transformed_matrix = pipeline.fit_transform(preproc_input)
    feature_names = pipeline.named_steps["preprocessing"].get_feature_names_out()

    # Check validation requirements
    assert transformed_matrix.shape[0] == 3
    assert not np.isnan(
        transformed_matrix
    ).any(), "NaN values found after imputation layer!"
    assert any(name.startswith("num__") for name in feature_names)
    assert any(name.startswith("cat__") for name in feature_names)
    assert any(name.startswith("bin__") for name in feature_names)
