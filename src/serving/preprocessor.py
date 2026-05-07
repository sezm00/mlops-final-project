import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Expected output columns in the exact order the model was trained on
FEATURE_COLUMNS = [
    "num__tenure",
    "num__MonthlyCharges",
    "num__TotalCharges",
    "num__AvgMonthlyCharges",
    "num__ChargeDeviation",
    "num__TotalCharges_log",
    "num__NumServices",
    "num__IsHighValueCustomer",
    "cat__MultipleLines_Yes",
    "cat__InternetService_Fiber optic",
    "cat__InternetService_No",
    "cat__OnlineSecurity_Yes",
    "cat__OnlineBackup_Yes",
    "cat__DeviceProtection_Yes",
    "cat__TechSupport_Yes",
    "cat__StreamingTV_Yes",
    "cat__StreamingMovies_Yes",
    "cat__Contract_One year",
    "cat__Contract_Two year",
    "cat__PaymentMethod_Credit card (automatic)",
    "cat__PaymentMethod_Electronic check",
    "cat__PaymentMethod_Mailed check",
    "bin__gender_Male",
    "bin__Partner_Yes",
    "bin__Dependents_Yes",
    "bin__PhoneService_Yes",
    "bin__PaperlessBilling_Yes",
    "bin__SeniorCitizen_1",
]


class PreprocessingAdapter:
    """
    Wraps the two joblib pipeline artifacts so the API can go from
    raw customer dict → model-ready DataFrame in one call.
    """

    def __init__(
        self,
        feature_engineer_path: str = "models/feature_engineer.joblib",
        preprocessing_pipeline_path: str = "models/preprocessing_pipeline.joblib",
    ):
        self.feature_engineer_path = Path(feature_engineer_path)
        self.preprocessing_pipeline_path = Path(preprocessing_pipeline_path)
        self._feature_engineer = None
        self._preprocessing_pipeline = None
        self._loaded = False

    def load(self):
        """Load both pipeline artifacts from disk."""
        if not self.feature_engineer_path.exists():
            raise FileNotFoundError(
                f"Feature engineer not found: {self.feature_engineer_path}"
            )
        if not self.preprocessing_pipeline_path.exists():
            raise FileNotFoundError(
                f"Preprocessing pipeline not found: {self.preprocessing_pipeline_path}"
            )

        self._feature_engineer = joblib.load(self.feature_engineer_path)
        raw_pipeline = joblib.load(self.preprocessing_pipeline_path)

        # preprocessing_pipeline.joblib is saved as a dict with a 'pipeline' key
        if isinstance(raw_pipeline, dict):
            self._preprocessing_pipeline = raw_pipeline["pipeline"]
        else:
            self._preprocessing_pipeline = raw_pipeline

        self._loaded = True
        logger.info("Preprocessing adapter loaded successfully")

    def transform(self, customer_dict: Dict[str, Any]) -> pd.DataFrame:
        """
        Transform a raw customer dict into a model-ready DataFrame.

        Parameters
        ----------
        customer_dict : dict
            Raw customer fields (matching CustomerInput schema).

        Returns
        -------
        pd.DataFrame
            Single-row DataFrame with 28 named feature columns.
        """
        if not self._loaded:
            raise RuntimeError("PreprocessingAdapter not loaded — call .load() first")

        df = pd.DataFrame([customer_dict])

        # Step 1: feature engineering (adds derived columns)
        df = self._apply_feature_engineering(df)

        # Step 2: preprocessing pipeline (scale + encode)
        transformed = self._preprocessing_pipeline.transform(df)

        # Wrap in DataFrame with named columns
        result = pd.DataFrame(transformed, columns=FEATURE_COLUMNS)
        return result

    def transform_batch(self, customer_dicts) -> pd.DataFrame:
        """Transform a list of raw customer dicts."""
        if not self._loaded:
            raise RuntimeError("PreprocessingAdapter not loaded — call .load() first")

        df = pd.DataFrame(customer_dicts)
        df = self._apply_feature_engineering(df)
        transformed = self._preprocessing_pipeline.transform(df)
        return pd.DataFrame(transformed, columns=FEATURE_COLUMNS)

    def _apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering.

        If the loaded artifact is a fitted sklearn transformer, use .transform().
        Otherwise fall back to manual computation matching featurize.py logic.
        """
        try:
            # Try sklearn-style transform first
            if hasattr(self._feature_engineer, "transform"):
                return self._feature_engineer.transform(df)
            # Some pipelines expose a fit_transform or __call__
            if callable(self._feature_engineer):
                return self._feature_engineer(df)
        except Exception as e:
            logger.warning(
                f"Feature engineer transform failed ({e}), using manual fallback"
            )

        # Manual fallback — mirrors src/features/feature_engineer.py logic
        return self._manual_feature_engineering(df)

    def _manual_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Manual feature engineering matching the training pipeline."""
        df = df.copy()

        # Fix TotalCharges: if 0 and tenure > 0, estimate from MonthlyCharges
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        mask = (df["TotalCharges"].isna()) | (df["TotalCharges"] == 0)
        df.loc[mask, "TotalCharges"] = (
            df.loc[mask, "MonthlyCharges"] * df.loc[mask, "tenure"]
        )

        # Derived numeric features
        df["AvgMonthlyCharges"] = df["TotalCharges"] / (df["tenure"] + 1)
        df["ChargeDeviation"] = df["MonthlyCharges"] - df["AvgMonthlyCharges"]
        df["TotalCharges_log"] = np.log1p(df["TotalCharges"])

        # Count of active services
        service_cols = [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "MultipleLines",
        ]
        df["NumServices"] = df[service_cols].apply(
            lambda row: sum(1 for v in row if v == "Yes"), axis=1
        )

        # High value customer flag (top 25% by TotalCharges)
        q75 = df["TotalCharges"].quantile(0.75)
        df["IsHighValueCustomer"] = (df["TotalCharges"] >= q75).astype(int)

        # Additional flags used by preprocessing (kept for pipeline compatibility)
        df["IsNewCustomer"] = (df["tenure"] <= 12).astype(int)
        df["IsRiskySegment"] = (df["Contract"] == "Month-to-month").astype(int)

        return df
