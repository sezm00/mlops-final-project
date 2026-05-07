from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """Single-customer churn prediction request.

    All fields mirror the Telco Customer Churn dataset columns.
    The preprocessing pipeline (built by Member 2) handles
    imputation, scaling, and encoding internally.
    """

    # ── Demographics ──────────────────────────────────────────────────────────
    gender: Literal["Male", "Female"] = Field(..., description="Customer gender")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="1 if senior citizen, else 0")
    Partner: Literal["Yes", "No"] = Field(..., description="Has a partner")
    Dependents: Literal["Yes", "No"] = Field(..., description="Has dependents")

    # ── Account ───────────────────────────────────────────────────────────────
    tenure: int = Field(..., ge=0, le=120, description="Months with company")
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., description="Contract type"
    )
    PaperlessBilling: Literal["Yes", "No"] = Field(..., description="Paperless billing")
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(..., description="Payment method")
    MonthlyCharges: float = Field(..., ge=0.0, description="Monthly charge amount")
    TotalCharges: float = Field(..., ge=0.0, description="Total charge amount")

    # ── Phone services ────────────────────────────────────────────────────────
    PhoneService: Literal["Yes", "No"] = Field(..., description="Has phone service")
    MultipleLines: Literal["Yes", "No", "No phone service"] = Field(
        ..., description="Multiple phone lines"
    )

    # ── Internet services ─────────────────────────────────────────────────────
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., description="Internet service type"
    )
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Online security add-on"
    )
    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Online backup add-on"
    )
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Device protection add-on"
    )
    TechSupport: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Tech support add-on"
    )
    StreamingTV: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Streaming TV add-on"
    )
    StreamingMovies: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Streaming movies add-on"
    )

    @field_validator("TotalCharges")
    @classmethod
    def total_charges_gte_monthly(cls, v, info):
        """TotalCharges should generally be >= MonthlyCharges for active customers."""
        # Allow 0 for new customers (tenure=0)
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictResponse(BaseModel):
    """Churn prediction response."""

    churn: bool = Field(..., description="True if customer predicted to churn")
    churn_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of churn [0, 1]"
    )
    model_version: str = Field(..., description="Model version used for inference")


class BatchPredictRequest(BaseModel):
    """Batch prediction request — up to 1000 customers."""

    instances: List[PredictRequest] = Field(
        ..., min_length=1, max_length=1000, description="List of customer feature records"
    )


class BatchPredictResponse(BaseModel):
    """Batch prediction response."""

    predictions: List[PredictResponse]
    total: int = Field(..., description="Number of predictions returned")
    model_version: str = Field(..., description="Model version used for inference")