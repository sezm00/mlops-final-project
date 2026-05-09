from typing import List, Optional, Literal
from pydantic import BaseModel, Field, conint, confloat


# =========================
# RAW CUSTOMER INPUT
# =========================

class CustomerInput(BaseModel):
    """
    Raw Telco customer record (pre-processing stage).
    Strictly validated for production safety.
    """

    # ── Numeric features ─────────────────────────────

    tenure: conint(ge=0, le=100) = Field(
        ...,
        description="Months the customer has been with the company (>= 0)",
    )

    MonthlyCharges: confloat(ge=0, le=1000) = Field(
        ...,
        description="Monthly charge amount in USD (must be >= 0)",
    )

    TotalCharges: confloat(ge=0) = Field(
        ...,
        description="Total charges accumulated (must be >= 0)",
    )

    # ── Demographics ────────────────────────────────

    gender: Literal["Male", "Female"]

    SeniorCitizen: Literal[0, 1]

    Partner: Literal["Yes", "No"]

    Dependents: Literal["Yes", "No"]

    # ── Phone service ────────────────────────────────

    PhoneService: Literal["Yes", "No"]

    MultipleLines: Literal["Yes", "No", "No phone service"]

    # ── Internet service ─────────────────────────────

    InternetService: Literal["DSL", "Fiber optic", "No"]

    OnlineSecurity: Literal["Yes", "No", "No internet service"]

    OnlineBackup: Literal["Yes", "No", "No internet service"]

    DeviceProtection: Literal["Yes", "No", "No internet service"]

    TechSupport: Literal["Yes", "No", "No internet service"]

    StreamingTV: Literal["Yes", "No", "No internet service"]

    StreamingMovies: Literal["Yes", "No", "No internet service"]

    # ── Contract & billing ───────────────────────────

    Contract: Literal["Month-to-month", "One year", "Two year"]

    PaperlessBilling: Literal["Yes", "No"]

    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]

    model_config = {
        "extra": "ignore",
        "json_schema_extra": {
            "example": {
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
        },
    }


# =========================
# REQUEST / RESPONSE
# =========================

class PredictRequest(BaseModel):
    customer: CustomerInput


class PredictResponse(BaseModel):
    churn: Literal["Yes", "No"]
    churn_probability: Optional[float] = Field(None, ge=0.0, le=1.0)
    model_version: str


class BatchPredictRequest(BaseModel):
    records: List[PredictRequest] = Field(..., min_length=1, max_length=500)


class SinglePrediction(BaseModel):
    churn: Literal["Yes", "No"]
    churn_probability: Optional[float]


class BatchPredictResponse(BaseModel):
    predictions: List[SinglePrediction]
    count: int
    model_version: str