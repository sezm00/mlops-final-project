from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

# ── Raw customer input ────────────────────────────────────────────────────────


class CustomerInput(BaseModel):
    """
    Raw customer record — exactly as it would appear before any preprocessing.
    All fields match the original Telco dataset columns (minus customerID).
    """

    # Numeric
    tenure: float = Field(
        ..., ge=0, description="Months the customer has been with the company"
    )
    MonthlyCharges: float = Field(
        ..., ge=0, description="Current monthly charge amount ($)"
    )
    TotalCharges: float = Field(
        ..., ge=0, description="Total amount charged to date ($)"
    )

    # Demographics
    gender: str = Field(..., description="'Male' or 'Female'")
    SeniorCitizen: int = Field(
        ..., ge=0, le=1, description="1 if senior citizen, 0 otherwise"
    )
    Partner: str = Field(..., description="'Yes' or 'No'")
    Dependents: str = Field(..., description="'Yes' or 'No'")

    # Phone service
    PhoneService: str = Field(..., description="'Yes' or 'No'")
    MultipleLines: str = Field(..., description="'Yes', 'No', or 'No phone service'")

    # Internet service
    InternetService: str = Field(..., description="'DSL', 'Fiber optic', or 'No'")
    OnlineSecurity: str = Field(
        ..., description="'Yes', 'No', or 'No internet service'"
    )
    OnlineBackup: str = Field(..., description="'Yes', 'No', or 'No internet service'")
    DeviceProtection: str = Field(
        ..., description="'Yes', 'No', or 'No internet service'"
    )
    TechSupport: str = Field(..., description="'Yes', 'No', or 'No internet service'")
    StreamingTV: str = Field(..., description="'Yes', 'No', or 'No internet service'")
    StreamingMovies: str = Field(
        ..., description="'Yes', 'No', or 'No internet service'"
    )

    # Contract & billing
    Contract: str = Field(
        ..., description="'Month-to-month', 'One year', or 'Two year'"
    )
    PaperlessBilling: str = Field(..., description="'Yes' or 'No'")
    PaymentMethod: str = Field(
        ...,
        description="'Electronic check', 'Mailed check', 'Bank transfer (automatic)', or 'Credit card (automatic)'",
    )

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v):
        if v not in ("Male", "Female"):
            raise ValueError("gender must be 'Male' or 'Female'")
        return v

    @field_validator("Partner", "Dependents", "PhoneService", "PaperlessBilling")
    @classmethod
    def validate_yes_no(cls, v):
        if v not in ("Yes", "No"):
            raise ValueError("must be 'Yes' or 'No'")
        return v

    @field_validator("MultipleLines")
    @classmethod
    def validate_multiple_lines(cls, v):
        if v not in ("Yes", "No", "No phone service"):
            raise ValueError("MultipleLines must be 'Yes', 'No', or 'No phone service'")
        return v

    @field_validator("InternetService")
    @classmethod
    def validate_internet_service(cls, v):
        if v not in ("DSL", "Fiber optic", "No"):
            raise ValueError("InternetService must be 'DSL', 'Fiber optic', or 'No'")
        return v

    @field_validator(
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    )
    @classmethod
    def validate_internet_addon(cls, v):
        if v not in ("Yes", "No", "No internet service"):
            raise ValueError("must be 'Yes', 'No', or 'No internet service'")
        return v

    @field_validator("Contract")
    @classmethod
    def validate_contract(cls, v):
        if v not in ("Month-to-month", "One year", "Two year"):
            raise ValueError(
                "Contract must be 'Month-to-month', 'One year', or 'Two year'"
            )
        return v

    @field_validator("PaymentMethod")
    @classmethod
    def validate_payment_method(cls, v):
        valid = (
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        )
        if v not in valid:
            raise ValueError(f"PaymentMethod must be one of: {valid}")
        return v

    model_config = {"extra": "ignore"}


# ── Request / Response ────────────────────────────────────────────────────────


class PredictRequest(BaseModel):
    """Single prediction request."""

    customer: CustomerInput

    model_config = {
        "json_schema_extra": {
            "example": {
                "customer": {
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
            }
        }
    }


class PredictResponse(BaseModel):
    churn: str = Field(..., description="'Yes' if predicted to churn, 'No' otherwise")
    churn_probability: Optional[float] = Field(None, ge=0.0, le=1.0)
    model_version: str


class BatchPredictRequest(BaseModel):
    records: List[PredictRequest] = Field(..., min_length=1, max_length=500)


class SinglePrediction(BaseModel):
    churn: str
    churn_probability: Optional[float]


class BatchPredictResponse(BaseModel):
    predictions: List[SinglePrediction]
    count: int
    model_version: str
