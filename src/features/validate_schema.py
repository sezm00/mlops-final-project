# src/features/validate_schema.py

"""
Stage 2 — Schema Validation & Handoff
======================================

Pure validation stage (NO transformations).

Ensures:
- engineered features exist
- values are valid
- consistency rules are respected

Acts as a contract gate between:
featurize.py and preprocess.py

"""
import pandas as pd
import yaml
import os

def load_params():
    with open("configs/params.yaml") as f:
        return yaml.safe_load(f)


def validate_features(df: pd.DataFrame, label: str, params: dict):

    engineered = params["validation"]["engineered_features"]

    threshold = params["feature_engineering"]["new_customer_tenure_threshold"]

    # ─────────────────────────────
    # 1. Missing columns
    # ─────────────────────────────
    missing = [c for c in engineered if c not in df.columns]
    if missing:
        raise ValueError(f"[validate_schema] {label} missing: {missing}")

    # ─────────────────────────────
    # 2. Null checks
    # ─────────────────────────────
    for col in engineered:
        if df[col].isnull().any():
            raise ValueError(f"[validate_schema] {label} nulls in {col}")

    # ─────────────────────────────
    # 3. Numeric validation
    # ─────────────────────────────
    if (df["NumServices"] < 0).any():
        raise ValueError(f"[validate_schema] {label} NumServices < 0")

    if (df["NumServices"] > 7).any():
        raise ValueError(f"[validate_schema] {label} NumServices > 7")

    if (df["AvgMonthlyCharges"] < 0).any():
        raise ValueError(f"[validate_schema] {label} negative AvgMonthlyCharges")

    if (df["TotalCharges_log"] < 0).any():
        raise ValueError(f"[validate_schema] {label} negative log charges")

    # ─────────────────────────────
    # 4. Binary checks
    # ─────────────────────────────
    binary_cols = ["IsNewCustomer", "IsRiskySegment", "IsHighValueCustomer"]

    for col in binary_cols:
        if not df[col].dropna().isin([0, 1]).all():
            raise ValueError(f"[validate_schema] {label} invalid binary {col}")

    # ─────────────────────────────
    # 5. Logical consistency
    # ─────────────────────────────
    if ((df["IsNewCustomer"] == 1) & (df["tenure"] > threshold)).any():
        raise ValueError(f"[validate_schema] {label} invalid IsNewCustomer logic")

    print(f"[validate_schema] {label} passed ✔")


def validate_schema():
    params = load_params()

    ref = pd.read_csv(params["data"]["reference_path"])
    prod = pd.read_csv(params["data"]["production_path"])

    validate_features(ref, "reference", params)
    validate_features(prod, "production", params)

      # ─────────────────────────────
    # CREATE FLAG FILE
    # ─────────────────────────────
    flag_path = "data/splits/.schema_validated.flag"
    os.makedirs(os.path.dirname(flag_path), exist_ok=True)

    with open(flag_path, "w") as f:
        f.write("schema_validated=True\n")

    print(f"[validate_schema] flag created → {flag_path}")
    print("[validate_schema] schema validation complete ✔")

if __name__ == "__main__":
    validate_schema()