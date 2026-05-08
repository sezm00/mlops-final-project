# src/data/prepare.py
"""
Stage 1 — Cleaning, Target Encoding & Positional Split
=======================================================

Strict responsibility boundary: this stage performs ONLY data cleaning,
target encoding, and the reference/production positional split.

NO feature engineering of any kind is performed here — not row-level,
not statistical. All feature engineering is exclusively owned by
featurize.py (Stage 2).



  clean()
          — fix types, correct known invalid values, standardise
            categorical labels, drop the identifier column.
          NOTE: TotalCharges NaN rows that arise from pd.to_numeric
            (non-tenure=0 whitespace entries) are left as NaN here.
            They will be imputed by the sklearn pipeline in preprocess.py
            using the reference-set median computed in featurize.py.

   encode_target()
          — map Churn Yes → 1, No → 0 on the full dataset before the split
            so both halves share the same integer encoding with no fitting.

  split_reference_production()
          — deterministic positional 70/30 split.
            The first 70 % of rows (older customers) become the reference
            set used for training; the remaining 30 % become production,
            simulating real-world temporal holdout.

Leakage guarantees:
  • No randomness in the split (purely positional).
  • No aggregation across rows.
  • No use of dataset statistics (mean, median, quantiles).
  • No dependency on the reference/production relationship.

"""

import os

import pandas as pd
import yaml

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


def load_params() -> dict:
    with open("configs/params.yaml") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Cleaning
# ─────────────────────────────────────────────────────────────────────────────


def clean(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Fix types, correct known invalid values, consolidate labels, drop ID.

    All operations are row-level or schema-level; no dataset statistics are
    used or computed.

    TotalCharges NaN handling
    ─────────────────────────
    pd.to_numeric coerces whitespace-only strings to NaN. Rows where
    tenure == 0 are factually corrected to 0.0 (customers never billed).
    Any remaining NaN rows (data entry errors unrelated to tenure) are
    intentionally left as NaN so the reference-median imputer in
    preprocess.py handles them in a leakage-safe, reproducible way.
    """

    # 1a. TotalCharges stored as object due to whitespace entries → coerce
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # 1b. tenure = 0 means customer was never billed → factual correction,
    #     NOT statistical imputation.
    fix_col = params["cleaning"]["totalcharges_fix_col"]
    df.loc[df[fix_col] == 0, "TotalCharges"] = 0.0

    # 1c. Consolidate redundant category labels to reduce OHE cardinality
    for col in params["cleaning"]["internet_service_cols"]:
        df[col] = df[col].replace("No internet service", "No")
    for col in params["cleaning"]["phone_service_cols"]:
        df[col] = df[col].replace("No phone service", "No")

    # 1d. Drop customerID — unique per row, zero predictive signal
    cols_to_drop = [c for c in params["cleaning"]["drop_columns"] if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Target Encoding
# ─────────────────────────────────────────────────────────────────────────────


def encode_target(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Encode Churn: Yes → 1, No → 0.

    Applied on the full dataset before the split so both halves share
    the same integer encoding without any fitting step.
    """
    df[target] = df[target].map({"Yes": 1, "No": 0})
    if df[target].isnull().any():
        raise ValueError(
            f"[prepare] '{target}' contains unexpected values after encoding. "
            "Only 'Yes' and 'No' are valid in the raw data."
        )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Reference / Production Split
# ─────────────────────────────────────────────────────────────────────────────


def split_reference_production(
    df: pd.DataFrame,
    reference_ratio: float,
    target: str,
) -> tuple:
    """
    Positional split — first N rows become reference, the rest production.

    Why positional (not random)?
    The Telco dataset has no timestamp column, but row order approximates
    customer acquisition order (older customers first). A positional split
    therefore simulates the real-world pattern of training on historical
    data and monitoring on newer arrivals, which is required for meaningful
    drift simulation in the monitoring component.

    Returns
    -------
    (reference_df, production_df)
    """
    split_idx = int(len(df) * reference_ratio)
    reference_df = df.iloc[:split_idx].copy()
    production_df = df.iloc[split_idx:].copy()

    ref_churn = reference_df[target].mean()
    prod_churn = production_df[target].mean()

    print(f"[prepare] Split index : {split_idx} of {len(df)}")
    print(f"[prepare] Reference   : {reference_df.shape} | churn rate: {ref_churn:.3f}")
    print(
        f"[prepare] Production  : {production_df.shape} | churn rate: {prod_churn:.3f}"
    )

    gap = abs(ref_churn - prod_churn)
    if gap > 0.05:
        print(
            f"[prepare] ⚠️  Churn rate gap {gap:.3f} — "
            "natural concept drift between sets. EXPECTED for drift simulation."
        )

    return reference_df, production_df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def prepare() -> None:
    params = load_params()

    raw_path = params["data"]["raw_path"]
    cleaned_reference_path = params["data"]["cleaned_reference_path"]
    production_path = params["data"]["production_path"]
    target = params["data"]["target_column"]
    reference_ratio = params["split"]["reference_ratio"]

    for path in [cleaned_reference_path, production_path]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    # ── Step 1: Load raw ──────────────────────────────────────────────────────
    print(f"[prepare] Loading raw data from: {raw_path}")
    df = pd.read_csv(raw_path)
    print(f"[prepare] Raw shape: {df.shape}")

    # ── Step 2: Target encoding (before split) ────────────────────────────────
    df = encode_target(df, target)
    print(f"[prepare] Target encoded — {df[target].value_counts().to_dict()}")

    # ── Step 3: Split ─────────────────────────────────────────────────────────
    reference_df, production_df = split_reference_production(
        df, reference_ratio, target
    )

    # ── Step 4: Clean ONLY reference ──────────────────────────────────────────
    reference_df = clean(reference_df, params)

    nan_total = reference_df.isnull().sum().sum()
    print(
        f"[prepare] Reference after cleaning : {reference_df.shape} | NaN total: {nan_total}"
    )

    if nan_total > 0:
        print(
            f"[prepare]   NaN breakdown:\n{reference_df.isnull().sum()[reference_df.isnull().sum() > 0]}"
        )

    # ── Production remains raw ────────────────────────────────────────────────
    print(f"[prepare] Production kept raw : {production_df.shape}")

    # ── Save outputs ───────────────────────────────────────────────────────────
    reference_df.to_csv(cleaned_reference_path, index=False)
    production_df.to_csv(production_path, index=False)

    print(f"[prepare] Reference saved  → {cleaned_reference_path}")
    print(f"[prepare] Production saved → {production_path}")
    print("[prepare] DONE — feature engineering delegated to featurize.py")


if __name__ == "__main__":
    prepare()
