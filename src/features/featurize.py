import numpy as np
import pandas as pd
import json
import os


# ============================================================
# Row-Level Feature Engineering
# ============================================================

def add_row_level_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    fe = params["feature_engineering"]

    df = df.copy()

    df["AvgMonthlyCharges"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"],
    )

    df["ChargeDeviation"] = df["MonthlyCharges"] - df["AvgMonthlyCharges"]

    df["TotalCharges_log"] = np.log1p(df["TotalCharges"])

    df["NumServices"] = df[fe["service_cols"]].eq("Yes").sum(axis=1)

    threshold = fe["new_customer_tenure_threshold"]
    df["IsNewCustomer"] = (df["tenure"] <= threshold).astype(int)

    df["IsRiskySegment"] = (
        (df["Contract"] == fe["risky_contract_value"]) &
        (df["tenure"] <= threshold)
    ).astype(int)

    return df


# ============================================================
# Statistical Feature Engineering
# ============================================================

def compute_reference_stats(reference_df: pd.DataFrame, params: dict) -> dict:
    high_value_q = params["stat_features"]["high_value_quantile"]

    stats = {
        "high_value_threshold": float(
            reference_df["MonthlyCharges"].quantile(high_value_q)
        ),
        "quantile_used": high_value_q,
        "computed_on_n_rows": len(reference_df),
    }

    stats_path = params["data"]["stats_path"]
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[featurise] stats saved → {stats_path}")

    return stats


def apply_stat_features(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    df = df.copy()

    df["IsHighValueCustomer"] = (
        df["MonthlyCharges"] >= stats["high_value_threshold"]
    ).astype(int)

    return df


# ============================================================
# MAIN PIPELINE FUNCTION (DVC ENTRY POINT)
# ============================================================

def run_featurisation():
    import yaml

    with open("configs/params.yaml") as f:
        params = yaml.safe_load(f)

    # INPUTS (from prepare stage)
    ref_path = params["data"]["cleaned_reference_path"]
    prod_path = params["data"]["cleaned_production_path"]

    # OUTPUTS (next stage inputs)
    out_ref_path = params["data"]["reference_path"]
    out_prod_path = params["data"]["production_path"]

    # load data
    reference_df = pd.read_csv(ref_path)
    production_df = pd.read_csv(prod_path)

    # row-level features
    reference_df = add_row_level_features(reference_df, params)
    production_df = add_row_level_features(production_df, params)

    # stats (reference only)
    stats = compute_reference_stats(reference_df, params)

    reference_df = apply_stat_features(reference_df, stats)
    production_df = apply_stat_features(production_df, stats)

    # save outputs
    os.makedirs(os.path.dirname(out_ref_path), exist_ok=True)

    reference_df.to_csv(out_ref_path, index=False)
    production_df.to_csv(out_prod_path, index=False)

    print("[featurise] DONE ✔")


if __name__ == "__main__":
    run_featurisation()