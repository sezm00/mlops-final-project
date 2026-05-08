import os

import joblib
import pandas as pd
import yaml

from src.features.feature_engineer import FeatureEngineer


# -------------------------
# Load config
# -------------------------
def load_params():
    with open("configs/params.yaml") as f:
        return yaml.safe_load(f)


# -------------------------
# Main DVC stage
# -------------------------
def run_featurisation():

    params = load_params()

    ref_path = params["data"]["cleaned_reference_path"]
    out_path = params["data"]["reference_path"]
    artifact_path = params["data"]["feature_engineer_path"]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)

    # -------------------------
    # Load reference data
    # -------------------------
    df = pd.read_csv(ref_path)
    print(f"[featurize] input shape: {df.shape}")

    # -------------------------
    # Fit + transform
    # -------------------------
    engine = FeatureEngineer(params)
    df = engine.fit_transform(df)

    # -------------------------
    # Save engineered data
    # -------------------------
    df.to_csv(out_path, index=False)

    # -------------------------
    # Save transformer artifact
    # -------------------------
    joblib.dump(
        {
            "engine": engine,
            "params": params,
            "stats": engine.stats_,
        },
        artifact_path,
    )

    print(f"[featurize] saved data → {out_path}")
    print(f"[featurize] saved model → {artifact_path}")
    print("[featurize] DONE ✔")


if __name__ == "__main__":
    run_featurisation()
