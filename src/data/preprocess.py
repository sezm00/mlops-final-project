import os
import yaml
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from imblearn.over_sampling import SMOTE ,SMOTENC


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
def load_params():
    with open("configs/params.yaml") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────
# Pipeline Builder
# ─────────────────────────────────────────────────────────────


def build_pipeline(params):

    numeric_features = params["preprocessing"]["numeric_features"]
    categorical_features = params["encoding"]["ohe_columns"]
    binary_features = params["encoding"]["binary_columns"]

    imputer_strategy = params["preprocessing"]["imputer_strategy"]

    # optional safety check
    if imputer_strategy not in ["mean", "median", "most_frequent", "constant"]:
        raise ValueError(f"Invalid imputer strategy: {imputer_strategy}")

    scaler_type = params["preprocessing"]["scaler"]
    if scaler_type != "standard":
        raise ValueError(f"Only 'standard' scaler supported, got {scaler_type}")

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=imputer_strategy)),
        ("scaler", StandardScaler())
    ])

    categorical_ohe = OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
        sparse_output=False
    )

    binary_ohe = OneHotEncoder(
        drop="if_binary",
        handle_unknown="ignore",
        sparse_output=False
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_ohe, categorical_features),
            ("bin", binary_ohe, binary_features),
        ],
        remainder="drop"
    )

    return Pipeline([
        ("preprocessing", preprocessor)
    ])

# ─────────────────────────────────────────────────────────────
# Split
# ─────────────────────────────────────────────────────────────
def split_data(df, target, params):
    X = df.drop(columns=[target])
    y = df[target]

    return train_test_split(
        X,
        y,
        test_size=params["split"]["test_size"],
        random_state=params["split"]["random_state"],
        stratify=y
    )


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def preprocess():

    params = load_params()

    ref_path = params["data"]["reference_path"]

    train_path = params["data"]["train_path"]
    test_path = params["data"]["test_path"]

    pipeline_path = params["data"]["pipeline_path"]
    target = params["data"]["target_column"]

    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    # ─────────────────────────────
    # Load ONLY reference data
    # ─────────────────────────────
    reference_df = pd.read_csv(ref_path)

    print(f"[INFO] Reference: {reference_df.shape}")

    # ─────────────────────────────
    # Split reference
    # ─────────────────────────────
    X_train, X_test, y_train, y_test = split_data(reference_df, target, params)

    # ─────────────────────────────
    # Build pipeline
    # ─────────────────────────────
    pipeline = build_pipeline(params)

    # ─────────────────────────────
    # Fit + transform (REFERENCE ONLY)
    # ─────────────────────────────
    X_train_proc = pipeline.fit_transform(X_train, y_train)
    X_test_proc = pipeline.transform(X_test)

    feature_names = pipeline.named_steps["preprocessing"].get_feature_names_out()

    X_train_df = pd.DataFrame(X_train_proc, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_proc, columns=feature_names)

    print("[INFO] Pipeline fitted and applied on reference only")

    # # ─────────────────────────────
    # # SMOTE
    # # ─────────────────────────────
    # if params["preprocessing"]["use_smote"]:
    #     smote = SMOTE(
    #         random_state=params["split"]["random_state"],
    #         k_neighbors=params["preprocessing"]["smote_k_neighbors"]
    #     )
    #     X_train_final, y_train_final = smote.fit_resample(X_train_df, y_train)
    # else:
    #     X_train_final, y_train_final = X_train_df, y_train

    # -------------------------------------------------
    # SMOTE (SMOTENC for mixed features) - FIXED
    # -------------------------------------------------
    if params["preprocessing"]["use_smote"]:

        numeric_features = params["preprocessing"]["numeric_features"]

        feature_cols = X_train_df.columns.tolist()

        categorical_features = []

        for i, col in enumerate(feature_cols):

            # detect numeric columns after transformer prefix
            is_numeric = any(num in col for num in numeric_features)

            if not is_numeric:
                categorical_features.append(i)

        # SAFETY CHECK (VERY IMPORTANT)
        if len(categorical_features) == len(feature_cols):
            raise ValueError(
                "SMOTENC error: all features detected as categorical. "
                "Check numeric feature mapping after preprocessing."
            )

        smote = SMOTENC(
            categorical_features=categorical_features,
            random_state=params["split"]["random_state"],
            k_neighbors=params["preprocessing"]["smote_k_neighbors"]
        )

        X_train_final, y_train_final = smote.fit_resample(
            X_train_df,
            y_train
        )

    else:
        X_train_final, y_train_final = X_train_df, y_train
    # ─────────────────────────────
    # Save datasets
    # ─────────────────────────────
    train_out = pd.DataFrame(X_train_final, columns=feature_names)
    train_out[target] = y_train_final.values
    train_out.to_csv(train_path, index=False)

    test_out = X_test_df.copy()
    test_out[target] = y_test.values
    test_out.to_csv(test_path, index=False)

    # ─────────────────────────────
    # Save pipeline
    # ─────────────────────────────
    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_names": feature_names,
            "params": params
        },
        pipeline_path
    )

    print(f"[INFO] Pipeline saved → {pipeline_path}")
    print("[INFO] DONE")


if __name__ == "__main__":
    preprocess()