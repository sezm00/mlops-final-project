# =========================
# Baseline model training
# =========================

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.evaluation.evaluate import detect_problem_type, evaluate_model


def load_split_data(train_path, test_path, target_column):
    """
    Load train/test split files and separate features from target.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if target_column not in train_df.columns:
        raise ValueError(f"Target column '{target_column}' was not found in train data.")

    if target_column not in test_df.columns:
        raise ValueError(f"Target column '{target_column}' was not found in test data.")

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    return X_train, y_train, X_test, y_test


def build_preprocessor(X_train):
    """
    Build preprocessing pipeline for numeric and categorical features.
    """
    numeric_features = X_train.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features


def build_baseline_model(problem_type, random_state):
    """
    Build baseline model based on the problem type.
    """
    if problem_type == "classification":
        return RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        )

    return RandomForestRegressor(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
    )


def train_baseline(args):
    """
    Train baseline model and save artifact.
    """
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.report_dir).mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test = load_split_data(
        train_path=args.train_path,
        test_path=args.test_path,
        target_column=args.target_column,
    )

    problem_type = detect_problem_type(y_train)

    preprocessor, numeric_features, categorical_features = build_preprocessor(X_train)
    model = build_baseline_model(problem_type, args.random_state)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    metrics = evaluate_model(
        model=pipeline,
        X_test=X_test,
        y_test=y_test,
        output_dir=args.report_dir,
    )

    model_path = Path(args.model_dir) / "baseline_model.pkl"
    joblib.dump(pipeline, model_path)

    metadata = {
        "target_column": args.target_column,
        "problem_type": problem_type,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "input_columns": X_train.columns.tolist(),
        "model_path": str(model_path),
        "metrics": metrics,
    }

    metadata_path = Path(args.model_dir) / "baseline_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("Baseline training completed.")
    print(f"Model saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(metrics)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-path", default="data/splits/train.csv")
    parser.add_argument("--test-path", default="data/splits/test.csv")
    parser.add_argument("--target-column", default="target")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--random-state", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    train_baseline(parse_args())
