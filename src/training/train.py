# =========================
# Training with MLflow experiment logging, diagnostics, and feature-importance analysis
# =========================

import argparse
import importlib
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from src.evaluation.diagnostics import (
    check_data_leakage,
    evaluate_overfit_underfit,
    save_model_diagnostics_report,
)
from src.evaluation.evaluate import detect_problem_type, evaluate_model
from src.training.mlflow_setup import configure_mlflow


# =========================
# Optional imports (safe)
# =========================
def optional_import(module_name, class_name):
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except Exception:
        return None


# =========================
# Encoder compatibility
# =========================
def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# =========================
# Data loading
# =========================
def load_split_data(train_path, test_path, target_column):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if target_column not in train_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in train data.")
    if target_column not in test_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in test data.")

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    return train_df, test_df, X_train, y_train, X_test, y_test


# =========================
# Preprocessor
# =========================
def build_preprocessor(X_train):
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
            ("encoder", make_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features


# =========================
# Main training function
# =========================
def train_with_mlflow(args):
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.report_dir).mkdir(parents=True, exist_ok=True)

    configure_mlflow(
        experiment_name=args.experiment_name,
        backend_store_path=args.backend_store_path,
    )

    train_df, test_df, X_train, y_train, X_test, y_test = load_split_data(
        args.train_path, args.test_path, args.target_column
    )

    problem_type = detect_problem_type(y_train)

    # =========================
    # DATA LEAKAGE (FIXED: now used)
    # =========================
    leakage_report = check_data_leakage(
        train_df=train_df,
        test_df=test_df,
        target_column=args.target_column,
        output_dir=args.report_dir,
    )

    print(f"[LEAKAGE STATUS]: {leakage_report['status']}")

    models = {
        "log_reg": LogisticRegression(max_iter=1000),
        "rf": RandomForestClassifier(n_estimators=200),
        "gb": GradientBoostingClassifier(),
        "svm": SVC(probability=True),
    }

    best_score = None
    best_model = None

    run_summaries = []
    diagnostics_all = []

    for name, estimator in models.items():
        preprocessor, _, _ = build_preprocessor(X_train)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )

        with mlflow.start_run(run_name=name):
            pipeline.fit(X_train, y_train)

            train_metrics = evaluate_model(
                pipeline, X_train, y_train, args.report_dir
            )
            test_metrics = evaluate_model(
                pipeline, X_test, y_test, args.report_dir
            )

            diagnostics = evaluate_overfit_underfit(
                model_name=name,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                problem_type=problem_type,
                primary_metric="f1_macro",
                higher_is_better=True,
            )

            diagnostics_all.append(diagnostics)

            score = test_metrics.get("f1_macro", 0)

            mlflow.log_metric("f1_macro", score)
            mlflow.log_param("leakage_status", leakage_report["status"])

            mlflow.sklearn.log_model(pipeline, "model")

            run_summaries.append(
                {
                    "model": name,
                    "score": score,
                    "verdict": diagnostics["verdict"],
                }
            )

            if best_score is None or score > best_score:
                best_score = score
                best_model = pipeline

    # =========================
    # Save final model
    # =========================
    model_path = Path(args.model_dir) / "best_model.pkl"
    joblib.dump(best_model, model_path)

    # =========================
    # Save reports
    # =========================
    pd.DataFrame(run_summaries).to_csv(
        Path(args.report_dir) / "run_summary.csv", index=False
    )

    save_model_diagnostics_report(
        diagnostics=diagnostics_all,
        output_dir=args.report_dir,
        filename="diagnostics.json",
    )

    print("Training completed successfully.")
    print(f"Best model score: {best_score}")


# =========================
# CLI
# =========================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-path", default="data/splits/train.csv")
    parser.add_argument("--test-path", default="data/splits/test.csv")
    parser.add_argument("--target-column", default="target")

    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--report-dir", default="reports")

    parser.add_argument("--experiment-name", default="mlops_experiment")
    parser.add_argument("--backend-store-path", default="mlruns/mlflow.db")

    return parser.parse_args()


if __name__ == "__main__":
    train_with_mlflow(parse_args())