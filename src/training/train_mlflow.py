# =========================
# Training with MLflow experiment logging
# =========================

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.evaluation.evaluate import detect_problem_type, evaluate_model
from src.training.mlflow_setup import configure_mlflow


def load_split_data(train_path, test_path, target_column):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    return X_train, y_train, X_test, y_test


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


def get_candidate_models(problem_type, random_state):
    if problem_type == "classification":
        return {
            "logistic_regression": LogisticRegression(max_iter=1000),
            "random_forest": RandomForestClassifier(
                n_estimators=200,
                random_state=random_state,
                n_jobs=-1,
            ),
            "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
        }

    return {
        "ridge_regression": Ridge(),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=random_state),
    }


def get_primary_metric(problem_type):
    if problem_type == "classification":
        return "f1_macro", True

    return "rmse", False


def train_with_mlflow(args):
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.report_dir).mkdir(parents=True, exist_ok=True)

    configure_mlflow(
        experiment_name=args.experiment_name,
        backend_store_path=args.backend_store_path,
    )

    X_train, y_train, X_test, y_test = load_split_data(
        train_path=args.train_path,
        test_path=args.test_path,
        target_column=args.target_column,
    )

    problem_type = detect_problem_type(y_train)
    primary_metric, higher_is_better = get_primary_metric(problem_type)

    preprocessor, numeric_features, categorical_features = build_preprocessor(X_train)
    candidate_models = get_candidate_models(problem_type, args.random_state)

    best_score = None
    best_model_name = None
    best_model = None
    best_metrics = None

    for model_name, estimator in candidate_models.items():
        with mlflow.start_run(run_name=f"experiment_{model_name}"):
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", estimator),
                ]
            )

            pipeline.fit(X_train, y_train)

            metrics = evaluate_model(
                model=pipeline,
                X_test=X_test,
                y_test=y_test,
                output_dir=args.report_dir,
            )

            score = metrics[primary_metric]

            mlflow.log_param("model_name", model_name)
            mlflow.log_param("problem_type", problem_type)
            mlflow.log_param("target_column", args.target_column)
            mlflow.log_param("train_rows", X_train.shape[0])
            mlflow.log_param("test_rows", X_test.shape[0])
            mlflow.log_param("numeric_features", len(numeric_features))
            mlflow.log_param("categorical_features", len(categorical_features))

            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(metric_name, metric_value)

            metrics_path = Path(args.report_dir) / "metrics.json"
            report_path = Path(args.report_dir) / "classification_report.txt"
            confusion_path = Path(args.report_dir) / "confusion_matrix.csv"

            if metrics_path.exists():
                mlflow.log_artifact(str(metrics_path))

            if report_path.exists():
                mlflow.log_artifact(str(report_path))

            if confusion_path.exists():
                mlflow.log_artifact(str(confusion_path))

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
            )

            should_replace = (
                best_score is None
                or (higher_is_better and score > best_score)
                or (not higher_is_better and score < best_score)
            )

            if should_replace:
                best_score = score
                best_model_name = model_name
                best_model = pipeline
                best_metrics = metrics

    best_model_path = Path(args.model_dir) / "best_logged_model.pkl"
    joblib.dump(best_model, best_model_path)

    summary = {
        "best_model_name": best_model_name,
        "primary_metric": primary_metric,
        "best_score": float(best_score),
        "metrics": best_metrics,
        "model_path": str(best_model_path),
    }

    summary_path = Path(args.report_dir) / "best_logged_model_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("MLflow experiment logging completed.")
    print(summary)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-path", default="data/splits/train.csv")
    parser.add_argument("--test-path", default="data/splits/test.csv")
    parser.add_argument("--target-column", default="target")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--experiment-name", default="mlops_training_experiments")
    parser.add_argument("--backend-store-path", default="mlruns/mlflow.db")
    parser.add_argument("--random-state", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    train_with_mlflow(parse_args())
