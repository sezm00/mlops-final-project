# =========================
# Training with MLflow experiment logging and diagnostics
# =========================

import argparse
import importlib
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
from sklearn.svm import SVC

from src.evaluation.diagnostics import (
    check_data_drift,
    check_data_leakage,
    evaluate_overfit_underfit,
    save_model_diagnostics_report,
)
from src.evaluation.evaluate import detect_problem_type, evaluate_model
from src.training.mlflow_setup import configure_mlflow


def optional_import(module_name, class_name):
    """
    Import optional model classes safely.

    This allows the pipeline to continue even if xgboost, lightgbm,
    or catboost are not installed in a specific environment.
    """
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except Exception as error:
        print(f"Optional model skipped: {class_name} from {module_name}. Reason: {error}")
        return None


def make_one_hot_encoder():
    """
    Create OneHotEncoder with compatibility across sklearn versions.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


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

    return train_df, test_df, X_train, y_train, X_test, y_test


def build_preprocessor(X_train):
    """
    Build preprocessing for numeric and categorical features.

    This is leakage-safe because the preprocessor is inside an sklearn Pipeline.
    It is fitted only on training data during pipeline.fit().
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
            ("encoder", make_one_hot_encoder()),
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
    """
    Return candidate models for MLflow experiment comparison.

    Classification models:
    - Logistic Regression
    - Random Forest
    - Gradient Boosting
    - SVC
    - XGBoost
    - LightGBM
    - CatBoost

    Regression models:
    - Ridge Regression
    - Random Forest
    - Gradient Boosting
    - XGBoost
    - LightGBM
    - CatBoost
    """
    models = {}

    if problem_type == "classification":
        models["logistic_regression"] = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        )

        models["random_forest"] = RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        )

        models["gradient_boosting"] = GradientBoostingClassifier(
            random_state=random_state,
        )

        models["svc"] = SVC(
            probability=True,
            random_state=random_state,
        )

        XGBClassifier = optional_import("xgboost", "XGBClassifier")
        LGBMClassifier = optional_import("lightgbm", "LGBMClassifier")
        CatBoostClassifier = optional_import("catboost", "CatBoostClassifier")

        if XGBClassifier is not None:
            models["xgboost"] = XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=random_state,
            )

        if LGBMClassifier is not None:
            models["lightgbm"] = LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=-1,
                random_state=random_state,
                verbosity=-1,
            )

        if CatBoostClassifier is not None:
            models["catboost"] = CatBoostClassifier(
                iterations=200,
                learning_rate=0.05,
                depth=4,
                random_seed=random_state,
                verbose=False,
            )

    else:
        models["ridge_regression"] = Ridge()

        models["random_forest"] = RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        )

        models["gradient_boosting"] = GradientBoostingRegressor(
            random_state=random_state,
        )

        XGBRegressor = optional_import("xgboost", "XGBRegressor")
        LGBMRegressor = optional_import("lightgbm", "LGBMRegressor")
        CatBoostRegressor = optional_import("catboost", "CatBoostRegressor")

        if XGBRegressor is not None:
            models["xgboost"] = XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=random_state,
            )

        if LGBMRegressor is not None:
            models["lightgbm"] = LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=-1,
                random_state=random_state,
                verbosity=-1,
            )

        if CatBoostRegressor is not None:
            models["catboost"] = CatBoostRegressor(
                iterations=200,
                learning_rate=0.05,
                depth=4,
                random_seed=random_state,
                verbose=False,
            )

    return models


def get_primary_metric(problem_type):
    """
    Select metric used to choose the best model.
    """
    if problem_type == "classification":
        return "f1_macro", True

    return "rmse", False


def train_with_mlflow(args):
    """
    Train multiple models, log experiments, run diagnostics, and save the best model.
    """
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.report_dir).mkdir(parents=True, exist_ok=True)

    configure_mlflow(
        experiment_name=args.experiment_name,
        backend_store_path=args.backend_store_path,
    )

    train_df, test_df, X_train, y_train, X_test, y_test = load_split_data(
        train_path=args.train_path,
        test_path=args.test_path,
        target_column=args.target_column,
    )

    problem_type = detect_problem_type(y_train)
    primary_metric, higher_is_better = get_primary_metric(problem_type)

    leakage_report = check_data_leakage(
        train_df=train_df,
        test_df=test_df,
        target_column=args.target_column,
        output_dir=args.report_dir,
    )

    drift_report = check_data_drift(
        train_df=train_df,
        test_df=test_df,
        target_column=args.target_column,
        output_dir=args.report_dir,
    )

    preprocessor, numeric_features, categorical_features = build_preprocessor(X_train)
    candidate_models = get_candidate_models(problem_type, args.random_state)

    best_score = None
    best_model_name = None
    best_model = None
    best_metrics = None

    feature_metadata = {
        "target_column": args.target_column,
        "problem_type": problem_type,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "input_columns": X_train.columns.tolist(),
    }

    feature_metadata_path = Path(args.model_dir) / "feature_columns.json"

    with open(feature_metadata_path, "w", encoding="utf-8") as file:
        json.dump(feature_metadata, file, indent=4)

    run_summaries = []
    diagnostic_summaries = []

    for model_name, estimator in candidate_models.items():
        with mlflow.start_run(run_name=f"experiment_{model_name}"):
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", estimator),
                ]
            )

            pipeline.fit(X_train, y_train)

            train_metrics = evaluate_model(
                model=pipeline,
                X_test=X_train,
                y_test=y_train,
                output_dir=args.report_dir,
            )

            test_metrics = evaluate_model(
                model=pipeline,
                X_test=X_test,
                y_test=y_test,
                output_dir=args.report_dir,
            )

            diagnostics = evaluate_overfit_underfit(
                model_name=model_name,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                problem_type=problem_type,
                primary_metric=primary_metric,
                higher_is_better=higher_is_better,
            )

            diagnostic_summaries.append(diagnostics)

            score = test_metrics[primary_metric]

            mlflow.log_param("model_name", model_name)
            mlflow.log_param("problem_type", problem_type)
            mlflow.log_param("target_column", args.target_column)
            mlflow.log_param("train_rows", X_train.shape[0])
            mlflow.log_param("test_rows", X_test.shape[0])
            mlflow.log_param("numeric_features", len(numeric_features))
            mlflow.log_param("categorical_features", len(categorical_features))
            mlflow.log_param("primary_metric", primary_metric)
            mlflow.log_param("quality_verdict", diagnostics["verdict"])
            mlflow.log_param("leakage_status", leakage_report["status"])
            mlflow.log_param("drift_status", drift_report["status"])

            for metric_name, metric_value in train_metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(f"train_{metric_name}", metric_value)

            for metric_name, metric_value in test_metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(f"test_{metric_name}", metric_value)
                    mlflow.log_metric(metric_name, metric_value)

            if diagnostics["gap"] is not None:
                mlflow.log_metric("train_test_gap", diagnostics["gap"])

            metrics_path = Path(args.report_dir) / "metrics.json"
            report_path = Path(args.report_dir) / "classification_report.txt"
            confusion_path = Path(args.report_dir) / "confusion_matrix.csv"
            leakage_path = Path(args.report_dir) / "data_leakage_report.json"
            drift_path = Path(args.report_dir) / "data_drift_report.json"

            for artifact_path in [
                metrics_path,
                report_path,
                confusion_path,
                leakage_path,
                drift_path,
                feature_metadata_path,
            ]:
                if artifact_path.exists():
                    mlflow.log_artifact(str(artifact_path))

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
            )

            run_summary = {
                "model_name": model_name,
                "primary_metric": primary_metric,
                "score": score,
                "quality_verdict": diagnostics["verdict"],
                "diagnostic_warning": " | ".join(diagnostics["warnings"]),
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }

            run_summaries.append(run_summary)

            should_replace = (
                best_score is None
                or (higher_is_better and score > best_score)
                or (not higher_is_better and score < best_score)
            )

            if should_replace:
                best_score = score
                best_model_name = model_name
                best_model = pipeline
                best_metrics = test_metrics

    best_model_path = Path(args.model_dir) / "best_model.pkl"
    joblib.dump(best_model, best_model_path)

    summary = {
        "best_model_name": best_model_name,
        "primary_metric": primary_metric,
        "best_score": float(best_score),
        "metrics": best_metrics,
        "model_path": str(best_model_path),
        "leakage_status": leakage_report["status"],
        "drift_status": drift_report["status"],
    }

    summary_path = Path(args.report_dir) / "best_model_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    run_summary_path = Path(args.report_dir) / "mlflow_run_summary.csv"
    pd.DataFrame(run_summaries).to_csv(run_summary_path, index=False)

    diagnostics_path = save_model_diagnostics_report(
        diagnostics=diagnostic_summaries,
        output_dir=args.report_dir,
        filename="model_diagnostics_report.json",
    )

    print("MLflow experiment logging completed.")
    print(json.dumps(summary, indent=4))

    print("\nModel quality diagnostics:")
    for item in diagnostic_summaries:
        print(f"- {item['model_name']}: {item['verdict']} | {' | '.join(item['warnings'])}")

    if leakage_report["warnings"]:
        print("\nData leakage warnings:")
        for warning in leakage_report["warnings"]:
            print(f"- {warning}")

    if drift_report["warnings"]:
        print("\nData drift warnings:")
        for warning in drift_report["warnings"]:
            print(f"- {warning}")


def parse_args():
    """
    Parse command line arguments.
    """
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
