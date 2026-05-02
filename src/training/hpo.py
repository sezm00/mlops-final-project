# =========================
# Hyperparameter tuning with MLflow
# =========================

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
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

    return preprocessor


def build_search(problem_type, random_state):
    if problem_type == "classification":
        estimator = RandomForestClassifier(random_state=random_state, n_jobs=-1)
        scoring = "f1_macro"
    else:
        estimator = RandomForestRegressor(random_state=random_state, n_jobs=-1)
        scoring = "neg_root_mean_squared_error"

    param_distributions = {
        "model__n_estimators": [100, 200, 300, 500],
        "model__max_depth": [None, 5, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    }

    return estimator, param_distributions, scoring


def run_hpo(args):
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

    preprocessor = build_preprocessor(X_train)
    estimator, param_distributions, scoring = build_search(problem_type, args.random_state)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=args.n_iter,
        scoring=scoring,
        cv=args.cv,
        n_jobs=-1,
        random_state=args.random_state,
        verbose=1,
    )

    with mlflow.start_run(run_name="random_forest_hpo"):
        search.fit(X_train, y_train)

        best_model = search.best_estimator_

        metrics = evaluate_model(
            model=best_model,
            X_test=X_test,
            y_test=y_test,
            output_dir=args.report_dir,
        )

        model_path = Path(args.model_dir) / "tuned_model.pkl"
        joblib.dump(best_model, model_path)

        summary = {
            "problem_type": problem_type,
            "scoring": scoring,
            "best_cv_score": float(search.best_score_),
            "best_params": search.best_params_,
            "test_metrics": metrics,
            "model_path": str(model_path),
        }

        summary_path = Path(args.report_dir) / "hpo_summary.json"

        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=4)

        mlflow.log_param("model_name", "random_forest_hpo")
        mlflow.log_param("problem_type", problem_type)
        mlflow.log_param("scoring", scoring)
        mlflow.log_param("cv", args.cv)
        mlflow.log_param("n_iter", args.n_iter)

        for param_name, param_value in search.best_params_.items():
            mlflow.log_param(param_name, param_value)

        mlflow.log_metric("best_cv_score", float(search.best_score_))

        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(metric_name, metric_value)

        mlflow.log_artifact(str(summary_path))
        mlflow.log_artifact(str(model_path))

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
        )

    print("Hyperparameter tuning completed.")
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
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--n-iter", type=int, default=10)

    return parser.parse_args()


if __name__ == "__main__":
    run_hpo(parse_args())
