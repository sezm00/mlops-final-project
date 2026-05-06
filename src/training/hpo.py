# =========================
# Hyperparameter tuning with MLflow and diagnostics
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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.evaluation.diagnostics import evaluate_overfit_underfit, save_model_diagnostics_report
from src.evaluation.evaluate import detect_problem_type, evaluate_model
from src.training.mlflow_setup import configure_mlflow


def optional_import(module_name, class_name):
    """
    Import optional HPO model classes safely.
    """
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except Exception as error:
        print(f"Optional HPO model skipped: {class_name} from {module_name}. Reason: {error}")
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

    return X_train, y_train, X_test, y_test


def build_preprocessor(X_train):
    """
    Build preprocessing for numeric and categorical features.
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

    return preprocessor


def get_hpo_candidates(problem_type, random_state):
    """
    Return HPO candidates and parameter grids.
    """
    candidates = {}

    if problem_type == "classification":
        candidates["random_forest"] = {
            "estimator": RandomForestClassifier(random_state=random_state, n_jobs=-1),
            "params": {
                "model__n_estimators": [100, 200, 300, 500],
                "model__max_depth": [None, 5, 10, 20, 30],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", None],
            },
        }

        XGBClassifier = optional_import("xgboost", "XGBClassifier")
        LGBMClassifier = optional_import("lightgbm", "LGBMClassifier")
        CatBoostClassifier = optional_import("catboost", "CatBoostClassifier")

        if XGBClassifier is not None:
            candidates["xgboost"] = {
                "estimator": XGBClassifier(
                    eval_metric="logloss",
                    random_state=random_state,
                ),
                "params": {
                    "model__n_estimators": [100, 200, 300],
                    "model__learning_rate": [0.01, 0.05, 0.1],
                    "model__max_depth": [3, 4, 5],
                    "model__subsample": [0.8, 0.9, 1.0],
                    "model__colsample_bytree": [0.8, 0.9, 1.0],
                },
            }

        if LGBMClassifier is not None:
            candidates["lightgbm"] = {
                "estimator": LGBMClassifier(
                    random_state=random_state,
                    verbosity=-1,
                ),
                "params": {
                    "model__n_estimators": [100, 200, 300],
                    "model__learning_rate": [0.01, 0.05, 0.1],
                    "model__num_leaves": [15, 31, 63],
                    "model__max_depth": [-1, 3, 5, 7],
                    "model__subsample": [0.8, 0.9, 1.0],
                },
            }

        if CatBoostClassifier is not None:
            candidates["catboost"] = {
                "estimator": CatBoostClassifier(
                    random_seed=random_state,
                    verbose=False,
                ),
                "params": {
                    "model__iterations": [100, 200, 300],
                    "model__learning_rate": [0.01, 0.05, 0.1],
                    "model__depth": [3, 4, 5, 6],
                    "model__l2_leaf_reg": [1, 3, 5, 7],
                },
            }

        scoring = "f1_macro"
        primary_metric = "f1_macro"
        higher_is_better = True

    else:
        candidates["random_forest"] = {
            "estimator": RandomForestRegressor(random_state=random_state, n_jobs=-1),
            "params": {
                "model__n_estimators": [100, 200, 300, 500],
                "model__max_depth": [None, 5, 10, 20, 30],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", None],
            },
        }

        XGBRegressor = optional_import("xgboost", "XGBRegressor")
        LGBMRegressor = optional_import("lightgbm", "LGBMRegressor")
        CatBoostRegressor = optional_import("catboost", "CatBoostRegressor")

        if XGBRegressor is not None:
            candidates["xgboost"] = {
                "estimator": XGBRegressor(random_state=random_state),
                "params": {
                    "model__n_estimators": [100, 200, 300],
                    "model__learning_rate": [0.01, 0.05, 0.1],
                    "model__max_depth": [3, 4, 5],
                    "model__subsample": [0.8, 0.9, 1.0],
                    "model__colsample_bytree": [0.8, 0.9, 1.0],
                },
            }

        if LGBMRegressor is not None:
            candidates["lightgbm"] = {
                "estimator": LGBMRegressor(random_state=random_state, verbosity=-1),
                "params": {
                    "model__n_estimators": [100, 200, 300],
                    "model__learning_rate": [0.01, 0.05, 0.1],
                    "model__num_leaves": [15, 31, 63],
                    "model__max_depth": [-1, 3, 5, 7],
                    "model__subsample": [0.8, 0.9, 1.0],
                },
            }

        if CatBoostRegressor is not None:
            candidates["catboost"] = {
                "estimator": CatBoostRegressor(random_seed=random_state, verbose=False),
                "params": {
                    "model__iterations": [100, 200, 300],
                    "model__learning_rate": [0.01, 0.05, 0.1],
                    "model__depth": [3, 4, 5, 6],
                    "model__l2_leaf_reg": [1, 3, 5, 7],
                },
            }

        scoring = "neg_root_mean_squared_error"
        primary_metric = "rmse"
        higher_is_better = False

    return candidates, scoring, primary_metric, higher_is_better


def run_hpo(args):
    """
    Run HPO across multiple candidate model families.
    """
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

    candidates, scoring, primary_metric, higher_is_better = get_hpo_candidates(
        problem_type=problem_type,
        random_state=args.random_state,
    )

    best_model = None
    best_model_name = None
    best_score = None
    best_summary = None
    hpo_summaries = []
    diagnostics_summaries = []

    for model_name, candidate in candidates.items():
        preprocessor = build_preprocessor(X_train)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", candidate["estimator"]),
            ]
        )

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=candidate["params"],
            n_iter=args.n_iter,
            scoring=scoring,
            cv=args.cv,
            n_jobs=-1,
            random_state=args.random_state,
            verbose=1,
            return_train_score=True,
        )

        with mlflow.start_run(run_name=f"hpo_{model_name}"):
            search.fit(X_train, y_train)

            tuned_model = search.best_estimator_

            train_metrics = evaluate_model(
                model=tuned_model,
                X_test=X_train,
                y_test=y_train,
                output_dir=args.report_dir,
            )

            test_metrics = evaluate_model(
                model=tuned_model,
                X_test=X_test,
                y_test=y_test,
                output_dir=args.report_dir,
            )

            diagnostics = evaluate_overfit_underfit(
                model_name=f"hpo_{model_name}",
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                problem_type=problem_type,
                primary_metric=primary_metric,
                higher_is_better=higher_is_better,
            )

            diagnostics_summaries.append(diagnostics)

            test_score = test_metrics[primary_metric]

            cv_results_path = Path(args.report_dir) / f"hpo_cv_results_{model_name}.csv"
            pd.DataFrame(search.cv_results_).to_csv(cv_results_path, index=False)

            model_path = Path(args.model_dir) / f"tuned_{model_name}_model.pkl"
            joblib.dump(tuned_model, model_path)

            summary = {
                "model_name": model_name,
                "problem_type": problem_type,
                "scoring": scoring,
                "primary_metric": primary_metric,
                "best_cv_score": float(search.best_score_),
                "test_score": float(test_score),
                "best_params": search.best_params_,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "diagnostics": diagnostics,
                "model_path": str(model_path),
            }

            summary_path = Path(args.report_dir) / f"hpo_summary_{model_name}.json"

            with open(summary_path, "w", encoding="utf-8") as file:
                json.dump(summary, file, indent=4)

            mlflow.log_param("model_name", f"hpo_{model_name}")
            mlflow.log_param("problem_type", problem_type)
            mlflow.log_param("scoring", scoring)
            mlflow.log_param("cv", args.cv)
            mlflow.log_param("n_iter", args.n_iter)
            mlflow.log_param("quality_verdict", diagnostics["verdict"])

            for param_name, param_value in search.best_params_.items():
                mlflow.log_param(param_name, param_value)

            mlflow.log_metric("best_cv_score", float(search.best_score_))

            for metric_name, metric_value in train_metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(f"train_{metric_name}", metric_value)

            for metric_name, metric_value in test_metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(f"test_{metric_name}", metric_value)
                    mlflow.log_metric(metric_name, metric_value)

            if diagnostics["gap"] is not None:
                mlflow.log_metric("train_test_gap", diagnostics["gap"])

            mlflow.log_artifact(str(summary_path))
            mlflow.log_artifact(str(cv_results_path))
            mlflow.log_artifact(str(model_path))

            mlflow.sklearn.log_model(
                sk_model=tuned_model,
                artifact_path="model",
            )

            hpo_summaries.append(summary)

            should_replace = (
                best_score is None
                or (higher_is_better and test_score > best_score)
                or (not higher_is_better and test_score < best_score)
            )

            if should_replace:
                best_score = test_score
                best_model = tuned_model
                best_model_name = model_name
                best_summary = summary

    final_model_path = Path(args.model_dir) / "best_tuned_model.pkl"
    joblib.dump(best_model, final_model_path)

    final_summary = {
        "best_hpo_model_name": best_model_name,
        "primary_metric": primary_metric,
        "best_score": float(best_score),
        "best_model_path": str(final_model_path),
        "best_summary": best_summary,
        "all_hpo_summaries": hpo_summaries,
    }

    final_summary_path = Path(args.report_dir) / "hpo_summary.json"

    with open(final_summary_path, "w", encoding="utf-8") as file:
        json.dump(final_summary, file, indent=4)

    save_model_diagnostics_report(
        diagnostics=diagnostics_summaries,
        output_dir=args.report_dir,
        filename="hpo_diagnostics_report.json",
    )

    print("Hyperparameter tuning completed.")
    print(json.dumps(final_summary, indent=4))


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
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--n-iter", type=int, default=6)

    return parser.parse_args()


if __name__ == "__main__":
    run_hpo(parse_args())
