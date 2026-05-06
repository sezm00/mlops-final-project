# =========================
# Training with MLflow experiment logging, diagnostics, and feature-importance analysis
# =========================

import argparse
import importlib
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
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


def optional_import(module_name, class_name):
    """
    Import optional model classes safely.
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
    Select the primary metric used for model comparison.
    """
    if problem_type == "classification":
        return "f1_macro", True

    return "rmse", False


def get_scorer_name(problem_type):
    """
    Return sklearn scoring name for permutation importance.
    """
    if problem_type == "classification":
        return "f1_macro"

    return "neg_root_mean_squared_error"


def make_train_validation_split(X_train, y_train, problem_type, validation_size, random_state):
    """
    Create train-core and validation split for feature-importance analysis only.
    """
    stratify_values = None

    if problem_type == "classification":
        class_counts = pd.Series(y_train).value_counts()

        if class_counts.min() >= 2:
            stratify_values = y_train

    return train_test_split(
        X_train,
        y_train,
        test_size=validation_size,
        random_state=random_state,
        stratify=stratify_values,
    )


def compute_permutation_feature_importance(
    pipeline,
    X_valid,
    y_valid,
    problem_type,
    output_dir,
    random_state,
    n_repeats,
):
    """
    Compute permutation importance on original input features.

    This is more reliable than basic tree split importance because it measures
    the validation performance drop when each feature is shuffled.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    scoring = get_scorer_name(problem_type)

    result = permutation_importance(
        estimator=pipeline,
        X=X_valid,
        y=y_valid,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": X_valid.columns.tolist(),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )

    importance_df["importance_positive"] = importance_df["importance_mean"].clip(lower=0)
    importance_df = importance_df.sort_values("importance_mean", ascending=False)

    total_positive_importance = importance_df["importance_positive"].sum()

    if total_positive_importance > 0:
        importance_df["importance_ratio"] = (
            importance_df["importance_positive"] / total_positive_importance
        )
        importance_df["cumulative_importance"] = importance_df["importance_ratio"].cumsum()
    else:
        importance_df["importance_ratio"] = 0.0
        importance_df["cumulative_importance"] = 0.0

    importance_df.to_csv(output_path / "all_feature_importance.csv", index=False)
    importance_df.head(10).to_csv(output_path / "top_10_feature_importance.csv", index=False)

    top_10_payload = {
        "method": "permutation_importance",
        "scoring": scoring,
        "top_10_features": importance_df.head(10)["feature"].tolist(),
    }

    with open(output_path / "top_10_features.json", "w", encoding="utf-8") as file:
        json.dump(top_10_payload, file, indent=4)

    return importance_df, top_10_payload


def build_feature_cutoff_candidates(importance_df):
    """
    Build multiple selected-feature candidates instead of blindly using top 10.
    """
    ranked_features = importance_df["feature"].tolist()
    feature_count = len(ranked_features)

    candidates = {}

    for cutoff in [5, 10, 15, 20]:
        safe_cutoff = min(cutoff, feature_count)

        if safe_cutoff > 0:
            candidates[f"top_{safe_cutoff}"] = ranked_features[:safe_cutoff]

    if "cumulative_importance" in importance_df.columns:
        for threshold in [0.90, 0.95]:
            covered = importance_df[
                importance_df["cumulative_importance"] <= threshold
            ]["feature"].tolist()

            if len(covered) == 0 and feature_count > 0:
                covered = [ranked_features[0]]

            if len(covered) < feature_count:
                next_index = len(covered)

                if next_index < feature_count:
                    covered = ranked_features[: next_index + 1]

            candidates[f"cumulative_{int(threshold * 100)}"] = covered

    unique_candidates = {}
    seen = set()

    for candidate_name, features in candidates.items():
        feature_tuple = tuple(features)

        if feature_tuple not in seen:
            unique_candidates[candidate_name] = features
            seen.add(feature_tuple)

    return unique_candidates


def select_best_feature_subset(
    reference_estimator,
    X_train_core,
    y_train_core,
    X_valid,
    y_valid,
    feature_candidates,
    problem_type,
    primary_metric,
    higher_is_better,
):
    """
    Train the best all-feature model family using multiple feature subsets.
    """
    results = []
    best_result = None

    for candidate_name, selected_features in feature_candidates.items():
        preprocessor, numeric_features, categorical_features = build_preprocessor(
            X_train_core[selected_features]
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", clone(reference_estimator)),
            ]
        )

        pipeline.fit(X_train_core[selected_features], y_train_core)

        train_metrics = evaluate_model(
            model=pipeline,
            X_test=X_train_core[selected_features],
            y_test=y_train_core,
            output_dir="reports",
        )

        validation_metrics = evaluate_model(
            model=pipeline,
            X_test=X_valid[selected_features],
            y_test=y_valid,
            output_dir="reports",
        )

        score = validation_metrics[primary_metric]

        result = {
            "candidate_name": candidate_name,
            "feature_count": len(selected_features),
            "selected_features": selected_features,
            "validation_score": float(score),
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
        }

        results.append(result)

        if best_result is None:
            best_result = result

        else:
            if higher_is_better:
                if score > best_result["validation_score"]:
                    best_result = result
                elif score == best_result["validation_score"] and len(selected_features) < best_result["feature_count"]:
                    best_result = result
            else:
                if score < best_result["validation_score"]:
                    best_result = result
                elif score == best_result["validation_score"] and len(selected_features) < best_result["feature_count"]:
                    best_result = result

    return best_result, results


def train_with_mlflow(args):
    """
    Train all-feature models, run feature-importance analysis, compare selected-feature
    performance against all-feature performance, and save the true final best model.
    """
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.report_dir).mkdir(parents=True, exist_ok=True)
    Path(args.feature_importance_dir).mkdir(parents=True, exist_ok=True)

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

    candidate_models = get_candidate_models(problem_type, args.random_state)

    best_all_feature_score = None
    best_all_feature_model_name = None
    best_all_feature_model = None
    best_all_feature_estimator = None
    best_all_feature_train_metrics = None
    best_all_feature_test_metrics = None
    best_all_feature_diagnostics = None

    run_summaries = []
    diagnostic_summaries = []

    for model_name, estimator in candidate_models.items():
        preprocessor, numeric_features, categorical_features = build_preprocessor(X_train)

        with mlflow.start_run(run_name=f"experiment_{model_name}_all_features"):
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
                model_name=f"{model_name}_all_features",
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                problem_type=problem_type,
                primary_metric=primary_metric,
                higher_is_better=higher_is_better,
            )

            diagnostic_summaries.append(diagnostics)

            score = test_metrics[primary_metric]

            mlflow.log_param("model_name", model_name)
            mlflow.log_param("model_stage", "all_features_reference")
            mlflow.log_param("problem_type", problem_type)
            mlflow.log_param("target_column", args.target_column)
            mlflow.log_param("train_rows", X_train.shape[0])
            mlflow.log_param("test_rows", X_test.shape[0])
            mlflow.log_param("numeric_features", len(numeric_features))
            mlflow.log_param("categorical_features", len(categorical_features))
            mlflow.log_param("primary_metric", primary_metric)
            mlflow.log_param("quality_verdict", diagnostics["verdict"])
            mlflow.log_param("leakage_status", leakage_report["status"])
            mlflow.log_param("feature_selection_used", "false")
            mlflow.log_param("is_final_model", "false")

            for metric_name, metric_value in train_metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(f"train_{metric_name}", metric_value)

            for metric_name, metric_value in test_metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(f"test_{metric_name}", metric_value)
                    mlflow.log_metric(metric_name, metric_value)

            if diagnostics["gap"] is not None:
                mlflow.log_metric("train_test_gap", diagnostics["gap"])

            for artifact_path in [
                Path(args.report_dir) / "metrics.json",
                Path(args.report_dir) / "classification_report.txt",
                Path(args.report_dir) / "confusion_matrix.csv",
                Path(args.report_dir) / "data_leakage_report.json",
            ]:
                if artifact_path.exists():
                    mlflow.log_artifact(str(artifact_path))

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
            )

            run_summary = {
                "model_name": model_name,
                "model_stage": "all_features_reference",
                "primary_metric": primary_metric,
                "score": score,
                "test_score": score,
                "feature_selection_used": False,
                "quality_verdict": diagnostics["verdict"],
                "diagnostic_warning": " | ".join(diagnostics["warnings"]),
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }

            run_summaries.append(run_summary)

            should_replace = (
                best_all_feature_score is None
                or (higher_is_better and score > best_all_feature_score)
                or (not higher_is_better and score < best_all_feature_score)
            )

            if should_replace:
                best_all_feature_score = score
                best_all_feature_model_name = model_name
                best_all_feature_model = pipeline
                best_all_feature_estimator = estimator
                best_all_feature_train_metrics = train_metrics
                best_all_feature_test_metrics = test_metrics
                best_all_feature_diagnostics = diagnostics

    best_all_features_model_path = Path(args.model_dir) / "best_all_features_model.pkl"
    best_reference_model_path = Path(args.model_dir) / "best_reference_model.pkl"

    joblib.dump(best_all_feature_model, best_all_features_model_path)
    joblib.dump(best_all_feature_model, best_reference_model_path)

    X_train_core, X_valid, y_train_core, y_valid = make_train_validation_split(
        X_train=X_train,
        y_train=y_train,
        problem_type=problem_type,
        validation_size=args.validation_size,
        random_state=args.random_state,
    )

    reference_preprocessor, _, _ = build_preprocessor(X_train_core)

    reference_pipeline_for_importance = Pipeline(
        steps=[
            ("preprocessor", reference_preprocessor),
            ("model", clone(best_all_feature_estimator)),
        ]
    )

    reference_pipeline_for_importance.fit(X_train_core, y_train_core)

    importance_df, top_10_payload = compute_permutation_feature_importance(
        pipeline=reference_pipeline_for_importance,
        X_valid=X_valid,
        y_valid=y_valid,
        problem_type=problem_type,
        output_dir=args.feature_importance_dir,
        random_state=args.random_state,
        n_repeats=args.permutation_repeats,
    )

    feature_candidates = build_feature_cutoff_candidates(importance_df)

    selected_feature_result, feature_selection_results = select_best_feature_subset(
        reference_estimator=best_all_feature_estimator,
        X_train_core=X_train_core,
        y_train_core=y_train_core,
        X_valid=X_valid,
        y_valid=y_valid,
        feature_candidates=feature_candidates,
        problem_type=problem_type,
        primary_metric=primary_metric,
        higher_is_better=higher_is_better,
    )

    selected_features = selected_feature_result["selected_features"]

    feature_selection_results_path = Path(args.feature_importance_dir) / "feature_selection_results.json"

    with open(feature_selection_results_path, "w", encoding="utf-8") as file:
        json.dump(feature_selection_results, file, indent=4)

    selected_features_payload = {
        "method": "permutation_importance_with_validation_cutoff_selection",
        "reference_best_model_name": best_all_feature_model_name,
        "primary_metric": primary_metric,
        "selected_candidate_name": selected_feature_result["candidate_name"],
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
        "top_10_reference_features": top_10_payload["top_10_features"],
        "feature_selection_validation_score": selected_feature_result["validation_score"],
    }

    selected_features_path = Path(args.feature_importance_dir) / "selected_features.json"

    with open(selected_features_path, "w", encoding="utf-8") as file:
        json.dump(selected_features_payload, file, indent=4)

    selected_preprocessor, selected_numeric_features, selected_categorical_features = build_preprocessor(
        X_train[selected_features]
    )

    selected_feature_pipeline = Pipeline(
        steps=[
            ("preprocessor", selected_preprocessor),
            ("model", clone(best_all_feature_estimator)),
        ]
    )

    selected_feature_pipeline.fit(X_train[selected_features], y_train)

    selected_train_metrics = evaluate_model(
        model=selected_feature_pipeline,
        X_test=X_train[selected_features],
        y_test=y_train,
        output_dir=args.report_dir,
    )

    selected_test_metrics = evaluate_model(
        model=selected_feature_pipeline,
        X_test=X_test[selected_features],
        y_test=y_test,
        output_dir=args.report_dir,
    )

    selected_diagnostics = evaluate_overfit_underfit(
        model_name=f"{best_all_feature_model_name}_selected_features",
        train_metrics=selected_train_metrics,
        test_metrics=selected_test_metrics,
        problem_type=problem_type,
        primary_metric=primary_metric,
        higher_is_better=higher_is_better,
    )

    diagnostic_summaries.append(selected_diagnostics)

    selected_score = selected_test_metrics[primary_metric]

    with mlflow.start_run(run_name=f"analysis_{best_all_feature_model_name}_selected_features"):
        mlflow.log_param("model_name", f"{best_all_feature_model_name}_selected_features")
        mlflow.log_param("model_stage", "selected_features_analysis")
        mlflow.log_param("reference_best_model_name", best_all_feature_model_name)
        mlflow.log_param("problem_type", problem_type)
        mlflow.log_param("target_column", args.target_column)
        mlflow.log_param("primary_metric", primary_metric)
        mlflow.log_param("feature_importance_method", "permutation_importance")
        mlflow.log_param("feature_selection_method", "validation_cutoff_selection")
        mlflow.log_param("selected_candidate_name", selected_feature_result["candidate_name"])
        mlflow.log_param("selected_feature_count", len(selected_features))
        mlflow.log_param("selected_features", json.dumps(selected_features))
        mlflow.log_param("top_10_reference_features", json.dumps(top_10_payload["top_10_features"]))
        mlflow.log_param("quality_verdict", selected_diagnostics["verdict"])
        mlflow.log_param("leakage_status", leakage_report["status"])
        mlflow.log_param("feature_selection_used", "true")
        mlflow.log_param("is_final_model", "false")

        for metric_name, metric_value in selected_train_metrics.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(f"selected_train_{metric_name}", metric_value)

        for metric_name, metric_value in selected_test_metrics.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(f"selected_test_{metric_name}", metric_value)
                mlflow.log_metric(metric_name, metric_value)

        if selected_diagnostics["gap"] is not None:
            mlflow.log_metric("selected_train_test_gap", selected_diagnostics["gap"])

        selected_summary = {
            "reference_best_model_name": best_all_feature_model_name,
            "selected_model_name": f"{best_all_feature_model_name}_selected_features",
            "primary_metric": primary_metric,
            "all_features_score": float(best_all_feature_score),
            "selected_features_score": float(selected_score),
            "selected_features": selected_features,
            "selected_feature_count": len(selected_features),
            "selected_candidate_name": selected_feature_result["candidate_name"],
            "selected_train_metrics": selected_train_metrics,
            "selected_test_metrics": selected_test_metrics,
            "selected_diagnostics": selected_diagnostics,
            "feature_importance_kept_as_analysis": True,
        }

        selected_summary_path = Path(args.report_dir) / "selected_feature_model_summary.json"

        with open(selected_summary_path, "w", encoding="utf-8") as file:
            json.dump(selected_summary, file, indent=4)

        for artifact_path in [
            selected_summary_path,
            Path(args.feature_importance_dir) / "all_feature_importance.csv",
            Path(args.feature_importance_dir) / "top_10_feature_importance.csv",
            Path(args.feature_importance_dir) / "top_10_features.json",
            feature_selection_results_path,
            selected_features_path,
            Path(args.report_dir) / "data_leakage_report.json",
        ]:
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path))

        mlflow.sklearn.log_model(
            sk_model=selected_feature_pipeline,
            artifact_path="model",
        )

    selected_feature_model_path = Path(args.model_dir) / "best_selected_feature_model.pkl"
    joblib.dump(selected_feature_pipeline, selected_feature_model_path)

    run_summaries.append(
        {
            "model_name": f"{best_all_feature_model_name}_selected_features",
            "model_stage": "selected_features_analysis",
            "primary_metric": primary_metric,
            "score": selected_score,
            "test_score": selected_score,
            "feature_selection_used": True,
            "quality_verdict": selected_diagnostics["verdict"],
            "diagnostic_warning": " | ".join(selected_diagnostics["warnings"]),
            **{f"train_{k}": v for k, v in selected_train_metrics.items()},
            **{f"test_{k}": v for k, v in selected_test_metrics.items()},
        }
    )

    selected_is_better = (
        (higher_is_better and selected_score > best_all_feature_score)
        or ((not higher_is_better) and selected_score < best_all_feature_score)
    )

    if selected_is_better:
        final_model_choice = "selected_features"
        final_model_name = f"{best_all_feature_model_name}_selected_features"
        final_model = selected_feature_pipeline
        final_features = selected_features
        final_feature_selection_used = True
        final_reason = (
            "Selected-feature model outperformed the all-feature model on the primary metric."
        )
    else:
        final_model_choice = "all_features"
        final_model_name = f"{best_all_feature_model_name}_all_features"
        final_model = best_all_feature_model
        final_features = X_train.columns.tolist()
        final_feature_selection_used = False
        final_reason = (
            "Selected-feature model did not outperform the all-feature model, "
            "so feature importance is kept as analysis only."
        )

    if final_feature_selection_used:
        final_X_train_for_eval = X_train[final_features]
        final_X_test_for_eval = X_test[final_features]
    else:
        final_X_train_for_eval = X_train
        final_X_test_for_eval = X_test

    final_train_metrics = evaluate_model(
        model=final_model,
        X_test=final_X_train_for_eval,
        y_test=y_train,
        output_dir=args.report_dir,
    )

    final_test_metrics = evaluate_model(
        model=final_model,
        X_test=final_X_test_for_eval,
        y_test=y_test,
        output_dir=args.report_dir,
    )

    final_diagnostics = evaluate_overfit_underfit(
        model_name=final_model_name,
        train_metrics=final_train_metrics,
        test_metrics=final_test_metrics,
        problem_type=problem_type,
        primary_metric=primary_metric,
        higher_is_better=higher_is_better,
    )

    with mlflow.start_run(run_name=f"final_{final_model_name}"):
        mlflow.log_param("model_name", final_model_name)
        mlflow.log_param("model_stage", "final_model")
        mlflow.log_param("final_model_choice", final_model_choice)
        mlflow.log_param("final_reason", final_reason)
        mlflow.log_param("reference_best_all_feature_model_name", best_all_feature_model_name)
        mlflow.log_param("problem_type", problem_type)
        mlflow.log_param("target_column", args.target_column)
        mlflow.log_param("primary_metric", primary_metric)
        mlflow.log_param("all_features_score", float(best_all_feature_score))
        mlflow.log_param("selected_features_score", float(selected_score))
        mlflow.log_param("feature_selection_used", str(final_feature_selection_used).lower())
        mlflow.log_param("final_features", json.dumps(final_features))
        mlflow.log_param("selected_features_analysis", json.dumps(selected_features))
        mlflow.log_param("top_10_reference_features", json.dumps(top_10_payload["top_10_features"]))
        mlflow.log_param("quality_verdict", final_diagnostics["verdict"])
        mlflow.log_param("leakage_status", leakage_report["status"])
        mlflow.log_param("is_final_model", "true")

        for metric_name, metric_value in final_train_metrics.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(f"final_train_{metric_name}", metric_value)

        for metric_name, metric_value in final_test_metrics.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(f"final_test_{metric_name}", metric_value)
                mlflow.log_metric(metric_name, metric_value)

        if final_diagnostics["gap"] is not None:
            mlflow.log_metric("final_train_test_gap", final_diagnostics["gap"])

        final_run_id = mlflow.active_run().info.run_id

        for artifact_path in [
            Path(args.report_dir) / "metrics.json",
            Path(args.report_dir) / "classification_report.txt",
            Path(args.report_dir) / "confusion_matrix.csv",
            Path(args.report_dir) / "data_leakage_report.json",
            Path(args.feature_importance_dir) / "all_feature_importance.csv",
            Path(args.feature_importance_dir) / "top_10_feature_importance.csv",
            Path(args.feature_importance_dir) / "top_10_features.json",
            feature_selection_results_path,
            selected_features_path,
        ]:
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path))

        mlflow.sklearn.log_model(
            sk_model=final_model,
            artifact_path="model",
        )

    best_model_path = Path(args.model_dir) / "best_model.pkl"
    joblib.dump(final_model, best_model_path)

    feature_metadata = {
        "target_column": args.target_column,
        "problem_type": problem_type,
        "final_model_choice": final_model_choice,
        "final_model_name": final_model_name,
        "feature_selection_used": final_feature_selection_used,
        "final_features": final_features,
        "selected_features_analysis": selected_features,
        "top_10_reference_features": top_10_payload["top_10_features"],
        "feature_importance_method": "permutation_importance",
        "feature_selection_method": "validation_cutoff_selection",
        "all_input_columns": X_train.columns.tolist(),
        "final_run_id": final_run_id,
        "model_path": str(best_model_path),
    }

    feature_metadata_path = Path(args.model_dir) / "feature_columns.json"

    with open(feature_metadata_path, "w", encoding="utf-8") as file:
        json.dump(feature_metadata, file, indent=4)

    summary = {
        "final_model_choice": final_model_choice,
        "final_model_name": final_model_name,
        "final_reason": final_reason,
        "primary_metric": primary_metric,
        "all_features_best_model_name": best_all_feature_model_name,
        "all_features_score": float(best_all_feature_score),
        "all_features_metrics": best_all_feature_test_metrics,
        "selected_features_score": float(selected_score),
        "selected_features_metrics": selected_test_metrics,
        "feature_importance_kept_as_analysis": not selected_is_better,
        "selected_features_analysis": selected_features,
        "top_10_reference_features": top_10_payload["top_10_features"],
        "final_features": final_features,
        "final_score": float(final_test_metrics[primary_metric]),
        "final_metrics": final_test_metrics,
        "model_path": str(best_model_path),
        "best_all_features_model_path": str(best_all_features_model_path),
        "selected_feature_model_path": str(selected_feature_model_path),
        "final_run_id": final_run_id,
        "leakage_status": leakage_report["status"],
    }

    summary_path = Path(args.report_dir) / "best_model_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    run_summaries.append(
        {
            "model_name": final_model_name,
            "model_stage": "final_model",
            "primary_metric": primary_metric,
            "score": final_test_metrics[primary_metric],
            "test_score": final_test_metrics[primary_metric],
            "feature_selection_used": final_feature_selection_used,
            "quality_verdict": final_diagnostics["verdict"],
            "diagnostic_warning": " | ".join(final_diagnostics["warnings"]),
            **{f"train_{k}": v for k, v in final_train_metrics.items()},
            **{f"test_{k}": v for k, v in final_test_metrics.items()},
        }
    )

    run_summary_path = Path(args.report_dir) / "mlflow_run_summary.csv"
    pd.DataFrame(run_summaries).to_csv(run_summary_path, index=False)

    save_model_diagnostics_report(
        diagnostics=diagnostic_summaries,
        output_dir=args.report_dir,
        filename="model_diagnostics_report.json",
    )

    print("MLflow experiment logging completed.")
    print(json.dumps(summary, indent=4))

    print("\nTop 10 reference features:")
    for feature in top_10_payload["top_10_features"]:
        print(f"- {feature}")

    print("\nSelected features tested through feature importance:")
    for feature in selected_features:
        print(f"- {feature}")

    print("\nFinal model decision:")
    print(final_reason)

    print("\nModel quality diagnostics:")
    for item in diagnostic_summaries:
        print(f"- {item['model_name']}: {item['verdict']} | {' | '.join(item['warnings'])}")

    if leakage_report["warnings"]:
        print("\nData leakage warnings:")
        for warning in leakage_report["warnings"]:
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
    parser.add_argument("--feature-importance-dir", default="reports/feature_importance")
    parser.add_argument("--experiment-name", default="mlops_training_experiments")
    parser.add_argument("--backend-store-path", default="mlruns/mlflow.db")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--validation-size", type=float, default=0.20)
    parser.add_argument("--permutation-repeats", type=int, default=8)

    return parser.parse_args()


if __name__ == "__main__":
    train_with_mlflow(parse_args())
