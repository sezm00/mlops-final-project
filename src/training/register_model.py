# =========================
# Register best model in MLflow
# =========================

import argparse
import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from src.training.mlflow_setup import configure_mlflow


def find_best_run(experiment_name, metric_name, higher_is_better=True):
    """
    Find the best MLflow run based on the selected metric.

    Logic:
    1. Search all runs that contain the selected metric.
    2. Sort by the metric.
    3. If tied, prefer the run marked is_final_model=true.
    """
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(
            f"Experiment '{experiment_name}' does not exist. "
            "Run training or HPO before registering the model."
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=1000,
    )

    valid_runs = []

    for run in runs:
        if metric_name in run.data.metrics:
            valid_runs.append(run)

    if not valid_runs:
        available_metrics = sorted(
            {metric for run in runs for metric in run.data.metrics.keys()}
        )

        raise ValueError(
            f"No MLflow runs found with metric '{metric_name}'. "
            f"Available metrics are: {available_metrics}"
        )

    valid_runs = sorted(
        valid_runs,
        key=lambda run: run.data.metrics[metric_name],
        reverse=higher_is_better,
    )

    best_metric_value = valid_runs[0].data.metrics[metric_name]

    tied_best_runs = [
        run for run in valid_runs if run.data.metrics[metric_name] == best_metric_value
    ]

    final_model_runs = [
        run for run in tied_best_runs if run.data.params.get("is_final_model") == "true"
    ]

    if final_model_runs:
        return final_model_runs[0]

    return valid_runs[0]


def load_json_if_exists(path):
    """
    Load a JSON file if it exists.
    """
    path = Path(path)

    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def register_and_promote_model(args):
    """
    Register the best MLflow model and promote it.

    Feature importance metadata is stored if available, but the registered model
    is chosen by best metric performance.
    """
    configure_mlflow(
        experiment_name=args.experiment_name,
        backend_store_path=args.backend_store_path,
    )

    best_run = find_best_run(
        experiment_name=args.experiment_name,
        metric_name=args.metric_name,
        higher_is_better=args.higher_is_better,
    )

    run_id = best_run.info.run_id
    best_metric_value = best_run.data.metrics[args.metric_name]

    model_uri = f"runs:/{run_id}/model"

    result = mlflow.register_model(
        model_uri=model_uri,
        name=args.registered_model_name,
    )

    client = MlflowClient()

    selected_features_payload = load_json_if_exists(args.selected_features_path)
    best_model_summary = load_json_if_exists(args.best_model_summary_path)

    selected_features = selected_features_payload.get("selected_features", [])
    top_10_reference_features = selected_features_payload.get(
        "top_10_reference_features", []
    )

    final_model_choice = best_model_summary.get(
        "final_model_choice",
        best_run.data.params.get("final_model_choice", "best_metric_run"),
    )

    final_reason = best_model_summary.get(
        "final_reason",
        "Registered the run with the best selected metric.",
    )

    feature_importance_kept_as_analysis = best_model_summary.get(
        "feature_importance_kept_as_analysis",
        "unknown",
    )

    model_version_tags = {
        "registered_metric_name": args.metric_name,
        "registered_metric_value": str(best_metric_value),
        "registered_run_id": run_id,
        "registered_model_name_from_run": best_run.data.params.get(
            "model_name", "unknown"
        ),
        "registered_model_stage_from_run": best_run.data.params.get(
            "model_stage", "unknown"
        ),
        "final_model_choice": str(final_model_choice),
        "final_reason": str(final_reason),
        "feature_importance_kept_as_analysis": str(feature_importance_kept_as_analysis),
        "selected_features_analysis": json.dumps(selected_features),
        "top_10_reference_features": json.dumps(top_10_reference_features),
        "feature_importance_method": "permutation_importance",
        "feature_selection_method": "validation_cutoff_selection",
    }

    for tag_key, tag_value in model_version_tags.items():
        client.set_model_version_tag(
            name=args.registered_model_name,
            version=result.version,
            key=tag_key,
            value=tag_value,
        )

    promotion_message = ""

    try:
        client.transition_model_version_stage(
            name=args.registered_model_name,
            version=result.version,
            stage="Staging",
            archive_existing_versions=True,
        )

        client.transition_model_version_stage(
            name=args.registered_model_name,
            version=result.version,
            stage="Production",
            archive_existing_versions=True,
        )

        promotion_message = "Model moved to Production."

    except Exception as error:
        client.set_registered_model_alias(
            name=args.registered_model_name,
            alias="staging",
            version=result.version,
        )

        client.set_registered_model_alias(
            name=args.registered_model_name,
            alias="production",
            version=result.version,
        )

        promotion_message = (
            "Classic stage transition was not available. "
            "Aliases 'staging' and 'production' were set instead."
        )

        print(f"Stage transition warning: {error}")

    print("Best model registered successfully.")
    print(f"Best run ID: {run_id}")
    print(f"Metric used: {args.metric_name}")
    print(f"Metric value: {best_metric_value}")
    print(f"Registered model name: {args.registered_model_name}")
    print(f"Registered model version: {result.version}")
    print(f"Model name from run: {best_run.data.params.get('model_name', 'unknown')}")
    print(f"Model stage from run: {best_run.data.params.get('model_stage', 'unknown')}")
    print(f"Final model choice: {final_model_choice}")
    print(f"Feature importance kept as analysis: {feature_importance_kept_as_analysis}")
    print(f"Selected features analysis: {selected_features}")
    print(f"Top 10 reference features: {top_10_reference_features}")
    print(promotion_message)


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment-name", default="mlops_training_experiments")
    parser.add_argument("--backend-store-path", default="mlruns/mlflow.db")
    parser.add_argument("--registered-model-name", default="BestMLOpsModel")
    parser.add_argument("--metric-name", default="f1_macro")
    parser.add_argument("--higher-is-better", action="store_true")
    parser.add_argument(
        "--selected-features-path",
        default="reports/feature_importance/selected_features.json",
    )
    parser.add_argument(
        "--best-model-summary-path", default="reports/best_model_summary.json"
    )

    return parser.parse_args()


if __name__ == "__main__":
    register_and_promote_model(parse_args())
