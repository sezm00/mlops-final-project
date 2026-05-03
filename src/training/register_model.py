# =========================
# Register best model in MLflow
# =========================

import argparse

import mlflow
from mlflow.tracking import MlflowClient

from src.training.mlflow_setup import configure_mlflow


def find_best_run(experiment_name, metric_name, higher_is_better=True):
    """
    Find the best MLflow run based on the selected metric.

    This function avoids fragile MLflow filter syntax by:
    1. Getting all runs in the experiment.
    2. Keeping only runs that contain the selected metric.
    3. Sorting them in Python.
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
        max_results=500,
    )

    valid_runs = []

    for run in runs:
        if metric_name in run.data.metrics:
            valid_runs.append(run)

    if not valid_runs:
        available_metrics = sorted(
            {
                metric
                for run in runs
                for metric in run.data.metrics.keys()
            }
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

    return valid_runs[0]


def register_and_promote_model(args):
    """
    Register the best MLflow model and promote it.

    The code first tries classic MLflow stages:
    - Staging
    - Production

    If the installed MLflow version does not support classic stage transitions,
    it falls back to aliases:
    - staging
    - production
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

    return parser.parse_args()


if __name__ == "__main__":
    register_and_promote_model(parse_args())
