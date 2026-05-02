# =========================
# Register best model in MLflow
# =========================

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.training.mlflow_setup import configure_mlflow


def find_best_run(experiment_name, metric_name, higher_is_better=True):
    """
    Find the best MLflow run based on the selected metric.
    """
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' does not exist.")

    order_by = [f"metrics.{metric_name} {'DESC' if higher_is_better else 'ASC'}"]

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=order_by,
        max_results=1,
    )

    if not runs:
        raise ValueError("No MLflow runs found.")

    return runs[0]


def register_and_promote_model(args):
    """
    Register best model and promote it to production.
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
    model_uri = f"runs:/{run_id}/model"

    result = mlflow.register_model(
        model_uri=model_uri,
        name=args.registered_model_name,
    )

    client = MlflowClient()

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

    print("Best model registered and promoted.")
    print(f"Run ID: {run_id}")
    print(f"Model name: {args.registered_model_name}")
    print(f"Model version: {result.version}")
    print("Stage: Production")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment-name", default="mlops_training_experiments")
    parser.add_argument("--backend-store-path", default="mlruns/mlflow.db")
    parser.add_argument("--registered-model-name", default="BestMLOpsModel")
    parser.add_argument("--metric-name", default="f1_macro")
    parser.add_argument("--higher-is-better", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    register_and_promote_model(parse_args())
