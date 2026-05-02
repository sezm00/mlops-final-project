# =========================
# MLflow setup utilities
# =========================

from pathlib import Path

import mlflow


def configure_mlflow(
    experiment_name="mlops_training_experiments",
    backend_store_path="mlruns/mlflow.db",
):
    """
    Configure MLflow tracking using a local SQLite backend.

    SQLite is used instead of a file-only backend because it supports
    stronger experiment tracking and model registry behavior.
    """
    backend_store_path = Path(backend_store_path)
    backend_store_path.parent.mkdir(parents=True, exist_ok=True)

    tracking_uri = f"sqlite:///{backend_store_path.resolve()}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    return tracking_uri
