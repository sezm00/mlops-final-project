import os
from pathlib import Path

import mlflow


def configure_mlflow(
    experiment_name="mlops_training_experiments",
    backend_store_path="mlruns/mlflow.db",
):
    """
    Configure MLflow tracking.

    Priority:
    1. MLFLOW_TRACKING_URI env var (e.g. your running server)
    2. Local SQLite fallback (for offline / standalone runs)
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

    if not tracking_uri:
        # Fallback to local SQLite
        backend_store_path = Path(backend_store_path)
        backend_store_path.parent.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"sqlite:///{backend_store_path.resolve()}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    return tracking_uri
