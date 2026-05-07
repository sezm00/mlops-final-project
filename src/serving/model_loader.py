import logging
import os
from pathlib import Path
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)


def load_model() -> Tuple[Any, Dict]:
    """
    Load the churn model.

    Returns
    -------
    model : sklearn-compatible model
    info  : dict with version, source, name
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

    if tracking_uri:
        try:
            return _load_from_mlflow(tracking_uri)
        except Exception as e:
            logger.warning(f"MLflow load failed ({e}), falling back to local pkl")

    return _load_local()


def _load_from_mlflow(tracking_uri: str) -> Tuple[Any, Dict]:
    import mlflow
    import mlflow.sklearn

    mlflow.set_tracking_uri(tracking_uri)
    model_name = os.environ.get("MLFLOW_MODEL_NAME", "BestMLOpsModel")

    # Try Production stage first (legacy), then @champion alias
    for ref in [f"models:/{model_name}/Production", f"models:/{model_name}@champion"]:
        try:
            model = mlflow.sklearn.load_model(ref)
            client = mlflow.MlflowClient()

            # Get version info
            try:
                versions = client.get_latest_versions(model_name, stages=["Production"])
                version = versions[0].version if versions else "unknown"
            except Exception:
                version = "unknown"

            logger.info(f"Loaded model from MLflow: {ref} (version {version})")
            return model, {
                "source": "mlflow",
                "name": model_name,
                "version": version,
                "uri": ref,
            }
        except Exception as e:
            logger.debug(f"Could not load {ref}: {e}")
            continue

    raise RuntimeError(
        f"No Production model found in MLflow registry for '{model_name}'"
    )


def _load_local() -> Tuple[Any, Dict]:
    import joblib

    candidates = [
        Path("models/best_model.pkl"),
        Path("models/best_all_features_model.pkl"),
    ]
    for path in candidates:
        if path.exists():
            model = joblib.load(path)
            logger.info(f"Loaded model from local path: {path}")
            return model, {
                "source": "local",
                "name": path.stem,
                "version": "local",
                "uri": str(path),
            }

    raise FileNotFoundError(
        "No model found. Tried MLflow registry and local paths: "
        + ", ".join(str(p) for p in candidates)
    )


def get_model_info() -> Dict:
    """Convenience wrapper — returns empty dict if model not loaded."""
    return {}
