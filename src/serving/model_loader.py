import os
import logging
import pickle
from typing import Tuple, Any, Optional
 
import mlflow
import mlflow.pyfunc
 
logger = logging.getLogger(__name__)
 
# Config from environment (set in docker-compose or CI)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "churn-model")
MLFLOW_MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Production")
LOCAL_MODEL_PATH = os.getenv("MLFLOW_MODEL_PATH", "model.pkl")
 
 
def load_model() -> Tuple[Any, str, Optional[str]]:
    """Load production model.
 
    Returns:
        (model, version, run_id)
 
    Raises:
        RuntimeError if no model source is available.
    """
    # ── Try MLflow Registry first ─────────────────────────────────────────────
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{MLFLOW_MODEL_NAME}/{MLFLOW_MODEL_STAGE}"
        logger.info(f"Loading model from MLflow: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        version, run_id = get_model_version(MLFLOW_MODEL_NAME, MLFLOW_MODEL_STAGE)
        logger.info(f"MLflow model loaded — version={version}, run_id={run_id}")
        return model, version, run_id
    except Exception as e:
        logger.warning(f"MLflow model load failed ({e}), falling back to local pkl...")
 
    # ── Fallback: local pickle ────────────────────────────────────────────────
    if os.path.exists(LOCAL_MODEL_PATH):
        try:
            with open(LOCAL_MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            logger.info(f"Loaded model from local pkl: {LOCAL_MODEL_PATH}")
            return model, "local-pkl", None
        except Exception as e:
            raise RuntimeError(f"Failed to load local model from {LOCAL_MODEL_PATH}: {e}")
 
    raise RuntimeError(
        f"No model available. MLflow registry unreachable and "
        f"local model not found at '{LOCAL_MODEL_PATH}'."
    )
 
 
def get_model_version(model_name: str, stage: str) -> Tuple[str, Optional[str]]:
    """Query MLflow registry for model version & run_id."""
    try:
        client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        versions = client.get_latest_versions(model_name, stages=[stage])
        if versions:
            v = versions[0]
            return v.version, v.run_id
        return "unknown", None
    except Exception as e:
        logger.warning(f"Could not fetch model version info: {e}")
        return "unknown", None