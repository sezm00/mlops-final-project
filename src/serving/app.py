import os
import time
import logging
from contextlib import asynccontextmanager
from typing import List, Optional
 
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
 
from src.serving.schemas import PredictRequest, PredictResponse, BatchPredictRequest, BatchPredictResponse
from src.serving.model_loader import load_model, get_model_version
 
# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
 
# ─── Prometheus Metrics ───────────────────────────────────────────────────────
INFERENCE_COUNTER = Counter(
    "model_inference_total",
    "Total number of inference requests",
    ["endpoint", "status"],
)
INFERENCE_LATENCY = Histogram(
    "model_inference_latency_seconds",
    "Inference latency in seconds",
    ["endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
MODEL_VERSION_GAUGE = Gauge(
    "model_version_info",
    "Current model version in production",
    ["version", "run_id"],
)
CHURN_PREDICTION_COUNTER = Counter(
    "churn_prediction_total",
    "Total predictions by class",
    ["predicted_class"],
)
FEATURE_MEAN_GAUGE = Gauge(
    "feature_mean_value",
    "Running mean of key feature values",
    ["feature_name"],
)
 
# ─── Global model state ───────────────────────────────────────────────────────
MODEL_STATE: dict = {}
 
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    logger.info("Starting up — loading model...")
    try:
        model, version, run_id = load_model()
        MODEL_STATE["model"] = model
        MODEL_STATE["version"] = version
        MODEL_STATE["run_id"] = run_id
        MODEL_STATE["loaded_at"] = time.time()
        MODEL_VERSION_GAUGE.labels(version=str(version), run_id=run_id or "unknown").set(1)
        logger.info(f"Model loaded successfully — version={version}, run_id={run_id}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        MODEL_STATE["model"] = None
        MODEL_STATE["version"] = "unknown"
        MODEL_STATE["run_id"] = "unknown"
    yield
    logger.info("Shutting down — releasing model resources.")
    MODEL_STATE.clear()
 
 
# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Telco Churn Prediction API",
    description="MLOps Member 3 — Model serving with monitoring & CI/CD",
    version="1.0.0",
    lifespan=lifespan,
)
 
 
# ─── Routes ───────────────────────────────────────────────────────────────────
 
@app.get("/health", tags=["Health"])
def health_check():
    """Liveness + readiness probe."""
    model_loaded = MODEL_STATE.get("model") is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_version": MODEL_STATE.get("version", "unknown"),
        "run_id": MODEL_STATE.get("run_id", "unknown"),
        "uptime_seconds": round(time.time() - MODEL_STATE.get("loaded_at", time.time()), 2),
    }
 
 
@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(request: PredictRequest):
    """Single-sample churn prediction."""
    if MODEL_STATE.get("model") is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
 
    start = time.time()
    try:
        df = pd.DataFrame([request.model_dump()])
        prediction = MODEL_STATE["model"].predict(df)
        proba = MODEL_STATE["model"].predict_proba(df)
 
        churn = bool(prediction[0])
        churn_prob = float(proba[0][1])
 
        # Track metrics
        latency = time.time() - start
        INFERENCE_COUNTER.labels(endpoint="/predict", status="success").inc()
        INFERENCE_LATENCY.labels(endpoint="/predict").observe(latency)
        CHURN_PREDICTION_COUNTER.labels(predicted_class="churn" if churn else "no_churn").inc()
        _track_feature_means(df)
 
        logger.info(f"Predict: churn={churn}, prob={churn_prob:.4f}, latency={latency:.4f}s")
        return PredictResponse(
            churn=churn,
            churn_probability=churn_prob,
            model_version=MODEL_STATE.get("version", "unknown"),
        )
    except Exception as e:
        INFERENCE_COUNTER.labels(endpoint="/predict", status="error").inc()
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
 
 
@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
def predict_batch(request: BatchPredictRequest):
    """Batch churn prediction (bonus endpoint)."""
    if MODEL_STATE.get("model") is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
 
    if len(request.instances) == 0:
        raise HTTPException(status_code=422, detail="instances list must not be empty")
    if len(request.instances) > 1000:
        raise HTTPException(status_code=422, detail="Batch size cannot exceed 1000")
 
    start = time.time()
    try:
        df = pd.DataFrame([inst.model_dump() for inst in request.instances])
        predictions = MODEL_STATE["model"].predict(df)
        probas = MODEL_STATE["model"].predict_proba(df)
 
        results = [
            PredictResponse(
                churn=bool(pred),
                churn_probability=float(prob[1]),
                model_version=MODEL_STATE.get("version", "unknown"),
            )
            for pred, prob in zip(predictions, probas)
        ]
 
        latency = time.time() - start
        INFERENCE_COUNTER.labels(endpoint="/predict/batch", status="success").inc()
        INFERENCE_LATENCY.labels(endpoint="/predict/batch").observe(latency)
        _track_feature_means(df)
 
        logger.info(f"Batch predict: n={len(results)}, latency={latency:.4f}s")
        return BatchPredictResponse(
            predictions=results,
            total=len(results),
            model_version=MODEL_STATE.get("version", "unknown"),
        )
    except Exception as e:
        INFERENCE_COUNTER.labels(endpoint="/predict/batch", status="error").inc()
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
 
 
@app.get("/metrics", tags=["Monitoring"])
def metrics():
    """Prometheus metrics scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
 
 
# ─── Helpers ──────────────────────────────────────────────────────────────────
 
def _track_feature_means(df: pd.DataFrame):
    """Update Prometheus gauges for numeric feature distributions."""
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    for col in numeric_cols:
        if col in df.columns:
            try:
                val = float(df[col].mean())
                FEATURE_MEAN_GAUGE.labels(feature_name=col).set(val)
            except Exception:
                pass