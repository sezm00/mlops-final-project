import logging
import os
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Gauge, Histogram,
                               generate_latest)

from src.serving.model_loader import load_model
from src.serving.preprocessor import PreprocessingAdapter
from src.serving.schemas import (BatchPredictRequest, BatchPredictResponse,
                                 PredictRequest, PredictResponse,
                                 SinglePrediction)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INFERENCE_COUNTER = Counter(
    "churn_inference_total",
    "Total number of inference requests",
    ["endpoint", "status"],
)
INFERENCE_LATENCY = Histogram(
    "churn_inference_latency_seconds",
    "Inference latency in seconds",
    ["endpoint"],
)
MODEL_VERSION_GAUGE = Gauge(
    "churn_model_version_info",
    "Currently loaded model version exposed as a labelled gauge",
    ["version", "name"],
)
CHURN_PREDICTION_COUNTER = Counter(
    "churn_prediction_label_total",
    "Count of churn predictions by label",
    ["label"],
)
CONFIDENCE_HISTOGRAM = Histogram(
    "churn_prediction_confidence",
    "Prediction confidence for the selected class",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
TENURE_HISTOGRAM = Histogram(
    "churn_input_tenure_months",
    "Distribution of tenure values received by the API",
    buckets=[0, 1, 6, 12, 24, 36, 48, 60, 72, 100],
)
MONTHLY_CHARGES_HISTOGRAM = Histogram(
    "churn_input_monthly_charges",
    "Distribution of MonthlyCharges values received by the API",
    buckets=[0, 20, 35, 50, 65, 80, 95, 110, 130, 160, 200],
)
BATCH_SIZE_HISTOGRAM = Histogram(
    "churn_batch_size",
    "Batch size for batch predictions",
    buckets=[1, 5, 10, 25, 50, 100, 250, 500],
)

MODEL = None
MODEL_INFO = {}
PREPROCESSOR: PreprocessingAdapter | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, MODEL_INFO, PREPROCESSOR

    logger.info("Loading preprocessing pipeline...")
    PREPROCESSOR = PreprocessingAdapter(
        feature_engineer_path=os.environ.get(
            "FEATURE_ENGINEER_PATH", "models/feature_engineer.joblib"
        ),
        preprocessing_pipeline_path=os.environ.get(
            "PIPELINE_PATH", "models/preprocessing_pipeline.joblib"
        ),
    )
    PREPROCESSOR.load()

    logger.info("Loading model...")
    MODEL, MODEL_INFO = load_model()
    MODEL_VERSION_GAUGE.labels(
        version=str(MODEL_INFO.get("version", "unknown")),
        name=str(MODEL_INFO.get("name", "unknown")),
    ).set(1)

    logger.info("Ready. Model: %s", MODEL_INFO)
    yield

    logger.info("Shutting down.")


app = FastAPI(
    title="Telco Churn Prediction API",
    description=(
        "Predicts customer churn for a telecom company.\n\n"
        "Send raw customer data; the API handles all preprocessing internally."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Monitoring"])
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "preprocessor_loaded": PREPROCESSOR is not None and PREPROCESSOR._loaded,
        "model_info": MODEL_INFO,
    }


@app.get("/metrics", tags=["Monitoring"])
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model/info", tags=["Model"])
def model_info():
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return MODEL_INFO


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(request: PredictRequest):
    _check_ready()
    start = time.perf_counter()

    try:
        customer_dict = request.customer.model_dump()
        _observe_input_metrics(customer_dict)
        features_df = PREPROCESSOR.transform(customer_dict)
        label, prob = _run_inference(features_df)

        CHURN_PREDICTION_COUNTER.labels(label=label).inc()
        _observe_confidence(label, prob)
        INFERENCE_COUNTER.labels(endpoint="/predict", status="success").inc()
        INFERENCE_LATENCY.labels(endpoint="/predict").observe(
            time.perf_counter() - start
        )

        return PredictResponse(
            churn=label,
            churn_probability=prob,
            model_version=str(MODEL_INFO.get("version", "unknown")),
        )

    except HTTPException:
        raise
    except Exception as exc:
        INFERENCE_COUNTER.labels(endpoint="/predict", status="error").inc()
        logger.exception("Prediction error")
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
def predict_batch(request: BatchPredictRequest):
    _check_ready()
    start = time.perf_counter()

    try:
        customer_dicts = [record.customer.model_dump() for record in request.records]
        for customer_dict in customer_dicts:
            _observe_input_metrics(customer_dict)
        BATCH_SIZE_HISTOGRAM.observe(len(customer_dicts))

        features_df = PREPROCESSOR.transform_batch(customer_dicts)
        predictions = MODEL.predict(features_df)

        try:
            probabilities = MODEL.predict_proba(features_df)
        except AttributeError:
            probabilities = None

        results = []
        for index, pred in enumerate(predictions):
            label = "Yes" if pred == 1 else "No"
            prob = float(probabilities[index][1]) if probabilities is not None else None
            CHURN_PREDICTION_COUNTER.labels(label=label).inc()
            _observe_confidence(label, prob)
            results.append(SinglePrediction(churn=label, churn_probability=prob))

        INFERENCE_COUNTER.labels(endpoint="/predict/batch", status="success").inc()
        INFERENCE_LATENCY.labels(endpoint="/predict/batch").observe(
            time.perf_counter() - start
        )

        return BatchPredictResponse(
            predictions=results,
            count=len(results),
            model_version=str(MODEL_INFO.get("version", "unknown")),
        )

    except HTTPException:
        raise
    except Exception as exc:
        INFERENCE_COUNTER.labels(endpoint="/predict/batch", status="error").inc()
        logger.exception("Batch prediction error")
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _check_ready():
    if MODEL is None or PREPROCESSOR is None or not PREPROCESSOR._loaded:
        raise HTTPException(
            status_code=503, detail="Service not ready; model or pipeline not loaded"
        )


def _observe_input_metrics(customer_dict: dict):
    TENURE_HISTOGRAM.observe(float(customer_dict["tenure"]))
    MONTHLY_CHARGES_HISTOGRAM.observe(float(customer_dict["MonthlyCharges"]))


def _observe_confidence(label: str, churn_probability: float | None):
    if churn_probability is None:
        return
    confidence = churn_probability if label == "Yes" else 1.0 - churn_probability
    CONFIDENCE_HISTOGRAM.observe(confidence)


def _run_inference(features_df: pd.DataFrame):
    prediction = MODEL.predict(features_df)[0]
    label = "Yes" if prediction == 1 else "No"

    try:
        proba = MODEL.predict_proba(features_df)[0]
        prob = float(proba[1])
    except AttributeError:
        prob = None

    return label, prob
