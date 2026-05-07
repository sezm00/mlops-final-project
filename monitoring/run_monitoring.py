import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_COLUMN = "Churn"
PREDICTION_COLUMN = "prediction"
DEFAULT_THRESHOLD = 0.20
DRIFTED_FEATURE_OUTPUT = "monitoring/evidently_reports/drifted_features_latest.json"
LOG_PATH = "monitoring/evidently_reports/drift_alerts.log"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_monitoring(
    reference_path: str = "data/processed/cleaned_reference.csv",
    production_path: str = "data/splits/production.csv",
    output_dir: str = "monitoring/evidently_reports",
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    reference = _normalize_monitoring_dtypes(pd.read_csv(reference_path))
    clean_current = _normalize_monitoring_dtypes(pd.read_csv(production_path))
    drifted_current = inject_drift(clean_current)

    reference = _prepare_for_model_performance(reference)
    clean_current = _prepare_for_model_performance(clean_current)
    drifted_current = _prepare_for_model_performance(drifted_current)

    baseline_summary = _run_evidently_report(
        reference,
        clean_current,
        output_path / "baseline_report.html",
        threshold,
    )
    drift_summary = _run_evidently_report(
        reference,
        drifted_current,
        output_path / "drift_report.html",
        threshold,
    )

    summary = {
        "threshold": threshold,
        "baseline": baseline_summary,
        "drift": drift_summary,
    }
    (output_path / "monitoring_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_alert_if_needed(drift_summary, output_path / "drift_alerts.log")
    return summary


def inject_drift(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    drifted = df.copy()
    rng = np.random.default_rng(random_state)

    if "MonthlyCharges" in drifted:
        monthly_charges = pd.to_numeric(drifted["MonthlyCharges"], errors="coerce")
        drifted["MonthlyCharges"] = monthly_charges * 1.35 + 20
    if "tenure" in drifted:
        tenure = pd.to_numeric(drifted["tenure"], errors="coerce")
        drifted["tenure"] = np.maximum(0, tenure * 0.45)
    if "TotalCharges" in drifted:
        total_charges = pd.to_numeric(drifted["TotalCharges"], errors="coerce")
        drifted["TotalCharges"] = total_charges * 1.25
    if "Contract" in drifted:
        drifted["Contract"] = "Month-to-month"
    if "InternetService" in drifted:
        drifted["InternetService"] = rng.choice(
            ["Fiber optic", "DSL"], size=len(drifted), p=[0.85, 0.15]
        )

    return drifted


def _run_evidently_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    html_path: Path,
    threshold: float,
) -> dict:
    from evidently.legacy.metric_preset import (ClassificationPreset,
                                                DataDriftPreset,
                                                DataQualityPreset)
    from evidently.legacy.metrics import DataDriftTable, DatasetDriftMetric
    from evidently.legacy.pipeline.column_mapping import ColumnMapping
    from evidently.legacy.report import Report

    column_mapping = ColumnMapping()
    if TARGET_COLUMN in reference.columns and PREDICTION_COLUMN in reference.columns:
        column_mapping.target = TARGET_COLUMN
        column_mapping.prediction = PREDICTION_COLUMN

    metrics = [
        DataDriftPreset(),
        DataQualityPreset(),
        DatasetDriftMetric(),
        DataDriftTable(),
    ]
    if column_mapping.target and column_mapping.prediction:
        metrics.append(ClassificationPreset())

    report = Report(metrics=metrics)
    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping,
    )
    report.save_html(str(html_path))

    summary = _extract_drift_summary(report.as_dict(), threshold)
    logger.info(
        "%s written. Drifted %s/%s features.",
        html_path,
        summary["drifted_feature_count"],
        summary["total_feature_count"],
    )
    return summary


def _prepare_for_model_performance(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    if TARGET_COLUMN in prepared.columns:
        prepared[PREDICTION_COLUMN] = prepared[TARGET_COLUMN]
    return prepared


def _normalize_monitoring_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for column in ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized


def _extract_drift_summary(report_dict: dict, threshold: float) -> dict:
    metrics = report_dict.get("metrics", [])
    dataset_metric = next(
        (
            metric
            for metric in metrics
            if metric.get("metric") in {"DataDriftTable", "DatasetDriftMetric"}
            and "drift_by_columns" in metric.get("result", {})
        ),
        {},
    )
    result = dataset_metric.get("result", {})
    drift_by_columns = result.get("drift_by_columns", {})

    drifted_features = []
    for feature, feature_result in drift_by_columns.items():
        if feature in {TARGET_COLUMN, PREDICTION_COLUMN}:
            continue
        if feature_result.get("drift_detected"):
            drifted_features.append(
                {
                    "feature": feature,
                    "score": _as_float(feature_result.get("drift_score")),
                    "test": feature_result.get("stattest_name"),
                }
            )

    total_features = max(result.get("number_of_columns", 0) - 2, 0)
    if total_features == 0:
        total_features = max(len(drift_by_columns) - 2, 0)
    drift_share = len(drifted_features) / total_features if total_features else 0.0

    return {
        "drift_detected": drift_share > threshold,
        "drift_share": round(drift_share, 4),
        "threshold": threshold,
        "drifted_feature_count": len(drifted_features),
        "total_feature_count": total_features,
        "drifted_features": drifted_features,
    }


def _as_float(value):
    if value is None:
        return None
    return float(value)


def _write_alert_if_needed(summary: dict, log_path: Path):
    if not summary["drift_detected"]:
        logger.info(
            "No drift alert: %.1f%% of features drifted.",
            summary["drift_share"] * 100,
        )
        return

    warning = {
        "event": "data_drift_alert",
        "message": (
            f"Drift detected on {summary['drift_share']:.1%} of features; "
            f"threshold is {summary['threshold']:.0%}."
        ),
        "drifted_features": summary["drifted_features"],
    }
    print(json.dumps(warning, indent=2))
    logger.warning(warning["message"])
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(warning) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Evidently monitoring reports."
    )
    parser.add_argument("--reference", default="data/processed/cleaned_reference.csv")
    parser.add_argument("--production", default="data/splits/production.csv")
    parser.add_argument("--output", default="monitoring/evidently_reports")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    run_monitoring(args.reference, args.production, args.output, args.threshold)
