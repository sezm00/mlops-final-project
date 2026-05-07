import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
 
import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, ClassificationPreset
from evidently.metrics import (
    DatasetDriftMetric,
    DataDriftTable,
    ColumnDriftMetric,
)
 
logger = logging.getLogger(__name__)
 
REPORTS_DIR = Path("monitoring/evidently_reports")
DRIFT_THRESHOLD = 0.20  # >20% features drifted → flag
 
TARGET_COLUMN = "Churn"
NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
 
 
def get_column_mapping() -> ColumnMapping:
    return ColumnMapping(
        target=TARGET_COLUMN,
        numerical_features=NUMERICAL_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )
 
 
def generate_baseline_report(
    reference_data: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Generate a data quality + drift baseline report from the training set.
    This is the reference snapshot for future drift comparisons.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = output_path or REPORTS_DIR / "baseline_report.html"
 
    column_mapping = get_column_mapping()
 
    report = Report(metrics=[
        DataQualityPreset(),
        DataDriftPreset(),
    ])
    # Baseline uses reference as both reference and current (self-comparison)
    report.run(
        reference_data=reference_data,
        current_data=reference_data,
        column_mapping=column_mapping,
    )
    report.save_html(str(output_path))
    logger.info(f"Baseline report saved → {output_path}")
    return output_path
 
 
def generate_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> Tuple[Path, bool, float]:
    """
    Compare current production data to the reference baseline.
 
    Returns:
        (report_path, drift_detected, drift_share)
        drift_detected = True if drift_share > DRIFT_THRESHOLD
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_path or REPORTS_DIR / f"drift_report_{timestamp}.html"
 
    column_mapping = get_column_mapping()
 
    report = Report(metrics=[
        DatasetDriftMetric(),
        DataDriftTable(),
        *[ColumnDriftMetric(column_name=col) for col in NUMERICAL_FEATURES],
    ])
    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )
    report.save_html(str(output_path))
 
    # ── Extract drift share ────────────────────────────────────────────────────
    report_dict = report.as_dict()
    drift_share = _extract_drift_share(report_dict)
    drift_detected = drift_share > DRIFT_THRESHOLD
 
    if drift_detected:
        logger.warning(
            f"⚠️  DRIFT DETECTED: {drift_share:.1%} of features drifted "
            f"(threshold={DRIFT_THRESHOLD:.0%})"
        )
        _save_drift_alert(drift_share, timestamp)
    else:
        logger.info(f"✅ No drift: {drift_share:.1%} of features drifted (threshold={DRIFT_THRESHOLD:.0%})")
 
    logger.info(f"Drift report saved → {output_path}")
    return output_path, drift_detected, drift_share
 
 
def _extract_drift_share(report_dict: dict) -> float:
    """Pull dataset drift share from Evidently report dict."""
    try:
        for metric in report_dict.get("metrics", []):
            if metric.get("metric") == "DatasetDriftMetric":
                result = metric.get("result", {})
                return float(result.get("drift_share", 0.0))
        return 0.0
    except Exception as e:
        logger.warning(f"Could not extract drift share: {e}")
        return 0.0
 
 
def _save_drift_alert(drift_share: float, timestamp: str):
    """Persist drift alert to JSON for downstream alerting / CI checks."""
    alert_path = REPORTS_DIR / "latest_drift_alert.json"
    alert = {
        "timestamp": timestamp,
        "drift_share": drift_share,
        "threshold": DRIFT_THRESHOLD,
        "alert": True,
    }
    with open(alert_path, "w") as f:
        json.dump(alert, f, indent=2)
    logger.info(f"Drift alert saved → {alert_path}")
 
 
def check_drift_alert() -> bool:
    """Return True if latest saved drift alert indicates drift."""
    alert_path = REPORTS_DIR / "latest_drift_alert.json"
    if not alert_path.exists():
        return False
    with open(alert_path) as f:
        alert = json.load(f)
    return alert.get("alert", False)
 
 
# ─── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
 
    train_path = sys.argv[1] if len(sys.argv) > 1 else "data/splits/train.csv"
    current_path = sys.argv[2] if len(sys.argv) > 2 else "data/splits/test.csv"
 
    ref = pd.read_csv(train_path)
    cur = pd.read_csv(current_path)
 
    generate_baseline_report(ref)
    report_path, drifted, share = generate_drift_report(ref, cur)
 
    print(f"\nDrift share: {share:.1%}")
    print(f"Drift detected: {drifted}")
    sys.exit(1 if drifted else 0)
 