import argparse
import json
from pathlib import Path

from monitoring.run_monitoring import DEFAULT_THRESHOLD, run_monitoring


def check_drift_threshold(
    summary_path: str = "monitoring/evidently_reports/monitoring_summary.json",
) -> bool:
    path = Path(summary_path)
    if not path.exists():
        return False

    summary = json.loads(path.read_text(encoding="utf-8"))
    return not summary.get("drift", {}).get("drift_detected", False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Evidently drift detection.")
    parser.add_argument("--reference", default="data/processed/cleaned_reference.csv")
    parser.add_argument("--current", default="data/splits/production.csv")
    parser.add_argument("--output", default="monitoring/evidently_reports")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    run_monitoring(args.reference, args.current, args.output, args.threshold)
