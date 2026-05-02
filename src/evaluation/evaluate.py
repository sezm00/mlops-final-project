# =========================
# Model evaluation utilities
# =========================

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def detect_problem_type(y):
    """
    Detect whether the target is classification or regression.
    """
    y_series = pd.Series(y)
    unique_count = y_series.nunique()

    if y_series.dtype == "object" or unique_count <= 20:
        return "classification"

    return "regression"


def evaluate_model(model, X_test, y_test, output_dir="reports"):
    """
    Evaluate a trained model and save metrics/reports.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    problem_type = detect_problem_type(y_test)
    y_pred = model.predict(X_test)

    metrics = {
        "problem_type": problem_type,
    }

    if problem_type == "classification":
        metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
        metrics["precision_macro"] = float(
            precision_score(y_test, y_pred, average="macro", zero_division=0)
        )
        metrics["recall_macro"] = float(
            recall_score(y_test, y_pred, average="macro", zero_division=0)
        )
        metrics["f1_macro"] = float(
            f1_score(y_test, y_pred, average="macro", zero_division=0)
        )

        report = classification_report(y_test, y_pred, zero_division=0)
        confusion = confusion_matrix(y_test, y_pred)

        with open(output_path / "classification_report.txt", "w", encoding="utf-8") as file:
            file.write(report)

        pd.DataFrame(confusion).to_csv(output_path / "confusion_matrix.csv", index=False)

    else:
        metrics["mae"] = float(mean_absolute_error(y_test, y_pred))
        metrics["rmse"] = float(mean_squared_error(y_test, y_pred, squared=False))
        metrics["r2"] = float(r2_score(y_test, y_pred))

    with open(output_path / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    return metrics
