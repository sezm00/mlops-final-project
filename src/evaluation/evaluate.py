# =========================
# Model evaluation utilities
# =========================

import json
from pathlib import Path

import numpy as np
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
    roc_auc_score,
)


def detect_problem_type(y):
    """
    Detect whether the target represents classification or regression.

    Classification is assumed when:
    - target dtype is object/category/bool, or
    - the number of unique values is reasonably small.
    """
    y_series = pd.Series(y)
    unique_count = y_series.nunique()

    if str(y_series.dtype) in ["object", "category", "bool"] or unique_count <= 20:
        return "classification"

    return "regression"


def evaluate_model(model, X_test, y_test, output_dir="reports"):
    """
    Evaluate a trained model and save metrics/reports.

    For classification:
    - accuracy
    - macro precision
    - macro recall
    - macro F1
    - ROC-AUC when probability predictions are available
    - classification report
    - confusion matrix

    For regression:
    - MAE
    - MSE
    - RMSE
    - R2
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

        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)

                if y_proba.shape[1] == 2:
                    metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba[:, 1]))

                elif y_proba.shape[1] > 2:
                    metrics["roc_auc_ovr"] = float(
                        roc_auc_score(y_test, y_proba, multi_class="ovr")
                    )

            except Exception:
                pass

        report = classification_report(y_test, y_pred, zero_division=0)
        confusion = confusion_matrix(y_test, y_pred)

        with open(output_path / "classification_report.txt", "w", encoding="utf-8") as file:
            file.write(report)

        pd.DataFrame(confusion).to_csv(
            output_path / "confusion_matrix.csv",
            index=False,
        )

    else:
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        metrics["mae"] = float(mean_absolute_error(y_test, y_pred))
        metrics["mse"] = float(mse)
        metrics["rmse"] = float(rmse)
        metrics["r2"] = float(r2_score(y_test, y_pred))

    with open(output_path / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    return metrics
