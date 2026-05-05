# =========================
# Model, leakage, and drift diagnostics
# =========================

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def _safe_float(value):
    """
    Convert values to normal Python floats for JSON serialization.
    """
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def check_data_leakage(
    train_df,
    test_df,
    target_column,
    output_dir="reports",
    high_corr_threshold=0.98,
):
    """
    Run practical data leakage checks.

    Checks included:
    1. Target column accidentally included as a feature.
    2. Duplicate feature rows between train and test.
    3. Feature columns that are identical to the target.
    4. Numeric features that are suspiciously correlated with the target.
    5. Categorical features that almost perfectly identify the target.

    These checks cannot prove there is no leakage, but they catch common problems.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    warnings = []
    details = {}

    if target_column not in train_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in train data.")

    if target_column not in test_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in test data.")

    feature_columns = [col for col in train_df.columns if col != target_column]

    if target_column in feature_columns:
        warnings.append("Target column appears inside feature columns.")

    train_feature_hashes = pd.util.hash_pandas_object(
        train_df[feature_columns].astype(str),
        index=False,
    )

    test_feature_hashes = pd.util.hash_pandas_object(
        test_df[feature_columns].astype(str),
        index=False,
    )

    overlap_count = int(len(set(train_feature_hashes).intersection(set(test_feature_hashes))))
    details["train_test_feature_overlap_count"] = overlap_count

    if overlap_count > 0:
        warnings.append(
            f"Possible leakage: {overlap_count} duplicated feature rows found across train and test."
        )

    identical_to_target = []

    for col in feature_columns:
        try:
            if train_df[col].reset_index(drop=True).equals(
                train_df[target_column].reset_index(drop=True)
            ):
                identical_to_target.append(col)
        except Exception:
            pass

    details["features_identical_to_target"] = identical_to_target

    if identical_to_target:
        warnings.append(
            f"High leakage risk: features identical to target found: {identical_to_target}"
        )

    high_target_correlations = {}

    numeric_columns = train_df[feature_columns].select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    if pd.api.types.is_numeric_dtype(train_df[target_column]):
        for col in numeric_columns:
            try:
                corr = train_df[col].corr(train_df[target_column])
                corr_abs = abs(corr)

                if corr_abs >= high_corr_threshold:
                    high_target_correlations[col] = _safe_float(corr)

            except Exception:
                pass

    details["high_target_correlations"] = high_target_correlations

    if high_target_correlations:
        warnings.append(
            f"Possible leakage: very high feature-target correlations found: {high_target_correlations}"
        )

    suspicious_categorical_features = {}

    categorical_columns = train_df[feature_columns].select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    for col in categorical_columns:
        try:
            crosstab = pd.crosstab(train_df[col], train_df[target_column], normalize="index")

            if not crosstab.empty:
                max_class_purity = crosstab.max(axis=1).max()

                if max_class_purity >= high_corr_threshold:
                    suspicious_categorical_features[col] = _safe_float(max_class_purity)

        except Exception:
            pass

    details["suspicious_categorical_features"] = suspicious_categorical_features

    if suspicious_categorical_features:
        warnings.append(
            "Possible leakage: categorical features almost perfectly identify the target: "
            f"{suspicious_categorical_features}"
        )

    status = "PASS" if not warnings else "WARNING"

    report = {
        "status": status,
        "warnings": warnings,
        "details": details,
    }

    report_path = output_path / "data_leakage_report.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    return report


def check_data_drift(
    train_df,
    test_df,
    target_column,
    output_dir="reports",
    numeric_pvalue_threshold=0.05,
    numeric_stat_threshold=0.20,
    categorical_tvd_threshold=0.20,
):
    """
    Compare train and test distributions to detect possible data drift.

    Numeric features:
    - Uses Kolmogorov-Smirnov test.
    - Flags drift when p-value is low and KS statistic is meaningful.

    Categorical features:
    - Uses total variation distance between distributions.
    - Flags drift when distribution shift is above threshold.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    feature_columns = [col for col in train_df.columns if col != target_column]

    drift_results = {}
    warnings = []

    for col in feature_columns:
        train_col = train_df[col].dropna()
        test_col = test_df[col].dropna()

        if train_col.empty or test_col.empty:
            continue

        if pd.api.types.is_numeric_dtype(train_col):
            try:
                ks_stat, p_value = ks_2samp(train_col, test_col)

                drift_detected = (
                    p_value < numeric_pvalue_threshold
                    and ks_stat >= numeric_stat_threshold
                )

                drift_results[col] = {
                    "type": "numeric",
                    "ks_statistic": _safe_float(ks_stat),
                    "p_value": _safe_float(p_value),
                    "drift_detected": bool(drift_detected),
                }

                if drift_detected:
                    warnings.append(
                        f"Numeric drift detected in '{col}' "
                        f"(KS={ks_stat:.4f}, p={p_value:.4f})."
                    )

            except Exception as error:
                drift_results[col] = {
                    "type": "numeric",
                    "error": str(error),
                }

        else:
            try:
                train_dist = train_col.astype(str).value_counts(normalize=True)
                test_dist = test_col.astype(str).value_counts(normalize=True)

                all_categories = sorted(set(train_dist.index).union(set(test_dist.index)))

                train_probs = np.array([train_dist.get(cat, 0.0) for cat in all_categories])
                test_probs = np.array([test_dist.get(cat, 0.0) for cat in all_categories])

                tvd = 0.5 * np.abs(train_probs - test_probs).sum()

                drift_detected = tvd >= categorical_tvd_threshold

                drift_results[col] = {
                    "type": "categorical",
                    "total_variation_distance": _safe_float(tvd),
                    "drift_detected": bool(drift_detected),
                }

                if drift_detected:
                    warnings.append(
                        f"Categorical drift detected in '{col}' "
                        f"(TVD={tvd:.4f})."
                    )

            except Exception as error:
                drift_results[col] = {
                    "type": "categorical",
                    "error": str(error),
                }

    status = "PASS" if not warnings else "WARNING"

    report = {
        "status": status,
        "warnings": warnings,
        "drift_results": drift_results,
    }

    report_path = output_path / "data_drift_report.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    return report


def evaluate_overfit_underfit(
    model_name,
    train_metrics,
    test_metrics,
    problem_type,
    primary_metric,
    higher_is_better=True,
    overfit_gap_threshold=0.10,
    classification_good_threshold=0.75,
    classification_underfit_threshold=0.60,
):
    """
    Create a model quality verdict based on train/test performance.

    Classification:
    - Good model: test score is high and train-test gap is acceptable.
    - Overfitting: train score is much higher than test score.
    - Underfitting: train and test scores are both weak.

    Regression:
    - Uses lower-is-better metric such as RMSE.
    - Overfitting means test RMSE is much worse than train RMSE.
    """
    warnings = []
    verdict = "GOOD"

    train_score = train_metrics.get(primary_metric)
    test_score = test_metrics.get(primary_metric)

    if train_score is None or test_score is None:
        return {
            "model_name": model_name,
            "verdict": "UNKNOWN",
            "warnings": [f"Could not evaluate overfitting because '{primary_metric}' is missing."],
            "primary_metric": primary_metric,
            "train_score": train_score,
            "test_score": test_score,
            "gap": None,
        }

    if higher_is_better:
        gap = train_score - test_score

        if gap > overfit_gap_threshold:
            verdict = "OVERFITTING_WARNING"
            warnings.append(
                f"Possible overfitting: train {primary_metric}={train_score:.4f}, "
                f"test {primary_metric}={test_score:.4f}, gap={gap:.4f}."
            )

        elif train_score < classification_underfit_threshold and test_score < classification_underfit_threshold:
            verdict = "UNDERFITTING_WARNING"
            warnings.append(
                f"Possible underfitting: both train and test {primary_metric} are low "
                f"(train={train_score:.4f}, test={test_score:.4f})."
            )

        elif test_score < classification_good_threshold:
            verdict = "WEAK_MODEL_WARNING"
            warnings.append(
                f"Model may not be strong enough: test {primary_metric}={test_score:.4f} "
                f"is below {classification_good_threshold:.2f}."
            )

        else:
            warnings.append(
                f"Model looks acceptable: test {primary_metric}={test_score:.4f}, "
                f"gap={gap:.4f}."
            )

    else:
        gap = test_score - train_score

        if gap > overfit_gap_threshold:
            verdict = "OVERFITTING_WARNING"
            warnings.append(
                f"Possible overfitting: train {primary_metric}={train_score:.4f}, "
                f"test {primary_metric}={test_score:.4f}, test is worse by {gap:.4f}."
            )

        else:
            warnings.append(
                f"Model looks acceptable based on {primary_metric}: "
                f"train={train_score:.4f}, test={test_score:.4f}."
            )

    return {
        "model_name": model_name,
        "verdict": verdict,
        "warnings": warnings,
        "primary_metric": primary_metric,
        "train_score": _safe_float(train_score),
        "test_score": _safe_float(test_score),
        "gap": _safe_float(gap),
    }


def save_model_diagnostics_report(
    diagnostics,
    output_dir="reports",
    filename="model_diagnostics_report.json",
):
    """
    Save model diagnostics to JSON.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report_path = output_path / filename

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(diagnostics, file, indent=4)

    return report_path
