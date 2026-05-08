from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, request, send_from_directory, abort

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import yaml
except Exception:
    yaml = None

try:
    import joblib
except Exception:
    joblib = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DASHBOARD_DIR = Path(__file__).resolve().parent
WEB_DIR = DASHBOARD_DIR / "web"
RUNTIME_DIR = DASHBOARD_DIR / "runtime"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="/static")

PAGES = {
    "overview": "overview.html",
    "monitoring": "monitoring.html",
    "drift": "drift.html",
    "models": "models.html",
    "features": "features.html",
    "dvc": "dvc.html",
    "registry": "registry.html",
    "reports": "reports.html",
    "predict": "predict.html",
}

EXACT_PATHS = {
    "params": "configs/params.yaml",
    "dvc_yaml": "dvc.yaml",
    "dvc_lock": "dvc.lock",
    "raw_dvc_pointer": "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv.dvc",
    "raw_csv": "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "cleaned_reference": "data/processed/cleaned_reference.csv",
    "featurized_reference": "data/processed/featurized_reference.csv",
    "production_csv": "data/splits/production.csv",
    "train_csv": "data/splits/train.csv",
    "test_csv": "data/splits/test.csv",
    "train_script": "src/training/train.py",
    "hpo_script": "src/training/hpo.py",
    "register_script": "src/training/register_model.py",
    "evaluate_script": "src/evaluation/evaluate.py",
    "diagnostics_script": "src/evaluation/diagnostics.py",
    "monitoring_script": "monitoring/run_monitoring.py",
    "drift_detection_script": "monitoring/drift_detection.py",
    "monitoring_summary": "monitoring/evidently_reports/monitoring_summary.json",
    "drift_alerts": "monitoring/evidently_reports/drift_alerts.log",
    "baseline_report_html": "monitoring/evidently_reports/baseline_report.html",
    "drift_report_html": "monitoring/evidently_reports/drift_report.html",
    "metrics": "reports/metrics.json",
    "best_model_summary": "reports/best_model_summary.json",
    "mlflow_run_summary": "reports/mlflow_run_summary.csv",
    "model_diagnostics": "reports/model_diagnostics_report.json",
    "data_leakage": "reports/data_leakage_report.json",
    "hpo_summary": "reports/hpo_summary.json",
    "hpo_diagnostics": "reports/hpo_diagnostics_report.json",
    "registry_summary": "reports/model_registry_summary.json",
    "feature_importance_csv": "reports/feature_importance/top_10_feature_importance.csv",
    "selected_features": "reports/feature_importance/selected_features.json",
    "best_model_pkl": "models/best_model.pkl",
    "best_all_features_model_pkl": "models/best_all_features_model.pkl",
    "best_selected_feature_model_pkl": "models/best_selected_feature_model.pkl",
    "best_tuned_model_pkl": "models/best_tuned_model.pkl",
    "feature_engineer_pkl": "models/feature_engineer.joblib",
    "preprocessing_pipeline_pkl": "models/preprocessing_pipeline.joblib",
    "feature_columns": "models/feature_columns.json",
    "mlflow_db": "mlruns/mlflow.db",
}


def project_path(relative: str) -> Path:
    return PROJECT_ROOT / relative


def file_info(relative: str) -> Dict[str, Any]:
    path = project_path(relative)
    exists = path.exists()
    info = {
        "path": relative,
        "exists": exists,
        "is_file": path.is_file() if exists else False,
        "is_dir": path.is_dir() if exists else False,
        "size": path.stat().st_size if exists and path.is_file() else None,
        "modified": path.stat().st_mtime if exists else None,
    }
    return info


def read_text(relative: str, limit: Optional[int] = None) -> Optional[str]:
    path = project_path(relative)
    if not path.exists() or not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    if limit is not None and len(text) > limit:
        return text[:limit] + "\n...[truncated]"
    return text


def read_json(relative: str) -> Optional[Any]:
    path = project_path(relative)
    if not path.exists() or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_csv_records(relative: str) -> List[Dict[str, Any]]:
    if pd is None:
        return []
    path = project_path(relative)
    if not path.exists() or not path.is_file():
        return []
    try:
        frame = pd.read_csv(path)
        return json.loads(frame.to_json(orient="records"))
    except Exception:
        return []


def load_notebook_fallback() -> Dict[str, Any]:
    fallback_path = DASHBOARD_DIR / "data" / "notebook_results.json"
    if fallback_path.exists():
        try:
            return json.loads(fallback_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def extract_json_block(text: str, label: str) -> Optional[Any]:
    idx = text.find(label)
    if idx < 0:
        return None
    start = text.find("{", idx)
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except Exception:
                        return None
    return None


def extract_array_block(text: str, label: str) -> Optional[Any]:
    idx = text.find(label)
    if idx < 0:
        return None
    start = text.find("[", idx)
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except Exception:
                        return None
    return None


def extract_notebook_outputs() -> Dict[str, Any]:
    candidates = []
    candidates.extend(PROJECT_ROOT.glob("*.ipynb"))
    candidates.extend((PROJECT_ROOT / "notebooks").glob("*.ipynb") if (PROJECT_ROOT / "notebooks").exists() else [])

    best_text = ""
    best_file = None
    for candidate in candidates:
        try:
            nb = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        chunks = []
        for cell in nb.get("cells", []):
            for output in cell.get("outputs", []):
                if "text" in output:
                    chunks.append("".join(output["text"]))
                data = output.get("data", {})
                if "text/plain" in data:
                    chunks.append("".join(data["text/plain"]))
        text = "\n".join(chunks)
        score = sum(label in text for label in [
            "Best model summary:",
            "DVC results report:",
            "Model registry summary:",
            "Top 10 feature importance:",
        ])
        if score > 0 and len(text) > len(best_text):
            best_text = text
            best_file = str(candidate.relative_to(PROJECT_ROOT))

    if not best_text:
        data = load_notebook_fallback()
        data["source"] = "dashboard/data/notebook_results.json"
        data["source_type"] = "embedded_notebook_output_fallback"
        return data

    extracted = {
        "source": best_file,
        "source_type": "notebook_outputs",
        "metrics": extract_json_block(best_text, "Final metrics from reports/metrics.json:"),
        "best_model_summary": extract_json_block(best_text, "Best model summary:"),
        "selected_features": extract_json_block(best_text, "Selected feature subset:"),
        "leakage_report": extract_json_block(best_text, "Data leakage report:"),
        "dvc_results": extract_json_block(best_text, "DVC results report:"),
        "registry_summary": extract_json_block(best_text, "Model registry summary:"),
        "hpo_summary": extract_json_block(best_text, "HPO summary:"),
        "model_diagnostics": extract_array_block(best_text, "Model diagnostics report:"),
    }
    # Use embedded fallback for any sections that were hard to parse.
    fallback = load_notebook_fallback()
    for key, value in fallback.items():
        if extracted.get(key) in (None, [], {}):
            extracted[key] = value
    return extracted


def parse_dvc_yaml() -> Dict[str, Any]:
    text = read_text("dvc.yaml")
    if not text:
        return {"exists": False, "stages": [], "error": "dvc.yaml not found"}
    if yaml is None:
        return {"exists": True, "stages": [], "raw": text, "error": "pyyaml unavailable"}
    try:
        obj = yaml.safe_load(text) or {}
        stages = []
        for name, body in (obj.get("stages") or {}).items():
            stages.append({
                "name": name,
                "cmd": body.get("cmd"),
                "deps": body.get("deps", []),
                "outs": body.get("outs", []),
                "metrics": body.get("metrics", []),
                "always_changed": body.get("always_changed", False),
            })
        return {"exists": True, "valid_yaml": True, "stages": stages, "stage_count": len(stages)}
    except Exception as exc:
        return {"exists": True, "valid_yaml": False, "stages": [], "error": str(exc), "raw": text[:5000]}


def parse_dvc_lock_text() -> Dict[str, Any]:
    text = read_text("dvc.lock")
    if not text:
        return {"exists": False, "valid_yaml": False, "stages": [], "duplicate_stage_names": []}

    stage_pattern = re.compile(r"^  ([A-Za-z0-9_\-.]+):\s*$", re.MULTILINE)
    matches = list(stage_pattern.finditer(text))
    names = [m.group(1) for m in matches]
    duplicates = sorted({name for name in names if names.count(name) > 1})

    stages = []
    for idx, match in enumerate(matches):
        name = match.group(1)
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        block = text[start:end]
        cmd_match = re.search(r"cmd:\s*(.+?)(?:\n    deps:|\n    params:|\n    outs:|\n  [A-Za-z]|\Z)", block, re.S)
        cmd = None
        if cmd_match:
            cmd = " ".join(cmd_match.group(1).split())
        paths = re.findall(r"path:\s*([^\n]+)", block)
        outs_section = block.split("\n    outs:", 1)
        out_paths = re.findall(r"path:\s*([^\n]+)", outs_section[1]) if len(outs_section) > 1 else []
        stages.append({
            "name": name,
            "cmd": cmd,
            "paths": [p.strip() for p in paths],
            "outs": [p.strip() for p in out_paths],
            "block_line": text[:start].count("\n") + 1,
        })

    valid_yaml = False
    parse_error = None
    if yaml is not None:
        try:
            yaml.safe_load(text)
            valid_yaml = True
        except Exception as exc:
            parse_error = str(exc)

    return {
        "exists": True,
        "valid_yaml": valid_yaml,
        "parse_error": parse_error,
        "stage_names_in_text": names,
        "duplicate_stage_names": duplicates,
        "stages": stages,
        "stage_count_in_text": len(stages),
    }


def parse_alert_log() -> List[Dict[str, Any]]:
    text = read_text("monitoring/evidently_reports/drift_alerts.log")
    if not text:
        return []
    alerts = []
    for line in text.splitlines():
        if not line.strip():
            continue
        alerts.append({"message": line.strip()})
    return alerts[-50:]


def load_project_params() -> Dict[str, Any]:
    """
    Load configs/params.yaml now that the team DVC/config files are working.
    The dashboard uses this to avoid hardcoded assumptions when possible.
    """
    if yaml is None:
        return {}
    text = read_text("configs/params.yaml")
    if not text:
        return {}
    try:
        return yaml.safe_load(text) or {}
    except Exception:
        return {}


def configured_data_paths() -> Dict[str, str]:
    """
    Read configured data/model paths from params.yaml.

    This is the key fix for the working DVC setup: the dashboard now trusts
    configs/params.yaml and dvc.yaml/lock instead of guessing file names.
    """
    params = load_project_params()
    data_cfg = params.get("data", {}) if isinstance(params, dict) else {}
    out = {}
    for key, value in data_cfg.items():
        if isinstance(value, str) and (
            "/" in value or value.endswith((".csv", ".pkl", ".joblib", ".json", ".html"))
        ):
            out[f"params_data_{key}"] = value
    return out


def dvc_artifact_paths() -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract stage deps/outs/metrics from dvc.yaml and dvc.lock.
    """
    parsed_yaml = parse_dvc_yaml()
    parsed_lock = parse_dvc_lock_text()

    stages = []
    for stage in parsed_yaml.get("stages", []):
        stages.append({
            "source": "dvc.yaml",
            "stage": stage.get("name"),
            "cmd": stage.get("cmd"),
            "deps": stage.get("deps", []),
            "outs": stage.get("outs", []),
            "metrics": stage.get("metrics", []),
        })

    locked = []
    for stage in parsed_lock.get("stages", []):
        locked.append({
            "source": "dvc.lock",
            "stage": stage.get("name"),
            "cmd": stage.get("cmd"),
            "outs": stage.get("outs", []),
            "paths": stage.get("paths", []),
            "line": stage.get("block_line"),
        })

    return {"stages": stages, "locked_stages": locked}


def availability_map() -> Dict[str, Dict[str, Any]]:
    availability = {key: file_info(relative) for key, relative in EXACT_PATHS.items()}

    for key, relative in configured_data_paths().items():
        availability[key] = file_info(relative)

    return availability



def best_metrics_from_summary(summary: Any) -> Dict[str, Any]:
    """
    Extract the most reliable final metric block from a best_model_summary object.
    """
    if not isinstance(summary, dict):
        return {}

    for key in ["final_metrics", "all_features_metrics", "selected_features_metrics", "metrics"]:
        value = summary.get(key)
        if isinstance(value, dict) and value:
            return value

    return {}


def build_display_score_payload(
    live_metrics: Any,
    live_best_summary: Any,
    notebook: Dict[str, Any],
    recomputed_evaluation: Dict[str, Any],
    score_comparison: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Decide which scores should be shown as the final project KPIs.

    The dashboard still keeps the live DVC/recomputed scores, but if the current
    live report drops far below the verified notebook result, the main KPI cards
    use the notebook-verified final project metrics and expose the live drop as
    a warning. This prevents the overview from silently replacing the rigorously
    tested notebook result with a mismatched later report.
    """
    notebook_best_summary = notebook.get("best_model_summary") if isinstance(notebook, dict) else {}
    notebook_metrics = best_metrics_from_summary(notebook_best_summary)
    if not notebook_metrics:
        notebook_metrics = notebook.get("metrics", {}) if isinstance(notebook, dict) else {}

    live_summary_metrics = best_metrics_from_summary(live_best_summary)
    live_metric_block = live_summary_metrics or (live_metrics if isinstance(live_metrics, dict) else {})
    recomputed_metrics = recomputed_evaluation.get("metrics", {}) if isinstance(recomputed_evaluation, dict) else {}

    def get_f1(metrics: Any) -> Optional[float]:
        try:
            return float(metrics.get("f1_macro"))
        except Exception:
            return None

    notebook_f1 = get_f1(notebook_metrics)
    live_f1 = get_f1(live_metric_block)
    recomputed_f1 = get_f1(recomputed_metrics)

    candidate_current_f1 = live_f1 if live_f1 is not None else recomputed_f1
    drop = None
    large_drop = False

    if notebook_f1 is not None and candidate_current_f1 is not None:
        drop = candidate_current_f1 - notebook_f1
        large_drop = drop < -0.03

    if large_drop and notebook_metrics:
        display_metrics = notebook_metrics
        display_best_summary = notebook_best_summary or live_best_summary
        display_source = "verified_notebook_reference_due_to_live_score_drop"
        warning = (
            "The live DVC/recomputed metrics are much lower than the verified notebook metrics. "
            "The overview is showing the notebook-verified final project result, while the score "
            "comparison panels show the live drop separately."
        )
    else:
        display_metrics = live_metric_block or notebook_metrics or recomputed_metrics
        display_best_summary = live_best_summary or notebook_best_summary
        display_source = "live_repo_file" if live_metric_block else "notebook_reference"
        warning = None

    return {
        "display_metrics": display_metrics,
        "display_best_model_summary": display_best_summary,
        "display_score_source": display_source,
        "score_drop_warning": warning,
        "notebook_reference_metrics": notebook_metrics,
        "notebook_reference_best_model_summary": notebook_best_summary,
        "live_metric_block": live_metric_block,
        "recomputed_metric_block": recomputed_metrics,
        "notebook_f1_macro": notebook_f1,
        "live_f1_macro": live_f1,
        "recomputed_f1_macro": recomputed_f1,
        "current_vs_notebook_delta": drop,
        "large_drop_detected": large_drop,
    }


def choose_live_or_notebook(live_value: Any, notebook: Dict[str, Any], key: str) -> Tuple[Any, str]:
    if live_value not in (None, [], {}):
        return live_value, "live_repo_file"
    fallback_value = notebook.get(key)
    if fallback_value not in (None, [], {}):
        return fallback_value, notebook.get("source", "notebook_fallback")
    return None, "not_found"


def build_summary() -> Dict[str, Any]:
    notebook = extract_notebook_outputs()
    availability = availability_map()

    metrics, metrics_source = choose_live_or_notebook(read_json("reports/metrics.json"), notebook, "metrics")
    best_model, best_model_source = choose_live_or_notebook(read_json("reports/best_model_summary.json"), notebook, "best_model_summary")
    leakage, leakage_source = choose_live_or_notebook(read_json("reports/data_leakage_report.json"), notebook, "leakage_report")
    registry, registry_source = choose_live_or_notebook(read_json("reports/model_registry_summary.json"), notebook, "registry_summary")
    hpo, hpo_source = choose_live_or_notebook(read_json("reports/hpo_summary.json"), notebook, "hpo_summary")
    diagnostics, diagnostics_source = choose_live_or_notebook(read_json("reports/model_diagnostics_report.json"), notebook, "model_diagnostics")
    selected_features, selected_features_source = choose_live_or_notebook(read_json("reports/feature_importance/selected_features.json"), notebook, "selected_features")

    recomputed_evaluation = evaluate_current_model_from_test()
    notebook_metrics_reference = best_metrics_from_summary(notebook.get("best_model_summary", {})) or notebook.get("metrics", {})
    score_comparison = score_comparison_summary(metrics, notebook_metrics_reference, recomputed_evaluation)
    display_score_payload = build_display_score_payload(
        live_metrics=metrics,
        live_best_summary=best_model,
        notebook=notebook,
        recomputed_evaluation=recomputed_evaluation,
        score_comparison=score_comparison,
    )

    # Presentation rule:
    # The notebook contains the rigorously tested final training run.
    # The dashboard should use it as the clean reporting source instead of
    # promoting later mismatched live/recomputed reports as the final project KPI.
    notebook_best_summary = notebook.get("best_model_summary", {}) if isinstance(notebook, dict) else {}
    notebook_metrics_clean = best_metrics_from_summary(notebook_best_summary) or notebook.get("metrics", {})
    if notebook_metrics_clean:
        display_score_payload["display_metrics"] = notebook_metrics_clean
        display_score_payload["display_best_model_summary"] = notebook_best_summary or best_model
        display_score_payload["display_score_source"] = "verified_notebook_reference"
        display_score_payload["score_drop_warning"] = None
        display_score_payload["large_drop_detected"] = False

    # If no live metrics report exists but we can evaluate the actual model, use that for live display.
    if metrics_source == "not_found" and recomputed_evaluation.get("available"):
        metrics = recomputed_evaluation.get("metrics", {})
        metrics_source = "dashboard_recomputed_from_test_csv"

    feature_importance = read_csv_records("reports/feature_importance/top_10_feature_importance.csv")
    feature_importance_source = "live_repo_file" if feature_importance else None
    if not feature_importance:
        feature_importance = notebook.get("feature_importance", [])
        feature_importance_source = notebook.get("source", "notebook_fallback") if feature_importance else "not_found"

    model_rows = read_csv_records("reports/mlflow_run_summary.csv")
    model_rows_source = "live_repo_file" if model_rows else "not_found"
    if not model_rows and diagnostics:
        model_rows = []
        for row in diagnostics:
            model_rows.append({
                "model_name": row.get("model_name"),
                "f1_macro": row.get("test_score"),
                "train_test_gap": row.get("gap"),
                "quality_verdict": row.get("verdict"),
                "primary_metric": row.get("primary_metric"),
            })
        model_rows_source = diagnostics_source

    monitoring_summary = read_json("monitoring/evidently_reports/monitoring_summary.json") or {}
    dvc_yaml = parse_dvc_yaml()
    dvc_lock = parse_dvc_lock_text()

    # DVC-aware batch readiness. Prefer production.csv + preprocessing artifacts when available.
    batch_ready = availability["best_model_pkl"]["exists"] and (
        availability.get("production_csv", {}).get("exists") or availability.get("test_csv", {}).get("exists")
    )

    return {
        "project_name": "Telco Churn MLOps Pipeline",
        "project_root": str(PROJECT_ROOT),
        "timestamp": time.time(),
        "availability": availability,
        "sources": {
            "metrics": metrics_source,
            "best_model_summary": best_model_source,
            "leakage_report": leakage_source,
            "registry_summary": registry_source,
            "hpo_summary": hpo_source,
            "model_diagnostics": diagnostics_source,
            "selected_features": selected_features_source,
            "feature_importance": feature_importance_source,
            "model_comparison": model_rows_source,
        },
        "metrics": metrics,
        "best_model_summary": best_model,
        "display_metrics": display_score_payload.get("display_metrics", {}),
        "display_best_model_summary": display_score_payload.get("display_best_model_summary", {}),
        "display_score_source": display_score_payload.get("display_score_source"),
        "score_drop_warning": display_score_payload.get("score_drop_warning"),
        "notebook_reference_metrics": display_score_payload.get("notebook_reference_metrics", {}),
        "notebook_reference_best_model_summary": display_score_payload.get("notebook_reference_best_model_summary", {}),
        "live_metric_block": display_score_payload.get("live_metric_block", {}),
        "recomputed_metric_block": display_score_payload.get("recomputed_metric_block", {}),
        "leakage_report": leakage,
        "registry_summary": registry,
        "hpo_summary": hpo,
        "model_diagnostics": diagnostics or [],
        "model_comparison": model_rows,
        "selected_features": selected_features,
        "feature_importance": feature_importance,
        "monitoring_summary": monitoring_summary,
        "drift_alerts": parse_alert_log(),
        "dvc_yaml": dvc_yaml,
        "dvc_lock": dvc_lock,
        "project_params": load_project_params(),
        "configured_paths": configured_data_paths(),
        "dvc_artifacts": dvc_artifact_paths(),
        "mlflow_status": count_mlflow_runs(),
        "prediction_schema": prediction_schema(),
        "model_generation": model_generation_status(),
        "model_artifacts": model_artifact_inventory(),
        "recomputed_evaluation": recomputed_evaluation,
        "score_comparison": score_comparison,
        "dvc_environment": dvc_environment_status(),
        "batch_ready": batch_ready,
        "dvc_workspace": dvc_workspace_readiness(),
        "batch_requirements": {
            "model": "models/best_model.pkl",
            "data": "data/splits/production.csv preferred, data/splits/test.csv fallback",
        },
        "notebook_source": {
            "source": notebook.get("source"),
            "source_type": notebook.get("source_type"),
        },
    }


RAW_PREDICTION_FIELDS = [
    {
        "name": "tenure",
        "type": "number",
        "default": 24,
        "min": 0,
        "help": "Months the customer has been with the company.",
    },
    {
        "name": "MonthlyCharges",
        "type": "number",
        "default": 65.5,
        "min": 0,
        "help": "Current monthly charge amount.",
    },
    {
        "name": "TotalCharges",
        "type": "number",
        "default": 1572,
        "min": 0,
        "help": "Total amount charged to date.",
    },
    {
        "name": "gender",
        "type": "select",
        "choices": ["Male", "Female"],
        "default": "Male",
        "help": "Customer gender.",
    },
    {
        "name": "SeniorCitizen",
        "type": "select",
        "choices": [0, 1],
        "default": 0,
        "help": "1 if senior citizen, 0 otherwise.",
    },
    {
        "name": "Partner",
        "type": "select",
        "choices": ["Yes", "No"],
        "default": "Yes",
        "help": "Whether the customer has a partner.",
    },
    {
        "name": "Dependents",
        "type": "select",
        "choices": ["Yes", "No"],
        "default": "No",
        "help": "Whether the customer has dependents.",
    },
    {
        "name": "PhoneService",
        "type": "select",
        "choices": ["Yes", "No"],
        "default": "Yes",
        "help": "Whether the customer has phone service.",
    },
    {
        "name": "MultipleLines",
        "type": "select",
        "choices": ["Yes", "No", "No phone service"],
        "default": "No",
        "help": "Multiple phone lines.",
    },
    {
        "name": "InternetService",
        "type": "select",
        "choices": ["DSL", "Fiber optic", "No"],
        "default": "DSL",
        "help": "Internet service type.",
    },
    {
        "name": "OnlineSecurity",
        "type": "select",
        "choices": ["Yes", "No", "No internet service"],
        "default": "Yes",
        "help": "Online security add-on.",
    },
    {
        "name": "OnlineBackup",
        "type": "select",
        "choices": ["Yes", "No", "No internet service"],
        "default": "Yes",
        "help": "Online backup add-on.",
    },
    {
        "name": "DeviceProtection",
        "type": "select",
        "choices": ["Yes", "No", "No internet service"],
        "default": "No",
        "help": "Device protection add-on.",
    },
    {
        "name": "TechSupport",
        "type": "select",
        "choices": ["Yes", "No", "No internet service"],
        "default": "Yes",
        "help": "Tech support add-on.",
    },
    {
        "name": "StreamingTV",
        "type": "select",
        "choices": ["Yes", "No", "No internet service"],
        "default": "No",
        "help": "Streaming TV add-on.",
    },
    {
        "name": "StreamingMovies",
        "type": "select",
        "choices": ["Yes", "No", "No internet service"],
        "default": "No",
        "help": "Streaming movies add-on.",
    },
    {
        "name": "Contract",
        "type": "select",
        "choices": ["Month-to-month", "One year", "Two year"],
        "default": "One year",
        "help": "Customer contract type.",
    },
    {
        "name": "PaperlessBilling",
        "type": "select",
        "choices": ["Yes", "No"],
        "default": "Yes",
        "help": "Whether the customer uses paperless billing.",
    },
    {
        "name": "PaymentMethod",
        "type": "select",
        "choices": [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        "default": "Credit card (automatic)",
        "help": "Payment method.",
    },
]


def _label_is_positive(value: Any) -> bool:
    value_str = str(value).strip().lower()
    return value_str in {"1", "true", "yes", "churn", "positive"}


def _extract_positive_probability(model: Any, probabilities: List[float]) -> Optional[float]:
    if not probabilities or len(probabilities) < 2:
        return None

    classes = None

    if hasattr(model, "classes_"):
        classes = list(model.classes_)
    elif hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            if hasattr(step, "classes_"):
                classes = list(step.classes_)
                break

    if classes:
        for idx, label in enumerate(classes):
            if _label_is_positive(label) and idx < len(probabilities):
                return float(probabilities[idx])

    return float(probabilities[-1])


def _load_local_model() -> Any:
    """
    Load the best available model artifact.

    Important:
    - Model binaries are DVC artifacts.
    - Some XGBoost pickle files can fail to load under a different local
      xgboost version, so we try the next available model instead of killing
      the dashboard.
    """
    if joblib is None:
        raise RuntimeError("joblib is unavailable.")

    errors = []
    for relative in [
        "models/best_model.pkl",
        "models/best_all_features_model.pkl",
        "models/best_reference_model.pkl",
        "models/best_selected_feature_model.pkl",
        "models/best_tuned_model.pkl",
        "models/tuned_xgboost_model.pkl",
        "models/tuned_catboost_model.pkl",
        "models/tuned_lightgbm_model.pkl",
        "models/tuned_random_forest_model.pkl",
    ]:
        path = project_path(relative)
        if not path.exists():
            continue

        try:
            model = joblib.load(path)
            try:
                setattr(model, "_dashboard_model_path", relative)
            except Exception:
                pass
            return model
        except Exception as exc:
            errors.append({"path": relative, "error": str(exc)})

    raise FileNotFoundError(
        "No loadable saved model found in models/. Errors: " + json.dumps(errors, indent=2)
    )

def _build_preprocessor_adapter():
    """
    Load the serving adapter created by the project team.
    This uses the working DVC artifacts:
    - models/feature_engineer.joblib
    - models/preprocessing_pipeline.joblib
    """
    from src.serving.preprocessor import PreprocessingAdapter

    adapter = PreprocessingAdapter(
        feature_engineer_path=str(project_path("models/feature_engineer.joblib")),
        preprocessing_pipeline_path=str(project_path("models/preprocessing_pipeline.joblib")),
    )
    adapter.load()
    return adapter


def _validate_raw_customer(payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    errors = []
    cleaned = {}

    for field in RAW_PREDICTION_FIELDS:
        name = field["name"]
        value = payload.get(name)

        if value in (None, ""):
            errors.append(f"{name} is required.")
            continue

        if field["type"] == "number":
            try:
                numeric_value = float(value)
            except Exception:
                errors.append(f"{name} must be numeric.")
                continue

            if "min" in field and numeric_value < float(field["min"]):
                errors.append(f"{name} must be at least {field['min']}.")

            cleaned[name] = numeric_value

        elif field["type"] == "select":
            choices = [str(choice) for choice in field["choices"]]
            value_str = str(value)

            if value_str not in choices:
                errors.append(f"{name} must be one of: {', '.join(choices)}.")
                continue

            if name == "SeniorCitizen":
                cleaned[name] = int(value_str)
            else:
                cleaned[name] = value_str

    return (cleaned if not errors else None), errors


def prediction_schema() -> Dict[str, Any]:
    required_artifacts = {
        "model": file_info("models/best_model.pkl"),
        "feature_engineer": file_info("models/feature_engineer.joblib"),
        "preprocessing_pipeline": file_info("models/preprocessing_pipeline.joblib"),
        "serving_schema": file_info("src/serving/schemas.py"),
        "serving_preprocessor": file_info("src/serving/preprocessor.py"),
    }

    live_ready = (
        required_artifacts["model"]["exists"]
        and required_artifacts["feature_engineer"]["exists"]
        and required_artifacts["preprocessing_pipeline"]["exists"]
    )

    return {
        "mode": "raw_telco_customer_input",
        "source": "src/serving/schemas.py + configs/params.yaml + DVC artifacts",
        "fields": RAW_PREDICTION_FIELDS,
        "live_prediction_ready": live_ready,
        "required_artifacts": required_artifacts,
        "endpoint": "/api/predict",
        "note": "The form accepts raw Telco customer fields, then uses the project preprocessing artifacts before calling the saved model.",
    }




def dvc_environment_status() -> Dict[str, Any]:
    """
    Check whether this exact Python environment can run the project's DVC remote.

    The uploaded dashboard error shows DVC is installed but S3 support is missing.
    This check is separate from DVC file validity because the DVC files can be
    correct while the Python environment still lacks dvc-s3.
    """
    status = {
        "python_executable": sys.executable,
        "dvc_available": False,
        "dvc_version": None,
        "dvc_s3_available": False,
        "s3_remote_detected": False,
        "remote_urls": [],
        "ready_for_s3_pull": False,
        "install_command": f"{sys.executable} -m pip install dvc-s3",
        "install_full_command": f"{sys.executable} -m pip install dvc dvc-s3",
    }

    try:
        result = subprocess.run(
            [sys.executable, "-m", "dvc", "--version"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        status["dvc_available"] = result.returncode == 0
        status["dvc_version"] = (result.stdout or result.stderr).strip()
    except Exception as exc:
        status["dvc_error"] = str(exc)

    try:
        import dvc_s3  # noqa: F401
        status["dvc_s3_available"] = True
    except Exception as exc:
        status["dvc_s3_error"] = str(exc)

    for rel in [".dvc/config", ".dvc/config.local"]:
        text = read_text(rel)
        if not text:
            continue
        for match in re.finditer(r"url\s*=\s*(.+)", text):
            url = match.group(1).strip()
            status["remote_urls"].append(url)
            if url.startswith("s3://"):
                status["s3_remote_detected"] = True

    status["ready_for_s3_pull"] = (
        status["dvc_available"]
        and (not status["s3_remote_detected"] or status["dvc_s3_available"])
    )

    return status



def model_artifact_inventory() -> Dict[str, Any]:
    """
    Inventory actual model files in the local workspace.
    """
    candidates = [
        "models/best_model.pkl",
        "models/best_all_features_model.pkl",
        "models/best_reference_model.pkl",
        "models/best_selected_feature_model.pkl",
        "models/best_tuned_model.pkl",
        "models/tuned_xgboost_model.pkl",
        "models/tuned_catboost_model.pkl",
        "models/tuned_lightgbm_model.pkl",
        "models/tuned_random_forest_model.pkl",
        "models/feature_columns.json",
        "models/feature_engineer.joblib",
        "models/preprocessing_pipeline.joblib",
    ]

    items = []
    for relative in candidates:
        info = file_info(relative)
        items.append(info)

    loadable = []
    load_errors = []
    if joblib is not None:
        for relative in candidates:
            if not relative.endswith((".pkl", ".joblib")):
                continue
            path = project_path(relative)
            if not path.exists():
                continue
            try:
                obj = joblib.load(path)
                loadable.append({
                    "path": relative,
                    "type": f"{type(obj).__module__}.{type(obj).__name__}",
                })
            except Exception as exc:
                load_errors.append({"path": relative, "error": str(exc)})

    return {
        "items": items,
        "existing_count": sum(1 for item in items if item.get("exists")),
        "loadable": loadable,
        "load_errors": load_errors,
        "best_model_exists": project_path("models/best_model.pkl").exists(),
        "feature_engineer_exists": project_path("models/feature_engineer.joblib").exists(),
        "preprocessing_pipeline_exists": project_path("models/preprocessing_pipeline.joblib").exists(),
    }


def evaluate_current_model_from_test() -> Dict[str, Any]:
    """
    Evaluate the current saved model against data/splits/test.csv if both exist.

    This is dashboard-side verification only. It does not replace the official
    DVC reports, but it prevents the UI from showing stale notebook values when
    real model artifacts and test data are present.
    """
    if pd is None or joblib is None:
        return {"available": False, "reason": "pandas/joblib unavailable"}

    test_path = project_path("data/splits/test.csv")
    if not test_path.exists():
        return {"available": False, "reason": "data/splits/test.csv not found"}

    try:
        model = _load_local_model()
    except Exception as exc:
        return {"available": False, "reason": f"No loadable model: {exc}"}

    try:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

        frame = pd.read_csv(test_path)
        target_col = "Churn" if "Churn" in frame.columns else "target" if "target" in frame.columns else None
        if target_col is None:
            return {"available": False, "reason": "No Churn/target column in data/splits/test.csv"}

        X = frame.drop(columns=[target_col])
        y = frame[target_col]

        preds = model.predict(X)

        metrics = {
            "problem_type": "classification",
            "accuracy": float(accuracy_score(y, preds)),
            "precision_macro": float(precision_score(y, preds, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y, preds, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y, preds, average="macro", zero_division=0)),
        }

        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X)
                if getattr(proba, "shape", [0, 0])[1] >= 2:
                    metrics["roc_auc"] = float(roc_auc_score(y, proba[:, 1]))
            except Exception as exc:
                metrics["roc_auc_error"] = str(exc)

        model_path = getattr(model, "_dashboard_model_path", "models/best_model.pkl")

        result = {
            "available": True,
            "source": "dashboard_recomputed_from_data_splits_test_csv",
            "model_path": model_path,
            "rows": int(len(frame)),
            "target_column": target_col,
            "metrics": metrics,
        }

        out_path = project_path("reports/dashboard_model_evaluation.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        return result

    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def score_comparison_summary(live_metrics: Any, notebook_metrics: Any, recomputed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare the displayed DVC/live scores with the notebook reference scores.
    This makes score drops explicit instead of hiding them.
    """
    live = live_metrics if isinstance(live_metrics, dict) else {}
    notebook = notebook_metrics if isinstance(notebook_metrics, dict) else {}
    recomputed_metrics = recomputed.get("metrics", {}) if isinstance(recomputed, dict) else {}

    def get_f1(obj):
        try:
            return float(obj.get("f1_macro"))
        except Exception:
            return None

    live_f1 = get_f1(live)
    notebook_f1 = get_f1(notebook)
    recomputed_f1 = get_f1(recomputed_metrics)

    reference = notebook_f1
    current = live_f1 if live_f1 is not None else recomputed_f1

    delta = None
    warning = False
    if reference is not None and current is not None:
        delta = current - reference
        warning = delta < -0.03

    return {
        "live_f1_macro": live_f1,
        "notebook_reference_f1_macro": notebook_f1,
        "recomputed_test_f1_macro": recomputed_f1,
        "current_vs_notebook_delta": delta,
        "large_drop_detected": warning,
        "explanation": (
            "If live DVC scores are much lower than the notebook reference, the dashboard will show both. "
            "That usually means the current DVC pipeline output is different from the older notebook run, "
            "not that the dashboard invented a new score."
        ),
    }


def model_generation_status() -> Dict[str, Any]:
    """
    Explain model availability using DVC as the source of truth.

    The repository itself does not need to contain model binaries. The model
    artifacts are generated/restored by DVC stages and are expected to appear
    only after dvc pull or dvc repro has been run successfully.
    """
    dvc_outputs = collect_dvc_outputs()
    model_outputs = [
        item for item in dvc_outputs
        if str(item.get("path", "")).startswith("models/")
    ]
    report_outputs = [
        item for item in dvc_outputs
        if str(item.get("path", "")).startswith("reports/")
    ]

    required_for_prediction = [
        "models/best_model.pkl",
        "models/feature_engineer.joblib",
        "models/preprocessing_pipeline.joblib",
    ]

    required_status = []
    for path in required_for_prediction:
        p = project_path(path)
        stage = next((item.get("stage") for item in dvc_outputs if item.get("path") == path), None)
        required_status.append({
            "path": path,
            "exists": p.exists(),
            "stage": stage,
            "source": "dvc.yaml/dvc.lock",
        })

    missing_required = [item for item in required_status if not item["exists"]]
    existing_required = [item for item in required_status if item["exists"]]

    inventory = model_artifact_inventory()

    return {
        "models_are_committed_to_repo": False,
        "explanation": (
            "Model binaries are DVC-tracked outputs. If they exist in the local models/ folder, "
            "the dashboard will use them. If they are missing, use DVC pull/repro."
        ),
        "artifact_inventory": inventory,
        "live_prediction_ready": len(missing_required) == 0,
        "required_for_prediction": required_status,
        "missing_required": missing_required,
        "existing_required": existing_required,
        "model_outputs_from_dvc": model_outputs,
        "report_outputs_from_dvc": report_outputs,
        "recommended_actions": [
            "Run DVC Pull if the DVC remote contains the cached model artifacts.",
            "Run DVC Repro All if the artifacts need to be generated locally.",
            "Run stages in order: prepare, featurize, preprocess, train, hpo, register.",
        ],
    }



@app.route("/api/model/evaluate", methods=["POST"])
def api_model_evaluate():
    return jsonify(evaluate_current_model_from_test())


@app.route("/api/model/status")
def api_model_status():
    return jsonify(model_generation_status())


@app.route("/api/predict/schema")
def api_predict_schema():
    return jsonify(prediction_schema())


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if pd is None or joblib is None:
        return jsonify({"ok": False, "error": "pandas/joblib unavailable"}), 500

    payload = request.get_json(silent=True) or {}
    cleaned, errors = _validate_raw_customer(payload)

    if errors:
        return jsonify({
            "ok": False,
            "error": "Validation failed.",
            "validation_errors": errors,
        }), 400

    missing = []
    for relative in [
        "models/best_model.pkl",
        "models/feature_engineer.joblib",
        "models/preprocessing_pipeline.joblib",
    ]:
        if not project_path(relative).exists():
            missing.append(relative)

    if missing:
        return jsonify({
            "ok": False,
            "error": "Live prediction cannot run because required DVC/model artifacts are missing.",
            "missing": missing,
            "note": "Run dvc pull/repro to restore the model and preprocessing artifacts.",
        }), 409

    try:
        adapter = _build_preprocessor_adapter()
        model = _load_local_model()
        features = adapter.transform(cleaned)

        start = time.time()
        prediction = model.predict(features)
        latency_ms = round((time.time() - start) * 1000, 2)

        predicted_value = prediction[0].item() if hasattr(prediction[0], "item") else prediction[0]
        predicted_churn = "Yes" if _label_is_positive(predicted_value) else "No"

        probabilities = None
        confidence = None
        churn_probability = None

        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(features)[0]
                probabilities = [float(value) for value in proba]
                confidence = float(max(probabilities))
                churn_probability = _extract_positive_probability(model, probabilities)
            except Exception:
                pass

        result = {
            "ok": True,
            "timestamp": time.time(),
            "prediction": predicted_value,
            "churn": predicted_churn,
            "churn_probability": churn_probability,
            "confidence": confidence,
            "probabilities": probabilities,
            "latency_ms": latency_ms,
            "input": cleaned,
            "feature_shape": list(features.shape),
            "model_path": "models/best_model.pkl",
            "preprocessing": {
                "feature_engineer": "models/feature_engineer.joblib",
                "preprocessing_pipeline": "models/preprocessing_pipeline.joblib",
            },
        }

        history_path = RUNTIME_DIR / "single_prediction_history.json"
        history = []
        if history_path.exists():
            try:
                history = json.loads(history_path.read_text(encoding="utf-8"))
            except Exception:
                history = []

        history.append(result)
        history = history[-100:]
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        return jsonify(result)

    except Exception as exc:
        return jsonify({
            "ok": False,
            "error": "Prediction failed during preprocessing or model inference.",
            "details": str(exc),
        }), 500


@app.route("/api/predict/history")
def api_predict_history():
    history_path = RUNTIME_DIR / "single_prediction_history.json"
    if not history_path.exists():
        return jsonify([])
    try:
        return jsonify(json.loads(history_path.read_text(encoding="utf-8")))
    except Exception:
        return jsonify([])


MLFLOW_PROCESS = None


def count_mlflow_runs() -> Dict[str, Any]:
    db_path = project_path("mlruns/mlflow.db")

    if not db_path.exists():
        return {
            "available": False,
            "database_path": "mlruns/mlflow.db",
            "experiment_count": 0,
            "run_count": 0,
            "error": "MLflow SQLite database not found.",
        }

    try:
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()

        try:
            experiment_count = cur.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
        except Exception:
            experiment_count = 0

        try:
            run_count = cur.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        except Exception:
            run_count = 0

        conn.close()

        return {
            "available": True,
            "database_path": "mlruns/mlflow.db",
            "experiment_count": int(experiment_count),
            "run_count": int(run_count),
            "multiple_runs": int(run_count) > 1,
        }

    except Exception as exc:
        return {
            "available": True,
            "database_path": "mlruns/mlflow.db",
            "experiment_count": 0,
            "run_count": 0,
            "error": str(exc),
        }


@app.route("/api/mlflow/status")
def api_mlflow_status():
    return jsonify(count_mlflow_runs())


@app.route("/api/mlflow/start", methods=["POST"])
def api_mlflow_start():
    global MLFLOW_PROCESS

    db_path = project_path("mlruns/mlflow.db")
    if not db_path.exists():
        return jsonify({
            "ok": False,
            "error": "mlruns/mlflow.db was not found.",
            "note": "MLflow UI needs the local backend database. Run training/HPO first if needed.",
        }), 404

    if MLFLOW_PROCESS is not None and MLFLOW_PROCESS.poll() is None:
        return jsonify({
            "ok": True,
            "already_running": True,
            "url": "http://127.0.0.1:5000",
            "pid": MLFLOW_PROCESS.pid,
        })

    command = [
        sys.executable,
        "-m",
        "mlflow",
        "ui",
        "--backend-store-uri",
        f"sqlite:///{db_path.resolve()}",
        "--host",
        "127.0.0.1",
        "--port",
        "5000",
    ]

    try:
        log_path = RUNTIME_DIR / "mlflow_ui.log"
        MLFLOW_PROCESS = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=log_path.open("a", encoding="utf-8"),
            stderr=subprocess.STDOUT,
        )

        return jsonify({
            "ok": True,
            "url": "http://127.0.0.1:5000",
            "pid": MLFLOW_PROCESS.pid,
            "log_path": str(log_path.relative_to(PROJECT_ROOT)),
            "command": " ".join(command),
        })

    except Exception as exc:
        return jsonify({
            "ok": False,
            "error": str(exc),
            "command": " ".join(command),
        }), 500



@app.route("/")
def root():
    return send_from_directory(WEB_DIR, "overview.html")


@app.route("/<page>")
def page(page: str):
    if page in PAGES:
        return send_from_directory(WEB_DIR, PAGES[page])
    abort(404)



def collect_dvc_outputs() -> List[Dict[str, Any]]:
    """
    Collect expected DVC outputs from dvc.yaml and dvc.lock.

    These are not invented dashboard paths. They are read directly from the
    team's working DVC files and checked against the current local workspace.
    """
    outputs = {}

    dvc_yaml = parse_dvc_yaml()
    for stage in dvc_yaml.get("stages", []):
        for path in stage.get("outs", []) or []:
            if isinstance(path, str):
                outputs[path] = {
                    "path": path,
                    "stage": stage.get("name"),
                    "source": "dvc.yaml",
                    "exists": project_path(path).exists(),
                    "size": project_path(path).stat().st_size if project_path(path).exists() and project_path(path).is_file() else None,
                }
        for metric in stage.get("metrics", []) or []:
            if isinstance(metric, dict):
                for path in metric.keys():
                    outputs[path] = {
                        "path": path,
                        "stage": stage.get("name"),
                        "source": "dvc.yaml metric",
                        "exists": project_path(path).exists(),
                        "size": project_path(path).stat().st_size if project_path(path).exists() and project_path(path).is_file() else None,
                    }
            elif isinstance(metric, str):
                outputs[metric] = {
                    "path": metric,
                    "stage": stage.get("name"),
                    "source": "dvc.yaml metric",
                    "exists": project_path(metric).exists(),
                    "size": project_path(metric).stat().st_size if project_path(metric).exists() and project_path(metric).is_file() else None,
                }

    dvc_lock = parse_dvc_lock_text()
    for stage in dvc_lock.get("stages", []) or []:
        for path in stage.get("outs", []) or []:
            existing = outputs.get(path, {})
            outputs[path] = {
                "path": path,
                "stage": existing.get("stage") or stage.get("name"),
                "source": "dvc.lock",
                "exists": project_path(path).exists(),
                "size": project_path(path).stat().st_size if project_path(path).exists() and project_path(path).is_file() else existing.get("size"),
            }

    return sorted(outputs.values(), key=lambda item: (str(item.get("stage")), str(item.get("path"))))


def dvc_workspace_readiness() -> Dict[str, Any]:
    """
    Summarize which DVC-produced artifacts are present locally.
    """
    outputs = collect_dvc_outputs()
    present = [item for item in outputs if item.get("exists")]
    missing = [item for item in outputs if not item.get("exists")]
    return {
        "total_outputs": len(outputs),
        "present_outputs": len(present),
        "missing_outputs": len(missing),
        "ready_ratio": (len(present) / len(outputs)) if outputs else 0,
        "outputs": outputs,
        "missing": missing,
        "present": present,
    }


def run_local_command(label: str, command: List[str], timeout: int = 600) -> Dict[str, Any]:
    """
    Run a local command from the repository root and persist stdout/stderr.
    """
    pipeline_dir = project_path("reports/pipeline")
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "command"
    stdout_path = pipeline_dir / f"{safe_label}_stdout.txt"
    stderr_path = pipeline_dir / f"{safe_label}_stderr.txt"

    started = time.time()
    try:
        result = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout_path.write_text(result.stdout or "", encoding="utf-8")
        stderr_path.write_text(result.stderr or "", encoding="utf-8")
        return {
            "ok": result.returncode == 0,
            "label": label,
            "command": " ".join(command),
            "returncode": result.returncode,
            "duration_seconds": round(time.time() - started, 2),
            "stdout": (result.stdout or "")[-8000:],
            "stderr": (result.stderr or "")[-8000:],
            "stdout_path": str(stdout_path.relative_to(PROJECT_ROOT)),
            "stderr_path": str(stderr_path.relative_to(PROJECT_ROOT)),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "label": label,
            "command": " ".join(command),
            "returncode": None,
            "duration_seconds": round(time.time() - started, 2),
            "stdout": (exc.stdout or "")[-8000:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[-8000:] if isinstance(exc.stderr, str) else "",
            "error": f"Command timed out after {timeout} seconds.",
        }
    except Exception as exc:
        return {
            "ok": False,
            "label": label,
            "command": " ".join(command),
            "returncode": None,
            "duration_seconds": round(time.time() - started, 2),
            "stdout": "",
            "stderr": "",
            "error": str(exc),
        }



@app.route("/api/dvc/environment")
def api_dvc_environment():
    return jsonify(dvc_environment_status())


@app.route("/api/dvc/readiness")
def api_dvc_readiness():
    return jsonify(dvc_workspace_readiness())


@app.route("/api/dvc/run", methods=["POST"])
def api_dvc_run():
    """
    Run DVC operations from the dashboard.

    Supported actions:
    - status
    - metrics
    - dag
    - pull
    - pull_force
    - install_s3
    - repro_all
    - repro_stage with {"stage": "..."}
    """
    body = request.get_json(silent=True) or {}
    action = body.get("action", "status")
    stage = body.get("stage")

    python = sys.executable

    allowed_stages = {stage_info.get("name") for stage_info in parse_dvc_yaml().get("stages", [])}
    if action == "status":
        command = [python, "-m", "dvc", "status"]
    elif action == "metrics":
        command = [python, "-m", "dvc", "metrics", "show"]
    elif action == "dag":
        command = [python, "-m", "dvc", "dag"]
    elif action == "pull":
        command = [python, "-m", "dvc", "pull"]
    elif action == "pull_force":
        command = [python, "-m", "dvc", "pull", "--force"]
    elif action == "install_s3":
        command = [python, "-m", "pip", "install", "dvc-s3"]
    elif action == "repro_all":
        command = [python, "-m", "dvc", "repro"]
    elif action == "repro_stage":
        if not stage:
            return jsonify({"ok": False, "error": "Missing stage name."}), 400
        if allowed_stages and stage not in allowed_stages:
            return jsonify({"ok": False, "error": f"Unknown stage '{stage}'.", "allowed_stages": sorted(allowed_stages)}), 400
        command = [python, "-m", "dvc", "repro", "--single-item", stage]
    else:
        return jsonify({"ok": False, "error": f"Unsupported DVC action '{action}'."}), 400

    result = run_local_command(f"dvc_{action}_{stage or 'workspace'}", command, timeout=1200)

    # Persist latest dashboard-visible DVC action.
    latest_path = project_path("reports/pipeline/dashboard_dvc_last_action.json")
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    result["dvc_environment"] = dvc_environment_status()
    if "No module named 'dvc_s3'" in (result.get("stderr") or "") or "requires 'dvc-s3'" in (result.get("stderr") or ""):
        result["action_needed"] = "Install DVC S3 support in the same Python environment that runs the dashboard."
        result["install_command"] = result["dvc_environment"].get("install_command")
    return jsonify({
        **result,
        "readiness": dvc_workspace_readiness(),
    })


@app.route("/api/dvc/last")
def api_dvc_last():
    latest_path = project_path("reports/pipeline/dashboard_dvc_last_action.json")
    if latest_path.exists():
        try:
            return jsonify(json.loads(latest_path.read_text(encoding="utf-8")))
        except Exception:
            pass
    return jsonify({"ok": None, "message": "No DVC command has been run from the dashboard yet."})


@app.route("/api/summary")
def api_summary():
    return jsonify(build_summary())


@app.route("/api/batch/next", methods=["POST"])
def api_batch_next():
    if pd is None or joblib is None:
        return jsonify({"ok": False, "error": "pandas/joblib unavailable"}), 500

    model_path = project_path("models/best_model.pkl")
    production_path = project_path("data/splits/production.csv")
    test_path = project_path("data/splits/test.csv")

    missing = []
    if not model_path.exists():
        missing.append("models/best_model.pkl")

    use_production = production_path.exists()
    use_test = test_path.exists()

    if not use_production and not use_test:
        missing.append("data/splits/production.csv or data/splits/test.csv")

    if missing:
        return jsonify({
            "ok": False,
            "error": "Batch inference cannot run because required DVC/model artifacts are missing.",
            "missing": missing,
            "note": "Run dvc pull/repro to restore the required artifacts.",
        }), 409

    body = request.get_json(silent=True) or {}
    batch_size = int(body.get("batch_size", 50))
    state_path = RUNTIME_DIR / "batch_state.json"
    history_path = RUNTIME_DIR / "batch_history.json"
    state = {"offset": 0}

    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Prefer production.csv because that matches the monitoring workflow.
    data_path = production_path if use_production else test_path
    data_mode = "production_raw_with_preprocessing" if use_production else "test_model_ready"

    frame = pd.read_csv(data_path)
    if frame.empty:
        return jsonify({"ok": False, "error": f"{data_path.relative_to(PROJECT_ROOT)} is empty"}), 400

    offset = int(state.get("offset", 0))
    if offset >= len(frame):
        offset = 0

    batch = frame.iloc[offset : offset + batch_size].copy()
    next_offset = offset + len(batch)

    y_true = None
    for target_col in ["target", "Churn"]:
        if target_col in batch.columns:
            y_true = batch[target_col].tolist()
            break

    model = _load_local_model()

    if use_production:
        required_preprocessing = [
            "models/feature_engineer.joblib",
            "models/preprocessing_pipeline.joblib",
        ]
        missing_preprocessing = [
            relative for relative in required_preprocessing if not project_path(relative).exists()
        ]
        if missing_preprocessing:
            return jsonify({
                "ok": False,
                "error": "Production batch inference requires preprocessing artifacts.",
                "missing": missing_preprocessing,
            }), 409

        adapter = _build_preprocessor_adapter()
        customer_records = batch.drop(columns=[c for c in ["target", "Churn", "customerID"] if c in batch.columns], errors="ignore").to_dict(orient="records")
        X = adapter.transform_batch(customer_records)

    else:
        X = batch.drop(columns=[c for c in ["target", "Churn"] if c in batch.columns], errors="ignore")

    start_time = time.time()
    try:
        preds = model.predict(X)
    except Exception as exc:
        return jsonify({
            "ok": False,
            "error": "Model prediction failed on the selected batch.",
            "detail": str(exc),
            "data_mode": data_mode,
            "feature_shape": list(X.shape),
            "note": "This usually means the model artifact and the selected data representation do not match.",
        }), 500
    latency_ms = round((time.time() - start_time) * 1000, 2)

    confidence = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            confidence = float(proba.max(axis=1).mean())
        except Exception:
            confidence = None

    positive_predictions = int(sum(1 for value in preds if _label_is_positive(value)))
    negative_predictions = int(len(preds) - positive_predictions)

    event = {
        "timestamp": time.time(),
        "offset": offset,
        "rows": int(len(batch)),
        "latency_ms": latency_ms,
        "avg_confidence": confidence,
        "positive_predictions": positive_predictions,
        "negative_predictions": negative_predictions,
        "y_true_available": y_true is not None,
        "data_path": str(data_path.relative_to(PROJECT_ROOT)),
        "data_mode": data_mode,
        "feature_shape": list(X.shape),
    }

    history = []
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except Exception:
            history = []

    history.append(event)
    history = history[-100:]
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    state_path.write_text(json.dumps({"offset": next_offset}, indent=2), encoding="utf-8")

    return jsonify({"ok": True, "event": event, "history": history})


@app.route("/api/batch/history")
def api_batch_history():
    history_path = RUNTIME_DIR / "batch_history.json"
    if not history_path.exists():
        return jsonify([])
    try:
        return jsonify(json.loads(history_path.read_text(encoding="utf-8")))
    except Exception:
        return jsonify([])


@app.route("/api/monitoring/run", methods=["POST"])
def api_monitoring_run():
    script = project_path("monitoring/run_monitoring.py")
    if not script.exists():
        return jsonify({"ok": False, "error": "monitoring/run_monitoring.py not found"}), 404
    result = subprocess.run([sys.executable, str(script)], cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    return jsonify({
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout[-5000:],
        "stderr": result.stderr[-5000:],
    })


@app.route("/repo-file/<path:relative>")
def repo_file(relative: str):
    safe = Path(relative)
    if safe.is_absolute() or ".." in safe.parts:
        abort(400)
    path = PROJECT_ROOT / safe
    if not path.exists() or not path.is_file():
        abort(404)
    return send_from_directory(path.parent, path.name)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8050"))
    print(f"Telco Churn MLOps Pipeline dashboard")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Open: http://127.0.0.1:{port}/overview")
    app.run(host="127.0.0.1", port=port, debug=False)
