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
}

EXACT_PATHS = {
    "params": "configs/params.yaml",
    "dvc_yaml": "dvc.yaml",
    "dvc_lock": "dvc.lock",
    "raw_dvc_pointer": "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv.dvc",
    "raw_csv": "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
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


def availability_map() -> Dict[str, Dict[str, Any]]:
    return {key: file_info(relative) for key, relative in EXACT_PATHS.items()}


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

    # Exact repo reality: if no data/model file exists, batch inference should be disabled.
    batch_ready = availability["best_model_pkl"]["exists"] and availability["test_csv"]["exists"]

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
        "batch_ready": batch_ready,
        "batch_requirements": {
            "model": "models/best_model.pkl",
            "data": "data/splits/test.csv",
        },
        "notebook_source": {
            "source": notebook.get("source"),
            "source_type": notebook.get("source_type"),
        },
    }


@app.route("/")
def root():
    return send_from_directory(WEB_DIR, "overview.html")


@app.route("/<page>")
def page(page: str):
    if page in PAGES:
        return send_from_directory(WEB_DIR, PAGES[page])
    abort(404)


@app.route("/api/summary")
def api_summary():
    return jsonify(build_summary())


@app.route("/api/batch/next", methods=["POST"])
def api_batch_next():
    if pd is None or joblib is None:
        return jsonify({"ok": False, "error": "pandas/joblib unavailable"}), 500

    model_path = project_path("models/best_model.pkl")
    data_path = project_path("data/splits/test.csv")
    missing = []
    if not model_path.exists():
        missing.append("models/best_model.pkl")
    if not data_path.exists():
        missing.append("data/splits/test.csv")
    if missing:
        return jsonify({
            "ok": False,
            "error": "Batch inference cannot run because required files are missing.",
            "missing": missing,
            "note": "This dashboard follows the exact repo. It will not invent production.csv or production_processed.csv.",
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

    frame = pd.read_csv(data_path)
    if frame.empty:
        return jsonify({"ok": False, "error": "data/splits/test.csv is empty"}), 400

    offset = int(state.get("offset", 0))
    if offset >= len(frame):
        offset = 0
    batch = frame.iloc[offset : offset + batch_size].copy()
    next_offset = offset + len(batch)

    X = batch.drop(columns=[c for c in ["target", "Churn"] if c in batch.columns], errors="ignore")
    y_true = None
    for target_col in ["target", "Churn"]:
        if target_col in batch.columns:
            y_true = batch[target_col].tolist()
            break

    model = joblib.load(model_path)
    start = time.time()
    preds = model.predict(X)
    latency_ms = round((time.time() - start) * 1000, 2)
    confidence = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            confidence = float(proba.max(axis=1).mean())
        except Exception:
            confidence = None

    event = {
        "timestamp": time.time(),
        "offset": offset,
        "rows": int(len(batch)),
        "latency_ms": latency_ms,
        "avg_confidence": confidence,
        "positive_predictions": int(sum(int(x) for x in preds)) if len(preds) else 0,
        "negative_predictions": int(len(preds) - sum(int(x) for x in preds)) if len(preds) else 0,
        "y_true_available": y_true is not None,
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
