# Telco Churn MLOps Pipeline Dashboard

This dashboard is built for the exact repository layout you provided. It does not require `production.csv`, `production_processed.csv`, or `reference.csv`.

Place the `dashboard/` folder directly inside the repository root:

```text
mlops-final-project-copy/
  configs/
  data/
  monitoring/
  notebooks/
  src/
  tests/
  dvc.yaml
  dvc.lock
  dashboard/
```

Run it locally:

```bash
cd "/Users/ibrahimlabib/Documents/GitHub/mlops-final-project-copy"
python3 -m pip install -r dashboard/requirements-dashboard.txt
python3 dashboard/server.py
```

Open:

```text
http://127.0.0.1:8050/overview
```

## What it reads from the exact repo

- `monitoring/evidently_reports/monitoring_summary.json`
- `monitoring/evidently_reports/drift_alerts.log`
- `monitoring/evidently_reports/baseline_report.html`
- `monitoring/evidently_reports/drift_report.html`
- `dvc.yaml`
- `dvc.lock`
- `configs/params.yaml`
- `src/training/train.py`
- `src/training/hpo.py`
- `src/training/register_model.py`
- `src/evaluation/evaluate.py`
- `src/evaluation/diagnostics.py`
- Optional generated outputs if they exist:
  - `reports/metrics.json`
  - `reports/best_model_summary.json`
  - `reports/mlflow_run_summary.csv`
  - `reports/model_registry_summary.json`
  - `reports/feature_importance/*.csv/json`
  - `models/best_model.pkl`

## Notebook result fallback

The repository zip did not contain generated `reports/` and `models/` folders. To still show the results from the training notebook, this dashboard includes a fallback copy of the extracted notebook outputs in:

```text
dashboard/data/notebook_results.json
```

If you later generate real `reports/` and `models/` locally, the dashboard will automatically prefer the live files over the fallback notebook results.

## Batch monitoring

The dashboard can run a scoring batch only when these two files exist:

```text
models/best_model.pkl
data/splits/test.csv
```

If they are missing, the batch page will show exactly which file is missing instead of pretending predictions were run.


## DVC-aware dashboard update

This dashboard now reads the working DVC setup directly:

- `configs/params.yaml`
- `dvc.yaml`
- `dvc.lock`

It also recognizes the current project outputs:

- `data/processed/cleaned_reference.csv`
- `data/processed/featurized_reference.csv`
- `data/splits/production.csv`
- `data/splits/train.csv`
- `data/splits/test.csv`
- `models/feature_engineer.joblib`
- `models/preprocessing_pipeline.joblib`
- `models/best_model.pkl`
- `reports/...`

The `/predict` page accepts raw Telco customer input and uses the same preprocessing artifacts and saved model used by the project pipeline.
