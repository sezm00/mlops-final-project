# 🚀 MLOps Final Project – Production Documentation

This repository implements an end-to-end MLOps pipeline for a Telco Customer Churn prediction system, including data versioning, model training, experiment tracking, deployment, monitoring, and a dashboard interface.

---

# 📌 Project Overview

This project demonstrates a production-grade MLOps workflow including:

* Data versioning with DVC
* Model training and evaluation pipeline
* Experiment tracking with MLflow
* FastAPI model serving
* Monitoring pipeline
* Interactive dashboard for analytics

---

# 👥 Team Roles

* **Shahd** — Data Engineering & Preprocessing
* **Ibrahim** — Deployment, API & Monitoring
* **Islam** — Model Training & Evaluation

---

# 🧰 Tech Stack

* Python 3.10+
* FastAPI
* MLflow
* DVC
* Pandas / NumPy
* Scikit-learn
* Uvicorn
* Dash (Dashboard UI)

---

# ⚙️ Setup Instructions

## 1. Clone Repository

```bash
git clone <repo-url>
cd mlops-final-project
```

---

## 2. Create Virtual Environment

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
```

---

## 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements-dev.txt
```

---

## 4. Pull Data (DVC)

```bash
dvc pull
```

---

# 📊 Running the Dashboard

## Step 1: Start Dashboard Server

```bash
python dashboard/server.py
```

## Step 2: Open Browser

```
http://127.0.0.1:8050/overview
```

---

# 📈 MLflow Tracking Server

Start MLflow in a separate terminal:

```bash
mlflow server \
  --backend-store-uri "sqlite:///mlruns/mlflow.db" \
  --default-artifact-root "mlruns/artifacts" \
  --host 127.0.0.1 \
  --port 5001
```

MLflow UI:

```
http://127.0.0.1:5001
```

---

# 🚀 API Deployment (FastAPI)

## Set Environment Variables

### macOS / Linux

```bash
export MLFLOW_TRACKING_URI="http://127.0.0.1:5001"
export PYTHONPATH=$PWD
```

### Windows (PowerShell)

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5001"
$env:PYTHONPATH = $PWD
```

## Run API Server

```bash
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload
```

API Docs:

```
http://127.0.0.1:8000/docs
```

---

# 📁 Project Structure

```
mlops-final-project/
│
├── data/                  # Raw & processed datasets (DVC tracked)
├── src/                   # Core ML pipeline
│   ├── training/          # Model training scripts
│   ├── preprocessing/     # Data cleaning & feature engineering
│   ├── serving/           # FastAPI application
│
├── monitoring/            # Monitoring pipeline
├── dashboard/             # Dash analytics dashboard
├── mlruns/                # MLflow tracking data
├── requirements-dev.txt   # Dependencies
└── dvc.yaml               # DVC pipeline definition
```

---

# 🔁 MLOps Pipeline Flow

1. Data ingestion (DVC)
2. Preprocessing & feature engineering
3. Model training
4. Experiment tracking (MLflow)
5. Model evaluation
6. Deployment via FastAPI
7. Monitoring pipeline
8. Dashboard visualization

---

# 📌 Contributing Guidelines

We follow a strict Git workflow to ensure clean collaboration.

## Rule

> 1 Issue = 1 Branch = 1 Pull Request

## Workflow

```bash
git checkout develop
git pull origin develop
git checkout -b feature/<task>-<issueID>
```

Then:

* Commit changes
* Push branch
* Open Pull Request to `develop`

---

# ⚠️ Common Issues

## 1. FileNotFoundError (data paths)

Run from project root:

```bash
cd mlops-final-project
```

## 2. DVC missing files

```bash
dvc pull
```

## 3. API cannot connect to MLflow

Ensure:

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5001
```

---

# 🎯 Production Notes

* Always run services in separate terminals
* Use virtual environment for isolation
* Ensure MLflow is running before API server
* Use DVC for reproducible data pipelines

---

# 🧠 Future Improvements

* Dockerization of full pipeline
* CI/CD automation with GitHub Actions
* Cloud deployment (AWS/GCP)
* Model drift detection

