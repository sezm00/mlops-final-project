# Model Card — Telco Churn MLOps Pipeline

## Model Overview

**Project:** Telco Churn MLOps Pipeline  
**Dataset:** IBM Telco Customer Churn  
**Task:** Binary classification  
**Target:** `Churn`  
**Final Model:** XGBoost Classifier  
**Registered Model Name:** `BestMLOpsModel`  
**Primary Metric:** F1 Macro  
**Model Stage:** Production  

This model predicts whether a telecom customer is likely to churn based on customer profile, account information, service subscriptions, contract type, billing method, tenure, and charge-related features.

---

## Intended Use

The model is intended for churn prediction and customer retention analysis in an MLOps demonstration pipeline.

It can be used to:

- Identify customers with higher churn risk
- Support retention campaign planning
- Demonstrate end-to-end MLOps practices
- Serve predictions through an API or dashboard interface
- Monitor production data drift over time

This model is intended for academic and demonstration purposes. It should not be used for real business decisions without additional validation on live company data.

---

## Dataset

The project uses the IBM Telco Customer Churn dataset.

| Item | Description |
|---|---|
| Rows | 7,043 customers |
| Target | `Churn` |
| Problem Type | Binary classification |
| Positive Class | Customer churned |
| Class Balance | Approximately 26.5% churned, 73.5% not churned |

The raw dataset includes demographic data, account details, phone and internet services, contract type, payment method, tenure, monthly charges, total charges, and churn label.

---

## Data Splitting

The dataset is split positionally to simulate a real-world timeline:

| Split | Purpose |
|---|---|
| Reference Set | Historical data used for fitting and training |
| Production Set | Newer data used for drift monitoring |
| Train/Test Split | Created from the reference data |

The production set is not used for model fitting. It is used only for monitoring and drift detection.

---

## Preprocessing

Preprocessing is handled through a serialized scikit-learn pipeline saved as:

`models/preprocessing_pipeline.joblib`

Main preprocessing steps:

- Missing value handling
- Median imputation for numeric columns
- Standard scaling for numeric columns
- One-hot encoding for categorical columns
- Binary encoding for binary columns
- SMOTENC applied only to the training split to handle class imbalance

---

## Feature Engineering

Feature engineering is performed before preprocessing using a custom transformer saved as:

`models/feature_engineer.joblib`

Engineered features include:

- `AvgMonthlyCharges`
- `ChargeDeviation`
- `TotalCharges_log`
- `NumServices`
- `IsNewCustomer`
- `IsRiskySegment`
- `IsHighValueCustomer`

Feature engineering and preprocessing parameters are controlled through:

`configs/params.yaml`

---

## Training Pipeline

The project uses DVC to manage the end-to-end pipeline.

Main DVC stages:

1. `prepare`
2. `featurize`
3. `preprocess`
4. `train`
5. `hpo`
6. `register`

Important commands:

- `python3 -m dvc pull`
- `python3 -m dvc repro`
- `python3 -m dvc status`
- `python3 -m dvc metrics show`

DVC is used to track data, model artifacts, reports, and pipeline dependencies.

---

## Models Evaluated

The following model families were tested:

- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Classifier
- XGBoost
- LightGBM
- CatBoost

The final selected model is the XGBoost all-features model.

---

## Final Model Performance

| Metric | Value |
|---|---:|
| Accuracy | 0.950 |
| Precision Macro | 0.950 |
| Recall Macro | 0.950 |
| F1 Macro | 0.950 |
| ROC-AUC | 0.989 |
| Data Leakage Check | PASS |
| Diagnostic Verdict | GOOD |

The primary selection metric was F1 Macro because the dataset is moderately imbalanced.

---

## Model Selection Decision

XGBoost was selected because it achieved the strongest overall test performance.

Feature importance analysis was performed, but the selected-feature model did not outperform the all-features model. Therefore, the all-features XGBoost model was retained as the final model.

---

## Experiment Tracking

MLflow is used for experiment tracking.

MLflow tracks:

- Parameters
- Metrics
- Model artifacts
- Evaluation reports
- Feature importance artifacts
- Registered model metadata

Experiment name:

`mlops_training_experiments`

Backend store:

`mlruns/mlflow.db`

---

## Model Registry

The final model is registered in MLflow as:

`BestMLOpsModel`

The registered model is promoted to:

`Production`

The registry provides a formal record of the selected model version used for serving and dashboard prediction.

---

## Monitoring

Monitoring is implemented using Evidently AI.

The monitoring pipeline compares reference data against production data to detect drift.

Production drift results:

| Item | Result |
|---|---:|
| Dataset Drift Detected | Yes |
| Drifted Features | 12 of 19 |
| Drift Share | 63.2% |
| Alert Triggered | Yes |
| Strongest Drift Feature | `MonthlyCharges` |

Main drifted features include:

- `MonthlyCharges`
- `tenure`
- `Contract`
- `InternetService`
- `StreamingTV`
- `TechSupport`
- `StreamingMovies`
- `OnlineBackup`
- `OnlineSecurity`
- `DeviceProtection`
- `TotalCharges`
- `MultipleLines`

The detected drift means the production cohort differs from the reference cohort. In a real deployment, this would trigger investigation or retraining.

---

## Prediction Interface

The dashboard includes a prediction form that allows a user to input customer information and receive a churn prediction.

The dashboard also provides:

- Model performance KPIs
- Model comparison results
- Feature importance analysis
- DVC pipeline status
- MLflow registry information
- Evidently drift reports
- Monitoring outputs

The dashboard should load saved artifacts for inference, not retrain the model during prediction.

---

## Model Output

The model outputs a binary churn prediction.

| Output | Meaning |
|---|---|
| `0` | Customer is not predicted to churn |
| `1` | Customer is predicted to churn |

The prediction interface may also return churn probability and confidence when available.

---

## Limitations

Known limitations:

- The model was trained on a public academic dataset, not live telecom company data.
- Results are valid only for the verified training and preprocessing setup.
- Performance may change if DVC regenerates different data splits or feature representations.
- Production drift was detected, so retraining may be needed in a real-world deployment.
- The model should support human decision-making, not replace it.
- Additional validation, security, access control, and monitoring are required before real production use.

---

## Ethical Considerations

Potential risks:

- Predictions may reflect historical bias in the dataset.
- Customers should not be treated unfairly based only on model output.
- The model should not be used for fully automated decisions without human review.
- Customer data should be handled securely and privately.

Recommended safeguards:

- Use predictions as decision support only.
- Monitor performance across different customer groups.
- Add access control before production deployment.
- Log predictions for auditability.
- Retrain and validate regularly using updated data.

---

## Reproducibility

The project supports reproducibility through:

- DVC pipeline stages
- Version-controlled data artifacts
- Central configuration in `params.yaml`
- MLflow experiment tracking
- Saved preprocessing pipeline
- Saved feature engineering transformer
- Saved model artifacts

---

## Summary

The final model is an XGBoost all-features classifier registered as `BestMLOpsModel`. It achieved strong verified performance with F1 Macro of 0.950 and ROC-AUC of 0.989. The project demonstrates a complete MLOps workflow using DVC for reproducibility, MLflow for experiment tracking and registry, Evidently for drift monitoring, and a dashboard for model visibility and prediction.