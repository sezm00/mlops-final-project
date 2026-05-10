# Data Card — Telco Customer Churn


> **Pipeline config:** `configs/params.yaml`


---

## 1. Dataset Identity

| Field | Value |
|---|---|
| **Name** | Telco Customer Churn |
| **Provider** | IBM Sample Data Sets (distributed via Kaggle) |
| **Kaggle URL** | https://www.kaggle.com/datasets/blastchar/telco-customer-churn |
| **File name** | `WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| **Format** | CSV, single flat table |
| **Rows** | 7 043 |
| **Columns (raw)** | 21 |
| **Task** | Binary classification — predict customer churn (`Churn` ∈ {Yes, No}) |
| **Domain** | Telecommunications / Customer Retention |

---

## 2. Source & Provenance

The dataset is a fictional IBM sample dataset created to illustrate customer retention analytics. It does not represent any real telecommunications company or real individual customers. It has been publicly available on Kaggle since at least 2018 under the file name above.

Because the data is synthetic/illustrative, no data-sharing agreement, data processing agreement, or IRB approval is required for academic and research use. However, the dataset's structure closely mirrors real telco data, and all modelling decisions should be validated against actual operational data before deployment.

---

## 3. Raw Schema

Each row represents one customer account. Columns in the original CSV:

| Column | Type | Description |
|---|---|---|
| `customerID` | string | Unique customer identifier — **dropped** in Stage 1 |
| `gender` | string | `Female` / `Male` |
| `SeniorCitizen` | int | `0` = not senior, `1` = senior (≥ 65) |
| `Partner` | string | `Yes` / `No` — has a partner |
| `Dependents` | string | `Yes` / `No` — has dependents |
| `tenure` | int | Months with the company (0 – 72) |
| `PhoneService` | string | `Yes` / `No` |
| `MultipleLines` | string | `Yes` / `No` / `No phone service` |
| `InternetService` | string | `DSL` / `Fiber optic` / `No` |
| `OnlineSecurity` | string | `Yes` / `No` / `No internet service` |
| `OnlineBackup` | string | `Yes` / `No` / `No internet service` |
| `DeviceProtection` | string | `Yes` / `No` / `No internet service` |
| `TechSupport` | string | `Yes` / `No` / `No internet service` |
| `StreamingTV` | string | `Yes` / `No` / `No internet service` |
| `StreamingMovies` | string | `Yes` / `No` / `No internet service` |
| `Contract` | string | `Month-to-month` / `One year` / `Two year` |
| `PaperlessBilling` | string | `Yes` / `No` |
| `PaymentMethod` | string | Four values: Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic) |
| `MonthlyCharges` | float | Current monthly billing amount (USD) |
| `TotalCharges` | object* | Cumulative charges — stored as string due to whitespace entries |
| `Churn` | string | **Target** — `Yes` / `No` |

*`TotalCharges` is stored as `object` in the raw CSV because 11 rows contain whitespace (`" "`) instead of a numeric value. These are all rows where `tenure == 0` (new customers who were never billed).

---

## 4. Preprocessing Decisions

All preprocessing is configuration-driven via `configs/params.yaml`. No values are hardcoded in source files.

### Stage 1 — `prepare.py`: Cleaning & Split

| Decision | Rationale |
|---|---|
| Drop `customerID` | Unique per row; no predictive signal; would cause data leakage if retained as a feature |
| Convert `TotalCharges` to float via `pd.to_numeric(errors='coerce')` | Raw CSV stores it as string; coercion converts whitespace to `NaN` |
| Set `TotalCharges = 0.0` where `tenure == 0` | Factual correction: customers who joined in the current billing cycle have `$0` cumulative charges by definition |
| Remaining `TotalCharges` NaN left as-is | Handled safely by the sklearn `SimpleImputer` (median strategy) in Stage 4 to avoid leakage |
| Replace `"No internet service"` → `"No"` in six service columns | Reduces one-hot encoding cardinality from 3 to 2 for these columns; semantically equivalent |
| Replace `"No phone service"` → `"No"` in `MultipleLines` | Same rationale as above |
| **Positional 70/30 split** (reference / production) | Row order approximates customer acquisition order. A positional split simulates temporal holdout: train on older customers, monitor on newer arrivals. This is intentional for drift simulation |
| Encode `Churn`: `Yes → 1`, `No → 0` before split | Both sets share the same encoding; no fitting step needed |

### Stage 2 — `featurize.py`: Feature Engineering

All row-level features below are computed statelessly (no dataset statistics). The single stateful feature (`IsHighValueCustomer`) is fitted **only on the reference set** and applied to all subsequent sets.

| Feature | Formula | Type |
|---|---|---|
| `AvgMonthlyCharges` | `TotalCharges / tenure` (= `MonthlyCharges` when `tenure == 0`) | float — row-level |
| `ChargeDeviation` | `MonthlyCharges − AvgMonthlyCharges` | float — row-level |
| `TotalCharges_log` | `log1p(TotalCharges)` | float — row-level |
| `NumServices` | Count of `Yes` across 7 optional service columns | int — row-level |
| `IsNewCustomer` | `1` if `tenure ≤ 12` else `0` | binary — row-level |
| `IsRiskySegment` | `1` if `Contract == "Month-to-month"` AND `IsNewCustomer == 1` | binary — row-level |
| `IsHighValueCustomer` | `1` if `MonthlyCharges ≥ 75th percentile` of reference set | binary — **stateful** |

The `high_value_threshold` is computed from the reference set and stored inside the serialised `FeatureEngineer` artifact (`models/feature_engineer.joblib`) to prevent train–serve skew.


### Stage 3 — `preprocess.py`: Encoding, Scaling & SMOTE

The entire preprocessing logic is encapsulated in a single serialisable sklearn `Pipeline` object saved to `models/preprocessing_pipeline.joblib`.

| Step | Transformer | Applied to |
|---|---|---|
| Median imputation | `SimpleImputer(strategy='median')` | Numeric columns |
| Standardisation | `StandardScaler` | Numeric columns |
| One-hot encoding (drop first) | `OneHotEncoder(drop='first')` | Multi-class categoricals (`Contract`, `PaymentMethod`, `InternetService`, …) |
| Binary encoding (drop if binary) | `OneHotEncoder(drop='if_binary')` | Binary categoricals (`gender`, `Partner`, `Dependents`, …) |
| Class balancing | `SMOTENC` | Training set only |

**SMOTE note:** `SMOTENC` (not vanilla SMOTE) is used because the training set contains both numeric and categorical columns after encoding. The categorical column indices are detected automatically from the feature names produced by the `ColumnTransformer`.

**Leakage guarantee:** The sklearn `Pipeline` is fitted **exclusively on the training portion of the reference set** (`X_train`). The test set and production set are only transformed, never used in fitting.

---

## 5. Dataset Splits

| Split | Rows (approx.) | Purpose |
|---|---|---|
| Reference (70% of raw) | ~4 930 | Training + evaluation; source of all statistics and fitted transformers |
| Production (30% of raw) | ~2 113 | Held out to simulate real-world scoring and drift monitoring |
| Train (80% of reference) | ~3 944 | Fitted sklearn pipeline; SMOTE applied here |
| Test (20% of reference) | ~986 | Evaluation only; no SMOTE |

The reference/production split is **positional** (no random seed). The train/test split uses `random_state=42` and `stratify=y` to preserve the class ratio.

---

## 6. Class Distribution

| Label | Count (full dataset) | Percentage |
|---|---|---|
| `Churn = 0` (retained) | ~5 174 | 73.5% |
| `Churn = 1` (churned) | ~1 869 | 26.5% |

The dataset is **moderately imbalanced**. SMOTE is applied to the training set to address this. The test set retains the natural class distribution for unbiased evaluation. The production set similarly retains the natural distribution for realistic drift monitoring.

---

## 7. Known Biases & Limitations

### Class imbalance
The churn rate is approximately 26.5%, making `Churn = 0` the majority class. Without SMOTE, a naive classifier predicting "no churn" always would achieve 73.5% accuracy, misleading stakeholders. All evaluation should use AUC-ROC, F1, and precision-recall curves rather than raw accuracy.

### Gender representation
`gender` is binary (`Female` / `Male`) with no accommodation for non-binary identities. Any fairness evaluation using gender as a protected attribute is limited to this binary framing.

### SeniorCitizen definition
`SeniorCitizen` is a binary flag with no documented age threshold. The assumed convention (≥ 65) is not confirmed by the dataset documentation. Fairness analyses using this attribute should note this ambiguity.

### No timestamp column
The dataset has no explicit timestamp. The positional split assumes row order approximates acquisition order, which is not verified. The resulting "temporal" split is a simulation, not a true temporal holdout.

### Synthetic / illustrative origin
The dataset was created by IBM for educational purposes. Its distributions may not reflect the real-world telco market, and models trained on it should not be deployed without validation against real operational data.

### Geographic and cultural context
No geographic information is provided. Usage patterns and churn drivers vary significantly by market. The dataset should be treated as US-centric unless stated otherwise.

### Proxy discrimination risk
`Contract` type, `PaymentMethod`, and `tenure` are correlated with income and financial stability. A model using these features may produce disparate outcomes for lower-income customers even if `SeniorCitizen` and `gender` are excluded.

---

## 8. Privacy & Sensitive Data

| Consideration | Status |
|---|---|
| Personally Identifiable Information (PII) | **None** — `customerID` is a synthetic key with no real-world mapping; it is dropped in Stage 1 |
| Special category data (GDPR Art. 9) | None present |
| Financial data | `MonthlyCharges` and `TotalCharges` are aggregate billing figures, not transaction records |
| GDPR applicability | Not applicable — dataset is synthetic; no real data subjects |
| CCPA applicability | Not applicable — same rationale |
| Data minimisation | `customerID` is explicitly dropped; no other columns are removed because all are predictive features or the target |

---

## 9. Licensing

| Field | Details |
|---|---|
| **License** | [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/) (as listed on Kaggle) |
| **Original source** | IBM Sample Data Sets — publicly released for educational use |
| **Commercial use** | Permitted under CC0 |
| **Attribution required** | Not required under CC0, but recommended: *"Telco Customer Churn dataset, IBM Sample Data, distributed via Kaggle (blastchar/telco-customer-churn)"* |
| **Redistribution** | Permitted |
| **Modification** | Permitted — all derived artifacts in this pipeline are covered by the project's own license |

---

## 10. Intended Use & Out-of-Scope Uses

**Intended use:** Academic MLOps pipeline development and experimentation. Demonstrating end-to-end pipeline capabilities including versioning, preprocessing, drift monitoring, and fairness testing.

**Out-of-scope:**
- Production deployment for real customer churn decisions without retraining on proprietary operational data.
- Legal, financial, or credit decisions of any kind.
- Inferences about real individuals.

---

## 11. Artifact Inventory

| Artifact | Path | DVC tracked | Description |
|---|---|---|---|
| Raw data | `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` | ✅ | Unmodified source file |
| Cleaned reference | `data/processed/cleaned_reference.csv` | ✅ | Post Stage 1: cleaned, target-encoded, no features |
| Featurized reference | `data/processed/reference.csv` | ✅ | Post Stage 2: engineered features added |
| Production set | `data/splits/production.csv` | ✅ | Held-out 30%; raw format for monitoring |
| Train split | `data/splits/train.csv` | ✅ | Encoded, scaled, SMOTE-balanced |
| Test split | `data/splits/test.csv` | ✅ | Encoded, scaled; no SMOTE |
| Preprocessing pipeline | `models/preprocessing_pipeline.joblib` | ✅ | Serialised sklearn Pipeline; fitted on train only |
| Feature engineer | `models/feature_engineer.joblib` | ✅ | Serialised FeatureEngineer; stores reference-set statistics |
