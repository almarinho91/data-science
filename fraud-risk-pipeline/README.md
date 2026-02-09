# Fraud Risk Pipeline (DuckDB + scikit-learn + FastAPI)

End-to-end fraud scoring project built on a public European credit card transactions dataset.

The goal is to demonstrate a realistic data science + data engineering workflow:
raw ingestion → feature engineering → model training/evaluation → online risk scoring API.

---

## What’s inside

- **raw layer**: `raw.transactions` (original CSV + lineage)
- **staging layer**: `stg.transactions` (cleaned data + `log_amount`)
- **features layer**: `features.transactions_features` (behavioral features like rolling stats and z-scores)
- **model**: Logistic Regression with class weighting, evaluated using PR-AUC and Precision@K
- **API**: FastAPI service exposing `POST /score`

Fraud is extremely rare (~0.17%), so the project focuses on metrics that matter in practice:
ROC-AUC, PR-AUC, and Precision@K instead of accuracy.

---

## Example result

Using a baseline model:

- ROC-AUC ≈ 0.98  
- PR-AUC ≈ 0.74  

At an operating point of 100 alerts:
- Precision ≈ 59%
- Recall ≈ 79%

Meaning: by reviewing only the top 100 riskiest transactions, the system catches most fraud with a manageable number of false positives.

---

## How to run

### 1) Setup

```bash
pip install -r requirements.txt
python -m ingestion.init_db
python -m ingestion.download_data
python -m ingestion.ingest_raw
```

### 2) Build staging + features

```bash
python -m ingestion.run_sql
```

### 3) Train model + save artifacts

```bash
python -m model.baseline
```

This creates:

- `models/fraud_model.joblib`
- `models/feature_cols.json`

### 4) Run API

```bash
uvicorn api.main:app --reload
```

Open:

- http://127.0.0.1:8000/docs  
- GET `/health`  
- POST `/score`  

### 5) Demo request

Generate a demo payload:

```bash
python -m ingestion.make_demo_payload
```

Send it to the API:

```bash
curl -X POST http://127.0.0.1:8000/score -H "Content-Type: application/json" -d @exports/demo_payload.json
```

---

## Project notes

- Dataset is highly imbalanced, which reflects real-world fraud.
- It shoudbe included in a 'data' folder inside the project: Luqi Liu. (2022). Credit Card Fraud Detection [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7395559
- Evaluation uses PR-AUC and Precision@K instead of accuracy.
- Feature engineering includes behavioral deviation (`amount_zscore_24h`), which emerged as the strongest signal.
- Velocity features are global due to missing customer/device identifiers; in production these would be computed per user or device.

---
