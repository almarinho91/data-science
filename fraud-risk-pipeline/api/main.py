from pathlib import Path
import json
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = Path("models/fraud_model.joblib")

app = FastAPI(title="Fraud Risk API")

model = joblib.load(MODEL_PATH)

FEATURE_COLS = json.loads((Path("models/feature_cols.json")).read_text(encoding="utf-8"))


class Transaction(BaseModel):
    Time: float
    Amount: float
    log_amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    tx_count_1h: float
    tx_count_24h: float
    amount_zscore_24h: float
    is_high_amount: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score")
def score(tx: Transaction):
    row = pd.DataFrame([tx.dict()])[FEATURE_COLS]
    prob = model.predict_proba(row)[0, 1]

    risk = "low"
    if prob > 0.9:
        risk = "high"
    elif prob > 0.5:
        risk = "medium"

    return {
        "fraud_probability": float(prob),
        "risk_level": risk,
    }
