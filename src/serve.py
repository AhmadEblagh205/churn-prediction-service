"""FastAPI inference service for the calibrated churn decision model.

Run locally:
    uvicorn serve:app --app-dir src --port 8000
Then:
    curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
         -d @examples/customer.json
"""
import os
from pathlib import Path

import joblib
import pandas as pd

from explain import explain_prediction
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.environ.get("MODEL_PATH",
                                 ROOT / "models" / "churn-2.0.0.joblib"))
ARTIFACT = joblib.load(MODEL_PATH)

app = FastAPI(title="Churn prediction service",
              version=ARTIFACT["version"])


class Customer(BaseModel):
    gender: str
    SeniorCitizen: str = Field(description="Yes or No")
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0, le=120)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(gt=0)
    TotalCharges: float = Field(ge=0)


def _check_categories(c: Customer):
    for col, allowed in ARTIFACT["allowed_values"].items():
        value = getattr(c, col)
        if value not in allowed:
            raise HTTPException(
                status_code=422,
                detail=f"{col}={value!r} not in known categories {allowed}")


@app.get("/health")
def health():
    return {"status": "ok", "model": ARTIFACT["model_name"],
            "version": ARTIFACT["version"],
            "trained": ARTIFACT["trained"]}


@app.post("/predict_batch")
def predict_batch(customers: list[Customer]):
    """Score up to 500 customers with the same validation as /predict."""
    if len(customers) > 500:
        raise HTTPException(status_code=422,
                            detail="batch limited to 500 customers")
    results = [predict(c) for c in customers]
    flagged = sum(r["decision"] != "no action" for r in results)
    return {"n": len(results), "flagged": flagged,
            "flag_rate": round(flagged / max(len(results), 1), 3),
            "results": results}


@app.post("/predict")
def predict(customer: Customer):
    _check_categories(customer)
    row = pd.DataFrame([customer.model_dump()])
    explanation = explain_prediction(ARTIFACT["pipeline"], row)
    proba = explanation["calibrated_probability"]
    threshold = float(ARTIFACT["threshold"])
    return {
        "churn_probability": round(proba, 4),
        "raw_logistic_probability": round(explanation["raw_probability"], 4),
        "threshold": round(threshold, 4),
        "threshold_policy": ARTIFACT.get("threshold_policy", "configured threshold"),
        "decision": ("target retention offer" if proba >= threshold
                     else "no action"),
        # Backward-compatible positive reasons plus explicit protective factors.
        "top_factors": explanation["raising_factors"],
        "protective_factors": explanation["lowering_factors"],
        "explanation_basis": explanation["basis"],
        "model_version": ARTIFACT["version"],
    }
