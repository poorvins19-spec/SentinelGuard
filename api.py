
import os
import joblib
import pandas as pd

from typing import Dict, Any
from fastapi import FastAPI, HTTPException

# =====================================
# 1. CREATE FASTAPI APPLICATION
# =====================================

app = FastAPI(
    title="SentinelGuard API",
    description="Banking Fraud Detection and Risk Prediction API",
    version="1.0"
)

MODEL_PATH = "models/fraud_model.pkl"

# =====================================
# 2. LOAD TRAINED MODEL
# =====================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model not found. Run compare_models.py first."
    )

saved = joblib.load(MODEL_PATH)

model = saved["model"]
features = saved["features"]
model_name = saved.get("model_name", "Saved ML Model")

# =====================================
# 3. HOME ENDPOINT
# =====================================

@app.get("/")
def home():

    return {
        "project": "SentinelGuard",
        "message": "Banking Fraud Detection API is running",
        "model": model_name,
        "status": "Active"
    }

# =====================================
# 4. HEALTH CHECK
# =====================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True
    }

# =====================================
# 5. FRAUD PREDICTION ENDPOINT
# =====================================

@app.post("/predict")
def predict_transaction(data: Dict[str, Any]):

    try:

        # Check required features
        missing = [
            feature for feature in features
            if feature not in data
        ]

        if missing:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Missing transaction features",
                    "missing_features": missing
                }
            )

        # Prepare transaction
        transaction = pd.DataFrame(
            [[data[feature] for feature in features]],
            columns=features
        )

        # Convert values to numeric
        transaction = transaction.apply(
            pd.to_numeric,
            errors="coerce"
        )

        if transaction.isnull().any().any():
            raise HTTPException(
                status_code=400,
                detail="All transaction features must be valid numbers."
            )

        # Generate prediction
        prediction = int(model.predict(transaction)[0])

        probability = float(
            model.predict_proba(transaction)[0][1]
        )

        risk_score = probability * 100

        # Risk classification
        if risk_score >= 50:
            risk_level = "HIGH"
            alert = "HIGH RISK - Immediate review recommended"

        elif risk_score >= 20:
            risk_level = "MEDIUM"
            alert = "MEDIUM RISK - Manual review recommended"

        else:
            risk_level = "LOW"
            alert = "LOW RISK"

        return {
            "project": "SentinelGuard",
            "model": model_name,
            "prediction": (
                "FRAUD" if prediction == 1 else "LEGITIMATE"
            ),
            "fraud_probability": round(probability * 100, 2),
            "risk_score": round(risk_score, 2),
            "risk_level": risk_level,
            "alert": alert
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
