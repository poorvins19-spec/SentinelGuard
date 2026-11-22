
import requests
import pandas as pd
import joblib

API_URL = "http://127.0.0.1:8000/predict"

def predict_transaction(transaction):

    saved = joblib.load("models/fraud_model.pkl")
    features = saved["features"]

    payload = {
        feature: float(transaction[feature])
        for feature in features
    }

    try:
        response = requests.post(
            API_URL,
            json=payload,
            timeout=30
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.RequestException as error:

        return {
            "error": str(error)
        }
