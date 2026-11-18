
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="SentinelGuard Fraud Alerts",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ SentinelGuard")
st.subheader("Banking Fraud Intelligence & Analyst Dashboard")

# 1. LOAD DATASET
DATA_PATH = "dataset/creditcard.csv"
MODEL_PATH = "models/xgboost_best_model.pkl"
THRESHOLD_PATH = "models/optimal_threshold.pkl"

if not os.path.exists(DATA_PATH):
    st.error("Dataset not found: " + DATA_PATH)
    st.stop()

if not os.path.exists(MODEL_PATH):
    st.error("XGBoost model not found. Run xgboost_tuning.py first.")
    st.stop()

df = pd.read_csv(DATA_PATH)

if "Class" not in df.columns:
    st.error("Class column is missing.")
    st.stop()

# 2. LOAD MODEL
saved = joblib.load(MODEL_PATH)

model = saved["model"]
features = saved["features"]

if not all(feature in df.columns for feature in features):
    st.error("Dataset features do not match the saved model.")
    st.stop()

# 3. GET OPTIMAL THRESHOLD
threshold = 0.50

if os.path.exists(THRESHOLD_PATH):
    threshold_data = joblib.load(THRESHOLD_PATH)
    threshold = float(threshold_data["threshold"])

# 4. PREDICT ALL TRANSACTIONS
X = df[features].apply(
    pd.to_numeric,
    errors="coerce"
).fillna(0)

with st.spinner("Analyzing all transactions..."):
    probabilities = model.predict_proba(X)[:, 1]

results = df.copy()

results["Fraud_Probability"] = probabilities
results["Risk_Score"] = (probabilities * 100).round(2)

results["Prediction"] = (
    probabilities >= threshold
).astype(int)

results["Risk_Level"] = results["Risk_Score"].apply(
    lambda score:
        "HIGH" if score >= 70
        else "MEDIUM" if score >= 30
        else "LOW"
)

results["Alert_Status"] = results["Prediction"].apply(
    lambda value:
        "🚨 FRAUD ALERT" if value == 1
        else "✅ NORMAL"
)

# 5. SIDEBAR FILTERS
st.sidebar.header("Analyst Filters")

risk_filter = st.sidebar.selectbox(
    "Risk Level",
    ["ALL", "HIGH", "MEDIUM", "LOW"]
)

status_filter = st.sidebar.selectbox(
    "Transaction Status",
    ["ALL", "FRAUD ALERT", "NORMAL"]
)

filtered = results.copy()

if risk_filter != "ALL":
    filtered = filtered[
        filtered["Risk_Level"] == risk_filter
    ]

if status_filter == "FRAUD ALERT":
    filtered = filtered[
        filtered["Prediction"] == 1
    ]

elif status_filter == "NORMAL":
    filtered = filtered[
        filtered["Prediction"] == 0
    ]

# 6. DASHBOARD METRICS
total = len(results)
alerts = int(results["Prediction"].sum())
normal = total - alerts

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Transactions", f"{total:,}")
c2.metric("Fraud Alerts", f"{alerts:,}")
c3.metric("Normal Transactions", f"{normal:,}")
c4.metric("Alert Rate", f"{alerts / total * 100:.2f}%" if total else "0%")

st.caption(f"Model decision threshold: {threshold:.2f}")

# 7. ALERT SUMMARY
st.divider()
st.subheader("🚨 Fraud Alert Summary")

if alerts > 0:
    st.warning(
        f"{alerts:,} transactions were flagged by the model."
    )
else:
    st.success("No transactions were flagged by the model.")

# 8. TRANSACTION TABLE
st.subheader("Transaction Monitoring")

display_columns = [
    column for column in [
        "Time",
        "Amount",
        "Fraud_Probability",
        "Risk_Score",
        "Risk_Level",
        "Alert_Status",
        "Class"
    ]
    if column in filtered.columns
]

st.dataframe(
    filtered[display_columns],
    use_container_width=True,
    height=450
)

# 9. DOWNLOAD ALERTS
st.subheader("Download Reports")

alerts_df = results[
    results["Prediction"] == 1
]

csv_data = alerts_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Fraud Alerts CSV",
    data=csv_data,
    file_name="sentinelguard_fraud_alerts.csv",
    mime="text/csv"
)

all_data = results.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download All Transaction Results",
    data=all_data,
    file_name="sentinelguard_all_transactions.csv",
    mime="text/csv"
)

# 10. ACTUAL FRAUD COMPARISON
st.divider()
st.subheader("Actual Fraud vs Model Alerts")

if "Class" in results.columns:
    actual_fraud = int(results["Class"].sum())
    st.write("Actual fraud transactions in dataset:", actual_fraud)
    st.write("Model-flagged transactions:", alerts)

    st.caption(
        "Class is the known dataset label. "
        "Prediction is the model's output. "
        "They are not the same."
    )

st.success("SentinelGuard analyst dashboard loaded.")
