
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import wcwidth

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="SentinelGuard",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ SentinelGuard")
st.subheader("Banking Fraud Detection & Intelligence System")

st.markdown("---")

# ---------------- FILE PATHS ----------------

DATA_PATH = "dataset/creditcard.csv"
MODEL_PATH = "models/fraud_model.pkl"

# ---------------- LOAD DATASET ----------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

if not os.path.exists(DATA_PATH):
    st.error("CSV dataset not found. Check dataset/creditcard.csv")
    st.stop()

df = load_data()

# ---------------- LOAD MODEL ----------------

if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found. Run your model training code first.")
    st.stop()

saved_model = joblib.load(MODEL_PATH)

# Support either a saved dictionary or a direct model
if isinstance(saved_model, dict):
    model = saved_model.get("model")
    features = saved_model.get("features")
else:
    model = saved_model
    features = None

if model is None:
    st.error("Model could not be loaded.")
    st.stop()

# ---------------- IDENTIFY TARGET ----------------

target_candidates = ["Class", "class", "is_fraud", "fraud"]
target = next(
    (col for col in target_candidates if col in df.columns),
    None
)

if target is None:
    st.error("Fraud target column not found in CSV.")
    st.stop()

df = df.copy()
df = df.dropna(subset=[target])

# ---------------- PREPARE FEATURES ----------------

if features is None:
    features = [c for c in df.columns if c != target]

missing = [c for c in features if c not in df.columns]

if missing:
    st.error(f"Dataset is missing model features: {missing}")
    st.stop()

X = df[features].apply(pd.to_numeric, errors="coerce")

# Fill missing values
X = X.fillna(0)

# ---------------- MODEL PREDICTIONS ----------------

@st.cache_data
def generate_predictions(_model, data):
    predictions = _model.predict(data)

    if hasattr(_model, "predict_proba"):
        probabilities = _model.predict_proba(data)

        # Find probability column for fraud label 1
        classes = list(_model.classes_)
        fraud_index = classes.index(1) if 1 in classes else -1

        if fraud_index >= 0:
            fraud_probability = probabilities[:, fraud_index]
        else:
            fraud_probability = np.zeros(len(data))
    else:
        fraud_probability = np.zeros(len(data))

    return predictions, fraud_probability

try:
    predictions, probabilities = generate_predictions(model, X)
except Exception as e:
    st.error(f"Prediction error: {e}")
    st.stop()

# ---------------- RESULT DATAFRAME ----------------

result = df.copy()

result["ML_Prediction"] = predictions
result["Fraud_Probability"] = probabilities * 100

result["Risk_Level"] = np.select(
    [
        result["Fraud_Probability"] >= 50,
        result["Fraud_Probability"] >= 20
    ],
    [
        "HIGH",
        "MEDIUM"
    ],
    default="LOW"
)

result["Actual_Status"] = np.where(
    pd.to_numeric(result[target], errors="coerce") == 1,
    "FRAUD",
    "LEGITIMATE"
)

result["Predicted_Status"] = np.where(
    result["ML_Prediction"] == 1,
    "FRAUD",
    "LEGITIMATE"
)

# ---------------- DASHBOARD METRICS ----------------

total = len(result)
actual_fraud = (result[target] == 1).sum()
legitimate = total - actual_fraud
predicted_fraud = (result["ML_Prediction"] == 1).sum()

st.markdown("### 📊 Transaction Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Transactions", f"{total:,}")
c2.metric("Actual Fraud", f"{actual_fraud:,}")
c3.metric("Legitimate", f"{legitimate:,}")
c4.metric("ML Predicted Fraud", f"{predicted_fraud:,}")

st.markdown("---")

# ---------------- CHARTS ----------------

left, right = st.columns(2)

with left:
    st.markdown("### Actual Transaction Distribution")

    actual_counts = result["Actual_Status"].value_counts()
    st.bar_chart(actual_counts)

with right:
    st.markdown("### ML Prediction Distribution")

    prediction_counts = result["Predicted_Status"].value_counts()
    st.bar_chart(prediction_counts)

st.markdown("---")

# ---------------- RISK DISTRIBUTION ----------------

st.markdown("### 🚨 Risk Level Distribution")

risk_counts = result["Risk_Level"].value_counts()

st.bar_chart(risk_counts)

# ---------------- FILTERS ----------------

st.markdown("---")
st.markdown("### 🔍 Search and Filter Transactions")

f1, f2, f3 = st.columns(3)

with f1:
    actual_filter = st.selectbox(
        "Actual Status",
        ["ALL", "FRAUD", "LEGITIMATE"]
    )

with f2:
    prediction_filter = st.selectbox(
        "ML Prediction",
        ["ALL", "FRAUD", "LEGITIMATE"]
    )

with f3:
    risk_filter = st.selectbox(
        "Risk Level",
        ["ALL", "HIGH", "MEDIUM", "LOW"]
    )

filtered = result.copy()

if actual_filter != "ALL":
    filtered = filtered[
        filtered["Actual_Status"] == actual_filter
    ]

if prediction_filter != "ALL":
    filtered = filtered[
        filtered["Predicted_Status"] == prediction_filter
    ]

if risk_filter != "ALL":
    filtered = filtered[
        filtered["Risk_Level"] == risk_filter
    ]

st.markdown(f"**Transactions displayed: {len(filtered):,}**")

# ---------------- TRANSACTION TABLE ----------------

display_columns = (
    ["Actual_Status", "Predicted_Status",
     "Fraud_Probability", "Risk_Level"]
    + features
)

st.dataframe(
    filtered[display_columns],
    width="stretch",
    height=400
)

# ---------------- DOWNLOAD REPORT ----------------

st.markdown("---")
st.markdown("### 📥 Download Fraud Detection Report")

csv_data = filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download CSV Report",
    data=csv_data,
    file_name="sentinelguard_fraud_report.csv",
    mime="text/csv"
)

st.success("SentinelGuard dashboard loaded successfully!")

# ---------------- CONFUSION MATRIX ----------------

st.markdown("---")
st.header("📊 Model Evaluation - Confusion Matrix")

# Actual labels and ML predictions
actual = pd.to_numeric(result[target], errors="coerce").astype(int)
predicted = pd.to_numeric(
    result["ML_Prediction"], errors="coerce"
).astype(int)

cm = confusion_matrix(
    actual,
    predicted,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

# Display matrix
st.subheader("Confusion Matrix")

fig, ax = plt.subplots(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Legitimate", "Fraud"],
    yticklabels=["Legitimate", "Fraud"],
    ax=ax
)

ax.set_xlabel("Predicted Status")
ax.set_ylabel("Actual Status")
ax.set_title("SentinelGuard Confusion Matrix")

st.pyplot(fig)
plt.close(fig)

# Display values
st.subheader("Confusion Matrix Explanation")

a, b = st.columns(2)

a.metric("True Negative (TN)", f"{tn:,}")
a.metric("False Positive (FP)", f"{fp:,}")

b.metric("False Negative (FN)", f"{fn:,}")
b.metric("True Positive (TP)", f"{tp:,}")

# Explanation
st.markdown("### What do these values mean?")

st.write(
    f"✅ **True Negative:** {tn:,} legitimate transactions "
    "correctly identified as legitimate."
)

st.write(
    f"⚠️ **False Positive:** {fp:,} legitimate transactions "
    "incorrectly flagged as fraud."
)

st.write(
    f"❌ **False Negative:** {fn:,} fraudulent transactions "
    "incorrectly identified as legitimate."
)

st.write(
    f"✅ **True Positive:** {tp:,} fraudulent transactions "
    "correctly detected as fraud."
)

# Calculate metrics
accuracy = (tn + tp) / max(tn + fp + fn + tp, 1)
precision = tp / max(tp + fp, 1)
recall = tp / max(tp + fn, 1)
f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall > 0 else 0
)

st.markdown("### Model Performance")

m1, m2, m3, m4 = st.columns(4)

m1.metric("Accuracy", f"{accuracy * 100:.2f}%")
m2.metric("Precision", f"{precision * 100:.2f}%")
m3.metric("Recall", f"{recall * 100:.2f}%")
m4.metric("F1 Score", f"{f1 * 100:.2f}%")
