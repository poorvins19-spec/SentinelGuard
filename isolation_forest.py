
import os
import time
import joblib
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

print("\n====================================")
print(" SENTINELGUARD - ISOLATION FOREST")
print("====================================")

# 1. LOAD DATASET
df = pd.read_csv("dataset/creditcard.csv")

if "Class" not in df.columns:
    raise ValueError("Class column not found.")

df = df.dropna(subset=["Class"]).copy()

X = df.drop(columns=["Class"])
y = df["Class"].astype(int)

X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

print("\nDataset shape:", df.shape)
print("\nActual class distribution:")
print(y.value_counts())

# 2. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 3. TRAIN ISOLATION FOREST
# This model learns normal patterns without using fraud labels.
model = IsolationForest(
    n_estimators=100,
    contamination=0.002,
    random_state=42,
    n_jobs=2
)

print("\nTraining Isolation Forest...")

start = time.perf_counter()

model.fit(X_train)

training_time = time.perf_counter() - start

# 4. PREDICT ANOMALIES
start = time.perf_counter()

raw_predictions = model.predict(X_test)

prediction_time = time.perf_counter() - start

# Isolation Forest:
# 1 = normal
# -1 = anomaly

# Convert to:
# 0 = normal
# 1 = anomaly

predictions = (raw_predictions == -1).astype(int)

# 5. EVALUATION
tn, fp, fn, tp = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
).ravel()

precision = precision_score(
    y_test, predictions, zero_division=0
)

recall = recall_score(
    y_test, predictions, zero_division=0
)

f1 = f1_score(
    y_test, predictions, zero_division=0
)

fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

print("\n========== RESULTS ==========")

print("Precision:", round(precision * 100, 2), "%")
print("Recall:", round(recall * 100, 2), "%")
print("F1 Score:", round(f1 * 100, 2), "%")
print("False Positive Rate:", round(fpr * 100, 2), "%")

print("\nConfusion Matrix:")
print("True Negatives :", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives :", tp)

print("\nAnomaly predictions:")
print(pd.Series(predictions).value_counts().rename({
    0: "Normal",
    1: "Anomaly"
}))

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Normal", "Fraud"],
    zero_division=0
))

print("\nTraining time:", round(training_time, 2), "seconds")
print("Prediction time:", round(prediction_time, 4), "seconds")

# 6. SAVE MODEL
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

joblib.dump(
    {
        "model": model,
        "features": list(X.columns),
        "contamination": 0.002
    },
    "models/isolation_forest.pkl"
)

# Save actual and predicted results
output = X_test.copy()

output["Actual_Fraud"] = y_test.values
output["Predicted_Anomaly"] = predictions

output.to_csv(
    "reports/isolation_forest_predictions.csv",
    index=False
)

metrics = pd.DataFrame([{
    "Model": "Isolation Forest",
    "Precision": precision,
    "Recall": recall,
    "F1_Score": f1,
    "False_Positive_Rate": fpr,
    "Training_Time_Seconds": training_time
}])

metrics.to_csv(
    "reports/isolation_forest_metrics.csv",
    index=False
)

print("\nSaved model: models/isolation_forest.pkl")
print("Saved predictions: reports/isolation_forest_predictions.csv")
print("Saved metrics: reports/isolation_forest_metrics.csv")

print("\nIsolation Forest completed successfully!")
