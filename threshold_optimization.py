
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score
)

print("\n======================================")
print(" SENTINELGUARD - THRESHOLD OPTIMIZATION")
print("======================================")

# 1. LOAD DATASET
df = pd.read_csv("dataset/creditcard.csv")

if "Class" not in df.columns:
    raise ValueError("Class column not found.")

df = df.dropna(subset=["Class"]).copy()

X = df.drop(columns=["Class"])
y = df["Class"].astype(int)

X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

# 2. SPLIT DATA INTO TRAIN, VALIDATION AND TEST
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=42,
    stratify=y
)

X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nTraining transactions:", len(X_train))
print("Validation transactions:", len(X_validation))
print("Testing transactions:", len(X_test))

# 3. LOAD PREVIOUSLY TUNED XGBOOST MODEL
saved_data = joblib.load("models/xgboost_best_model.pkl")

model = saved_data["model"]
features = saved_data["features"]

# Match the model's original feature order
X_validation = X_validation[features]
X_test = X_test[features]

# 4. GET VALIDATION PROBABILITIES
validation_probabilities = model.predict_proba(
    X_validation
)[:, 1]

# 5. TEST DIFFERENT THRESHOLDS
thresholds = np.arange(0.05, 1.00, 0.05)

results = []

for threshold in thresholds:

    predictions = (
        validation_probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_validation,
        predictions,
        labels=[0, 1]
    ).ravel()

    recall = recall_score(
        y_validation,
        predictions,
        zero_division=0
    )

    precision = precision_score(
        y_validation,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_validation,
        predictions,
        zero_division=0
    )

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    results.append({
        "Threshold": round(float(threshold), 2),
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "False_Positive_Rate": fpr,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn
    })

results_df = pd.DataFrame(results)

print("\n========== VALIDATION RESULTS ==========")
print(results_df.round(4).to_string(index=False))

# 6. SELECT THRESHOLD
# First prefer thresholds meeting the PDF's FPR target.
eligible = results_df[
    results_df["False_Positive_Rate"] < 0.05
]

if not eligible.empty:
    best_row = eligible.sort_values(
        ["Recall", "F1_Score"],
        ascending=False
    ).iloc[0]
else:
    # If no threshold meets the target, choose the lowest FPR,
    # then use recall and F1 as tie-breakers.
    best_row = results_df.sort_values(
        ["False_Positive_Rate", "Recall", "F1_Score"],
        ascending=[True, False, False]
    ).iloc[0]

best_threshold = float(best_row["Threshold"])

print("\nSelected threshold:", best_threshold)
print(
    "Validation recall:",
    round(best_row["Recall"] * 100, 2), "%"
)
print(
    "Validation false-positive rate:",
    round(best_row["False_Positive_Rate"] * 100, 2), "%"
)

# 7. FINAL EVALUATION ON UNSEEN TEST SET
test_probabilities = model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
).ravel()

test_precision = precision_score(
    y_test, test_predictions, zero_division=0
)

test_recall = recall_score(
    y_test, test_predictions, zero_division=0
)

test_f1 = f1_score(
    y_test, test_predictions, zero_division=0
)

test_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

test_ap = average_precision_score(
    y_test, test_probabilities
)

print("\n========== FINAL TEST RESULTS ==========")
print("Threshold :", best_threshold)
print("Precision :", round(test_precision * 100, 2), "%")
print("Recall    :", round(test_recall * 100, 2), "%")
print("F1 Score  :", round(test_f1 * 100, 2), "%")
print("FPR       :", round(test_fpr * 100, 2), "%")
print("Average Precision:", round(test_ap, 4))

print("\nConfusion Matrix:")
print("True Negatives :", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives :", tp)

# 8. SAVE RESULTS
os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)

results_df.to_csv(
    "reports/threshold_comparison.csv",
    index=False
)

# Save selected threshold separately.
joblib.dump(
    {
        "threshold": best_threshold,
        "validation_metrics": best_row.to_dict(),
        "test_metrics": {
            "precision": test_precision,
            "recall": test_recall,
            "f1_score": test_f1,
            "false_positive_rate": test_fpr,
            "average_precision": test_ap
        }
    },
    "models/optimal_threshold.pkl"
)

print("\nSaved: reports/threshold_comparison.csv")
print("Saved: models/optimal_threshold.pkl")
print("\nThreshold optimization completed!")
