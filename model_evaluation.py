
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt

# 1. Load dataset
df = pd.read_csv("dataset/creditcard.csv")

# 2. Separate features and target
target = "Class"

X = df.drop(columns=[target])
y = df[target]

# 3. Load saved model
saved = joblib.load("models/fraud_model.pkl")

if isinstance(saved, dict):
    model = saved["model"]
    features = saved.get("features", list(X.columns))
else:
    model = saved
    features = list(X.columns)

X = X[features]

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. Predict test transactions
y_pred = model.predict(X_test)

# 6. Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n===== SENTINELGUARD MODEL EVALUATION =====")

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")

# 7. Confusion matrix
cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

print("\n===== CONFUSION MATRIX =====")
print(cm)

print("\nTrue Negative :", tn)
print("False Positive:", fp)
print("False Negative:", fn)
print("True Positive :", tp)

# 8. Detailed report
print("\n===== CLASSIFICATION REPORT =====")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Legitimate", "Fraud"],
        zero_division=0
    )
)

# 9. Save confusion matrix image
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Legitimate", "Fraud"]
)

display.plot(cmap="Blues", values_format="d")
plt.title("SentinelGuard Confusion Matrix")
plt.tight_layout()

os.makedirs("reports", exist_ok=True)

plt.savefig("reports/confusion_matrix.png")
plt.show()

# 10. Save evaluation results
metrics = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "True Negative",
        "False Positive",
        "False Negative",
        "True Positive"
    ],
    "Value": [
        accuracy,
        precision,
        recall,
        f1,
        tn,
        fp,
        fn,
        tp
    ]
})

metrics.to_csv(
    "reports/model_evaluation.csv",
    index=False
)

print("\nEvaluation completed successfully!")
print("Reports saved inside reports folder.")
