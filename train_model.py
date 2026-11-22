
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# 1. Load dataset
df = pd.read_csv("dataset/creditcard.csv")

# 2. Identify target column
target_column = "Class"

if target_column not in df.columns:
    raise ValueError(
        f"Target column '{target_column}' not found. "
        f"Available columns: {df.columns.tolist()}"
    )

# 3. Prepare features and target
df = df.dropna(subset=[target_column]).copy()

X = df.drop(columns=[target_column])
y = df[target_column]

# Keep numeric features only
X = X.select_dtypes(include=["number"])

# Fill missing feature values
X = X.fillna(X.median())

if y.nunique() != 2:
    raise ValueError("This version requires a binary fraud target.")

if not set(y.unique()).issubset({0, 1}):
    raise ValueError("Target labels must be 0 (legitimate) and 1 (fraud).")

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Create model
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# 6. Train model
print("Training SentinelGuard model...")
model.fit(X_train, y_train)

# 7. Test model on unseen test data
y_pred = model.predict(X_test)

# 8. Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

tn, fp, fn, tp = confusion_matrix(
    y_test, y_pred, labels=[0, 1]
).ravel()

fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

# 9. Display results
print("\n========== SENTINELGUARD MODEL EVALUATION ==========")

print(f"Accuracy:          {accuracy * 100:.2f}%")
print(f"Precision:         {precision * 100:.2f}%")
print(f"Recall:            {recall * 100:.2f}%")
print(f"F1 Score:          {f1 * 100:.2f}%")
print(f"False Positive Rate: {fpr * 100:.2f}%")

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

# 10. Save model and evaluation results
os.makedirs("models", exist_ok=True)

joblib.dump(
    {
        "model": model,
        "features": X.columns.tolist(),
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": fpr,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
    },
    "models/fraud_model.pkl"
)

print("\nMODEL AND EVALUATION RESULTS SAVED SUCCESSFULLY!")
