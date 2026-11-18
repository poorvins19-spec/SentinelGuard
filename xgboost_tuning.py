
import os
import time
import joblib
import pandas as pd

from xgboost import XGBClassifier

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score
)

print("\n======================================")
print(" SENTINELGUARD - XGBOOST MODEL TUNING")
print("======================================")

# 1. LOAD DATASET
df = pd.read_csv("dataset/creditcard.csv")

if "Class" not in df.columns:
    raise ValueError("The CSV must contain a 'Class' target column.")

df = df.dropna(subset=["Class"]).copy()

X = df.drop(columns=["Class"])
y = df["Class"].astype(int)

# Keep numeric features and handle missing values
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

print("\nDataset shape:", df.shape)
print("\nClass distribution:")
print(y.value_counts())

# 2. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# Calculate class imbalance using training data only
negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())

scale_weight = negative_count / positive_count

print("\nScale positive weight:", round(scale_weight, 2))

# 3. CREATE XGBOOST MODEL
xgb_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    random_state=42,
    n_jobs=2,
    scale_pos_weight=scale_weight
)

# 4. DEFINE SMALL PARAMETER GRID
# A compact grid is used to reduce training time on an 8 GB RAM laptop.
parameter_grid = {
    "n_estimators": [100],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1]
}

# 5. GRID SEARCH
cross_validation = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=parameter_grid,
    scoring="average_precision",
    cv=cross_validation,
    n_jobs=1,
    verbose=2,
    refit=True
)

print("\nStarting GridSearchCV...")
print("This may take some time.")

start_time = time.perf_counter()

grid_search.fit(X_train, y_train)

tuning_time = time.perf_counter() - start_time

best_model = grid_search.best_estimator_

print("\nBest parameters:")
print(grid_search.best_params_)

print(
    "\nBest cross-validation average precision:",
    round(grid_search.best_score_, 4)
)

# 6. EVALUATE ON UNSEEN TEST DATA
start_time = time.perf_counter()

probabilities = best_model.predict_proba(X_test)[:, 1]

prediction_time = time.perf_counter() - start_time

# Default probability threshold = 0.50
predictions = (probabilities >= 0.50).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
).ravel()

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(
    y_test, predictions, zero_division=0
)
recall = recall_score(
    y_test, predictions, zero_division=0
)
f1 = f1_score(
    y_test, predictions, zero_division=0
)
average_precision = average_precision_score(
    y_test, probabilities
)

false_positive_rate = (
    fp / (fp + tn) if (fp + tn) > 0 else 0
)

# Estimate average time for one transaction.
sample_count = min(1000, len(X_test))
latency_per_transaction_ms = (
    prediction_time / sample_count
) * 1000

print("\n========== TEST RESULTS ==========")
print("Accuracy :", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall   :", round(recall * 100, 2), "%")
print("F1 Score :", round(f1 * 100, 2), "%")
print("Average Precision:", round(average_precision, 4))
print("False Positive Rate:", round(false_positive_rate * 100, 2), "%")

print("\nConfusion Matrix:")
print("True Negatives :", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives :", tp)

print(
    "\nEstimated prediction time per transaction:",
    round(latency_per_transaction_ms, 4),
    "ms"
)

print("Total tuning time:", round(tuning_time, 2), "seconds")

# 7. SAVE RESULTS
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

results = pd.DataFrame([{
    "Model": "XGBoost",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1_Score": f1,
    "Average_Precision": average_precision,
    "False_Positive_Rate": false_positive_rate,
    "Estimated_Latency_ms": latency_per_transaction_ms,
    "Tuning_Time_Seconds": tuning_time
}])

results.to_csv(
    "reports/xgboost_tuning_results.csv",
    index=False
)

# Save the fitted model and feature names together
joblib.dump(
    {
        "model": best_model,
        "features": list(X.columns),
        "best_parameters": grid_search.best_params_,
        "metrics": results.iloc[0].to_dict()
    },
    "models/xgboost_best_model.pkl"
)

print("\nSaved model:")
print("models/xgboost_best_model.pkl")

print("\nSaved results:")
print("reports/xgboost_tuning_results.csv")

print("\nXGBoost tuning completed successfully!")