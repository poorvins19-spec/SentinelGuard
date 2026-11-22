
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import pandas as pd
import numpy as np
import joblib

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =====================================
# 1. LOAD DATASET
# =====================================

print("Loading SentinelGuard dataset...")

df = pd.read_csv("dataset/creditcard.csv")

target_column = "Class"

if target_column not in df.columns:
    raise ValueError(
        f"Target column {target_column} not found. "
        f"Available columns: {df.columns.tolist()}"
    )

df = df.dropna(subset=[target_column]).copy()

X = df.drop(columns=[target_column])
X = X.select_dtypes(include=["number"])
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median()).fillna(0)

y = pd.to_numeric(df[target_column])

if not set(y.unique()).issubset({0, 1}):
    raise ValueError("Target must contain 0 and 1 only.")

y = y.astype(int)

print("Dataset loaded:", X.shape)

# =====================================
# 2. TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# =====================================
# 3. DEFINE MODELS
# =====================================

class_weight_ratio = (
    (y_train == 0).sum() / max((y_train == 1).sum(), 1)
)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=class_weight_ratio,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )
}

# =====================================
# 4. TRAIN AND EVALUATE
# =====================================

results = []
trained_models = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

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

    fpr = fp / (fp + tn) if (fp + tn) else 0

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "FPR": fpr,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    })

    trained_models[name] = model

    print(name, "training completed!")

# =====================================
# 5. COMPARE MODELS
# =====================================

results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")

print(
    results_df[
        ["Model", "Accuracy", "Precision",
         "Recall", "F1 Score", "FPR"]
    ].to_string(index=False)
)

# =====================================
# 6. SELECT BEST MODEL
# =====================================

# Prefer models meeting the project FPR target.
eligible = results_df[results_df["FPR"] < 0.05]

if not eligible.empty:
    best_row = eligible.sort_values(
        ["Recall", "F1 Score"],
        ascending=False
    ).iloc[0]
else:
    # If neither meets the target, choose highest F1.
    best_row = results_df.sort_values(
        "F1 Score",
        ascending=False
    ).iloc[0]

best_name = best_row["Model"]
best_model = trained_models[best_name]

best_metrics = {
    "accuracy": float(best_row["Accuracy"]),
    "precision": float(best_row["Precision"]),
    "recall": float(best_row["Recall"]),
    "f1": float(best_row["F1 Score"]),
    "fpr": float(best_row["FPR"]),
    "tn": int(best_row["TN"]),
    "fp": int(best_row["FP"]),
    "fn": int(best_row["FN"]),
    "tp": int(best_row["TP"])
}

print("\nBEST MODEL:", best_name)

print("\nSelected Model Metrics:")
for key, value in best_metrics.items():
    print(key, ":", value)

# =====================================
# 7. SAVE BEST MODEL
# =====================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    {
        "model": best_model,
        "features": X.columns.tolist(),
        "metrics": best_metrics,
        "model_name": best_name
    },
    "models/fraud_model.pkl"
)

# Save comparison report
results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)

print("\nBEST MODEL SAVED SUCCESSFULLY!")
print("Comparison report saved successfully!")
