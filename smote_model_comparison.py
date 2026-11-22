
import os
import time
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ---------------- 1. LOAD DATA ----------------

print("\nSENTINELGUARD - SMOTE MODEL COMPARISON")

df = pd.read_csv("dataset/creditcard.csv")

target = "Class"

if target not in df.columns:
    raise ValueError("Target column 'Class' not found in dataset.")

df = df.dropna(subset=[target]).copy()

X = df.drop(columns=[target])
y = df[target].astype(int)

# Keep numeric features only
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

print("\nDataset shape:", df.shape)
print("\nOriginal class distribution:")
print(y.value_counts())

# ---------------- 2. TRAIN TEST SPLIT ----------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data before SMOTE:")
print(y_train.value_counts())

print("\nTesting data:")
print(y_test.value_counts())

# ---------------- 3. APPLY SMOTE ----------------

minority_count = y_train.value_counts().min()

if minority_count < 2:
    raise ValueError("Not enough minority-class samples for SMOTE.")

k = min(5, minority_count - 1)

smote = SMOTE(
    random_state=42,
    k_neighbors=k
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nTraining data after SMOTE:")
print(pd.Series(y_train_smote).value_counts())

# ---------------- 4. DEFINE MODELS ----------------

models = {
    "Logistic Regression": ImbPipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ]),

    "Random Forest": ImbPipeline([
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight=None
        ))
    ])
}

# ---------------- 5. TRAIN AND EVALUATE ----------------

results = []
trained_models = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    start = time.perf_counter()

    model.fit(X_train_smote, y_train_smote)

    training_time = time.perf_counter() - start

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
        "F1_Score": f1,
        "False_Positive_Rate": fpr,
        "Training_Time_Seconds": training_time
    })

    trained_models[name] = model

    print("Accuracy :", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision * 100, 2), "%")
    print("Recall   :", round(recall * 100, 2), "%")
    print("F1 Score :", round(f1 * 100, 2), "%")
    print("FPR      :", round(fpr * 100, 2), "%")
    print("TN, FP, FN, TP:", tn, fp, fn, tp)

# ---------------- 6. COMPARE MODELS ----------------

results_df = pd.DataFrame(results)

print("\n===== MODEL COMPARISON =====")
print(results_df.to_string(index=False))

os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)

results_df.to_csv(
    "reports/smote_model_comparison.csv",
    index=False
)

# ---------------- 7. SELECT MODEL ----------------

# Prefer models meeting the PDF's FPR target.
eligible = results_df[
    results_df["False_Positive_Rate"] < 0.05
]

if len(eligible) > 0:
    best_row = eligible.sort_values(
        ["Recall", "F1_Score"],
        ascending=False
    ).iloc[0]
else:
    best_row = results_df.sort_values(
        ["Recall", "F1_Score"],
        ascending=False
    ).iloc[0]

best_name = best_row["Model"]
best_model = trained_models[best_name]

# Save the complete pipeline and feature names
joblib.dump(
    {
        "model": best_model,
        "features": list(X.columns),
        "metrics": best_row.to_dict()
    },
    "models/smote_best_model.pkl"
)

print("\nBest selected model:", best_name)
print("Saved: models/smote_best_model.pkl")
print("Saved: reports/smote_model_comparison.csv")
print("\nSMOTE model comparison completed!")
