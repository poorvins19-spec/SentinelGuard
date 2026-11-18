
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# 1. Load dataset
df = pd.read_csv("dataset/creditcard.csv")

# 2. Load trained model
saved = joblib.load("models/fraud_model.pkl")

model = saved["model"]
features = saved["features"]

# 3. Prepare sample transactions
X = df[features].copy()
X = X.fillna(X.median())

# Use a sample to reduce SHAP computation time
sample = X.sample(
    n=min(100, len(X)),
    random_state=42
)

print("Calculating SHAP values...")

# 4. Create SHAP explainer
explainer = shap.TreeExplainer(model)

raw_values = explainer.shap_values(sample)

# 5. Select fraud-class SHAP values
if isinstance(raw_values, list):
    shap_values = raw_values[1]
else:
    shap_values = np.asarray(raw_values)

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

# 6. Calculate feature importance
importance = np.abs(shap_values).mean(axis=0)

importance_df = pd.DataFrame({
    "Feature": features,
    "SHAP Importance": importance
})

importance_df = importance_df.sort_values(
    by="SHAP Importance",
    ascending=False
)

print("\nTOP 15 FRAUD PREDICTION FEATURES")
print(importance_df.head(15).to_string(index=False))

# 7. Plot SHAP feature importance
top_features = importance_df.head(15).sort_values(
    by="SHAP Importance"
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["SHAP Importance"]
)

plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("Transaction Features")

plt.title(
    "SentinelGuard - SHAP Feature Importance"
)

plt.tight_layout()
plt.savefig(
    "models/shap_feature_importance.png",
    dpi=300
)

plt.show()

# 8. Explain one transaction
transaction_index = 0

local_values = shap_values[transaction_index]

explanation = pd.DataFrame({
    "Feature": features,
    "Transaction Value": sample.iloc[transaction_index].values,
    "SHAP Contribution": local_values
})

explanation["Absolute Contribution"] = (
    explanation["SHAP Contribution"].abs()
)

explanation = explanation.sort_values(
    by="Absolute Contribution",
    ascending=False
)

print("\nINDIVIDUAL TRANSACTION EXPLANATION")
print(explanation.head(10).to_string(index=False))

print("\nSHAP ANALYSIS COMPLETED SUCCESSFULLY!")
