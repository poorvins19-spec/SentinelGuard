
import os
import time
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

print("\n================================")
print(" SENTINELGUARD - AUTOENCODER")
print("================================")

# Use CPU for compatibility with ordinary laptops
device = torch.device("cpu")

# 1. LOAD DATASET
df = pd.read_csv("dataset/creditcard.csv")

if "Class" not in df.columns:
    raise ValueError("Class column not found in CSV.")

df = df.dropna(subset=["Class"]).copy()

X = df.drop(columns=["Class"])
y = df["Class"].astype(int)

X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

features = list(X.columns)

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

# Train only on legitimate transactions
X_normal = X_train[y_train == 0]

# Limit training size to reduce memory use
if len(X_normal) > 100000:
    X_normal = X_normal.sample(
        n=100000,
        random_state=42
    )

# 3. SCALE DATA
scaler = StandardScaler()

X_normal_scaled = scaler.fit_transform(X_normal)
X_test_scaled = scaler.transform(X_test)

X_normal_scaled = np.nan_to_num(X_normal_scaled)
X_test_scaled = np.nan_to_num(X_test_scaled)

train_tensor = torch.tensor(
    X_normal_scaled,
    dtype=torch.float32
)

train_loader = DataLoader(
    TensorDataset(train_tensor),
    batch_size=512,
    shuffle=True
)

# 4. BUILD AUTOENCODER
class Autoencoder(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_size)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


model = Autoencoder(X_normal_scaled.shape[1]).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# 5. TRAIN MODEL
print("\nTraining Autoencoder...")

start = time.perf_counter()

epochs = 10

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for (batch,) in train_loader:

        batch = batch.to(device)

        reconstructed = model(batch)

        loss = criterion(reconstructed, batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"- Loss: {average_loss:.6f}"
    )

training_time = time.perf_counter() - start

# 6. CALCULATE RECONSTRUCTION ERROR
model.eval()

with torch.no_grad():

    test_tensor = torch.tensor(
        X_test_scaled,
        dtype=torch.float32
    )

    reconstructed = model(test_tensor)

    errors = torch.mean(
        (test_tensor - reconstructed) ** 2,
        dim=1
    ).numpy()

# Set anomaly threshold using normal training errors
with torch.no_grad():

    normal_reconstructed = model(train_tensor)

    normal_errors = torch.mean(
        (train_tensor - normal_reconstructed) ** 2,
        dim=1
    ).numpy()

threshold = float(np.percentile(normal_errors, 99.8))

# 7. CLASSIFY ANOMALIES
predictions = (errors > threshold).astype(int)

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

fpr = fp / (fp + tn) if (fp + tn) else 0

# 8. DISPLAY RESULTS
print("\n========== AUTOENCODER RESULTS ==========")

print("Anomaly threshold:", threshold)
print("Precision:", round(precision * 100, 2), "%")
print("Recall:", round(recall * 100, 2), "%")
print("F1 Score:", round(f1 * 100, 2), "%")
print("False Positive Rate:", round(fpr * 100, 2), "%")

print("\nConfusion Matrix:")
print("True Negatives :", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives :", tp)

print("\nPrediction counts:")
print(pd.Series(predictions).value_counts())

# 9. SAVE MODEL AND RESULTS
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

joblib.dump(
    {
        "model_state": model.state_dict(),
        "scaler": scaler,
        "features": features,
        "threshold": threshold
    },
    "models/autoencoder.pkl"
)

output = X_test.copy()
output["Actual_Fraud"] = y_test.values
output["Reconstruction_Error"] = errors
output["Predicted_Anomaly"] = predictions

output.to_csv(
    "reports/autoencoder_predictions.csv",
    index=False
)

pd.DataFrame([{
    "Model": "Autoencoder",
    "Precision": precision,
    "Recall": recall,
    "F1_Score": f1,
    "False_Positive_Rate": fpr,
    "Threshold": threshold,
    "Training_Time": training_time
}]).to_csv(
    "reports/autoencoder_metrics.csv",
    index=False
)

print("\nSaved: models/autoencoder.pkl")
print("Saved: reports/autoencoder_predictions.csv")
print("Saved: reports/autoencoder_metrics.csv")

print("\nAutoencoder completed successfully!")
