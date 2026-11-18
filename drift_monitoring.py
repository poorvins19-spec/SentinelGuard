
import os
import json
import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset

print("\n====================================")
print(" SENTINELGUARD - DATA DRIFT MONITOR")
print("====================================")

# 1. LOAD DATASET
DATA_PATH = "dataset/creditcard.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "Dataset not found: " + DATA_PATH
    )

df = pd.read_csv(DATA_PATH)

if "Class" not in df.columns:
    raise ValueError("Class column is missing.")

# 2. PREPARE DATA
df = df.dropna(subset=["Class"]).copy()

# Class is a target label, not a transaction input feature.
# We exclude it from drift comparison.
features = df.drop(columns=["Class"])

features = features.select_dtypes(
    include=["number"]
).copy()

features = features.replace(
    [float("inf"), float("-inf")],
    float("nan")
).fillna(0)

# 3. CREATE REFERENCE AND CURRENT DATA
split_index = int(len(features) * 0.70)

reference_data = features.iloc[:split_index].copy()
current_data = features.iloc[split_index:].copy()

if len(reference_data) == 0 or len(current_data) == 0:
    raise ValueError("Not enough data to compare.")

print("\nReference records:", len(reference_data))
print("Comparison records:", len(current_data))

# 4. CREATE DRIFT REPORT
print("\nGenerating Evidently drift report...")

report = Report([
    DataDriftPreset()
])

result = report.run(
    reference_data=reference_data,
    current_data=current_data
)

# 5. SAVE HTML REPORT
os.makedirs("reports", exist_ok=True)

report_path = "reports/data_drift_report.html"

result.save_html(report_path)

print("\nDrift report saved:")
print(report_path)

# 6. SAVE JSON REPORT
json_path = "reports/data_drift_report.json"

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(
        result.dict(),
        file,
        indent=4,
        default=str
    )

print("\nJSON report saved:")
print(json_path)

print("\nReference and comparison dataset details:")
print("Number of features:", len(features.columns))

print("\nDrift monitoring completed successfully!")
