
# SentinelGuard
## Banking Fraud Intelligence and Explainable Risk Scoring Platform

### Project Overview
SentinelGuard is a machine-learning-based banking fraud detection platform. It analyzes transaction data, predicts potentially fraudulent transactions, assigns risk scores, and helps analysts understand model predictions.

### Problem Statement
Banks process large numbers of transactions every day. Detecting fraudulent transactions is challenging because fraud cases are much less common than legitimate transactions.

SentinelGuard uses machine learning to identify suspicious transaction patterns and support fraud investigation.

### Main Features
- Automatic analysis of transactions from a CSV dataset
- Fraud prediction using machine learning
- Legitimate and potentially fraudulent transaction classification
- Random Forest and XGBoost model comparison
- SMOTE-based class imbalance handling
- Probability-based risk scoring
- Decision threshold optimization
- SHAP-based model explainability
- Isolation Forest anomaly detection
- Autoencoder-based anomaly detection
- Fraud alert dashboard
- FastAPI prediction service
- Data drift monitoring
- Downloadable prediction and alert reports

### Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn
- SHAP
- PyTorch
- Streamlit
- FastAPI
- Matplotlib and Seaborn
- Evidently AI

### Dataset Information
The project uses a credit-card transaction CSV dataset.

Important columns:
- Time: Transaction time information
- Amount: Transaction amount
- V1 to V28: Anonymized transaction features
- Class: Actual transaction label

Class labels:
- 0 = Legitimate transaction
- 1 = Fraudulent transaction

**Dataset limitation:** The commonly used credit-card dataset does not contain merchant category, country, device risk score, or IP risk score. These fields are not claimed as available in this dataset.

### Project Workflow
1. Load the transaction dataset.
2. Inspect and preprocess the data.
3. Explore the fraud and legitimate transaction distribution.
4. Train machine learning models.
5. Handle class imbalance.
6. Compare model performance.
7. Optimize the classification threshold.
8. Explain predictions using SHAP.
9. Detect anomalies using unsupervised learning.
10. Generate fraud risk scores and alerts.
11. Display results through a dashboard.
12. Monitor data drift.

### Project Structure
```text
SentinelGuard/
│
├── dataset/
│   ├── creditcard.csv
│   └── README.md
│
├── models/
│   └── README.md
│
├── reports/
│   └── README.md
│
├── app.py
├── api.py
├── api_client.py
├── test_integration.py
├── train_model.py
├── compare_models.py
├── smote_model_comparison.py
├── xgboost_tuning.py
├── threshold_optimization.py
├── isolation_forest.py
├── autoencoder.py
├── shap_analysis.py
├── drift_monitoring.py
├── fraud_alerts.py
├── requirements.txt
├── .gitignore
└── README.md
```

This is the intended structure. Keep only the files that actually exist in your project.

### Installation

1. Clone or download this repository.
2. Open the project folder in VS Code.
3. Create and activate a virtual environment.

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

4. Install the required packages:

```powershell
pip install -r requirements.txt
```

5. Place the dataset in the dataset folder.

### Run the Streamlit Dashboard

Run the dashboard file that exists in your project:

```powershell
streamlit run app.py
```

If your analyst dashboard is in `fraud_alerts.py`, run:

```powershell
streamlit run fraud_alerts.py
```

### Run the FastAPI Service

```powershell
uvicorn api:app --reload
```

Open this address in your browser to view the API documentation:

```text
http://127.0.0.1:8000/docs
```

### Model Evaluation
The project evaluates models using relevant classification metrics, including:

- Precision
- Recall
- F1-score
- Confusion matrix
- ROC-AUC
- Average precision

Add your actual results to this section after running the experiments. Do not enter estimated or invented performance values.

### Explainable AI
SHAP is used to explain how input features contribute to model predictions. This helps make fraud predictions easier to investigate.

### Risk Scoring
The platform converts predicted fraud probabilities into risk scores. Risk levels and alert thresholds are configurable and should be evaluated using validation data.

### Future Enhancements
- Advanced fraud-ring detection using graph neural networks
- Improved real-time monitoring
- Cloud-based deployment
- Online learning for changing fraud patterns
- Advanced analyst alert management

### Project Disclaimer
This project is developed for academic and educational purposes. Model predictions are decision-support outputs and should not be treated as confirmed fraud without further investigation.

### Author
Developed as an AI and Machine Learning project.
