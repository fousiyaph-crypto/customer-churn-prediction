
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# Project paths
ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models"
REPORT_PATH = ROOT / "reports"

MODEL_PATH.mkdir(parents=True, exist_ok=True)
REPORT_PATH.mkdir(parents=True, exist_ok=True)

# Load processed data
X_train = pd.read_csv(DATA_PATH / "X_train.csv")
X_test = pd.read_csv(DATA_PATH / "X_test.csv")

y_train = pd.read_csv(DATA_PATH / "y_train.csv").squeeze()
y_test = pd.read_csv(DATA_PATH / "y_test.csv").squeeze()

# MLflow configuration
mlflow.set_experiment("Customer-Churn-Prediction")

with mlflow.start_run(run_name="Logistic_Regression"):

    # Model parameters
    params = {
        "max_iter": 1000,
        "random_state": 42,
    }

    # Train only Logistic Regression
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluation metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    # Log parameters and metrics
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)

    # Save model
    model_file = MODEL_PATH / "logistic_regression.pkl"
    joblib.dump(model, model_file)

    # Log model to MLflow
    mlflow.sklearn.log_model(
        sk_model=model,
        name="logistic_regression_model",
    )

    # Save evaluation report
    report = classification_report(y_test, y_pred)
    report_file = REPORT_PATH / "classification_report.txt"
    report_file.write_text(report, encoding="utf-8")

    mlflow.log_artifact(str(report_file))

    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_file = REPORT_PATH / "confusion_matrix.csv"

    pd.DataFrame(
        cm,
        index=["Actual_0", "Actual_1"],
        columns=["Predicted_0", "Predicted_1"],
    ).to_csv(cm_file)

    mlflow.log_artifact(str(cm_file))

    print("\nLogistic Regression Training Successful!")
    print("\nEvaluation Metrics:")

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nModel saved:", model_file)
    print("Classification report saved:", report_file)
