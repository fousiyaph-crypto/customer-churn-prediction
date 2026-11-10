
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models"
REPORT_PATH = ROOT / "reports"

MODEL_PATH.mkdir(exist_ok=True)
REPORT_PATH.mkdir(exist_ok=True)

# Load training and testing datasets
X_train = pd.read_csv(DATA_PATH / "X_train.csv")
X_test = pd.read_csv(DATA_PATH / "X_test.csv")

y_train = pd.read_csv(DATA_PATH / "y_train.csv").squeeze()
y_test = pd.read_csv(DATA_PATH / "y_test.csv").squeeze()

# Define models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )
}

results = []

# Train and evaluate each model
for name, model in models.items():

    print("\nTraining:", name)

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_probability)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC-AUC:", round(auc, 4))

    # Save trained model
    filename = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, MODEL_PATH / filename)

    # Save results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": auc
    })

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"]
    )

    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    chart_name = filename.replace(".pkl", "_confusion_matrix.png")
    plt.savefig(REPORT_PATH / chart_name)
    plt.close()

# Compare model results
results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

results_df.to_csv(
    REPORT_PATH / "model_comparison.csv",
    index=False
)

print("\nModel training completed successfully!")
print("Models saved in:", MODEL_PATH)
print("Reports saved in:", REPORT_PATH)
