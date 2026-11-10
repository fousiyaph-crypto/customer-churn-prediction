
from pathlib import Path
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models"
REPORT_PATH = ROOT / "reports"

MODEL_PATH.mkdir(exist_ok=True)
REPORT_PATH.mkdir(exist_ok=True)

# Load processed data
X_train = pd.read_csv(DATA_PATH / "X_train.csv")
X_test = pd.read_csv(DATA_PATH / "X_test.csv")

y_train = pd.read_csv(DATA_PATH / "y_train.csv").squeeze()
y_test = pd.read_csv(DATA_PATH / "y_test.csv").squeeze()

print("Data loaded successfully!")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# Logistic Regression
model = LogisticRegression(max_iter=2000, random_state=42)

# Hyperparameter options
parameters = {
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["liblinear", "lbfgs"],
    "class_weight": [None, "balanced"]
}

# Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=model,
    param_grid=parameters,
    cv=5,
    scoring="roc_auc",
    n_jobs=1,
    verbose=1
)

print("\nStarting hyperparameter tuning...")

grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-validation ROC-AUC:")
print(round(grid_search.best_score_, 4))

# Test evaluation
y_pred = best_model.predict(X_test)
y_probability = best_model.predict_proba(X_test)[:, 1]

results = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1 Score": f1_score(y_test, y_pred, zero_division=0),
    "ROC AUC": roc_auc_score(y_test, y_probability)
}

print("\nTuned Model Performance:")
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Save best model
# Save model and selected threshold together
model_package = {
    "model": best_model,
    "threshold": best_threshold
}

joblib.dump(
    model_package,
    MODEL_PATH / "churn_threshold_model.pkl"
)

print("Improved model and threshold saved successfully!")
print(f"Selected threshold: {best_threshold}")
# Save results
pd.DataFrame([results]).to_csv(
    REPORT_PATH / "tuned_model_performance.csv",
    index=False
)

print("\nBest model saved successfully!")
print("Results saved successfully!")
