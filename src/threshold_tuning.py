import joblib
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models"

# Load training and test data
X_train = pd.read_csv(DATA_PATH / "X_train.csv")
X_test = pd.read_csv(DATA_PATH / "X_test.csv")
y_train = pd.read_csv(DATA_PATH / "y_train.csv").squeeze()
y_test = pd.read_csv(DATA_PATH / "y_test.csv").squeeze()

# Split training data into training and validation sets
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)

# Train Logistic Regression
model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_fit, y_fit)

# Find the positive churn class
positive_class = 1
positive_index = list(model.classes_).index(positive_class)

# Validation probabilities
val_probabilities = model.predict_proba(X_val)[:, positive_index]

# Compare thresholds
results = []

for threshold in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:

    predictions = (val_probabilities >= threshold).astype(int)

    results.append({
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_val, predictions),
        "Precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_val, predictions, zero_division=0
        )
    })

results_df = pd.DataFrame(results)

print("\nVALIDATION THRESHOLD COMPARISON")
print(results_df.round(3).to_string(index=False))

# Select the threshold with the highest validation F1 score
best_row = results_df.loc[results_df["F1 Score"].idxmax()]
best_threshold = best_row["Threshold"]

print("\nBest threshold:", best_threshold)

# Evaluate on untouched test data
test_probabilities = model.predict_proba(X_test)[:, positive_index]
test_predictions = (test_probabilities >= best_threshold).astype(int)

print("\nFINAL TEST RESULTS")
print("Accuracy:", round(accuracy_score(y_test, test_predictions), 3))
print("Precision:", round(precision_score(y_test, test_predictions, zero_division=0), 3))
print("Recall:", round(recall_score(y_test, test_predictions, zero_division=0), 3))
print("F1 Score:", round(f1_score(y_test, test_predictions, zero_division=0), 3))
print("ROC-AUC:", round(roc_auc_score(y_test, test_probabilities), 3))

# Save results
results_df.to_csv(
    ROOT / "reports" / "threshold_comparison.csv",
    index=False
)
import joblib

# Save trained model and selected threshold
MODEL_PATH.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "model": model,
        "threshold": float(best_threshold)
    },
    MODEL_PATH / "tuned_logistic_model.pkl"
)

print("Tuned model and threshold saved successfully!")

print("\nThreshold results saved successfully!")
# Train final model using all training data
final_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

final_model.fit(X_train, y_train)

# Save model and selected threshold together
model_bundle = {
    "model": final_model,
    "threshold": float(best_threshold),
    "feature_names": list(X_train.columns)
}

MODEL_PATH.mkdir(parents=True, exist_ok=True)

joblib.dump(
    model_bundle,
    MODEL_PATH / "churn_threshold_model.pkl"
)

print("\nImproved model saved successfully!")
print("Selected threshold:", best_threshold)
print("Saved file: models/churn_threshold_model.pkl")