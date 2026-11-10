
# ============================================================
# CUSTOMER CHURN PREDICTION
# MODEL TRAINING AND MLFLOW EXPERIMENT TRACKING
# ============================================================

# 1. IMPORT LIBRARIES

import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import traceback

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


# ============================================================
# 2. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models"
REPORT_PATH = ROOT / "reports"

MODEL_PATH.mkdir(parents=True, exist_ok=True)
REPORT_PATH.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 60)
print("CUSTOMER CHURN PREDICTION - MLFLOW")
print("=" * 60)


# ============================================================
# 3. MLFLOW CONFIGURATION
# ============================================================

tracking_db = ROOT / "mlflow.db"

mlflow.set_tracking_uri(
    f"sqlite:///{tracking_db.as_posix()}"
)

mlflow.set_experiment("Customer Churn Prediction")

print("\nMLflow initialized successfully!")
print("Tracking URI:", mlflow.get_tracking_uri())


# ============================================================
# 4. LOAD PROCESSED DATASET
# ============================================================

print("\nLoading processed dataset...")

X_train = pd.read_csv(DATA_PATH / "X_train.csv")
X_test = pd.read_csv(DATA_PATH / "X_test.csv")

y_train = pd.read_csv(
    DATA_PATH / "y_train.csv"
).squeeze()

y_test = pd.read_csv(
    DATA_PATH / "y_test.csv"
).squeeze()

print("\nDataset loaded successfully!")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Check dataset consistency

if X_train.shape[1] != X_test.shape[1]:
    raise ValueError("Training and testing features do not match.")

if list(X_train.columns) != list(X_test.columns):
    raise ValueError("Training and testing feature columns differ.")

if y_train.isnull().any() or y_test.isnull().any():
    raise ValueError("Target dataset contains missing values.")

print("\nDataset validation completed!")


# ============================================================
# 5. DEFINE MACHINE LEARNING MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

}


# ============================================================
# 6. TRAIN AND TRACK MODELS
# ============================================================

results = []

for name, model in models.items():

    model_key = name.lower().replace(" ", "_")

    print("\n" + "=" * 60)
    print("TRAINING MODEL:", name)
    print("=" * 60)

    try:

        with mlflow.start_run(run_name=name):

            # ------------------------------------------------
            # A. LOG MODEL PARAMETERS
            # ------------------------------------------------

            print("\nLogging model parameters...")

            mlflow.log_param("model_name", name)

            mlflow.log_params(model.get_params())

            mlflow.log_param(
                "training_samples",
                len(X_train)
            )

            mlflow.log_param(
                "testing_samples",
                len(X_test)
            )

            mlflow.log_param(
                "number_of_features",
                X_train.shape[1]
            )


            # ------------------------------------------------
            # B. TRAIN MODEL
            # ------------------------------------------------

            print("\nTraining started...")

            model.fit(X_train, y_train)

            print("Training completed successfully!")


            # ------------------------------------------------
            # C. MAKE PREDICTIONS
            # ------------------------------------------------

            print("\nGenerating predictions...")

            y_pred = model.predict(X_test)

            y_probability = model.predict_proba(X_test)[:, 1]


            # ------------------------------------------------
            # D. MODEL EVALUATION
            # ------------------------------------------------

            metrics = {

                "accuracy": accuracy_score(
                    y_test, y_pred
                ),

                "precision": precision_score(
                    y_test, y_pred, zero_division=0
                ),

                "recall": recall_score(
                    y_test, y_pred, zero_division=0
                ),

                "f1_score": f1_score(
                    y_test, y_pred, zero_division=0
                ),

                "roc_auc": roc_auc_score(
                    y_test, y_probability
                )

            }

            print("\nMODEL PERFORMANCE")

            for metric, value in metrics.items():

                print(f"{metric}: {value:.4f}")


            # ------------------------------------------------
            # E. LOG METRICS TO MLFLOW
            # ------------------------------------------------

            print("\nLogging metrics...")

            mlflow.log_metrics(metrics)

            print("Metrics logged successfully!")


            # ------------------------------------------------
            # F. SAVE MODEL LOCALLY
            # ------------------------------------------------

            model_filename = model_key + ".pkl"

            local_model_path = MODEL_PATH / model_filename

            joblib.dump(model, local_model_path)

            print("\nLocal model saved:")
            print(local_model_path)


            # ------------------------------------------------
            # G. LOG MODEL TO MLFLOW 3.x
            # ------------------------------------------------

            print("\nLogging model to MLflow...")

            mlflow.sklearn.log_model(
                sk_model=model,
                name=model_key
            )

            print("MLflow model logged successfully!")


            # ------------------------------------------------
            # H. GENERATE CONFUSION MATRIX
            # ------------------------------------------------

            print("\nGenerating confusion matrix...")

            cm = confusion_matrix(
                y_test,
                y_pred,
                labels=[0, 1]
            )

            display = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=["No Churn", "Churn"]
            )

            display.plot(cmap="Blues")

            plt.title("Confusion Matrix - " + name)

            chart_filename = (
                model_key + "_confusion_matrix.png"
            )

            chart_path = REPORT_PATH / chart_filename

            plt.savefig(
                chart_path,
                bbox_inches="tight"
            )

            plt.close()

            mlflow.log_artifact(str(chart_path))

            print("Confusion matrix saved and logged!")


            # ------------------------------------------------
            # I. GENERATE CLASSIFICATION REPORT
            # ------------------------------------------------

            report = classification_report(
                y_test,
                y_pred,
                zero_division=0
            )

            print("\nCLASSIFICATION REPORT")
            print(report)

            report_filename = (
                model_key + "_classification_report.txt"
            )

            report_path = REPORT_PATH / report_filename

            report_path.write_text(
                report,
                encoding="utf-8"
            )

            mlflow.log_artifact(str(report_path))

            print("Classification report logged!")


            # ------------------------------------------------
            # J. SAVE RESULTS FOR COMPARISON
            # ------------------------------------------------

            results.append({

                "Model": name,
                **metrics

            })

            print("\nSUCCESS:", name)


    except Exception as error:

        print("\nMODEL FAILED:", name)

        print("Error:", str(error))

        traceback.print_exc()

        continue


# ============================================================
# 7. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

if results:

    results_df = pd.DataFrame(results)

    print(results_df.to_string(index=False))

    comparison_path = (
        REPORT_PATH / "model_comparison.csv"
    )

    results_df.to_csv(
        comparison_path,
        index=False
    )

    print("\nModel comparison saved successfully!")


    # --------------------------------------------------------
    # FIND BEST MODEL
    # --------------------------------------------------------

    best_model = results_df.loc[
        results_df["roc_auc"].idxmax()
    ]

    print("\nBEST MODEL:", best_model["Model"])

    print(
        "BEST ROC-AUC:",
        round(best_model["roc_auc"], 4)
    )

else:

    print("\nNo models completed successfully!")


# ============================================================
# 8. FINAL TRAINING STATUS
# ============================================================

print("\n" + "=" * 60)
print("TRAINING PROCESS COMPLETED")
print("=" * 60)

print("\nModels saved in:", MODEL_PATH)

print("Reports saved in:", REPORT_PATH)

print("MLflow database:", tracking_db)

print("\nThank you!")
