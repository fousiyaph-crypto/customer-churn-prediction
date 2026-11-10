from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from pathlib import Path
from mlflow.tracking import MlflowClient
from sklearn.metrics import confusion_matrix

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
import mlflow

# -------------------------------------------------
# 1. PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "logistic_regression.pkl"

st.title("📊 Customer Churn Prediction Dashboard")
st.caption("Customer analytics powered by Logistic Regression")
# -------------------------------------------------
# MLFLOW MODEL PERFORMANCE
# -------------------------------------------------


# -------------------------------------------------
# MLFLOW MODEL PERFORMANCE
# -------------------------------------------------

st.divider()
st.header("📈 MLflow Model Performance")

st.write(
    "Evaluation results from the Logistic Regression experiment."
)

try:
    client = MlflowClient()

    experiments = client.search_experiments()

    experiment = next(
        (
            exp for exp in experiments
            if exp.name == "Customer-Churn-Prediction"
        ),
        None
    )

    if experiment is None:
        st.warning("Customer Churn experiment not found.")

    else:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1
        )

        if not runs:
            st.warning("No MLflow runs found.")

        else:
            latest_run = runs[0]
            metrics = latest_run.data.metrics

            metric_names = [
                ("Accuracy", "accuracy"),
                ("Precision", "precision"),
                ("Recall", "recall"),
                ("F1-score", "f1_score"),
                ("ROC-AUC", "roc_auc")
            ]

            # Display metric cards
            cols = st.columns(5)

            for col, (label, key) in zip(
                cols, metric_names
            ):
                value = metrics.get(key)

                if value is not None:
                    col.metric(label, f"{value:.2%}")
                else:
                    col.metric(label, "N/A")

            st.success(
                "Performance metrics loaded from MLflow successfully!"
            )

            # Performance comparison chart
            st.subheader("Model Performance Comparison")

            performance_df = pd.DataFrame({
                "Metric": [
                    label for label, key in metric_names
                ],
                "Score": [
                    metrics.get(key, 0)
                    for label, key in metric_names
                ]
            })

            fig = px.bar(
                performance_df,
                x="Metric",
                y="Score",
                text=performance_df["Score"].map(
                    lambda x: f"{x:.2%}"
                ),
                title="Logistic Regression Performance",
                range_y=[0, 1]
            )

            fig.update_traces(
                textposition="outside"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

except Exception as e:
    st.error(
        f"Could not retrieve MLflow metrics: {e}"
    )

# -----------------------------------------
# MODEL PERFORMANCE COMPARISON
# -----------------------------------------

st.subheader("📊 Model Performance Comparison")

original_metrics = {
    "Accuracy": 0.7320,
    "Precision": 0.7428,
    "Recall": 0.7428,
    "F1 Score": 0.7428,
    "ROC AUC": 0.8027
}

tuned_df = pd.read_csv(
    ROOT / "reports" / "tuned_model_performance.csv"
)

tuned_metrics = {
    "Accuracy": tuned_df.loc[0, "Accuracy"],
    "Precision": tuned_df.loc[0, "Precision"],
    "Recall": tuned_df.loc[0, "Recall"],
    "F1 Score": tuned_df.loc[0, "F1 Score"],
    "ROC AUC": tuned_df.loc[0, "ROC AUC"]
}

comparison_df = pd.DataFrame({
    "Metric": list(original_metrics.keys()),
    "Original Logistic Regression": list(original_metrics.values()),
    "Tuned Logistic Regression": list(tuned_metrics.values())
})

st.dataframe(
    comparison_df.style.format({
        "Original Logistic Regression": "{:.2%}",
        "Tuned Logistic Regression": "{:.2%}"
    }),
    use_container_width=True
)

chart_df = comparison_df.set_index("Metric")

st.bar_chart(chart_df)

st.info(
    "The original Logistic Regression model remains the "
    "current best model based on the comparison."
)

# -----------------------------------------
# ROC CURVE ANALYSIS
# -----------------------------------------

from sklearn.metrics import roc_curve, auc

st.subheader("📈 ROC Curve Analysis")

# Original model predictions
original_model = joblib.load(
    ROOT / "models" / "logistic_regression.pkl"
)
# Load test datasets
X_test = pd.read_csv(
    ROOT / "data" / "processed" / "X_test.csv"
)

y_test = pd.read_csv(
    ROOT / "data" / "processed" / "y_test.csv"
).squeeze("columns")

st.success("Test dataset loaded successfully!")
y_probability_original = original_model.predict_proba(X_test)[:, 1]

# Tuned model predictions
# Tuned model predictions
# Load improved model with tuned threshold
model_package = joblib.load(
    ROOT / "models" / "churn_threshold_model.pkl"
)

tuned_model = model_package["model"]
selected_threshold = float(model_package["threshold"])
feature_names = model_package["feature_names"]

# Get tuned model probabilities
y_probability_tuned = tuned_model.predict_proba(X_test)[:, 1]

# Convert probabilities into churn predictions
y_pred_tuned = (
    y_probability_tuned >= selected_threshold
).astype(int)

st.success(
    f"Tuned model loaded successfully! Threshold: {selected_threshold}"
)

# Calculate ROC curves
fpr_original, tpr_original, _ = roc_curve(
    y_test, y_probability_original
)

fpr_tuned, tpr_tuned, _ = roc_curve(
    y_test, y_probability_tuned
)

auc_original = auc(fpr_original, tpr_original)
auc_tuned = auc(fpr_tuned, tpr_tuned)

# Create chart
roc_df = pd.DataFrame({
    "False Positive Rate - Original": fpr_original,
    "True Positive Rate - Original": tpr_original
})

st.line_chart(
    roc_df,
    x="False Positive Rate - Original",
    y="True Positive Rate - Original"
)

st.write(f"Original Model ROC-AUC: {auc_original:.2%}")
st.write(f"Tuned Model ROC-AUC: {auc_tuned:.2%}")

# -------------------------------------------------
# 2. LOAD DATA AND MODEL
# -------------------------------------------------


@st.cache_data
def load_data():
    X_train = pd.read_csv(DATA_PATH / "X_train.csv")
    X_test = pd.read_csv(DATA_PATH / "X_test.csv")
    y_train = pd.read_csv(DATA_PATH / "y_train.csv").squeeze()
    y_test = pd.read_csv(DATA_PATH / "y_test.csv").squeeze()

    X = pd.concat([X_train, X_test], ignore_index=True)
    y = pd.concat([y_train, y_test], ignore_index=True)

    return X, y


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    X, y = load_data()
    model = load_model()
    # Load actual test data for model evaluation
    X_test = pd.read_csv(DATA_PATH / "X_test.csv")
    y_test = pd.read_csv(DATA_PATH / "y_test.csv").squeeze()
except Exception as e:
    st.error(f"Could not load project files: {e}")
    st.stop()

# -------------------------------------------------
# MODEL EVALUATION: CONFUSION MATRIX
# -------------------------------------------------

st.divider()
st.header("🎯 Confusion Matrix")

st.write(
    "Actual customer status compared with the model's predictions."
)

try:
    y_pred = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Stayed", "Actual Churned"],
        columns=["Predicted Stayed", "Predicted Churned"]
    )

    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        title="Logistic Regression Confusion Matrix",
        labels={
            "x": "Predicted Status",
            "y": "Actual Status",
            "color": "Number of Customers"
        },
        color_continuous_scale="Blues"
    )

    st.plotly_chart(
        fig_cm,
        use_container_width=True
    )

    # Classification report
    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Stayed", "Churned"],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )

except Exception as e:
    st.error(f"Model evaluation error: {e}")

# -------------------------------------------------
# 3. CUSTOMER OVERVIEW
# -------------------------------------------------

# -----------------------------------------
# CUSTOMER CHURN RISK KPIs
# -----------------------------------------

st.subheader("Customer Churn Risk Overview")

# Predict churn probability for test customers
churn_probabilities = tuned_model.predict_proba(X_test)[:, 1]

# Apply selected threshold
churn_predictions = (
    churn_probabilities >= selected_threshold
).astype(int)

# Calculate KPI values
total_customers = len(churn_predictions)

predicted_churn = int(churn_predictions.sum())

predicted_stay = total_customers - predicted_churn

churn_rate = (
    predicted_churn / total_customers * 100
    if total_customers > 0 else 0
)

# Display KPI cards
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Test Customers",
    f"{total_customers:,}"
)

col2.metric(
    "Predicted Churn",
    f"{predicted_churn:,}"
)

col3.metric(
    "Predicted Stay",
    f"{predicted_stay:,}"
)

col4.metric(
    "Predicted Churn Rate",
    f"{churn_rate:.2f}%"
)

# -----------------------------------------
# CHURN RISK DISTRIBUTION
# -----------------------------------------

st.subheader("Customer Churn Risk Distribution")

# Categorize customers by predicted churn probability
risk_levels = pd.cut(
    churn_probabilities,
    bins=[-0.001, 0.30, 0.60, 1.0],
    labels=["Low Risk", "Medium Risk", "High Risk"]
)

risk_counts = (
    risk_levels.value_counts()
    .reindex(["Low Risk", "Medium Risk", "High Risk"], fill_value=0)
)

# Create chart data
risk_df = pd.DataFrame({
    "Risk Level": risk_counts.index,
    "Number of Customers": risk_counts.values
}).set_index("Risk Level")

# Display chart
st.bar_chart(risk_df)

st.caption(
    "Risk bands: Low < 30%, Medium 30–60%, High > 60%. "
    "Based on predicted probabilities for test customers."
)

total_customers = len(X)
total_churn = int((y == 1).sum())
churn_rate = total_churn / total_customers if total_customers else 0

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Customers Churned", f"{total_churn:,}")
col3.metric("Churn Rate", f"{churn_rate:.2%}")

st.divider()

# -------------------------------------------------
# 4. CHURN DISTRIBUTION
# -------------------------------------------------

st.subheader("Customer Churn Distribution")

churn_df = pd.DataFrame({
    "Customer Status": y.map({
        0: "Stayed",
        1: "Churned"
    }).fillna(y.astype(str))
})

fig = px.pie(
    churn_df,
    names="Customer Status",
    title="Stayed vs Churned Customers",
    hole=0.45
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# 5. FEATURE ANALYSIS
# -------------------------------------------------

st.subheader("Customer Feature Analysis")

numeric_columns = X.select_dtypes(
    include="number"
).columns.tolist()

if numeric_columns:
    feature = st.selectbox(
        "Select a customer feature",
        numeric_columns
    )

    fig = px.histogram(
        X,
        x=feature,
        title=f"Distribution of {feature}"
    )

    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# 6. INDIVIDUAL CUSTOMER PREDICTION
# -------------------------------------------------

st.divider()
st.subheader("🔮 Predict Individual Customer Churn")

st.write("Enter customer details to estimate churn risk.")

# Use training data's original feature order
feature_names = list(X.columns)

with st.form("customer_prediction_form"):
    customer_values = {}

    for feature in feature_names:
        if pd.api.types.is_numeric_dtype(X[feature]):
            default_value = float(X[feature].median())

            customer_values[feature] = st.number_input(
                feature,
                value=default_value
            )
        else:
            options = X[feature].dropna().unique().tolist()

            customer_values[feature] = st.selectbox(
                feature,
                options=options
            )

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    input_df = pd.DataFrame(
        [customer_values],
        columns=feature_names
    )

    try:
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        classes = list(model.classes_)
        positive_index = classes.index(1)
        churn_probability = probabilities[positive_index]

        if prediction == 1:
            st.error("⚠️ Customer is predicted to churn")
        else:
            st.success("✅ Customer is predicted to stay")

        st.metric(
            "Estimated Churn Probability",
            f"{churn_probability:.2%}"
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")

# -----------------------------------------
# INDIVIDUAL CUSTOMER CHURN PREDICTION
# -----------------------------------------

st.subheader("Individual Customer Churn Prediction")

st.write("Select a test customer to check their churn risk.")

customer_index = st.number_input(
    "Select Customer Index",
    min_value=0,
    max_value=len(X_test) - 1,
    value=0,
    step=1
)

if st.button("Predict Customer Churn"):

    customer_data = X_test.iloc[[int(customer_index)]]

    customer_probability = tuned_model.predict_proba(
        customer_data
    )[0][class_index]

    customer_prediction = (
        customer_probability >= selected_threshold
    )

    st.metric(
        "Churn Probability",
        f"{customer_probability * 100:.2f}%"
    )

    if customer_prediction:
        st.error("High Risk: Customer is predicted to churn.")
    else:
        st.success("Low Risk: Customer is predicted to stay.")

# -----------------------------------------
# CONFUSION MATRIX
# -----------------------------------------

st.subheader("Model Evaluation - Confusion Matrix")

# Compare actual and predicted test labels
cm = confusion_matrix(
    y_test,
    churn_predictions,
    labels=[0, 1]
)

confusion_df = pd.DataFrame(
    cm,
    index=["Actual Stayed", "Actual Churned"],
    columns=["Predicted Stayed", "Predicted Churned"]
)

st.write("Confusion Matrix")
st.dataframe(confusion_df)

# Display interpretation
tn, fp, fn, tp = cm.ravel()

col1, col2 = st.columns(2)

col1.metric("Correctly Predicted", int(tn + tp))
col2.metric("Incorrectly Predicted", int(fp + fn))

st.write(f"True Negatives: {tn}")
st.write(f"False Positives: {fp}")
st.write(f"False Negatives: {fn}")
st.write(f"True Positives: {tp}")

# -----------------------------------------
# FEATURE IMPORTANCE
# -----------------------------------------

st.subheader("Customer Churn Feature Importance")

# Get Logistic Regression coefficients
coefficients = tuned_model.coef_[0]

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

# Sort by absolute coefficient
feature_importance["Importance"] = (
    feature_importance["Coefficient"].abs()
)

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

# Display top 10 features
top_features = feature_importance.head(10)

st.write("Top 10 features influencing churn predictions")

st.bar_chart(
    top_features.set_index("Feature")["Coefficient"]
)

st.dataframe(
    top_features[["Feature", "Coefficient"]]
)
