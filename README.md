# Customer Churn Prediction

A Machine Learning project that predicts customer churn using Logistic Regression and presents customer risk insights through an interactive Streamlit dashboard.

## Project Overview

This project aims to identify customers who are likely to leave a service. It includes model evaluation, churn risk analysis, individual customer predictions and an interactive dashboard.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Logistic Regression
- MLflow
- Streamlit
- Matplotlib / data visualization tools

## Key Features

- Customer churn prediction
- Customer churn risk distribution
- Individual customer prediction
- Model performance evaluation
- Confusion matrix
- Feature importance analysis
- Interactive Streamlit dashboard
- MLflow experiment tracking

## Model Evaluation

Baseline Logistic Regression metrics recorded through MLflow:

| Metric | Result |
|---|---:|
| Accuracy | 73.20% |
| Precision | 74.28% |
| Recall | 74.28% |
| F1-score | 74.28% |
| ROC-AUC | 80.27% |

Threshold-adjusted test evaluation:

| Metric | Result |
|---|---:|
| Test customers | 1,000 |
| Accuracy | 71.80% |
| True Positives | 430 |
| True Negatives | 288 |
| False Positives | 191 |
| False Negatives | 91 |

*These are separate evaluation results: the baseline MLflow metrics and threshold-adjusted confusion matrix should not be treated as the same model evaluation.*

## Project Structure

```text
customer-churn-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── reports/
│
├── src/
│   ├── eda.py
│   ├── preprocessing.py
│   ├── train.py
│   └── threshold_tuning.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit dashboard:

```bash
streamlit run app.py
```

## Dashboard

The dashboard provides customer churn KPIs, churn risk distribution, individual customer predictions, confusion matrix and feature importance.

## Future Improvements

- Improve model performance through additional experiments.
- Add more customer segmentation insights.
- Deploy the dashboard online.
- Explore additional classification algorithms.

## Author

Developed as a Machine Learning and Data Science portfolio project.
