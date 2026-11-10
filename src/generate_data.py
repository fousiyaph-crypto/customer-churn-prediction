
import pandas as pd
import numpy as np
from pathlib import Path

# Reproducibility
np.random.seed(42)

# Number of customers
n = 5000

# Generate customer information
data = {
    "customerID": [f"CUST{i:05d}" for i in range(1, n + 1)],
    "gender": np.random.choice(["Male", "Female"], n),
    "SeniorCitizen": np.random.choice([0, 1], n, p=[0.84, 0.16]),
    "Partner": np.random.choice(["Yes", "No"], n),
    "Dependents": np.random.choice(["Yes", "No"], n, p=[0.3, 0.7]),
    "tenure": np.random.randint(1, 73, n),
    "PhoneService": np.random.choice(["Yes", "No"], n, p=[0.9, 0.1]),
    "InternetService": np.random.choice(
        ["DSL", "Fiber optic", "No"], n,
        p=[0.35, 0.45, 0.20]
    ),
    "Contract": np.random.choice(
        ["Month-to-month", "One year", "Two year"],
        n, p=[0.55, 0.25, 0.20]
    ),
    "PaperlessBilling": np.random.choice(["Yes", "No"], n),
    "PaymentMethod": np.random.choice(
        ["Electronic check", "Mailed check",
         "Bank transfer", "Credit card"], n
    ),
    "MonthlyCharges": np.round(
        np.random.uniform(20, 120, n), 2
    )
}

df = pd.DataFrame(data)

# Calculate total charges
df["TotalCharges"] = (
    df["MonthlyCharges"] * df["tenure"]
).round(2)

# Generate churn probability using customer characteristics
logit = (
    1.5
    - 0.055 * df["tenure"]
    + 0.9 * (df["Contract"] == "Month-to-month").astype(int)
    - 0.9 * (df["Contract"] == "Two year").astype(int)
    + 0.35 * (df["InternetService"] == "Fiber optic").astype(int)
    + 0.4 * (df["PaymentMethod"] == "Electronic check").astype(int)
    + 0.25 * (df["SeniorCitizen"] == 1).astype(int)
)

probability = 1 / (1 + np.exp(-logit))

# Generate target variable
df["Churn"] = np.where(
    np.random.random(n) < probability,
    "Yes",
    "No"
)

# Save dataset in the data folder
project_root = Path(__file__).resolve().parents[1]
output_path = project_root / "data" / "customer_churn.csv"

output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

# Display dataset information
print("Customer Churn Dataset Created Successfully!")
print("-" * 45)
print("Total customers:", len(df))
print("Total columns:", len(df.columns))
print("\nChurn distribution:")
print(df["Churn"].value_counts())
print("\nDataset preview:")
print(df.head())
print("\nSaved location:", output_path)
