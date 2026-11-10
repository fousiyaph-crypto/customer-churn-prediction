
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Locate the dataset
project_root = Path(__file__).resolve().parents[1]
file_path = project_root / "data" / "customer_churn.csv"

# Load dataset
df = pd.read_csv(file_path)

# 1. Display first five rows
print("\nFIRST FIVE CUSTOMERS")
print(df.head())

# 2. Dataset shape
print("\nDATASET SHAPE")
print(df.shape)

# 3. Dataset information
print("\nDATASET INFORMATION")
df.info()

# 4. Statistical summary
print("\nSTATISTICAL SUMMARY")
print(df.describe())

# 5. Missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# 6. Churn distribution
print("\nCHURN DISTRIBUTION")
print(df["Churn"].value_counts())

print("\nCHURN PERCENTAGE")
print(df["Churn"].value_counts(normalize=True) * 100)

# 7. Churn visualization
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.tight_layout()

# Save chart
output_path = project_root / "reports" / "churn_distribution.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path)
plt.show()

print("\nEDA completed successfully!")
