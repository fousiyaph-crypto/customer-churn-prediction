
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
df = pd.read_csv(project_root / "data" / "customer_churn.csv")

# Churn by contract
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Contract", hue="Churn")
plt.title("Customer Churn by Contract Type")
plt.tight_layout()
plt.savefig(project_root / "reports" / "churn_by_contract.png")
plt.show()

# Churn by tenure
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Churn", y="tenure")
plt.title("Customer Churn by Tenure")
plt.tight_layout()
plt.savefig(project_root / "reports" / "churn_by_tenure.png")
plt.show()

# Average monthly charges
print("\nAverage Monthly Charges by Churn")
print(df.groupby("Churn")["MonthlyCharges"].mean())

# Churn rate by contract
print("\nChurn Rate by Contract")
print(
    df.groupby("Contract")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
)

print("\nCustomer analysis completed!")
