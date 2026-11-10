import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "customer_churn.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 1. Load dataset
df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)

# 2. Remove customer ID
df = df.drop(columns=["customerID"])

# 3. Separate features and target
X = df.drop(columns=["Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# 4. Identify column types
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical columns:", numeric_features)
print("\nCategorical columns:", categorical_features)

# 5. Create preprocessing pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        drop="first"
    ))
])

# 6. Combine preprocessing
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# 7. Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 8. Fit only on training data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# 9. Convert processed features into DataFrames
feature_names = preprocessor.get_feature_names_out()

X_train_processed = pd.DataFrame(
    X_train_processed,
    columns=feature_names
)

X_test_processed = pd.DataFrame(
    X_test_processed,
    columns=feature_names
)

# 10. Save processed datasets
X_train_processed.to_csv(
    OUTPUT_PATH / "X_train.csv", index=False
)

X_test_processed.to_csv(
    OUTPUT_PATH / "X_test.csv", index=False
)

y_train.to_csv(
    OUTPUT_PATH / "y_train.csv", index=False
)

y_test.to_csv(
    OUTPUT_PATH / "y_test.csv", index=False
)

# Save fitted preprocessing object
joblib.dump(
    preprocessor,
    MODEL_PATH / "preprocessor.pkl"
)

# 11. Display results
print("\nDATA PREPROCESSING COMPLETED!")
print("--------------------------------")
print("Training samples:", X_train_processed.shape[0])
print("Testing samples:", X_test_processed.shape[0])
print("Processed features:", X_train_processed.shape[1])
print("\nTraining churn distribution:")
print(y_train.value_counts())
print("\nTesting churn distribution:")
print(y_test.value_counts())

print("\nFiles saved successfully!")