# Train baseline ML model on OpenFDA data to predict whether an adverse event is serious.

import os
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
import joblib

DB_URI = os.getenv(
    "WAREHOUSE_DB_URI",
    "postgresql+psycopg2://airflow:airflow@localhost:5432/airflow"
)

engine = create_engine(DB_URI)
query = "SELECT * FROM openfda_events;"
print("Loading OpenFDA data from warehouse...")
df = pd.read_sql(query, engine)
print(f"Fetched {len(df)} rows")

# Feature selection / preprocessing
# Target variable: serious (0/1)
df = df.dropna(subset=["serious"])
df["serious"] = df["serious"].astype(int)

features = ["patientonsetage", "patientsex", "reaction", "brand_name"]
X = df[features]
y = df["serious"]

numeric = ["patientonsetage"]
categorical = ["patientsex", "reaction", "brand_name"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical),
    ]
)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

print("Training model...")
model.fit(X_train, y_train)

# Eval
y_pred = model.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Artifact
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/openfda_serious_predictor.joblib")
print("Model saved to models/openfda_serious_predictor.joblib")
