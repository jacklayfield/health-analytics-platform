# Train a Gradient Booster classifier to predict whether an adverse event is serious (2) or not (1)

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
import joblib

from pipelines.common.data_utils import load_openfda_events
from pipelines.common.preprocessing import make_preprocessor

def main():
    df = load_openfda_events()
    print(f"Fetched {len(df)} rows from warehouse")

    df = df.dropna(subset=["serious"])
    df["serious"] = df["serious"].astype(int)

    features = ["patientonsetage", "patientsex", "reaction", "brand_name"]
    X = df[features]
    y = df["serious"]

    numeric = ["patientonsetage"]
    categorical = ["patientsex", "reaction", "brand_name"]
    preprocessor = make_preprocessor(numeric, categorical)

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Gradient Boosting...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    Path("models").mkdir(exist_ok=True)
    out_path = Path("models/openfda_serious_gb.joblib")
    joblib.dump(model, out_path)
    print(f"Model saved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
