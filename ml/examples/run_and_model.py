import mlflow
import mlflow.sklearn

from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

mlflow.set_experiment("diabetes_prediction")

db = load_diabetes()

X_train, X_test, y_train, y_test = train_test_split(
    db.data, db.target, test_size=0.2, random_state=42
)

with mlflow.start_run():

    model = RandomForestRegressor(
        n_estimators=100, max_depth=6, max_features=3, random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)

    mlflow.sklearn.log_model(model, "model", registered_model_name="diabetes_model")
