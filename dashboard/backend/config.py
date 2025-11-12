import os
from pathlib import Path
import yaml

ML_CONFIG_PATH = Path("/app/ml/config/config.yaml")

def load_ml_config():
    with open(ML_CONFIG_PATH, "r") as fh:
        return yaml.safe_load(fh)

ML_CONFIG = load_ml_config()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", ML_CONFIG.get("mlflow", {}).get("tracking_uri"))
MODEL_REGISTRY = os.getenv("MODEL_REGISTRY", ML_CONFIG.get("ml", {}).get("model_registry_name", "health_models"))

POSTGRES_URI = os.getenv("POSTGRES_URI", ML_CONFIG.get("data", {}).get("warehouse_uri", "postgresql://user:pass@localhost:5432/db"))

AIRFLOW_BASE = os.getenv("AIRFLOW_BASE", "http://localhost:8080/api/v1")
AIRFLOW_USERNAME = os.getenv("AIRFLOW_USERNAME")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD")
