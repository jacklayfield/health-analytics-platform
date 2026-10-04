# Health Analytics ML Pipeline

A production-ready machine learning pipeline for health analytics, featuring comprehensive experiment tracking, model evaluation, hyperparameter optimization, and model serving capabilities.

## Features

### Core ML Pipeline

* **Configurable Training**: YAML-based configuration for easy customization
* **Multiple Algorithms**: Support for Logistic Regression, Random Forest, Gradient Boosting, and more
* **Data Validation**: Comprehensive data quality checks and validation
* **Feature Engineering**: Automated preprocessing with categorical encoding and scaling

### Experiment Tracking & MLOps

* **MLflow Integration**: Complete experiment tracking, model versioning, and registry
* **Hyperparameter Optimization**: Automated tuning using Optuna
* **Model Evaluation**: Comprehensive metrics, cross-validation, and comparison
* **Model Serving**: FastAPI-based REST API for model inference

### Production Features

* **Model Monitoring**: Built-in monitoring and drift detection capabilities
* **Testing Framework**: Comprehensive unit and integration tests
* **CLI Interface**: Command-line tools for all pipeline operations
* **Docker Support**: Containerized deployment ready
* **CI/CD Ready**: GitHub Actions and pre-commit hooks

## Project Structure

```
ml/
├── config/
│   └── config.yaml              # Main configuration file
├── src/
│   ├── config/
│   │   └── config_manager.py    # Configuration management
│   ├── data/
│   │   └── data_loader.py       # Enhanced data loading
│   ├── experiments/
│   │   └── experiment_tracker.py # MLflow integration
│   ├── evaluation/
│   │   └── model_evaluator.py   # Model evaluation framework
│   ├── optimization/
│   │   └── hyperparameter_tuner.py # Hyperparameter optimization
│   ├── training/
│   │   └── trainer.py           # Main training pipeline
│   ├── serving/
│   │   └── model_server.py      # Model serving API
│   ├── utils/
│   │   └── logger.py            # Logging utilities
│   └── cli/
│       └── main.py              # Command-line interface
├── tests/
│   ├── test_data_loader.py      # Data loading tests
│   └── test_model_evaluator.py  # Evaluation tests
├── pipelines/
│   ├── common/
│   │   ├── data_utils.py        # Legacy data utilities
│   │   └── preprocessing.py     # Legacy preprocessing
│   ├── serious_prediction/      # Serious event prediction
│   └── reaction_prediction/     # Reaction prediction
├── models/                      # Trained model artifacts
├── logs/                        # Log files
├── artifacts/                   # Evaluation artifacts
└── requirements.txt             # Dependencies
```

## Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd health-analytics-platform/ml
   ```

2. **Create virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:

   ```bash
   export WAREHOUSE_DB_URI="postgresql+psycopg2://user:pass@localhost:5432/db"
   export MLFLOW_TRACKING_URI="http://localhost:5000"
   ```

## Quick Start

### 1. Configuration

The pipeline uses YAML configuration for easy customization. Initialize a config file:

```bash
python -m src.cli.main config init config/my_config.yaml
```

### 2. Train Models

Train all models for serious event prediction:

```bash
python -m src.cli.main train serious_prediction
```

Training is dataset-first. OpenFDA is the default dataset, and a dataset can be
selected explicitly when more dataset pipelines are enabled:

```bash
python -m src.cli.main train serious_prediction --dataset openfda
```

Dataset metadata is defined in `config/config.yaml`, including the warehouse
table, schema version, entity grain, and supported tasks. Synthea is registered
there, but its task list remains empty until its patient/encounter feature view
is implemented.

Train with hyperparameter optimization:

```bash
python -m src.cli.main train serious_prediction --optimize
```

Train with data filters:

```bash
python -m src.cli.main train serious_prediction --filters "serious=1,patientsex=M"
```

### 3. Start Model Server

```bash
python -m src.cli.main serve --port 8080
```

### 4. Make Predictions

```bash
curl -X POST "http://localhost:8080/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "features": {
         "patientonsetage": 35,
         "patientsex": "F",
         "reaction": "nausea",
         "brand_name": "drug_a"
       }
     }'
```

## MLflow Integration

### Start MLflow Server

The Compose stack runs MLflow with PostgreSQL metadata and a persistent local
artifact volume. From the repository's `ml/` directory:

```bash
docker compose up --build -d mlflow-db mlflow
```

The `mlflow_db_data` and `mlflow_data` Docker volumes survive container
recreation. Back them up before treating this as production storage.

By default, artifacts stay on the MLflow volume, so no S3 account is needed.
To switch to an S3-compatible store later, set `MLFLOW_ARTIFACTS_DESTINATION`
to an `s3://<bucket>` URI and configure `MLFLOW_S3_ENDPOINT_URL` and AWS
credentials in the MLflow environment. The MLflow image includes the S3 client.

### Register and Serve a Model

Training registers each evaluated model under a name of the form
`health_models_<dataset>_<task>_<algorithm>` and assigns its latest version the
`candidate` alias. For example:

```text
models:/health_models_openfda_serious_prediction_random_forest@candidate
```

Candidates are not automatically promoted to `champion`. After reviewing a
version in the MLflow UI at [http://localhost:5000](http://localhost:5000),
promote it explicitly:

```python
from mlflow import MlflowClient

client = MlflowClient(tracking_uri="http://localhost:5000")
client.set_registered_model_alias(
  name="health_models_openfda_serious_prediction_random_forest",
  alias="champion",
  version="1",
)
```

Set `MLFLOW_MODEL_URI` in `ml/.env` to the chosen alias, then start or reload the
API service:

```text
MLFLOW_MODEL_URI=models:/health_models_openfda_serious_prediction_random_forest@champion
```

The API resolves the alias to a concrete version at startup and reports that
version with predictions. After changing the alias, reload the API model or
recreate the `ml-api` container to pick up the promoted version.

## Configuration

### Main Configuration (`config/config.yaml`)

```yaml
data:
  warehouse_uri: "${WAREHOUSE_DB_URI}"
  tables:
    openfda_events: "openfda_events"

features:
  serious_prediction:
    target: "serious"
    numeric_features: ["patientonsetage"]
    categorical_features: ["patientsex", "reaction", "brand_name"]

models:
  serious_prediction:
    algorithms:
      - name: "logistic_regression"
        class: "sklearn.linear_model.LogisticRegression"
        params:
          max_iter: 1000
          random_state: 42

training:
  test_size: 0.2
  random_state: 42
  cross_validation:
    cv_folds: 5
    scoring: ["accuracy", "f1_macro"]

mlflow:
  tracking_uri: "${MLFLOW_TRACKING_URI}"
  experiment_name: "health_analytics"
```

## Testing

Run the test suite:

```bash
pytest
pytest --cov=src
pytest tests/test_data_loader.py
```

## Model Evaluation

* **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
* **Cross-Validation**: K-fold validation with multiple metrics
* **Visualization**: Confusion matrices, ROC curves, feature importance
* **Comparison**: Side-by-side model performance comparison

## Hyperparameter Optimization

Automated hyperparameter tuning using Optuna:

```python
from src.optimization.hyperparameter_tuner import HyperparameterTuner, PARAM_SPACES

tuner = HyperparameterTuner(n_trials=100)

results = tuner.optimize_classification(
    model_class=RandomForestClassifier,
    X=X_train,
    y=y_train,
    param_space=PARAM_SPACES['RandomForestClassifier']
)

print(f"Best parameters: {results['best_params']}")
print(f"Best score: {results['best_score']}")
```

## Model Serving

### API Endpoints

* `GET /health` - Health check
* `GET /models` - List available models
* `POST /predict` - Single prediction
* `POST /predict_batch` - Batch predictions
* `POST /models/{name}/load` - Load specific model
* `POST /models/reload` - Reload all models

### Example Usage

```python
import requests

response = requests.post("http://localhost:8000/predict", json={
    "features": {
        "patientonsetage": 35,
        "patientsex": "F",
        "reaction": "nausea",
        "brand_name": "drug_a"
    },
    "return_probabilities": True
})

result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Probabilities: {result['probabilities']}")
```

## Docker Deployment

### Build Image

```bash
docker build -t health-ml-pipeline .
```

### Run Container

```bash
docker run -p 8000:8000 \
  -e WAREHOUSE_DB_URI="postgresql://..." \
  -e MLFLOW_TRACKING_URI="http://mlflow:5000" \
  health-ml-pipeline
```

## Development

### Code Quality

```bash
black src/ tests/
flake8 src/ tests/
mypy src/
pre-commit install
pre-commit run --all-files
```

### Adding New Models

1. Update configuration (`config/config.yaml`)
2. Add parameter space if using hyperparameter optimization
3. Train the model