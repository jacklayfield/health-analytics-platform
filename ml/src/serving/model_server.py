"""
Model serving infrastructure using FastAPI.
"""

import joblib
import mlflow
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime
import os
import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from mlflow.tracking import MlflowClient

from ..config.config_manager import config_manager
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PredictionRequest(BaseModel):
    """Request model for predictions."""

    features: Dict[str, Any] = Field(..., description="Feature values for prediction")
    model_name: Optional[str] = Field(None, description="Specific model to use")
    return_probabilities: bool = Field(
        False, description="Whether to return prediction probabilities"
    )


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    prediction: Union[int, float, str] = Field(..., description="Model prediction")
    probabilities: Optional[Dict[str, float]] = Field(
        None, description="Prediction probabilities"
    )
    model_name: str = Field(..., description="Name of the model used")
    model_version: Optional[str] = Field(None, description="Registered model version")
    timestamp: str = Field(..., description="Prediction timestamp")
    confidence: Optional[float] = Field(None, description="Prediction confidence")


class ModelInfo(BaseModel):
    """Model information response."""

    name: str
    model_version: Optional[str] = None
    task_type: str
    algorithm: str
    training_date: Optional[str]
    accuracy: Optional[float]
    features: List[str]
    is_loaded: bool


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: str
    models_loaded: int
    total_models: int


class ModelServer:
    """FastAPI-based model serving server."""

    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize model server.

        Args:
            models_dir: Directory containing trained models
        """
        self.app = FastAPI(
            title="Health Analytics ML API",
            description="Machine Learning API for Health Analytics Platform",
            version="1.0.0",
        )

        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Model storage
        if models_dir is None:
            config = config_manager.load_config()
            models_dir = config.paths.models

        self.models_dir = Path(models_dir)
        self.models: Dict[str, Any] = {}
        self.model_info: Dict[str, Dict[str, Any]] = {}
        self.model_uri = os.getenv("MLFLOW_MODEL_URI")
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(self.tracking_uri)

        # Setup routes
        self._setup_routes()

        # Load models on startup
        self._load_models()

    def _setup_routes(self):
        """Setup API routes."""

        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            return HealthResponse(
                status="healthy",
                timestamp=datetime.now().isoformat(),
                models_loaded=len(self.models),
                total_models=len(self.model_info),
            )

        @self.app.get("/models", response_model=List[ModelInfo])
        async def list_models():
            """List available models."""
            model_list = []
            for name, info in self.model_info.items():
                model_list.append(
                    ModelInfo(
                        name=name,
                        model_version=info.get("model_version"),
                        task_type=info.get("task_type", "unknown"),
                        algorithm=info.get("algorithm", "unknown"),
                        training_date=info.get("training_date"),
                        accuracy=info.get("accuracy"),
                        features=info.get("features", []),
                        is_loaded=name in self.models,
                    )
                )
            return model_list

        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest):
            """Make predictions using loaded models."""
            try:
                # Determine which model to use
                if request.model_name:
                    if request.model_name not in self.models:
                        raise HTTPException(
                            status_code=404,
                            detail=(
                                f"Model '{request.model_name}' not found or not loaded"
                            ),
                        )
                    model_name = request.model_name
                else:
                    # Use the first available model
                    if not self.models:
                        raise HTTPException(
                            status_code=503, detail="No models are currently loaded"
                        )
                    model_name = list(self.models.keys())[0]

                # Get model
                model = self.models[model_name]

                # Prepare features
                features_df = pd.DataFrame([request.features])

                # Make prediction
                prediction = model.predict(features_df)[0]

                # Get probabilities if requested and available
                probabilities = None
                confidence = None

                if request.return_probabilities and hasattr(model, "predict_proba"):
                    proba = model.predict_proba(features_df)[0]
                    class_names = (
                        model.classes_
                        if hasattr(model, "classes_")
                        else range(len(proba))
                    )
                    probabilities = {
                        str(cls): float(prob) for cls, prob in zip(class_names, proba)
                    }
                    confidence = float(max(proba))

                return PredictionResponse(
                    prediction=(
                        float(prediction)
                        if isinstance(prediction, (int, float))
                        else str(prediction)
                    ),
                    probabilities=probabilities,
                    model_name=self.model_info[model_name].get("registry_name", model_name),
                    model_version=self.model_info[model_name].get("model_version"),
                    timestamp=datetime.now().isoformat(),
                    confidence=confidence,
                )

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/predict_batch")
        async def predict_batch(requests: List[PredictionRequest]):
            """Make batch predictions."""
            try:
                results = []
                for request in requests:
                    response = await predict(request)
                    results.append(response)
                return results
            except Exception as e:
                logger.error(f"Batch prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/models/{model_name}/load")
        async def load_model(model_name: str):
            """Load a specific model."""
            try:
                self._load_model(model_name)
                return {"message": f"Model '{model_name}' loaded successfully"}
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/models/{model_name}/unload")
        async def unload_model(model_name: str):
            """Unload a specific model."""
            if model_name in self.models:
                del self.models[model_name]
                return {"message": f"Model '{model_name}' unloaded successfully"}
            else:
                raise HTTPException(
                    status_code=404, detail=f"Model '{model_name}' not found"
                )

        @self.app.post("/models/reload")
        async def reload_models():
            """Reload all models."""
            try:
                self._load_models()
                return {"message": "All models reloaded successfully"}
            except Exception as e:
                logger.error(f"Failed to reload models: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    def _load_models(self):
        """Load all available models."""
        if self.model_uri:
            self._load_registered_model(self.model_uri)
            return

        if not self.models_dir.exists():
            logger.warning(f"Models directory does not exist: {self.models_dir}")
            return

        model_files = list(self.models_dir.glob("*.joblib"))

        for model_file in model_files:
            model_name = model_file.stem
            try:
                self._load_model(model_name)
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")

    def _load_registered_model(self, model_uri: str) -> None:
        """Load a specific registered MLflow model alias or version."""
        match = re.fullmatch(r"models:/(.+?)(?:@([^/]+)|/(\d+))", model_uri)
        if match is None:
            raise ValueError(
                "MLFLOW_MODEL_URI must use models:/<name>@<alias> "
                "or models:/<name>/<version>"
            )

        registry_name, alias, explicit_version = match.groups()
        client = MlflowClient(tracking_uri=self.tracking_uri)
        if alias:
            version = client.get_model_version_by_alias(registry_name, alias).version
            model_key = f"{registry_name}@{alias}"
        else:
            version = explicit_version
            model_key = f"{registry_name}@{version}"

        versioned_model_uri = f"models:/{registry_name}/{version}"
        model = self._load_mlflow_model(versioned_model_uri)
        self.models[model_key] = model
        self.model_info[model_key] = {
            "registry_name": registry_name,
            "model_version": str(version),
            "task_type": "unknown",
            "algorithm": "MLflow model",
            "training_date": None,
            "features": [],
        }

    def _load_mlflow_model(self, model_uri: str):
        """Load a model artifact through MLflow."""
        return mlflow.pyfunc.load_model(model_uri)

    def _load_model(self, model_name: str):
        """Load a specific model."""
        model_path = self.models_dir / f"{model_name}.joblib"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load model
        model = joblib.load(model_path)
        self.models[model_name] = model

        # Store model info
        self.model_info[model_name] = {
            "task_type": "classification",  # Could be inferred from model type
            "algorithm": type(model.named_steps["classifier"]).__name__,
            "training_date": datetime.fromtimestamp(
                model_path.stat().st_mtime
            ).isoformat(),
            "features": self._get_model_features(model),
            "is_loaded": True,
        }

        logger.info(f"Model '{model_name}' loaded successfully")

    def _get_model_features(self, model) -> List[str]:
        """Extract feature names from model."""
        try:
            # Try to get feature names from preprocessor
            if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
                preprocessor = model.named_steps["preprocessor"]
                if hasattr(preprocessor, "get_feature_names_out"):
                    return preprocessor.get_feature_names_out().tolist()

            # Fallback: return generic feature names
            return [f"feature_{i}" for i in range(10)]  # Default number

        except Exception as e:
            logger.warning(f"Could not extract feature names: {e}")
            return []

    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """Run the model server."""
        logger.info(f"Starting model server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port, **kwargs)


def create_app(models_dir: Optional[str] = None) -> FastAPI:
    """Create FastAPI app for model serving."""
    server = ModelServer(models_dir)
    return server.app


if __name__ == "__main__":
    # Create and run server
    server = ModelServer()
    server.run(host="0.0.0.0", port=8000)
