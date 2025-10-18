"""
MLflow experiment tracking and model management.
"""

import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json

from ..config.config_manager import config_manager, MLflowConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ExperimentTracker:
    """MLflow experiment tracking and model management."""
    
    def __init__(self, config: Optional[MLflowConfig] = None):
        """
        Initialize experiment tracker.
        
        Args:
            config: MLflow configuration. If None, loads from config manager.
        """
        if config is None:
            config = config_manager.load_config().mlflow
        
        self.config = config
        self.client = MlflowClient(tracking_uri=config.tracking_uri)
        self._setup_experiment()
    
    def _setup_experiment(self) -> None:
        """Set up MLflow experiment."""
        try:
            # Set tracking URI
            mlflow.set_tracking_uri(self.config.tracking_uri)
            
            # Get or create experiment
            experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.config.experiment_name)
                logger.info(f"Created new experiment: {self.config.experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {self.config.experiment_name}")
            
            mlflow.set_experiment(self.config.experiment_name)
            
        except Exception as e:
            logger.error(f"Failed to setup MLflow experiment: {e}")
            raise
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> mlflow.ActiveRun:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for the run
            tags: Tags to associate with the run
            
        Returns:
            Active MLflow run
        """
        return mlflow.start_run(run_name=run_name, tags=tags)
    
    def log_parameters(self, params: Dict[str, Any]) -> None:
        """Log parameters to current run."""
        mlflow.log_params(params)
        logger.info(f"Logged parameters: {list(params.keys())}")
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics to current run."""
        mlflow.log_metrics(metrics)
        logger.info(f"Logged metrics: {list(metrics.keys())}")
    
    def log_model(
        self,
        model: Any,
        model_name: str,
        signature: Optional[mlflow.models.ModelSignature] = None,
        input_example: Optional[pd.DataFrame] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log model to MLflow.
        
        Args:
            model: Trained model object
            model_name: Name for the model
            signature: Model signature (optional)
            input_example: Example input data (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            Model URI
        """
        try:
            model_uri = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=model_name,
                signature=signature,
                input_example=input_example,
                metadata=metadata
            )
            logger.info(f"Logged model: {model_name}")
            return model_uri
        except Exception as e:
            logger.error(f"Failed to log model {model_name}: {e}")
            raise
    
    def log_artifacts(self, artifacts_dir: str, artifact_path: Optional[str] = None) -> None:
        """Log artifacts to current run."""
        mlflow.log_artifacts(artifacts_dir, artifact_path)
        logger.info(f"Logged artifacts from: {artifacts_dir}")
    
    def log_data_info(self, df: pd.DataFrame, data_name: str) -> None:
        """
        Log data information to current run.
        
        Args:
            df: DataFrame to log info about
            data_name: Name for the data
        """
        data_info = {
            f"{data_name}_rows": len(df),
            f"{data_name}_columns": len(df.columns),
            f"{data_name}_memory_usage": df.memory_usage(deep=True).sum()
        }
        
        # Log missing values
        missing_info = df.isnull().sum()
        for col, missing_count in missing_info.items():
            if missing_count > 0:
                data_info[f"{data_name}_{col}_missing"] = missing_count
        
        # Log data types
        for col, dtype in df.dtypes.items():
            data_info[f"{data_name}_{col}_dtype"] = str(dtype)
        
        self.log_metrics(data_info)
    
    def log_feature_importance(
        self,
        feature_names: List[str],
        importance_values: List[float],
        model_name: str
    ) -> None:
        """Log feature importance to current run."""
        importance_dict = dict(zip(feature_names, importance_values))
        
        # Log as metrics
        for feature, importance in importance_dict.items():
            mlflow.log_metric(f"feature_importance_{feature}", importance)
        
        # Log as artifact
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=False)
        
        importance_path = Path("artifacts") / f"{model_name}_feature_importance.csv"
        importance_path.parent.mkdir(parents=True, exist_ok=True)
        importance_df.to_csv(importance_path, index=False)
        
        mlflow.log_artifact(str(importance_path))
        logger.info(f"Logged feature importance for {model_name}")
    
    def log_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
        model_name: str = "model"
    ) -> None:
        """Log confusion matrix as artifact."""
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        cm_path = Path("artifacts") / f"{model_name}_confusion_matrix.png"
        cm_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        mlflow.log_artifact(str(cm_path))
        logger.info(f"Logged confusion matrix for {model_name}")
    
    def register_model(
        self,
        model_uri: str,
        model_name: str,
        version: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Register model in MLflow model registry.
        
        Args:
            model_uri: URI of the logged model
            model_name: Name for the registered model
            version: Model version (optional)
            description: Model description (optional)
            tags: Model tags (optional)
            
        Returns:
            Registered model version
        """
        try:
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                tags=tags
            )
            
            if description:
                self.client.update_model_version(
                    name=model_name,
                    version=registered_model.version,
                    description=description
                )
            
            logger.info(f"Registered model: {model_name} (version {registered_model.version})")
            return registered_model.version
            
        except Exception as e:
            logger.error(f"Failed to register model {model_name}: {e}")
            raise
    
    def get_best_model(
        self,
        model_name: str,
        metric: str = "accuracy",
        ascending: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Get the best model from registry based on metric.
        
        Args:
            model_name: Name of the model
            metric: Metric to use for comparison
            ascending: Whether to sort ascending (for loss metrics)
            
        Returns:
            Best model information or None
        """
        try:
            # Get all versions of the model
            model_versions = self.client.search_model_versions(f"name='{model_name}'")
            
            if not model_versions:
                logger.warning(f"No versions found for model: {model_name}")
                return None
            
            best_model = None
            best_metric_value = None
            
            for version in model_versions:
                # Get run information
                run = self.client.get_run(version.run_id)
                
                if metric in run.data.metrics:
                    metric_value = run.data.metrics[metric]
                    
                    if best_metric_value is None:
                        best_metric_value = metric_value
                        best_model = {
                            'version': version.version,
                            'run_id': version.run_id,
                            'metric_value': metric_value,
                            'model_uri': version.source
                        }
                    else:
                        if (ascending and metric_value < best_metric_value) or \
                           (not ascending and metric_value > best_metric_value):
                            best_metric_value = metric_value
                            best_model = {
                                'version': version.version,
                                'run_id': version.run_id,
                                'metric_value': metric_value,
                                'model_uri': version.source
                            }
            
            if best_model:
                logger.info(f"Best model for {model_name}: version {best_model['version']} "
                           f"({metric}={best_model['metric_value']})")
            
            return best_model
            
        except Exception as e:
            logger.error(f"Failed to get best model for {model_name}: {e}")
            return None
    
    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
        description: Optional[str] = None
    ) -> None:
        """
        Transition model to a new stage.
        
        Args:
            model_name: Name of the model
            version: Model version
            stage: Target stage (Staging, Production, Archived)
            description: Transition description (optional)
        """
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                description=description
            )
            logger.info(f"Transitioned {model_name} v{version} to {stage}")
        except Exception as e:
            logger.error(f"Failed to transition model {model_name} v{version}: {e}")
            raise


# Global experiment tracker instance
experiment_tracker = ExperimentTracker()

