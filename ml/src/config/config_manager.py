"""
Configuration management for ML pipeline.
Handles loading and validation of configuration files.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Data configuration settings."""
    warehouse_uri: str
    tables: Dict[str, str]


@dataclass
class FeatureConfig:
    """Feature configuration for a specific task."""
    target: str
    numeric_features: list[str]
    categorical_features: list[str]
    text_features: list[str]


@dataclass
class ModelConfig:
    """Model configuration settings."""
    algorithms: list[Dict[str, Any]]


@dataclass
class TrainingConfig:
    """Training configuration settings."""
    test_size: float
    validation_size: float
    random_state: int
    stratify: bool
    cross_validation: Dict[str, Any]


@dataclass
class EvaluationConfig:
    """Evaluation configuration settings."""
    metrics: Dict[str, list[str]]


@dataclass
class MLflowConfig:
    """MLflow configuration settings."""
    tracking_uri: str
    experiment_name: str
    model_registry_name: str


@dataclass
class PathsConfig:
    """Paths configuration settings."""
    data: str
    models: str
    logs: str
    artifacts: str


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str
    format: str
    file: str


@dataclass
class MLConfig:
    """Main ML pipeline configuration."""
    data: DataConfig
    features: Dict[str, FeatureConfig]
    models: Dict[str, ModelConfig]
    training: TrainingConfig
    evaluation: EvaluationConfig
    mlflow: MLflowConfig
    paths: PathsConfig
    logging: LoggingConfig


class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config: Optional[MLConfig] = None
    
    def load_config(self) -> MLConfig:
        """Load and parse configuration file."""
        if self._config is not None:
            return self._config
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        logger.info(f"Loading configuration from {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Expand environment variables
        config_dict = self._expand_env_vars(config_dict)
        
        # Create configuration objects
        self._config = self._create_config_objects(config_dict)
        
        # Validate configuration
        self._validate_config()
        
        logger.info("Configuration loaded successfully")
        return self._config
    
    def _expand_env_vars(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Expand environment variables with optional defaults."""
        def expand_recursive(obj):
            if isinstance(obj, dict):
                return {k: expand_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [expand_recursive(item) for item in obj]
            elif isinstance(obj, str):
                # Handle ${VAR:default} syntax
                if obj.startswith("${") and obj.endswith("}"):
                    body = obj[2:-1]
                    if ":" in body:
                        var, default = body.split(":", 1)
                    else:
                        var, default = body, None
                    return os.environ.get(var, default)
                return os.path.expandvars(obj)
            else:
                return obj

        return expand_recursive(config_dict)

    def _create_config_objects(self, config_dict: Dict[str, Any]) -> MLConfig:
        """Create configuration objects from dictionary."""
        return MLConfig(
            data=DataConfig(**config_dict['data']),
            features={
                name: FeatureConfig(**feat_config)
                for name, feat_config in config_dict['features'].items()
            },
            models={
                name: ModelConfig(**model_config)
                for name, model_config in config_dict['models'].items()
            },
            training=TrainingConfig(**config_dict['training']),
            evaluation=EvaluationConfig(**config_dict['evaluation']),
            mlflow=MLflowConfig(**config_dict['mlflow']),
            paths=PathsConfig(**config_dict['paths']),
            logging=LoggingConfig(**config_dict['logging'])
        )
    
    def _validate_config(self) -> None:
        """Validate configuration settings."""
        if not self._config:
            raise ValueError("Configuration not loaded")
        
        # Validate paths exist or can be created
        for path_name, path_value in self._config.paths.__dict__.items():
            path = Path(path_value)
            if not path.exists():
                logger.info(f"Creating directory: {path}")
                path.mkdir(parents=True, exist_ok=True)
        
        # Validate test and validation sizes
        total_size = self._config.training.test_size + self._config.training.validation_size
        if total_size >= 1.0:
            raise ValueError(f"Test size + validation size ({total_size}) must be < 1.0")
        
        logger.info("Configuration validation passed")
    
    def get_feature_config(self, task_name: str) -> FeatureConfig:
        """Get feature configuration for a specific task."""
        if task_name not in self._config.features:
            raise KeyError(f"Feature configuration not found for task: {task_name}")
        return self._config.features[task_name]
    
    def get_model_config(self, task_name: str) -> ModelConfig:
        """Get model configuration for a specific task."""
        if task_name not in self._config.models:
            raise KeyError(f"Model configuration not found for task: {task_name}")
        return self._config.models[task_name]


# Global configuration instance
config_manager = ConfigManager()

