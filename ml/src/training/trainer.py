"""
Comprehensive training pipeline with experiment tracking and evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path
import importlib
import json

from ..config.config_manager import config_manager, FeatureConfig, ModelConfig, TrainingConfig
from ..data.data_loader import DataLoader
from ..experiments.experiment_tracker import experiment_tracker
from ..evaluation.model_evaluator import ModelEvaluator
from ..optimization.hyperparameter_tuner import HyperparameterTuner, PARAM_SPACES
from ..utils.logger import get_logger

logger = get_logger(__name__)


class MLTrainer:
    """Comprehensive ML training pipeline."""
    
    def __init__(
        self,
        task_name: str,
        feature_config: Optional[FeatureConfig] = None,
        model_config: Optional[ModelConfig] = None,
        training_config: Optional[TrainingConfig] = None
    ):
        """
        Initialize ML trainer.
        
        Args:
            task_name: Name of the ML task
            feature_config: Feature configuration
            model_config: Model configuration
            training_config: Training configuration
        """
        self.task_name = task_name
        self.config = config_manager.load_config()
        
        self.feature_config = feature_config or self.config.features[task_name]
        self.model_config = model_config or self.config.models[task_name]
        self.training_config = training_config or self.config.training
        
        self.data_loader = DataLoader()
        self.evaluator = ModelEvaluator()
        self.tuner = HyperparameterTuner()
        
        self.models = {}
        self.results = {}
    
    def load_and_prepare_data(
        self,
        table_name: str = "openfda_events",
        filters: Optional[Dict[str, Any]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Load and prepare data for training.
        
        Args:
            table_name: Name of the table to load
            filters: Optional filters for data loading
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Loading data for task: {self.task_name}")
        
        # Load data
        df = self.data_loader.load_data(table_name, filters=filters)
        
        # Validate data quality
        validation_results = self.data_loader.validate_data_quality(
            df, self.feature_config.target
        )
        
        if not validation_results['is_valid']:
            logger.warning(f"Data quality issues: {validation_results['issues']}")
        
        # Prepare features and target
        features = (self.feature_config.numeric_features + 
                   self.feature_config.categorical_features + 
                   self.feature_config.text_features)
        
        X = df[features]
        y = df[self.feature_config.target]
        
        # Remove rows with missing target
        mask = y.notna()
        X = X[mask]
        y = y[mask]
        
        logger.info(f"Data prepared: {len(X)} samples, {len(features)} features")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.training_config.test_size,
            random_state=self.training_config.random_state,
            stratify=y if self.training_config.stratify else None
        )
        
        logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test")
        
        return X_train, X_test, y_train, y_test
    
    def create_preprocessor(self) -> ColumnTransformer:
        """Create preprocessing pipeline."""
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])
        
        preprocessor = ColumnTransformer([
            ("num", numeric_pipeline, self.feature_config.numeric_features),
            ("cat", categorical_pipeline, self.feature_config.categorical_features)
        ])
        
        return preprocessor
    
    def train_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        optimize_hyperparams: bool = False,
        run_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train a single model with optional hyperparameter optimization.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            optimize_hyperparams: Whether to optimize hyperparameters
            run_name: Name for the MLflow run
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training model: {model_name}")
        
        # Find model configuration
        model_config = None
        for algo in self.model_config.algorithms:
            if algo['name'] == model_name:
                model_config = algo
                break
        
        if model_config is None:
            raise ValueError(f"Model configuration not found: {model_name}")
        
        # Start MLflow run
        with experiment_tracker.start_run(run_name=run_name) as run:
            # Log parameters
            experiment_tracker.log_parameters({
                'task_name': self.task_name,
                'model_name': model_name,
                'algorithm': model_config['class'],
                **model_config['params']
            })
            
            # Create preprocessor
            preprocessor = self.create_preprocessor()
            
            # Get model class
            model_class = self._get_model_class(model_config['class'])
            
            # Hyperparameter optimization
            if optimize_hyperparams and model_name in PARAM_SPACES:
                logger.info(f"Optimizing hyperparameters for {model_name}")
                
                # Create base model for optimization
                base_model = model_class(**model_config['params'])
                
                # Optimize hyperparameters
                optimization_results = self.tuner.optimize_classification(
                    model_class=model_class,
                    X=X_train,
                    y=y_train,
                    param_space=PARAM_SPACES[model_name],
                    study_name=f"{self.task_name}_{model_name}_optimization"
                )
                
                # Use optimized parameters
                model_params = optimization_results['best_params']
                experiment_tracker.log_parameters({
                    f'optimized_{k}': v for k, v in model_params.items()
                })
                
                logger.info(f"Best hyperparameters: {model_params}")
            else:
                model_params = model_config['params']
            
            # Create final model
            model = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", model_class(**model_params))
            ])
            
            # Train model
            experiment_tracker.log_model_training(model_name)
            model.fit(X_train, y_train)
            
            # Evaluate model
            evaluation_results = self.evaluator.evaluate_classification(
                model=model,
                X_test=X_test,
                y_test=y_test,
                X_train=X_train,
                y_train=y_train,
                model_name=model_name
            )
            
            # Log metrics
            experiment_tracker.log_metrics(evaluation_results['metrics'])
            experiment_tracker.log_model_evaluation(
                model_name, evaluation_results['metrics']
            )
            
            # Log feature importance if available
            if hasattr(model.named_steps['classifier'], 'feature_importances_'):
                feature_names = self._get_feature_names(preprocessor, X_train)
                experiment_tracker.log_feature_importance(
                    feature_names=feature_names,
                    importance_values=model.named_steps['classifier'].feature_importances_,
                    model_name=model_name
                )
            
            # Log confusion matrix
            experiment_tracker.log_confusion_matrix(
                y_test.values,
                model.predict(X_test),
                model_name=model_name
            )
            
            # Save model
            model_path = self._save_model(model, model_name)
            
            # Log model
            model_uri = experiment_tracker.log_model(
                model=model,
                model_name=model_name,
                metadata={
                    'task_name': self.task_name,
                    'algorithm': model_config['class'],
                    'training_samples': len(X_train),
                    'test_samples': len(X_test)
                }
            )
            
            # Store results
            self.models[model_name] = model
            self.results[model_name] = {
                'model': model,
                'evaluation': evaluation_results,
                'model_path': model_path,
                'model_uri': model_uri,
                'run_id': run.info.run_id
            }
            
            logger.info(f"Model {model_name} training completed successfully")
            
            return self.results[model_name]
    
    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        optimize_hyperparams: bool = False
    ) -> Dict[str, Any]:
        """
        Train all configured models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            optimize_hyperparams: Whether to optimize hyperparameters
            
        Returns:
            Dictionary with all training results
        """
        logger.info(f"Training all models for task: {self.task_name}")
        
        all_results = {}
        
        for algo_config in self.model_config.algorithms:
            model_name = algo_config['name']
            
            try:
                result = self.train_model(
                    model_name=model_name,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    optimize_hyperparams=optimize_hyperparams
                )
                all_results[model_name] = result
                
            except Exception as e:
                logger.error(f"Failed to train model {model_name}: {e}")
                all_results[model_name] = {'error': str(e)}
        
        # Compare models
        successful_results = [
            result['evaluation'] for result in all_results.values()
            if 'evaluation' in result
        ]
        
        if successful_results:
            comparison_df = self.evaluator.compare_models(
                successful_results,
                primary_metric="accuracy"
            )
            
            logger.info("Model comparison:")
            logger.info(f"\n{comparison_df}")
            
            # Save comparison
            comparison_path = Path(self.config.paths.artifacts) / f"{self.task_name}_model_comparison.csv"
            comparison_df.to_csv(comparison_path, index=False)
            
            all_results['comparison'] = comparison_df
        
        return all_results
    
    def _get_model_class(self, class_path: str):
        """Get model class from string path."""
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    
    def _get_feature_names(self, preprocessor: ColumnTransformer, X: pd.DataFrame) -> List[str]:
        """Get feature names after preprocessing."""
        feature_names = []
        
        # Numeric features
        feature_names.extend(self.feature_config.numeric_features)
        
        # Categorical features (after one-hot encoding)
        for feature in self.feature_config.categorical_features:
            if feature in X.columns:
                unique_values = X[feature].dropna().unique()
                feature_names.extend([f"{feature}_{val}" for val in unique_values])
        
        return feature_names
    
    def _save_model(self, model: Pipeline, model_name: str) -> str:
        """Save model to disk."""
        models_dir = Path(self.config.paths.models)
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = models_dir / f"{self.task_name}_{model_name}.joblib"
        joblib.dump(model, model_path)
        
        logger.info(f"Model saved to: {model_path}")
        return str(model_path)
    
    def run_full_pipeline(
        self,
        table_name: str = "openfda_events",
        filters: Optional[Dict[str, Any]] = None,
        optimize_hyperparams: bool = False
    ) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Args:
            table_name: Name of the table to load
            filters: Optional filters for data loading
            optimize_hyperparams: Whether to optimize hyperparameters
            
        Returns:
            Dictionary with complete pipeline results
        """
        logger.info(f"Starting full training pipeline for task: {self.task_name}")
        
        try:
            # Load and prepare data
            X_train, X_test, y_train, y_test = self.load_and_prepare_data(
                table_name, filters
            )
            
            # Train all models
            results = self.train_all_models(
                X_train, y_train, X_test, y_test, optimize_hyperparams
            )
            
            # Generate summary report
            summary = self._generate_summary_report(results)
            
            logger.info(f"Full pipeline completed for task: {self.task_name}")
            
            return {
                'task_name': self.task_name,
                'data_info': {
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'features': len(X_train.columns)
                },
                'results': results,
                'summary': summary
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed for task {self.task_name}: {e}")
            raise
        finally:
            self.data_loader.close()
    
    def _generate_summary_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary report of training results."""
        successful_models = [
            name for name, result in results.items()
            if 'evaluation' in result
        ]
        
        if not successful_models:
            return {'error': 'No models trained successfully'}
        
        # Get best model
        best_model = None
        best_accuracy = 0
        
        for model_name in successful_models:
            accuracy = results[model_name]['evaluation']['metrics']['accuracy']
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model_name
        
        return {
            'total_models': len(self.model_config.algorithms),
            'successful_models': len(successful_models),
            'best_model': best_model,
            'best_accuracy': best_accuracy,
            'successful_model_names': successful_models
        }

