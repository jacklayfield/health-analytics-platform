"""
Hyperparameter optimization using Optuna.
"""

import optuna
import optuna.integration.mlflow
from typing import Dict, Any, List, Optional, Callable, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import make_scorer
import joblib
from pathlib import Path

from ..config.config_manager import config_manager, TrainingConfig
from ..utils.logger import get_logger
from ..experiments.experiment_tracker import experiment_tracker

logger = get_logger(__name__)


class HyperparameterTuner:
    """Hyperparameter optimization using Optuna."""
    
    def __init__(
        self,
        training_config: Optional[TrainingConfig] = None,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        direction: str = "maximize"
    ):
        """
        Initialize hyperparameter tuner.
        
        Args:
            training_config: Training configuration
            n_trials: Number of optimization trials
            timeout: Timeout in seconds (optional)
            direction: Optimization direction ("maximize" or "minimize")
        """
        if training_config is None:
            training_config = config_manager.load_config().training
        
        self.training_config = training_config
        self.n_trials = n_trials
        self.timeout = timeout
        self.direction = direction
        
        # Create study
        self.study = None
        self.best_params = None
        self.best_score = None
    
    def optimize_classification(
        self,
        model_class: type,
        X: pd.DataFrame,
        y: pd.Series,
        param_space: Dict[str, Any],
        scoring: str = "accuracy",
        cv_folds: Optional[int] = None,
        study_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters for classification model.
        
        Args:
            model_class: Model class to optimize
            X: Training features
            y: Training targets
            param_space: Parameter search space
            scoring: Scoring metric
            cv_folds: Number of CV folds (uses config default if None)
            study_name: Name for the study (optional)
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting hyperparameter optimization for {model_class.__name__}")
        
        if cv_folds is None:
            cv_folds = self.training_config.cross_validation['cv_folds']
        
        # Create study
        study_name = study_name or f"{model_class.__name__}_optimization"
        self.study = optuna.create_study(
            direction=self.direction,
            study_name=study_name
        )
        
        # Define objective function
        def objective(trial):
            # Sample parameters
            params = self._sample_parameters(trial, param_space)
            
            # Create model with sampled parameters
            model = model_class(**params)
            
            # Cross-validation
            cv = StratifiedKFold(
                n_splits=cv_folds,
                shuffle=True,
                random_state=self.training_config.random_state
            )
            
            scores = cross_val_score(
                model, X, y, cv=cv, scoring=scoring, n_jobs=-1
            )
            
            return scores.mean()
        
        # Optimize
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout
        )
        
        # Get results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        logger.info(f"Optimization completed. Best score: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': len(self.study.trials),
            'study_name': study_name
        }
    
    def optimize_regression(
        self,
        model_class: type,
        X: pd.DataFrame,
        y: pd.Series,
        param_space: Dict[str, Any],
        scoring: str = "neg_mean_squared_error",
        cv_folds: Optional[int] = None,
        study_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters for regression model.
        
        Args:
            model_class: Model class to optimize
            X: Training features
            y: Training targets
            param_space: Parameter search space
            scoring: Scoring metric
            cv_folds: Number of CV folds (uses config default if None)
            study_name: Name for the study (optional)
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting hyperparameter optimization for {model_class.__name__}")
        
        if cv_folds is None:
            cv_folds = self.training_config.cross_validation['cv_folds']
        
        # Create study
        study_name = study_name or f"{model_class.__name__}_optimization"
        self.study = optuna.create_study(
            direction=self.direction,
            study_name=study_name
        )
        
        # Define objective function
        def objective(trial):
            # Sample parameters
            params = self._sample_parameters(trial, param_space)
            
            # Create model with sampled parameters
            model = model_class(**params)
            
            # Cross-validation
            cv = KFold(
                n_splits=cv_folds,
                shuffle=True,
                random_state=self.training_config.random_state
            )
            
            scores = cross_val_score(
                model, X, y, cv=cv, scoring=scoring, n_jobs=-1
            )
            
            return scores.mean()
        
        # Optimize
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout
        )
        
        # Get results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        logger.info(f"Optimization completed. Best score: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': len(self.study.trials),
            'study_name': study_name
        }
    
    def _sample_parameters(
        self,
        trial: optuna.Trial,
        param_space: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Sample parameters from search space."""
        params = {}
        
        for param_name, param_config in param_space.items():
            if isinstance(param_config, dict):
                param_type = param_config.get('type', 'uniform')
                
                if param_type == 'uniform':
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high']
                    )
                elif param_type == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config['low'],
                        param_config['high']
                    )
                elif param_type == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config['choices']
                    )
                elif param_type == 'loguniform':
                    params[param_name] = trial.suggest_loguniform(
                        param_name,
                        param_config['low'],
                        param_config['high']
                    )
            else:
                # Direct value
                params[param_name] = param_config
        
        return params
    
    def optimize_with_mlflow(
        self,
        model_class: type,
        X: pd.DataFrame,
        y: pd.Series,
        param_space: Dict[str, Any],
        task_type: str = "classification",
        scoring: str = "accuracy",
        run_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters with MLflow integration.
        
        Args:
            model_class: Model class to optimize
            X: Training features
            y: Training targets
            param_space: Parameter search space
            task_type: Type of task ("classification" or "regression")
            scoring: Scoring metric
            run_name: Name for the MLflow run
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting MLflow-integrated optimization for {model_class.__name__}")
        
        # Create MLflow callback
        mlflow_callback = optuna.integration.mlflow.MLflowCallback(
            tracking_uri=experiment_tracker.config.tracking_uri,
            metric_name=scoring
        )
        
        # Create study with MLflow callback
        study_name = f"{model_class.__name__}_mlflow_optimization"
        self.study = optuna.create_study(
            direction=self.direction,
            study_name=study_name
        )
        
        # Define objective function
        def objective(trial):
            params = self._sample_parameters(trial, param_space)
            model = model_class(**params)
            
            if task_type == "classification":
                cv = StratifiedKFold(
                    n_splits=self.training_config.cross_validation['cv_folds'],
                    shuffle=True,
                    random_state=self.training_config.random_state
                )
            else:
                cv = KFold(
                    n_splits=self.training_config.cross_validation['cv_folds'],
                    shuffle=True,
                    random_state=self.training_config.random_state
                )
            
            scores = cross_val_score(
                model, X, y, cv=cv, scoring=scoring, n_jobs=-1
            )
            
            return scores.mean()
        
        # Optimize with MLflow callback
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            callbacks=[mlflow_callback]
        )
        
        # Get results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        logger.info(f"MLflow optimization completed. Best score: {self.best_score:.4f}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': len(self.study.trials),
            'study_name': study_name
        }
    
    def get_optimization_history(self) -> pd.DataFrame:
        """Get optimization history as DataFrame."""
        if self.study is None:
            raise ValueError("No optimization study found. Run optimization first.")
        
        trials_data = []
        for trial in self.study.trials:
            trial_data = {
                'trial_number': trial.number,
                'value': trial.value,
                'state': trial.state.name
            }
            trial_data.update(trial.params)
            trials_data.append(trial_data)
        
        return pd.DataFrame(trials_data)
    
    def save_study(self, filepath: str) -> None:
        """Save optimization study to file."""
        if self.study is None:
            raise ValueError("No optimization study found.")
        
        joblib.dump(self.study, filepath)
        logger.info(f"Study saved to: {filepath}")
    
    def load_study(self, filepath: str) -> None:
        """Load optimization study from file."""
        self.study = joblib.load(filepath)
        logger.info(f"Study loaded from: {filepath}")


# Predefined parameter spaces for common models
PARAM_SPACES = {
    'LogisticRegression': {
        'C': {'type': 'loguniform', 'low': 0.001, 'high': 100},
        'max_iter': {'type': 'int', 'low': 100, 'high': 2000},
        'class_weight': {'type': 'categorical', 'choices': [None, 'balanced']}
    },
    
    'RandomForestClassifier': {
        'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
        'max_depth': {'type': 'int', 'low': 3, 'high': 20},
        'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
        'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
        'class_weight': {'type': 'categorical', 'choices': [None, 'balanced']}
    },
    
    'GradientBoostingClassifier': {
        'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
        'learning_rate': {'type': 'uniform', 'low': 0.01, 'high': 0.3},
        'max_depth': {'type': 'int', 'low': 2, 'high': 10},
        'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
        'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10}
    },
    
    'XGBClassifier': {
        'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
        'learning_rate': {'type': 'uniform', 'low': 0.01, 'high': 0.3},
        'max_depth': {'type': 'int', 'low': 2, 'high': 10},
        'subsample': {'type': 'uniform', 'low': 0.6, 'high': 1.0},
        'colsample_bytree': {'type': 'uniform', 'low': 0.6, 'high': 1.0}
    }
}

