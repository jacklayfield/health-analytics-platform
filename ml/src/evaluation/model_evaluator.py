"""
Comprehensive model evaluation framework.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

from ..config.config_manager import config_manager, EvaluationConfig, TrainingConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation framework."""
    
    def __init__(
        self,
        eval_config: Optional[EvaluationConfig] = None,
        training_config: Optional[TrainingConfig] = None
    ):
        """
        Initialize model evaluator.
        
        Args:
            eval_config: Evaluation configuration
            training_config: Training configuration
        """
        if eval_config is None:
            eval_config = config_manager.load_config().evaluation
        if training_config is None:
            training_config = config_manager.load_config().training
        
        self.eval_config = eval_config
        self.training_config = training_config
    
    def evaluate_classification(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.Series] = None,
        model_name: str = "model"
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation for classification models.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            X_train: Training features (for cross-validation)
            y_train: Training targets (for cross-validation)
            model_name: Name of the model
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating classification model: {model_name}")
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = self._calculate_classification_metrics(
            y_test, y_pred, y_pred_proba
        )
        
        # Cross-validation if training data provided
        cv_scores = None
        if X_train is not None and y_train is not None:
            cv_scores = self._cross_validate_classification(
                model, X_train, y_train
            )
            metrics.update(cv_scores)
        
        # Generate detailed report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Create evaluation summary
        evaluation_results = {
            'model_name': model_name,
            'task_type': 'classification',
            'metrics': metrics,
            'classification_report': report,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'test_size': len(X_test),
            'num_classes': len(np.unique(y_test))
        }
        
        # Add cross-validation results if available
        if cv_scores:
            evaluation_results['cross_validation'] = cv_scores
        
        logger.info(f"Evaluation completed for {model_name}")
        return evaluation_results
    
    def evaluate_regression(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.Series] = None,
        model_name: str = "model"
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation for regression models.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            X_train: Training features (for cross-validation)
            y_train: Training targets (for cross-validation)
            model_name: Name of the model
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating regression model: {model_name}")
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = self._calculate_regression_metrics(y_test, y_pred)
        
        # Cross-validation if training data provided
        cv_scores = None
        if X_train is not None and y_train is not None:
            cv_scores = self._cross_validate_regression(
                model, X_train, y_train
            )
            metrics.update(cv_scores)
        
        # Create evaluation summary
        evaluation_results = {
            'model_name': model_name,
            'task_type': 'regression',
            'metrics': metrics,
            'test_size': len(X_test),
            'residuals': (y_test - y_pred).tolist()
        }
        
        # Add cross-validation results if available
        if cv_scores:
            evaluation_results['cross_validation'] = cv_scores
        
        logger.info(f"Evaluation completed for {model_name}")
        return evaluation_results
    
    def _calculate_classification_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate classification metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, (p, r, f) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
            metrics[f'precision_class_{i}'] = p
            metrics[f'recall_class_{i}'] = r
            metrics[f'f1_class_{i}'] = f
        
        # ROC AUC if probabilities available
        if y_pred_proba is not None:
            if y_pred_proba.shape[1] == 2:  # Binary classification
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            else:  # Multi-class
                metrics['roc_auc_macro'] = roc_auc_score(
                    y_true, y_pred_proba, multi_class='ovr', average='macro'
                )
        
        return metrics
    
    def _calculate_regression_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate regression metrics."""
        metrics = {}
        
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Additional metrics
        metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return metrics
    
    def _cross_validate_classification(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, Any]:
        """Perform cross-validation for classification."""
        cv_scores = {}
        
        # Use stratified k-fold for classification
        cv = StratifiedKFold(
            n_splits=self.training_config.cross_validation['cv_folds'],
            shuffle=True,
            random_state=self.training_config.random_state
        )
        
        # Calculate CV scores for each metric
        for metric in self.training_config.cross_validation['scoring']:
            scores = cross_val_score(
                model, X, y, cv=cv, scoring=metric, n_jobs=-1
            )
            cv_scores[f'cv_{metric}_mean'] = scores.mean()
            cv_scores[f'cv_{metric}_std'] = scores.std()
        
        return cv_scores
    
    def _cross_validate_regression(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, Any]:
        """Perform cross-validation for regression."""
        cv_scores = {}
        
        # Use k-fold for regression
        cv = KFold(
            n_splits=self.training_config.cross_validation['cv_folds'],
            shuffle=True,
            random_state=self.training_config.random_state
        )
        
        # Calculate CV scores for each metric
        for metric in self.eval_config.metrics['regression']:
            scores = cross_val_score(
                model, X, y, cv=cv, scoring=metric, n_jobs=-1
            )
            cv_scores[f'cv_{metric}_mean'] = scores.mean()
            cv_scores[f'cv_{metric}_std'] = scores.std()
        
        return cv_scores
    
    def compare_models(
        self,
        model_results: List[Dict[str, Any]],
        primary_metric: str = "accuracy"
    ) -> pd.DataFrame:
        """
        Compare multiple model evaluation results.
        
        Args:
            model_results: List of evaluation results from different models
            primary_metric: Primary metric for comparison
            
        Returns:
            DataFrame with model comparison
        """
        comparison_data = []
        
        for result in model_results:
            model_info = {
                'model_name': result['model_name'],
                'task_type': result['task_type'],
                'test_size': result['test_size']
            }
            
            # Add metrics
            model_info.update(result['metrics'])
            
            comparison_data.append(model_info)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by primary metric
        if primary_metric in comparison_df.columns:
            comparison_df = comparison_df.sort_values(
                primary_metric, ascending=False
            )
        
        logger.info(f"Model comparison completed. Best model by {primary_metric}: "
                   f"{comparison_df.iloc[0]['model_name']}")
        
        return comparison_df
    
    def generate_evaluation_report(
        self,
        evaluation_results: Dict[str, Any],
        output_dir: str = "artifacts"
    ) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            evaluation_results: Evaluation results
            output_dir: Output directory for report
            
        Returns:
            Path to generated report
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        model_name = evaluation_results['model_name']
        report_path = output_path / f"{model_name}_evaluation_report.json"
        
        # Save detailed results
        with open(report_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        logger.info(f"Evaluation report saved to: {report_path}")
        return str(report_path)
    
    def plot_evaluation_metrics(
        self,
        evaluation_results: Dict[str, Any],
        output_dir: str = "artifacts"
    ) -> List[str]:
        """
        Generate evaluation plots.
        
        Args:
            evaluation_results: Evaluation results
            output_dir: Output directory for plots
            
        Returns:
            List of generated plot paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        model_name = evaluation_results['model_name']
        plot_paths = []
        
        # Confusion Matrix
        if 'confusion_matrix' in evaluation_results:
            cm = np.array(evaluation_results['confusion_matrix'])
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            cm_path = output_path / f"{model_name}_confusion_matrix.png"
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(str(cm_path))
        
        # Metrics comparison (if multiple models)
        if 'metrics' in evaluation_results:
            metrics = evaluation_results['metrics']
            
            # Plot key metrics
            key_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
            available_metrics = [m for m in key_metrics if m in metrics]
            
            if available_metrics:
                plt.figure(figsize=(10, 6))
                metric_values = [metrics[m] for m in available_metrics]
                
                bars = plt.bar(available_metrics, metric_values)
                plt.title(f'Model Performance Metrics - {model_name}')
                plt.ylabel('Score')
                plt.ylim(0, 1)
                
                # Add value labels on bars
                for bar, value in zip(bars, metric_values):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{value:.3f}', ha='center', va='bottom')
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                metrics_path = output_path / f"{model_name}_metrics.png"
                plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
                plt.close()
                plot_paths.append(str(metrics_path))
        
        logger.info(f"Generated {len(plot_paths)} evaluation plots for {model_name}")
        return plot_paths

